# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

import os
from os.path import join as pjoin
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse

import warnings
warnings.filterwarnings("ignore")

# import torch

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence

from core.util.preprocessor.base import Preprocessor


class ArgoversePreprocessor(Preprocessor):
    def __init__(self, root_dir, algo="tnt", obs_horizon=20, obs_range=35, pred_horizon=30):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range)

        self.obs_horizon = obs_horizon
        self.obs_range = obs_range
        self.pred_horizon = pred_horizon
        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        content = os.listdir(self.root_dir)
        self.folders = [folder for folder in content if os.path.isdir(pjoin(self.root_dir, folder))]

        self.am = ArgoverseMap()
        self.loaders = []
        for folder in self.folders:
            self.loaders.append(ArgoverseForecastingLoader(pjoin(self.root_dir, folder)))

    def generate(self):
        for i, loader in enumerate(self.loaders):
            # add for debug
            if "test" in self.folders[i]:
                continue
            for f_path in tqdm(loader.seq_list, desc=f"Processing {self.folders[i]}..."):
                seq = loader.get(f_path)
                path, seq_f_name_ext = os.path.split(f_path)
                seq_f_name, ext = os.path.splitext(seq_f_name_ext)

                yield self.folders[i], seq_f_name, copy.deepcopy(seq.seq_df)

    def process(self, dataframe: pd.DataFrame,  map_feat=True):
        data = self.read_argo_data(dataframe)
        data = self.get_obj_feats(data)

        data['graph'] = self.get_lane_graph(data)
        return data

    def __len__(self):
        num_seq = 0
        for loader in self.loaders:
            num_seq += len(loader.seq_list)
        return num_seq

    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        city = df["CITY_NAME"].values[0]

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data

    def get_obj_feats(self, data):
        # get the origin and compute the oritentation of the target agent
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32)
        pre = data['trajs'][0][self.obs_horizon-2] - orig
        theta = np.pi - np.arctan2(pre[1], pre[0])
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        # get the target candidates and candidate gt
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)
        ctr_line_candts, _ = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'])
        tar_candts = self.lane_candidate_sampling(ctr_line_candts, viz=False)
        candts_gt, offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])

        # self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)

        if not np.all(offse_gt < self.LANE_WIDTH[data['city']]):
            self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue

            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1

            obs_mask = step < self.obs_horizon
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]

            for i in range(len(step)):
                # if step[i] == 19 - (len(step) - 1) + i:
                if step[i] == self.obs_horizon - len(step) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = np.matmul(rot, (tar_candts - orig.reshape(-1, 2)).T).T
        data['gt_candts'] = candts_gt
        data['gt_offset'] = offse_gt
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


if __name__ == "__main__":
    root = "/media/Data/autonomous_driving/Argoverse"
    raw_dir = os.path.join(root, "raw_data")
    # inter_dir = os.path.join(root, "intermediate")
    interm_dir = "/home/jb/projects/Data/traj_pred/interm_tnt_n_s_0717"
    argoverse_processor = ArgoversePreprocessor(raw_dir)

    for s_name, f_name, df in argoverse_processor.generate():
            argoverse_processor.process_and_save(df, s_name, f_name, interm_dir)
