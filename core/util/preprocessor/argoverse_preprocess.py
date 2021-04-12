# About: script to pre-process argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.01.30

import os
from os.path import join as pjoin
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

from core.util.preprocessor.base import Preprocessor
from core.util.preprocessor.object_utils import is_track_stationary

DEBUG = True


class ArgoversePreprocessor(Preprocessor):
    def __init__(self, root_dir, algo="tnt", obs_horizon=20, obs_range=30):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range)

        self.obs_horizon = obs_horizon
        self.obs_range = obs_range
        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        content = os.listdir(self.root_dir)
        self.folders = [folder for folder in content if os.path.isdir(pjoin(self.root_dir, folder))]

        self.map = ArgoverseMap()
        self.loaders = []
        for folder in self.folders:
            self.loaders.append(ArgoverseForecastingLoader(pjoin(self.root_dir, folder)))

    def __len__(self):
        num_seq = 0
        for loader in self.loaders:
            num_seq += len(loader.seq_list)
        return num_seq

    def generate(self):
        for i, loader in enumerate(self.loaders):
            # add for debug
            if "test" in self.folders[i] and DEBUG:
                continue
            for f_path in tqdm(loader.seq_list, desc=f"Processing {self.folders[i]}..."):
                seq = loader.get(f_path)
                path, seq_f_name_ext = os.path.split(f_path)
                seq_f_name, ext = os.path.splitext(seq_f_name_ext)

                yield self.folders[i], seq_f_name, seq.seq_df

    def extract_feature(self, dataframe: pd.DataFrame, map_feat=True):
        # normalize timestamps
        dataframe['TIMESTAMP'] -= np.min(dataframe['TIMESTAMP'].values)

        # select agent trajectory
        agent_df = dataframe[dataframe.OBJECT_TYPE == "AGENT"].sort_values(by="TIMESTAMP")
        norm_center = agent_df[['X', 'Y']].values[self.obs_horizon - 1]
        seq_ts = np.unique(agent_df['TIMESTAMP'].values)

        if not isinstance(agent_df, pd.DataFrame):
            return None, None, None         # return None if no agent in the sequence
        obj_df = dataframe[dataframe.OBJECT_TYPE != "AGENT"].sort_values(by="TIMESTAMP") # remove "AGENT from dataframe"
        obj_df = obj_df[obj_df['TIMESTAMP'] <= seq_ts[self.obs_horizon-1]]  # remove object record after observation

        # include object data within the detection range
        obj_feats, obj_df = self.__extract_obj_feat(obj_df, agent_df, norm_center)

        lane_feats = None
        if map_feat:
            # include lane data within the detection range
            lane_feats = self.__extract_lane_feat(agent_df, obj_df, norm_center)

        # extract the feature for the agent traj
        agent_feats = self.__extract_agent_feat(agent_df, norm_center)

        return agent_feats, obj_feats, lane_feats

    def encode_feature(self, *feats):
        # check if the features are valid
        assert len(feats) == 3, "[ArgoversePreprocessor]: Missing feature..."
        [agent_feature, obj_feature_ls, lane_feature_ls] = feats

        polyline_id = 0
        traj_id2mask, lane_id2mask = {}, {}
        gt = agent_feature[-1]
        traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 9))

        # encoding agent feature
        pre_traj_len = traj_nd.shape[0]
        agent_len = agent_feature[0].shape[0]
        agent_nd = np.hstack((agent_feature[0],                         # (xs, ys, xe, ye)
                              np.ones((agent_len, 1)),                  # object type, 1
                              agent_feature[2].reshape((-1, 1)),        # timestamp
                              np.ones((agent_len, 1)) * polyline_id))   # polyline id
        assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

        offset_gt = self.__trans_gt_offset_format(gt)
        traj_nd = np.vstack((traj_nd, agent_nd))
        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

        # encoding obj feature
        for obj_feature in obj_feature_ls:
            obj_len = obj_feature[0].shape[0]
            # assert obj_feature[2].shape[0] == obj_len, f"obs_len of obj is {obj_len}"
            if not obj_feature[2].shape[0] == obj_len:
                from pdb import set_trace;set_trace()
            obj_nd = np.hstack((obj_feature[0],                         # (xs, ys, xe, ye)
                                np.zeros((obj_len, 1)),                 # object type, 0
                                obj_feature[2].reshape((-1, 1)),        # timestamp
                                np.ones((obj_len, 1)) * polyline_id))   # polyline id
            assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
            traj_nd = np.vstack((traj_nd, obj_nd))

            traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
            pre_traj_len = traj_nd.shape[0]
            polyline_id += 1

        # now the features are:
        # (xs, ys, xe, ye, obejct_type, timestamp(avg_for_start_end?),polyline_id) for object
        # change object features to (xs, ys, xe, ye, timestamp, NULL, NULL, NULL, NULL, polyline_id)
        traj_nd = np.hstack(
            [traj_nd, np.zeros((traj_nd.shape[0], 4), dtype=traj_nd.dtype)])
        traj_nd = traj_nd[:, [0, 1, 2, 3, 5, 7, 8, 9, 10, 6]]

        # encodeing lane feature
        if lane_feature_ls:
            pre_lane_len = lane_nd.shape[0]
            for lane_feature in lane_feature_ls:
                l_lane_len = lane_feature[0].shape[0]
                l_lane_nd = np.hstack((
                    lane_feature[0],                                # (xs, ys, zs, xe, ye, ze)
                    (lane_feature[2]) * np.ones((l_lane_len, 1)),   # traffic control
                    (lane_feature[3]) * np.ones((l_lane_len, 1)),   # is intersaction
                    np.ones((l_lane_len, 1)) * polyline_id          # polyline id
                ))
                assert l_lane_nd.shape[1] == 9, "obj_traj feature dim 1 is not correct"
                lane_nd = np.vstack((lane_nd, l_lane_nd))
                lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
                _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
                pre_lane_len = lane_nd.shape[0]
                polyline_id += 1

                r_lane_len = lane_feature[1].shape[0]
                r_lane_nd = np.hstack((
                    lane_feature[1],                                # (xs, ys, zs, xe, ye, ze)
                    (lane_feature[2]) * np.ones((l_lane_len, 1)),   # traffic control
                    (lane_feature[3]) * np.ones((l_lane_len, 1)),   # is intersaction
                    np.ones((r_lane_len, 1)) * polyline_id          # polyline id
                ))
                assert r_lane_nd.shape[1] == 9, "obj_traj feature dim 1 is not correct"
                lane_nd = np.vstack((lane_nd, r_lane_nd))
                lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
                _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
                pre_lane_len = lane_nd.shape[0]
                polyline_id += 1

                assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"

            # FIXME: handling `nan` in lane_nd
            col_mean = np.nanmean(lane_nd, axis=0)
            if np.isnan(col_mean).any():
                lane_nd[:, 2].fill(.0)
                lane_nd[:, 5].fill(.0)
            else:
                inds = np.where(np.isnan(lane_nd))
                lane_nd[inds] = np.take(col_mean, inds[1])

            # (xs, ys, zs, xe, ye, ze, polyline_id) for lanes
            # change lanes feature to xs, ys, xe, ye, NULL, zs, ze, traffic_control, is_intersection, polyline_id)
            lane_nd = np.hstack(
                [lane_nd, np.zeros((lane_nd.shape[0], 1), dtype=lane_nd.dtype)])
            lane_nd = lane_nd[:, [0, 1, 3, 4, 9, 2, 5, 6, 7, 8]]

            # don't ignore the id
            polyline_features = np.vstack((traj_nd, lane_nd))
        else:
            polyline_features = traj_nd

        data = [[polyline_features.astype(np.float32),
                 offset_gt, agent_feature[-5], agent_feature[-4], agent_feature[-3], agent_feature[-2],
                 traj_id2mask, lane_id2mask, traj_nd.shape[0], lane_nd.shape[0]]]

        return pd.DataFrame(
            data,
            columns=["POLYLINE_FEATURES",
                     "GT", "CANDIDATES", "CANDIDATE_GT", "OFFSET_GT", "TARGET_GT",
                     "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK", "TARJ_LEN", "LANE_LEN"]
        )

    def __extract_agent_feat(self, agent_df, norm_center):
        xys, gt_xys = agent_df[["X", "Y"]].values[:self.obs_horizon], agent_df[["X", "Y"]].values[self.obs_horizon:]
        xys = self.__norm_and_vec__(xys, norm_center)

        ts = agent_df['TIMESTAMP'].values[:self.obs_horizon]
        ts = (ts[:-1] + ts[1:]) / 2

        candidates = self.uniform_candidate_sampling(50)
        gt_xys -= norm_center  # normalize to last observed timestamp point of agent

        # handle the gt
        if len(gt_xys) > 0:
            # todo: adjust the range for candidate sampling, now radian of 50 meter, 900 samples
            candidate_gt, offset_gt = self.get_candidate_gt(candidates, gt_xys[-1, :])
        else:
            candidate_gt, offset_gt = None, None

        return [xys, agent_df['OBJECT_TYPE'].iloc[0], ts, agent_df['TRACK_ID'].iloc[0],
                candidates, candidate_gt, offset_gt, gt_xys[-1, :], gt_xys]

    def __extract_obj_feat(self, obj_df, agnt_df, norm_center):
        obj_feat_ls = []
        for track_id, obj_sub_df in obj_df.groupby("TRACK_ID"):
            # skip object with timestamps less than obs_horizon
            if len(obj_sub_df) < self.obs_horizon or is_track_stationary(obj_sub_df):
                obj_df = obj_df[obj_df["TRACK_ID"] != track_id]
                continue

            xys = obj_sub_df[['X', 'Y']].values
            ts = obj_sub_df["TIMESTAMP"].values

            # check if there exist a ts that the obj is within the detection range of agent
            agnt_ts = agnt_df["TIMESTAMP"].values
            ids = np.concatenate([np.where(agnt_ts == t)[0] for t in ts])
            agnt_xys = agnt_df[['X', 'Y']].values[ids]
            diff = xys - agnt_xys
            dis = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
            if not np.any(dis <= self.obs_range):
                continue                            # skip this obj if it is not within the range for one single ts

            xys = self.__norm_and_vec__(xys, norm_center)
            ts = (ts[:-1] + ts[1:]) / 2

            obj_feat_ls.append(
                [xys, obj_sub_df['OBJECT_TYPE'].iloc[0], ts, track_id]
            )
        return obj_feat_ls, obj_df

    def __extract_lane_feat(self, agent_df, obj_df, norm_center):
        city_name = agent_df["CITY_NAME"].values[0]
        # include traj lane ids
        lane_ids = self.__get_lane_ids_base_traj(agent_df, self.obs_range)
        for _, obj_df in obj_df.groupby("TRACK_ID"):
            ids = self.__get_lane_ids_base_traj(obj_df, self.obs_range)
            lane_ids = np.hstack((lane_ids, ids))
        lane_ids_array = np.unique(lane_ids)

        lane_feat_ls = []
        for lane_id in lane_ids_array:
            traffic_control = self.map.lane_has_traffic_control_measure(lane_id, city_name)
            is_intersection = self.map.lane_is_in_intersection(lane_id, city_name)

            centerlane = self.map.get_lane_segment_centerline(lane_id, city_name)
            # normalize to last observed timestamp point of agent
            centerlane[:, :2] -= norm_center
            halluc_lane_1, halluc_lane_2 = self.__get_halluc_lane(centerlane, city_name)

            lane_feat_ls.append(
                [halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id]
            )

        return lane_feat_ls

    @staticmethod
    def __norm_and_vec__(traj, norm_center):
        """
        normalize the trajectory coordinate and vectorize it
        :param traj: np.array, list of points indicating a trajectory [x, y, z] or [x, y]
        :param norm_center: np.array, coordinate of centralized origen
        """
        traj_norm = traj - norm_center
        return np.hstack((traj_norm[:-1], traj_norm[1:]))

    def __get_halluc_lane(self, centerlane, city_name):
        """
        return left & right lane based on centerline
        args:
        returns:
            doubled_left_halluc_lane, doubled_right_halluc_lane, shaped in (N-1, 3)
        """
        if centerlane.shape[0] <= 1:
            raise ValueError('shape of centerlane error.')

        half_width = self.LANE_WIDTH[city_name] / 2
        rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
        halluc_lane_1, halluc_lane_2 = np.empty((0, centerlane.shape[1]*2)), np.empty((0, centerlane.shape[1]*2))
        for i in range(centerlane.shape[0]-1):
            st, en = centerlane[i][:2], centerlane[i+1][:2]
            dx = en - st
            norm = np.linalg.norm(dx)
            e1, e2 = rotate_quat @ dx / norm, rotate_quat.T @ dx / norm
            lane_1 = np.hstack(
                (st + e1 * half_width, centerlane[i][2], en + e1 * half_width, centerlane[i+1][2]))
            lane_2 = np.hstack(
                (st + e2 * half_width, centerlane[i][2], en + e2 * half_width, centerlane[i+1][2]))
            # print(halluc_lane_1, )
            halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
            halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
        return halluc_lane_1, halluc_lane_2

    def __get_lane_ids_base_traj(self, traj_df, lane_radius=20.0):
        """
        get corresponding lane ids based on trajectory
        :param am: Argoverse map object
        :param traj_df: DataFrame, trajectory dataframe
        :param lane_radius: float, the radius to include the lane
        :return:np.array, the related lane ids
        """
        # todo: change the get lane ids method to fit varaible traj length
        # get lane ids at the start
        traj_len = traj_df.shape[0]

        city_name = traj_df["CITY_NAME"].values[0]
        query_x, query_y = traj_df[['X', 'Y']].values[0]
        lane_ids_str = self.map.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

        # get lane ids in the middle
        query_x, query_y = traj_df[['X', 'Y']].values[int(traj_len / 2)]
        lane_ids_mid = self.map.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

        # get lane ids at the end
        query_x, query_y = traj_df[['X', 'Y']].values[-1]
        lane_ids_end = self.map.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

        # merge all the lane ids and remove duplicate
        return np.unique(lane_ids_str + lane_ids_mid + lane_ids_end)

    @staticmethod
    def __trans_gt_offset_format(gt):
        """
        Our predicted trajectories are parameterized as per-stepcoordinate offsets,
        starting from the last observed location.We rotate the coordinate system based on the heading of
        the target vehicle at the last observed location.
        """
        assert gt.shape == (30, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"

        # for test, no gt, just return a (0, 2) ndarray
        if gt.shape == (0, 2):
            return gt

        offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))
        assert (offset_gt.cumsum(axis=0) -gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) -gt).sum()}"

        return offset_gt

    def process_and_save(self, dataframe: pd.DataFrame, set_name, file_name, dir_=None, map_feat=True):
        enc_df = self.process(dataframe, map_feat)
        self.save(enc_df, set_name, file_name, dir_)


if __name__ == "__main__":
    root = "/media/Data/autonomous_driving/Argoverse"
    raw_dir = os.path.join(root, "raw_data")
    inter_dir = os.path.join(root, "intermediate")
    argoverse_processor = ArgoversePreprocessor(raw_dir)

    if not DEBUG:
        pool = Pool(16)
        # todo: multi-thread processing
        for s_name, f_name, df in argoverse_processor.generate():
            pool.apply_async(func=argoverse_processor.process_and_save,
                             args=(df, s_name, f_name, inter_dir))

        pool.close()
        pool.join()

    else:
        for s_name, f_name, df in argoverse_processor.generate():
            argoverse_processor.process_and_save(df, s_name, f_name)


