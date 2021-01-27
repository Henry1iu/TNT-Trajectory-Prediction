#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

from core.util.config import color_dict
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from core.util.object_utils import get_nearby_moving_obj_feature_ls
from core.util.lane_utils import get_nearby_lane_feature_ls, get_halluc_lane
from core.util.viz_utils import show_doubled_lane, show_traj
from core.util.agent_utils import get_agent_feature_ls
from core.util.viz_utils import *
import pdb


def compute_feature_for_one_seq(traj_df: pd.DataFrame, am: ArgoverseMap, obs_len: int = 20, lane_radius: int = 5, obj_radius: int = 10, viz: bool = False, mode='rect', query_bbox=[-100, 100, -100, 100]) -> List[List]:
    """
    return lane & track features
    args:
        mode: 'rect' or 'nearby'
    returns:
        agent_feature_ls:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        norm_center np.ndarray: (2, )
    """
    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    seq_len = seq_ts.shape[0]
    city_name = traj_df['CITY_NAME'].iloc[0]
    agent_df = None
    agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [
        None] * 7
    # agent traj & its start/end point
    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
        if obj_type == 'AGENT':
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
            norm_center = np.array([query_x, query_y])
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")

    # prune points after "obs_len" timestamp
    # [FIXED] test set length is only `obs_len`
    traj_df = traj_df[traj_df['TIMESTAMP'] <=
                      agent_df['TIMESTAMP'].values[obs_len-1]]

    assert (np.unique(traj_df["TIMESTAMP"].values).shape[0]
            == obs_len), "Obs len mismatch"

    # search nearby lane from the last observed point of agent
    # FIXME: nearby or rect?
    # lane_feature_ls = get_nearby_lane_feature_ls(
    #     am, agent_df, obs_len, city_name, lane_radius, norm_center)
    lane_feature_ls = get_nearby_lane_feature_ls(
        am, agent_df, obs_len, city_name, lane_radius, norm_center, mode=mode, query_bbox=query_bbox)
    # pdb.set_trace()

    # search nearby moving objects from the last observed point of agent
    obj_feature_ls = get_nearby_moving_obj_feature_ls(
        agent_df, traj_df, obs_len, seq_ts, norm_center)
    # get agent features
    agent_feature = get_agent_feature_ls(agent_df, obs_len, norm_center)

    # vis
    if viz:
        for features in lane_feature_ls:
            show_doubled_lane(
                np.vstack((features[0][:, :2], features[0][-1, 3:5])))
            show_doubled_lane(
                np.vstack((features[1][:, :2], features[1][-1, 3:5])))
        for features in obj_feature_ls:
            show_traj(
                np.vstack((features[0][:, :2], features[0][-1, 2:])), features[1])
        show_traj(np.vstack(
            (agent_feature[0][:, :2], agent_feature[0][-1, 2:])), agent_feature[1])

        plt.plot(agent_x_end - query_x, agent_y_end - query_y, 'o',
                 color=color_dict['AGENT'], markersize=7)
        plt.plot(0, 0, 'x', color='blue', markersize=4)
        plt.plot(start_x-query_x, start_y-query_y,
                 'x', color='blue', markersize=4)
        plt.show()

    return [agent_feature, obj_feature_ls, lane_feature_ls, norm_center]


def trans_gt_offset_format(gt):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets, starting from the last observed location.
    We rotate the coordinate system based on the heading of the target vehicle at the last observed location.
    
    """
    assert gt.shape == (30, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"

    # for test, no gt, just return a (0, 2) ndarray
    if gt.shape == (0, 2):
        return gt

    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))
    # import pdb
    # pdb.set_trace()
    assert (offset_gt.cumsum(axis=0) -
            gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) -gt).sum()}"

    return offset_gt


def encoding_features(agent_feature, obj_feature_ls, lane_feature_ls):
    """
    args:
        agent_feature:
            list of (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    returns:
        pd.DataFrame of (
            polyline_features: vstack[
                (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id),
                (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
                ]
            offset_gt: incremental offset from agent's last obseved point,
            traj_id2mask: Dict[int, int]
            lane_id2mask: Dict[int, int]
        )
        where obejct_type = {0 - others, 1 - agent}

    """
    polyline_id = 0
    traj_id2mask, lane_id2mask = {}, {}
    gt = agent_feature[-1]
    traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 9))

    # encoding agent feature
    pre_traj_len = traj_nd.shape[0]
    agent_len = agent_feature[0].shape[0]
    # print(agent_feature[0].shape, np.ones(
    # (agent_len, 1)).shape, agent_feature[2].shape, (np.ones((agent_len, 1)) * polyline_id).shape)
    agent_nd = np.hstack((agent_feature[0], np.ones(
        (agent_len, 1)), agent_feature[2].reshape((-1, 1)), np.ones((agent_len, 1)) * polyline_id))
    assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

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
        obj_nd = np.hstack((obj_feature[0], np.zeros(
            (obj_len, 1)), obj_feature[2].reshape((-1, 1)), np.ones((obj_len, 1)) * polyline_id))
        assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        traj_nd = np.vstack((traj_nd, obj_nd))

        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

    # incodeing lane feature
    pre_lane_len = lane_nd.shape[0]
    for lane_feature in lane_feature_ls:
        l_lane_len = lane_feature[0].shape[0]
        l_lane_nd = np.hstack(
            (lane_feature[0], (lane_feature[2]) * np.ones((l_lane_len, 1)),
             (lane_feature[3]) * np.ones((l_lane_len, 1)), np.ones((l_lane_len, 1)) * polyline_id))
        assert l_lane_nd.shape[1] == 9, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, l_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        r_lane_len = lane_feature[1].shape[0]
        r_lane_nd = np.hstack(
            (lane_feature[1], (lane_feature[2]) * np.ones((l_lane_len, 1)),
             (lane_feature[3]) * np.ones((l_lane_len, 1)), np.ones((r_lane_len, 1)) * polyline_id)
        )
        assert r_lane_nd.shape[1] == 9, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, r_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"
        # lane_nd = np.vstack((lane_nd, l_lane_nd, r_lane_nd))

    # FIXME: handling `nan` in lane_nd
    col_mean = np.nanmean(lane_nd, axis=0)
    if np.isnan(col_mean).any():
        # raise ValueError(
        # print(f"{col_mean}\nall z (height) coordinates are `nan`!!!!")
        lane_nd[:, 2].fill(.0)
        lane_nd[:, 5].fill(.0)
    else:
        inds = np.where(np.isnan(lane_nd))
        lane_nd[inds] = np.take(col_mean, inds[1])

    # traj_ls, lane_ls = reconstract_polyline(
    #     np.vstack((traj_nd, lane_nd)), traj_id2mask, lane_id2mask, traj_nd.shape[0])
    # type_ = 'AGENT'
    # for traj in traj_ls:
    #     show_traj(traj, type_)
    #     type_ = 'OTHERS'

    # for lane in lane_ls:
    #     show_doubled_lane(lane)
    # plt.show()

    # transform gt to offset_gt
    offset_gt = trans_gt_offset_format(gt)

    # now the features are:
    # (xs, ys, xe, ye, obejct_type, timestamp(avg_for_start_end?),polyline_id) for object
    # (xs, ys, zs, xe, ye, ze, polyline_id) for lanes

    # change lanes feature to xs, ys, xe, ye, NULL, zs, ze, traffic_control, is_intersection, polyline_id)
    lane_nd = np.hstack(
        [lane_nd, np.zeros((lane_nd.shape[0], 1), dtype=lane_nd.dtype)])
    lane_nd = lane_nd[:, [0, 1, 3, 4, 9, 2, 5, 6, 7, 8]]
    # change object features to (xs, ys, xe, ye, timestamp, NULL, NULL, NULL, NULL, polyline_id)
    traj_nd = np.hstack(
        [traj_nd, np.zeros((traj_nd.shape[0], 4), dtype=traj_nd.dtype)])
    traj_nd = traj_nd[:, [0, 1, 2, 3, 5, 7, 8, 9, 10, 6]]

    # don't ignore the id
    polyline_features = np.vstack((traj_nd, lane_nd))
    data = [[polyline_features.astype(
        np.float32), offset_gt, traj_id2mask, lane_id2mask, traj_nd.shape[0], lane_nd.shape[0]]]

    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "GT",
                 "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK", "TARJ_LEN", "LANE_LEN"]
    )


def save_features(df, name, dir_=None):
    if dir_ is None:
        dir_ = './input_data'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}.pkl"
    df.to_pickle(
        os.path.join(dir_, name)
    )
