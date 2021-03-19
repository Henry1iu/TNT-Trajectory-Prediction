#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

import numpy as np
from core.util.config import LANE_WIDTH


def get_halluc_lane(centerlane, city_name):
    """
    return left & right lane based on centerline
    args:
    returns:
        doubled_left_halluc_lane, doubled_right_halluc_lane, shaped in (N-1, 3)
    """
    if centerlane.shape[0] <= 1:
        raise ValueError('shape of centerlane error.')

    half_width = LANE_WIDTH[city_name] / 2
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


def get_rect_lane_id(lane_dict, x_min, x_max, y_min, y_max):
    lane_ids = []

    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in lane_dict.items():

        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_ids.append(lane_id)
    return lane_ids


def get_lane_ids_base_traj(am, traj_df, obs_horizon, lane_radius=20.0):
    """
    get corresponding lane ids based on trajectory
    :param am: Argoverse map object
    :param traj_df: DataFrame, trajectory dataframe
    :param obs_horizon: int, observation horizon(length)
    :param lane_radius: float, the radius to include the lane
    :return:np.array, the related lane ids
    """
    # get lane ids at the start
    city_name = traj_df["CITY_NAME"].values[0]
    query_x, query_y = traj_df[['X', 'Y']].values[0]
    lane_ids_str = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

    # get lane ids in the middle
    query_x, query_y = traj_df[['X', 'Y']].values[obs_horizon-1]
    lane_ids_mid = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

    # get lane ids at the end
    query_x, query_y = traj_df[['X', 'Y']].values[-1]
    lane_ids_end = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

    # merge all the lane ids and remove duplicate
    return np.unique([lane_ids_str, lane_ids_mid, lane_ids_end])


def get_nearby_lane_feature_ls(
        am,
        agent_df,
        obs_len,
        city_name,
        lane_radius,
        norm_center,
        has_attr=False,
        mode='nearby',
        query_bbox=None):
    '''
    compute lane features
    args:
        norm_center: np.ndarray
        mode: 'nearby' return nearby lanes within the radius; 'rect' return lanes within the query bbox
        **kwargs: query_bbox= List[int, int, int, int]
    returns:
        list of list of lane a segment feature,
        formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    '''

    lane_feature_ls = []
    if mode == 'nearby':
        query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
        nearby_lane_ids = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)

    elif mode == 'rect':
        lane_dict = am.city_lane_centerlines_dict[city_name]
        query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
        # nearby_lane_ids = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)
        nearby_lane_ids = get_rect_lane_id(
            lane_dict, query_bbox[0]+query_x, query_bbox[1]+query_x, query_bbox[2]+query_y, query_bbox[3]+query_y
        )

    else:
        raise ValueError(f"{mode} is not in {'rect', 'nearby'}")

    for lane_id in nearby_lane_ids:
        traffic_control = am.lane_has_traffic_control_measure(
            lane_id, city_name)
        is_intersection = am.lane_is_in_intersection(lane_id, city_name)

        centerlane = am.get_lane_segment_centerline(lane_id, city_name)
        # normalize to last observed timestamp point of agent
        centerlane[:, :2] -= norm_center
        halluc_lane_1, halluc_lane_2 = get_halluc_lane(centerlane, city_name)

        if has_attr:
            raise NotImplementedError()

        lane_feature_ls.append(
            [halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id])
    return lane_feature_ls
