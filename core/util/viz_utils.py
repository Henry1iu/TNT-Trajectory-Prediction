#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from core.util.config import color_dict
import torch


def show_doubled_lane(polygon):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    xs, ys = polygon[:, 0], polygon[:, 1]
    plt.plot(xs, ys, '--', color='grey')


def show_traj(traj, type_):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    plt.plot(traj[:, 0], traj[:, 1], color=color_dict[type_])


def reconstract_polyline(features, traj_mask, lane_mask, add_len):
    traj_ls, lane_ls = [], []
    for id_, mask in traj_mask.items():
        data = features[mask[0]: mask[1]]
        traj = np.vstack((data[:, 0:2], data[-1, 2:4]))
        traj_ls.append(traj)
    for id_, mask in lane_mask.items():
        data = features[mask[0]+add_len: mask[1]+add_len]
        # lane = np.vstack((data[:, 0:2], data[-1, 3:5]))
        # change lanes feature to (xs, ys, zs, xe, ye, ze, polyline_id)
        lane = np.vstack((data[:, 0:2], data[-1, 2:4]))
        lane_ls.append(lane)
    return traj_ls, lane_ls


def show_pred_and_gt(pred_y, y):
    plt.plot(y[:, 0], y[:, 1], color='r')
    plt.plot(pred_y[:, 0], pred_y[:, 1], lw=0, marker='o', fillstyle='none')


def show_predict_result(data, pred_y: torch.Tensor, y, add_len, show_lane=True):
    features, _ = data['POLYLINE_FEATURES'].values[0], data['GT'].values[0].astype(
        np.float32)
    traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]

    traj_ls, lane_ls = reconstract_polyline(
        features, traj_mask, lane_mask, add_len)

    type_ = 'AGENT'
    for traj in traj_ls:
        show_traj(traj, type_)
        type_ = 'OTHERS'

    if show_lane:
        for lane in lane_ls:
            show_doubled_lane(lane)

    pred_y = pred_y.numpy().reshape((-1, 2)).cumsum(axis=0)
    y = y.numpy().reshape((-1, 2)).cumsum(axis=0)
    show_pred_and_gt(pred_y, y)