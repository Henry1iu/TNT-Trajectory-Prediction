#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from core.util.config import VELOCITY_THRESHOLD, RAW_DATA_FORMAT, OBJ_RADIUS, EXIST_THRESHOLD


def compute_velocity(track_df: pd.DataFrame) -> List[float]:
    """Compute velocities for the given track.

    Args:
        track_df (pandas Dataframe): Data for the track
    Returns:
        vel (list of float): Velocity at each timestep

    """
    x_coord = track_df["X"].values
    y_coord = track_df["Y"].values
    timestamp = track_df["TIMESTAMP"].values
    vel_x, vel_y = zip(*[(
        (x_coord[i] - x_coord[i - 1]) /
        (float(timestamp[i]) - float(timestamp[i - 1])),
        (y_coord[i] - y_coord[i - 1]) /
        (float(timestamp[i]) - float(timestamp[i - 1])),
    ) for i in range(1, len(timestamp))])
    vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]

    return vel


def is_track_stationary(track_df: pd.DataFrame) -> bool:
    """Check if the track is stationary.

    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False

    """
    vel = compute_velocity(track_df)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[int(len(vel) / 2)]
    return True if threshold_vel < VELOCITY_THRESHOLD else False


def fill_track_lost_in_middle(
        track_array: np.ndarray,
        seq_timestamps: np.ndarray,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Handle the case where the object exited and then entered the frame but still retains the same track id.
        It'll be a rare case.

    Args:
        track_array (numpy array): Padded data for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        raw_data_format (Dict): Format of the sequence
    Returns:
        filled_track (numpy array): Track data filled with missing timestamps

    """
    curr_idx = 0
    filled_track = np.empty((0, track_array.shape[1]))
    try:
        for timestamp in seq_timestamps:
            filled_track = np.vstack((filled_track, track_array[curr_idx]))
            if timestamp in track_array[:, raw_data_format["TIMESTAMP"]]:
                curr_idx += 1
    except:
        from pdb import set_trace; set_trace()
    return filled_track


def pad_track(
        track_df: pd.DataFrame,
        seq_timestamps: np.ndarray,
        obs_len: int,
        raw_data_format: Dict[str, int], ) -> np.ndarray:
    """Pad incomplete tracks.

    Args:
        track_df (Dataframe): Dataframe for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        obs_len (int): Length of observed trajectory
        raw_data_format (Dict): Format of the sequence
    Returns:
            padded_track_array (numpy array): Track data padded in front and back

    """
    track_vals = track_df.values
    track_timestamps = track_df["TIMESTAMP"].values

    # start and index of the track in the sequence
    start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
    end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

    # Edge padding in front and rear, i.e., repeat the first and last coordinates
    # if self.PADDING_TYPE == "REPEAT"
    padded_track_array = np.pad(track_vals,
                                ((start_idx, obs_len - end_idx - 1),
                                    (0, 0)), "edge")
    if padded_track_array.shape[0] < obs_len:
        padded_track_array = fill_track_lost_in_middle(
            padded_track_array, seq_timestamps, raw_data_format)

    # Overwrite the timestamps in padded part
    for i in range(padded_track_array.shape[0]):
        padded_track_array[i, 0] = seq_timestamps[i]
    return padded_track_array


def get_nearby_moving_obj_feature_ls(agent_df, traj_df, obs_len, seq_ts, norm_center):
    """
    args:
    returns: list of list, (doubled_track, object_type, timestamp, track_id)
    """
    obj_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
    p0 = np.array([query_x, query_y])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df['OBJECT_TYPE'].iloc[0] == 'AGENT':
            continue

        if len(remain_df) < EXIST_THRESHOLD or is_track_stationary(remain_df):
            continue

        xys, ts = None, None
        # if len(remain_df) < obs_len:
        #     paded_nd = pad_track(remain_df, seq_ts, obs_len, RAW_DATA_FORMAT)
        #     xys = np.array(paded_nd[:, 3:5], dtype=np.float64)
        #     ts = np.array(paded_nd[:, 0], dtype=np.float64)  # FIXME: fix bug: not consider padding time_seq
        # else:
        xys = remain_df[['X', 'Y']].values
        ts = remain_df["TIMESTAMP"].values

        p1 = xys[-1]
        if np.linalg.norm(p0 - p1) > OBJ_RADIUS:
            continue

        xys -= norm_center  # normalize to last observed timestamp point of agent
        xys = np.hstack((xys[:-1], xys[1:]))
        
        ts = (ts[:-1] + ts[1:]) / 2
        # if not xys.shape[0] == ts.shape[0]:
        #     from pdb import set_trace;set_trace()

        obj_feature_ls.append(
            [xys, remain_df['OBJECT_TYPE'].iloc[0], ts, track_id])
    return obj_feature_ls
