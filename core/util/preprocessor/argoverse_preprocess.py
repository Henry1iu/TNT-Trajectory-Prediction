# About: script to pre-process argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.01.30

import os
from os.path import join as pjoin
from tqdm import tqdm
import pandas as pd
import numpy as np

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

from core.util.preprocessor.base import Preprocessor
from core.util.preprocessor.object_utils import is_track_stationary
from core.util.preprocessor.lane_utils import get_lane_ids_base_traj, get_halluc_lane


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
            for f_path in tqdm(loader.seq_list, desc=f"Processing {self.folders[i]}..."):
                seq = loader.get(f_path)
                path, seq_f_name_ext = os.path.split(f_path)
                seq_f_name, ext = os.path.splitext(seq_f_name_ext)

                yield self.folders[i], seq_f_name, seq.seq_df

    def select(self, dataframe: pd.DataFrame, map_feat=True):
        # normalize timestamps
        dataframe['TIMESTAMP'] -= np.min(dataframe['TIMESTAMP'].values)
        dataframe = dataframe.sort_values(by="TIMESTAMP")
        seq_ts = np.unique(dataframe['TIMESTAMP'].values)

        city_name = dataframe['CITY_NAME'].iloc[0]

        # select agent trajectory
        agent_df, obj_df = None, None
        for obj_type, sub_dataframe in dataframe.groupby("OBJECT_TYPE"):
            if obj_type == "AGENT":
                agent_df = sub_dataframe
                query_x, query_y = agent_df[['X', 'Y']].values[self.obs_horizon - 1]
                norm_center = np.array([query_x, query_y])
                obj_df = dataframe[dataframe.OBJECT_TYPE != "AGENT"]     # remove "AGENT from dataframe"
        if not agent_df:
            return None         # return None if no agent in the sequence

        # include object data within the detection range
        obj_df_selected = self.__select_object__(agent_df, obj_df)

        # include lane data within the detection range
        lane_df_selected = self.__select_lane__(agent_df, obj_df)

    def encode(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            return None

    def __select_object__(self, agent_df, remain_df, norm_center):
        for track_id, obj_df in remain_df.groupby("TRACK_ID"):
            if len(obj_df) < self.obs_horizon or is_track_stationary(obj_df):
                remain_df = remain_df[remain_df["TRACK_ID"] != track_id]
            # todo: the criteria to determine whether include the object
            pass        # currently, no filtering

            # todo: remove the norm from the object coordinate

        return remain_df

    def __select_lane__(self, agent_df, obj_df, norm_center):
        city_name = agent_df["CITY_NAME"].values[0]
        # include traj lane ids
        lane_ids = get_lane_ids_base_traj(self.map, agent_df, self.obs_horizon, self.obs_range)
        for _, obj_df in obj_df.groupby("TRACK_ID"):
            ids = get_lane_ids_base_traj(self.map, obj_df, self.obs_horizon, self.obs_range)
            lane_ids.extend(ids)
        lane_ids = np.unique(lane_ids)

        lane_feature_ls = []
        for lane_id in lane_ids:
            traffic_control = self.map.lane_has_traffic_control_measure(
                lane_id, city_name)
            is_intersection = self.map.lane_is_in_intersection(lane_id, city_name)

            centerlane = self.map.get_lane_segment_centerline(lane_id, city_name)
            # normalize to last observed timestamp point of agent
            centerlane[:, :2] -= norm_center
            halluc_lane_1, halluc_lane_2 = get_halluc_lane(centerlane, city_name)

            lane_feature_ls.append(
                [halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id])
        # todo: construct the dataframe to store the lane feature
        return pd.DataFrame()


if __name__ == "__main__":
    pth = "/Users/jb/projects/trajectory_prediction_algorithms/yet-another-vectornet/data/raw_data"
    processor = ArgoversePreprocessor(pth)

    for s_name, f_name, df in processor.generate():
        df = processor.select(df)
        encoded_df = processor.encode(df)
        processor.save(encoded_df, s_name, f_name)
