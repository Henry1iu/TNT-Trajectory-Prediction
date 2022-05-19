# About:    superclass for data preprocessor
# Author:   Jianbang LIU
# Date:     2021.01.30
import copy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.utils.mpl_plotting_utils import visualize_centerline

from core.util.cubic_spline import Spline2D


class Preprocessor(Dataset):
    """
    superclass for all the trajectory data preprocessor
    those preprocessor will reformat the data in a single sequence and feed to the system or store them
    """
    def __init__(self, root_dir, algo="tnt", obs_horizon=20, obs_range=30, pred_horizon=30):
        self.root_dir = root_dir            # root directory stored the dataset

        self.algo = algo                    # the name of the algorithm
        self.obs_horizon = obs_horizon      # the number of timestampe for observation
        self.obs_range = obs_range          # the observation range
        self.pred_horizon = pred_horizon    # the number of timestamp for prediction

        self.split = None

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        """ the total number of sequence in the dataset """
        raise NotImplementedError

    def process(self, dataframe: pd.DataFrame, seq_id: str, map_feat=True):
        """
        select filter the data frame, output filtered data frame
        :param dataframe: DataFrame, the data frame
        :param seq_id: str, the sequence id
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """
        raise NotImplementedError

    def extract_feature(self, dataframe: pd.DataFrame, map_feat=True):
        """
        select and filter the data frame, output filtered frame feature
        :param dataframe: DataFrame, the data frame
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """
        raise NotImplementedError

    def encode_feature(self, *feats):
        """
        encode the filtered features to specific format required by the algorithm
        :feats dataframe: DataFrame, the data frame containing the filtered data
        :return: DataFrame[POLYLINE_FEATURES, GT, TRAJ_ID_TO_MASK, LANE_ID_TO_MASK, TARJ_LEN, LANE_LEN]
        """
        raise NotImplementedError

    def save(self, dataframe: pd.DataFrame, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the dataframe encoded
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not isinstance(dataframe, pd.DataFrame):
            return

        if not dir_:
            dir_ = os.path.join(os.path.split(self.root_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split + "_intermediate", "raw")

        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = f"features_{file_name}.pkl"
        dataframe.to_pickle(os.path.join(dir_, fname))
        # print("[Preprocessor]: Saving data to {} with name: {}...".format(dir_, fname))

    def process_and_save(self, dataframe: pd.DataFrame, seq_id, dir_=None, map_feat=True):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the data frame
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        df_processed = self.process(dataframe, seq_id, map_feat)
        self.save(df_processed, seq_id, dir_)

        return []

    @staticmethod
    def uniform_candidate_sampling(sampling_range, rate=30):
        """
        uniformly sampling of the target candidate
        :param sampling_range: int, the maximum range of the sampling
        :param rate: the sampling rate (num. of samples)
        return rate^2 candidate samples
        """
        x = np.linspace(-sampling_range, sampling_range, rate)
        return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)

    # implement a candidate sampling with equal distance;
    def lane_candidate_sampling(self, centerline_list, orig, distance=0.5, viz=False):
        """the input are list of lines, each line containing"""
        candidates = []
        for lane_id, line in enumerate(centerline_list):
            sp = Spline2D(x=line[:, 0], y=line[:, 1])
            s_o, d_o = sp.calc_frenet_position(orig[0], orig[1])
            s = np.arange(s_o, sp.s[-1], distance)
            ix, iy = sp.calc_global_position_online(s)
            candidates.append(np.stack([ix, iy], axis=1))
        candidates = np.unique(np.concatenate(candidates), axis=0)

        if viz:
            fig = plt.figure(0, figsize=(8, 7))
            fig.clear()
            for centerline_coords in centerline_list:
                visualize_centerline(centerline_coords)
            plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list), len(candidates)))
            plt.show()

        return candidates

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidate, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy

    @staticmethod
    def plot_target_candidates(candidate_centerlines, traj_obs, traj_fut, candidate_targets):
        fig = plt.figure(1, figsize=(8, 7))
        fig.clear()

        # plot centerlines
        for centerline_coords in candidate_centerlines:
            visualize_centerline(centerline_coords)

        # plot traj
        plt.plot(traj_obs[:, 0], traj_obs[:, 1], "x-", color="#d33e4c", alpha=1, linewidth=1, zorder=15)
        # plot end point
        plt.plot(traj_obs[-1, 0], traj_obs[-1, 1], "o", color="#d33e4c", alpha=1, markersize=6, zorder=15)
        # plot future traj
        plt.plot(traj_fut[:, 0], traj_fut[:, 1], "+-", color="b", alpha=1, linewidth=1, zorder=15)

        # plot target sample
        plt.scatter(candidate_targets[:, 0], candidate_targets[:, 1], marker="*", c="green", alpha=1, s=6, zorder=15)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(candidate_centerlines),
                                                                                       len(candidate_targets)))
        # plt.show(block=False)
        # plt.pause(0.01)
        plt.show()


# example of preprocessing scripts
if __name__ == "__main__":
    processor = Preprocessor("raw_data")
    loader = DataLoader(processor,
                        batch_size=16,
                        num_workers=16,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

    for i, data in enumerate(tqdm(loader)):
        pass



