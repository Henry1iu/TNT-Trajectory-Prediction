import os
from os.path import join as pjoin
import numpy as np
import time
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader


# Argoverse Target Agent Trajectory Loader
class ArgoversePreprocessor(Dataset):
    def __init__(self,
                 root_dir,
                 obs_horizon=20,
                 pred_horizon=30,
                 split="train"):

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        self.split = split

        self.loader = ArgoverseForecastingLoader(pjoin(root_dir, split+"_obs" if split == "test" else split))

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        dataframe = seq.seq_df
        agent_df = dataframe[dataframe.OBJECT_TYPE == "AGENT"].sort_values(by="TIMESTAMP")
        trajs = np.concatenate((agent_df.X.to_numpy().reshape(-1, 1), agent_df.Y.to_numpy().reshape(-1, 1)), 1)

        orig = trajs[self.obs_horizon-1].copy().astype(np.float32)
        pre = (trajs[self.obs_horizon-3] - orig) / 2.0
        theta = np.pi - np.arctan2(pre[1], pre[0])
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        agt_norm = np.matmul(rot, (trajs - orig.reshape(-1, 2)).T).T

        return agt_norm.reshape(-1)

    def __len__(self):
        return len(self.loader)


def main():
    # Init loader
    root = "/media/Data/autonomous_driving/Argoverse/raw_data"
    dataset = ArgoversePreprocessor(root_dir=root, split="train")
    loader = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=False, pin_memory=False, drop_last=False)

    traj_array = np.empty((0, 50 * 2), dtype=np.float)
    # load all the target agent trajectory
    for i, data in enumerate(tqdm(loader)):
        traj_array = np.vstack([traj_array, data])

    # Apply PCA to reduce the dimension
    start_time = time.time()
    embedding = PCA(n_components=50).fit_transform(traj_array)
    print("Processing time of PCA: {}".format((time.time() - start_time)/60))

    # Apply DSCAN
    start_time = time.time()
    db = DBSCAN(eps=0.3, min_samples=10).fit(traj_array)
    print("Processing time of DBSCAN: {}".format((time.time() - start_time)/60))

    # Display via t-SNE
    start_time = time.time()
    traj_embedding = TSNE(n_components=2, init='pca').fit_transform(traj_array)
    print("Processing time of t-SNE: {}".format((time.time() - start_time)/60))


if __name__ == "__main__":
    main()
