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
from argoverse.map_representation.map_api import ArgoverseMap


def visualize_centerline(ax, centerlines, orig, rot, color="grey") -> None:
    """Visualize the computed centerline.

    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    for centerline in centerlines:
        centerline = np.matmul(rot, (centerline[:, :2] - orig.reshape(-1, 2)).T).T
        line_coords = list(zip(*centerline))
        lineX = line_coords[0]
        lineY = line_coords[1]
        ax.plot(lineX, lineY, "--", color=color, alpha=1, linewidth=1, zorder=0)
        ax.text(lineX[1], lineY[1], "s")
        ax.text(lineX[-2], lineY[-2], "e")
    # ax.axis("equal")


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
        self.am = ArgoverseMap()

        self.loader = ArgoverseForecastingLoader(pjoin(root_dir, split+"_obs" if split == "test" else split))

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        dataframe = seq.seq_df
        city = dataframe["CITY_NAME"].values[0]

        agent_df = dataframe[dataframe.OBJECT_TYPE == "AGENT"].sort_values(by="TIMESTAMP")
        trajs = np.concatenate((agent_df.X.to_numpy().reshape(-1, 1), agent_df.Y.to_numpy().reshape(-1, 1)), 1)

        orig = trajs[self.obs_horizon-1].copy().astype(np.float32)

        # get the road centrelines
        lanes = self.am.find_local_lane_centerlines(orig[0], orig[1], city_name=city)

        # get the rotation
        pre, conf = self.am.get_lane_direction(query_xy_city_coords=orig, city_name=city)
        _, conf, nearest = self.am.get_nearest_centerline(query_xy_city_coords=orig, city_name=city)
        nearest = nearest.reshape((-1, nearest.shape[0], nearest.shape[1]))

        if conf <= 0.9:
            pre = (trajs[self.obs_horizon-4] - orig) / 2.0
        theta = - np.arctan2(pre[1], pre[0])
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        agt_norm = np.matmul(rot, (trajs - orig.reshape(-1, 2)).T).T

        fig, axs = plt.subplots(1, 1)

        axs.plot(agt_norm[:self.obs_horizon, 0], agt_norm[:self.obs_horizon, 1], 'gx-')     # obs
        axs.plot(agt_norm[self.obs_horizon:, 0], agt_norm[self.obs_horizon:, 1], 'yx-')     # future
        axs.set_xlim([-120, 120])
        axs.set_ylim([-50, 50])

        visualize_centerline(axs, lanes, orig, rot)
        visualize_centerline(axs, nearest, orig, rot, color='red')

        plt.show()
        print("")

        return agt_norm.astype(np.float32)

    def __len__(self):
        return len(self.loader)


def main():
    # Init loader
    root = "/media/Data/autonomous_driving/Argoverse/raw_data"
    dataset = ArgoversePreprocessor(root_dir=root, split="train")
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=False, drop_last=False)

    fig, axs = plt.subplots(3, 1)

    traj_array_flatten = np.empty((0, 50 * 2), dtype=np.float)
    # load all the target agent trajectory
    for i, traj_batch in enumerate(tqdm(loader)):
        (batch_size, _, _) = traj_batch.shape
    #     for batch_id in range(batch_size):
    #         axs[0].plot(traj_batch[batch_id, :, 0], traj_batch[batch_id, :, 1], alpha=0.01)         # plot whole traj
    #         axs[1].plot(traj_batch[batch_id, :20, 0], traj_batch[batch_id, :20, 1], alpha=0.01)     # plot observed traj
    #         axs[2].plot(traj_batch[batch_id, 20:, 0], traj_batch[batch_id, 20:, 1], alpha=0.01)     # plot future traj
    #
        traj_array_flatten = np.vstack([traj_array_flatten, traj_batch.reshape(batch_size, -1)])
    # plt.show()

    # Apply PCA to reduce the dimension
    # start_time = time.time()
    # embedding = PCA(n_components=50).fit_transform(traj_array_flatten)
    # print("Processing time of PCA: {}".format((time.time() - start_time)/60))

    # Apply DSCAN
    for eps in np.linspace(1, 3, 20):
        start_time = time.time()
        db = DBSCAN(eps=eps, min_samples=1000).fit(traj_array_flatten)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print("\neps = {}".format(eps))
        print("Processing time of DBSCAN: {}".format((time.time() - start_time)/60))
        print("Num of Cluster: {}".format(n_clusters_))
        unique_labels = set(labels)
        for label in unique_labels:
            class_member_mask = (labels == label)
            print("Lable {}: No. of Trajectories: {};".format(label, sum(class_member_mask)))

    # plot the trajectories

    # Display via t-SNE
    # start_time = time.time()
    # traj_embedding = TSNE(n_components=2, init='pca').fit_transform(traj_array)
    # print("Processing time of t-SNE: {}".format((time.time() - start_time)/60))


if __name__ == "__main__":
    main()
