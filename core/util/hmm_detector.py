# The HMM model for detecting the centerline of a trajectory
# Author: Jianbang @ PRAI, EE, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: Newson, Paul, and John Krumm. "Hidden Markov map matching through noise and sparseness." In Proceedings of the 17th ACM SIGSPATIAL international conference on advances in geographic information systems, pp. 336-343. 2009.

from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

from core.util.traj_clustering import visualize_centerline


class HMMDetector(object):
    def __init__(self):
        # lines: a list of LineString object representing the centerlines
        # traj: a sequence of pts profiling a trajectory, np.array([num_timestamp, 2])
        # pt_projected: a array of pts projected from the traj pt to the centerlines, np.array([num_traj, num_timestamp, 2])
        self.traj = None
        self.num_timestamp = 0
        self.num_line = 0
        self.lines = []
        self.pt_projected = None

    def detect(self, traj, centerlines):
        """ detect the corresponding centerline for each timestamp

        Args:
            :param traj, np.ndarray((num_timestamp, 2)).float(), the sequence of sample points representing traj;
            :param centerlines, list[centerline], the list of point set representing the centerlines

        Returns:
            :param path, np.ndarray((num_timestamp, )).int32(), the sequence of centerline id
        """
        self.__reset()

        # init centerlines
        self.traj = traj
        self.num_timestamp = len(traj)
        for centerline in centerlines:
            self.lines.append(LineString(centerline))
        self.num_line = len(self.lines)
        self.pt_projected = np.empty((self.num_timestamp, self.num_line, 2))

        # project pts
        self.__project_pts()

        # compute the measurement prob.
        measurement = self.__compute_measurement()

        # compute the transition prob.
        transition = self.__compute_transition()

        # compute the best path and prob.
        path, _ = self.__viterbi(measurement[0, :], transition, measurement)

        return path

    def __project_pts(self):
        """ project the pts to each centerline
            self.pt_projected: np.array([num_timestamp, num_centerlines, 2])
        """
        assert self.num_line != 0, "Error! No centerlines!"

        for t, xy in enumerate(self.traj):
            pt = Point(xy[0], xy[1])
            closest_pts = np.zeros((self.num_line, 2))
            for i, line in enumerate(self.lines):
                closest_pt = line.interpolate(line.project(pt))
                closest_pts[i] = np.array([closest_pt.x, closest_pt.y])
            self.pt_projected[t] = closest_pts

    def __compute_measurement(self):
        """ compute the measurement probability of each trajectory point
        Returns:
            :return p, [num_timestamp, num_line]
        """

        distance_relative = np.abs(self.pt_projected - np.expand_dims(self.traj, 1))
        distance_relative = np.hypot(distance_relative[:, :, 0], distance_relative[:, :, 1])
        # sigma = 1.4826 * np.median(distance_relative, axis=0)

        # p = np.exp(-0.5 * (distance_relative / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        p = np.exp(-0.5 * distance_relative ** 2)
        return p

    def __compute_transition(self):
        """ compute the transition probability of two consecutive point
        Returns:
             :return p, [num_timestamp, num_line, num_line], dimension 1 represent the destination state
        """
        state_list = [i for i in range(self.num_line)]
        xx, yy = np.meshgrid(state_list, state_list)

        d_x = self.pt_projected[:-1, xx.reshape(-1)] - self.pt_projected[1:, yy.reshape(-1)]
        d_x = np.hypot(d_x[:, :, 0], d_x[:, :, 1])                  # [num_stamp - 1, num_centerlines x num_centerlines]

        d_z = self.traj[:-1] - self.traj[1:]
        d_z = np.hypot(d_z[:, 0], d_z[:, 1])                        # [num_stamp - 1, 1]

        d_t = np.abs(np.expand_dims(d_z, -1) - d_x)                 # [num_stamp - 1, num_centerlines x num_centerlines]
        # beta = np.median(d_t, axis=0) / np.log(2)

        # p = np.exp(-d_t / beta) / beta
        p = np.exp(-d_t)

        return p.reshape((len(self.traj) - 1, self.num_line, self.num_line))

    def __viterbi(self, p_start, p_trans, p_emit):
        """ viterbi algorithm to solve the HMM model

        Args:
            :param p_start, the initial state probabilities (prior)
            :param p_trans, the state transition probabilities, (num_timestamp, num_line, num_line)
            :param p_emit, the emitting probabilities, (num_timestamp, num_line)
        Returns:
            :return best_path,
            :return viterbi,
        """
        viterbi = np.zeros((self.num_timestamp, self.num_line))
        best_path_table = np.zeros((self.num_timestamp, self.num_line))
        best_path = np.zeros(self.num_timestamp).astype(np.int32)

        viterbi[0] = p_start * p_emit[0]
        viterbi[0] /= np.sum(viterbi[0])

        for t in range(1, self.num_timestamp):              # loop through time
            for s in range(0, self.num_line):               # loop through the states @(t-1)
                p = viterbi[t-1, :] * p_trans[t-1, s, :] * p_emit[t, s]
                viterbi[t, s] = np.max(p)
                best_path_table[t, s] = np.argmax(p)

            viterbi[t, :] /= np.sum(viterbi[t, :])          # normalization

        # Back-tracking
        best_path[-1] = viterbi[-1, :].argmax()             # last state
        for t in range(self.num_timestamp-1, 0, -1):       # states of (last-1)th to 0th time step
            best_path[t-1] = best_path_table[t, best_path[t]]
        return best_path, viterbi

    def __reset(self):
        """reset HMM detector"""
        self.traj = None
        self.num_line = 0
        self.lines = []
        self.pt_projected = None


# Argoverse Target Agent Trajectory Loader
class ArgoversePreprocessor(Dataset):
    def __init__(self,
                 root_dir,
                 obs_horizon=20,
                 pred_horizon=30,
                 split="train",
                 viz=False):

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        self.split = split
        self.am = ArgoverseMap()

        self.loader = ArgoverseForecastingLoader(pjoin(root_dir, split+"_obs" if split == "test" else split))

        self.hmm = HMMDetector()

        self.viz = viz

        if self.viz:
            self.fig, self.axs = plt.subplots(2, 1)

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        dataframe = seq.seq_df
        city = dataframe["CITY_NAME"].values[0]

        agent_df = dataframe[dataframe.OBJECT_TYPE == "AGENT"].sort_values(by="TIMESTAMP")
        traj = np.concatenate((agent_df.X.to_numpy().reshape(-1, 1), agent_df.Y.to_numpy().reshape(-1, 1)), 1)

        orig = traj[self.obs_horizon-1].copy().astype(np.float32)

        # get the road centrelines
        lanes = self.am.find_local_lane_centerlines(orig[0], orig[1], city_name=city)
        candidate_lanes = self.am.get_candidate_centerlines_for_traj(traj[:self.obs_horizon], city_name=city)
        candidate_lane_seq = self.hmm.detect(traj[:self.obs_horizon], candidate_lanes)
        print("candidate_lane_seq: ", candidate_lane_seq)
        # weight = (np.arange(self.obs_horizon) + 1) / self.obs_horizon
        # candidate_lane_id = np.bincount(candidate_lane_seq, weights=weight).argmax()
        candidate_lane_id = np.bincount(candidate_lane_seq).argmax()

        # get the rotation
        lane_dir_vector, conf, nearest = self.am.get_lane_direction_traj(traj=traj[:self.obs_horizon], city_name=city)

        if conf <= 0.1:
            lane_dir_vector = (orig - traj[self.obs_horizon-4]) / 2.0
        theta = - np.arctan2(lane_dir_vector[1], lane_dir_vector[0])
        # print("pre: {};".format(pre))
        # print("theta: {};".format(theta))

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        rot_ = np.asarray([
            [1, 0],
            [0, 1]
        ])

        agt_rot = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
        agt_ori = np.matmul(rot_, (traj - orig.reshape(-1, 2)).T).T

        if self.viz:
            # plot original seq
            self.axs[0].plot(agt_ori[:self.obs_horizon, 0], agt_ori[:self.obs_horizon, 1], 'gx-')     # obs
            self.axs[0].plot(agt_ori[self.obs_horizon:, 0], agt_ori[self.obs_horizon:, 1], 'yx-')     # future
            self.axs[0].set_xlim([-120, 120])
            self.axs[0].set_ylim([-50, 50])

            visualize_centerline(self.axs[0], lanes, orig, rot_)
            # visualize_centerline(self.axs[0], [nearest], orig, rot_, color='red')
            visualize_centerline(self.axs[0], [candidate_lanes[candidate_lane_id]], orig, rot_, color='red')

            self.axs[0].set_title("The Original")

            # plot rotated seq
            self.axs[1].plot(agt_rot[:self.obs_horizon, 0], agt_rot[:self.obs_horizon, 1], 'gx-')     # obs
            self.axs[1].plot(agt_rot[self.obs_horizon:, 0], agt_rot[self.obs_horizon:, 1], 'yx-')     # future
            self.axs[1].set_xlim([-120, 120])
            self.axs[1].set_ylim([-50, 50])

            visualize_centerline(self.axs[1], lanes, orig, rot)
            visualize_centerline(self.axs[1], [nearest], orig, rot, color='red')
            # visualize_centerline(self.axs[0], [candidate_lanes[np.bincount(candidate_lane_seq).argmax()]], orig, rot_, color='red')

            self.axs[1].set_title("The Rotated")

            self.fig.show()
            self.fig.waitforbuttonpress()
            for ax in tuple(self.axs):
                ax.cla()

        return agt_rot.astype(np.float32)

    def __len__(self):
        return len(self.loader)


if __name__ == "__main__":
    # config for initializing loader
    root = "/media/Data/autonomous_driving/Argoverse/raw_data"
    visualize = True

    # loader init
    dataset = ArgoversePreprocessor(root_dir=root, split="train", viz=visualize)
    if visualize:
        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=False, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=False, pin_memory=False, drop_last=False)

    for i, traj_batch in enumerate(tqdm(loader)):
        pass
