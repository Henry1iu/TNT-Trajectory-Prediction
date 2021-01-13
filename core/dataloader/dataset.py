# %%

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader


# %%
def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack(
            (edge_index, np.vstack((np.hstack([from_[:i], from_[i + 1:]]), np.hstack([to_[:i], to_[i + 1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start


# %%


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0


# %%


class GraphDataset(InMemoryDataset):
    """
    dataset object similar to `torchvision`
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass

    def process(self):

        def get_data_path_ls(dir_):
            return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]

        # make sure deterministic results
        data_path_ls = sorted(get_data_path_ls(self.root))

        valid_len_ls = []
        valid_len_ls = []
        data_ls = []
        for data_p in tqdm(data_path_ls):
            if not data_p.endswith('pkl'):
                continue
            x_ls = []
            y = None
            cluster = None
            edge_index_ls = []
            data = pd.read_pickle(data_p)
            all_in_features = data['POLYLINE_FEATURES'].values[0]
            add_len = data['TARJ_LEN'].values[0]
            cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
            valid_len_ls.append(cluster.max())
            y = data['GT'].values[0].reshape(-1).astype(np.float32)

            traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
            agent_id = 0
            edge_index_start = 0
            assert all_in_features[agent_id][
                       -1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

            for id_, mask_ in traj_mask.items():
                data_ = all_in_features[mask_[0]:mask_[1]]
                edge_index_, edge_index_start = get_fc_edge_index(
                    data_.shape[0], start=edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)

            for id_, mask_ in lane_mask.items():
                data_ = all_in_features[mask_[0] + add_len: mask_[1] + add_len]
                edge_index_, edge_index_start = get_fc_edge_index(
                    data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)
            edge_index = np.hstack(edge_index_ls)
            x = np.vstack(x_ls)
            data_ls.append([x, y, cluster, edge_index])

        # [x, y, cluster, edge_index, valid_len]
        g_ls = []
        padd_to_index = np.max(valid_len_ls)
        feature_len = data_ls[0][0].shape[1]
        for ind, tup in enumerate(data_ls):
            tup[0] = np.vstack(
                [tup[0], np.zeros((padd_to_index - tup[-2].max(), feature_len), dtype=tup[0].dtype)])
            tup[-2] = np.hstack(
                [tup[2], np.arange(tup[-2].max() + 1, padd_to_index + 1)])
            g_data = GraphData(
                x=torch.from_numpy(tup[0]),
                y=torch.from_numpy(tup[1]),
                cluster=torch.from_numpy(tup[2]),
                edge_index=torch.from_numpy(tup[3]),
                valid_len=torch.tensor([valid_len_ls[ind]]),
                time_step_len=torch.tensor([padd_to_index + 1])
            )
            g_ls.append(g_data)
        data, slices = self.collate(g_ls)
        torch.save((data, slices), self.processed_paths[0])


# %%
if __name__ == "__main__":
    # for folder in os.listdir("./data/interm_data"):
    INTERMEDIATE_DATA_DIR = "/Users/jb/projects/trajectory_prediction_algorithms/yet-another-vectornet/data/interm_data"

    for folder in ["train", "val", "test"]:
        dataset_input_path = os.path.join(
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")

        dataset = GraphDataset(dataset_input_path)
        batch_iter = DataLoader(dataset, batch_size=2)
        batch = next(iter(batch_iter))

# %%
