import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

import gc
from copy import deepcopy, copy

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
# from torch.utils.data import DataLoader

from core.dataloader.dataset import get_fc_edge_index

sys.path.append("core/dataloader")


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


# dataset loader which loads data from disk
class Argoverse(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Argoverse, self).__init__(root, transform, pre_transform)
        gc.collect()
        # self.data_list = [torch.load(osp.join(self.processed_dir, f_na)) for f_na in self.processed_file_names]

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return [file for file in os.listdir(self.processed_dir) if "data" in file and file.endswith(".pt")]

    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData """
        # counting the largest polyline id
        valid_len = []
        print("[Argoverse]: Counting the valid length...")
        for raw_path in tqdm(self.raw_paths):
            raw_data = pd.read_pickle(raw_path)
            poly_feat = raw_data["POLYLINE_FEATURES"].values[0]
            cluster = poly_feat[:, -1].reshape(-1).astype(np.int32)
            valid_len.append(cluster.max())
        index_to_pad = np.max(valid_len)
        print("[Argoverse]: The longest valid length is {}.".format(index_to_pad))
        print("[Argoverse]: The mean of valid length is {}.".format(np.mean(valid_len)))

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        print("[Argoverse]: Transforming the data to GraphData...")
        for ind, raw_path in enumerate(tqdm(self.raw_paths)):
            file_name = osp.split(raw_path)[1]
            file_id = re.findall(r"\d+", file_name)[0]

            # Read data from `raw_path`.
            raw_data = pd.read_pickle(raw_path)
            poly_feat = raw_data['POLYLINE_FEATURES'].values[0]
            add_len = raw_data['TARJ_LEN'].values[0]
            cluster = poly_feat[:, -1].reshape(-1).astype(np.int32)
            y = raw_data['GT'].values[0].reshape(-1).astype(np.float32)

            candidate = raw_data['CANDIDATES'].values[0].astype(np.float32)
            gt_candidate = raw_data['CANDIDATE_GT'].values[0].astype(np.float32)
            gt_offset = raw_data['OFFSET_GT'].values[0].astype(np.float32)
            gt_target = raw_data['TARGET_GT'].values[0].astype(np.float32)

            traj_mask, lane_mask = raw_data["TRAJ_ID_TO_MASK"].values[0], raw_data['LANE_ID_TO_MASK'].values[0]

            # rearrange x in the sequence of mask
            x_ls = []
            edge_index_ls = []
            edge_index_start = 0
            for id_, mask_ in traj_mask.items():
                data_ = poly_feat[mask_[0]:mask_[1]]
                edge_index_, edge_index_start = get_fc_edge_index(data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)

            for id_, mask_ in lane_mask.items():
                data_ = poly_feat[mask_[0]+add_len: mask_[1]+add_len]
                edge_index_, edge_index_start = get_fc_edge_index(data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)
            edge_index = np.hstack(edge_index_ls)
            x = np.vstack(x_ls)
            feature_len = x.shape[1]
            x = np.vstack([x, np.zeros((index_to_pad - cluster.max(), feature_len), dtype=x.dtype)])

            cluster = np.hstack([cluster, np.arange(valid_len[ind] + 1, index_to_pad + 1)])

            # subgraph input data
            graph_input = GraphData(
                x=torch.from_numpy(x),
                y=torch.from_numpy(y),
                cluster=torch.from_numpy(cluster).short(),
                edge_index=torch.from_numpy(edge_index).long(),
                valid_len=torch.tensor([valid_len[ind]]),
                time_step_len=torch.tensor([index_to_pad + 1]),
                candidate=torch.from_numpy(candidate).float(),
                candidate_gt=torch.from_numpy(gt_candidate).float(),
                offset_gt=torch.from_numpy(gt_offset).float(),
                target_gt=torch.from_numpy(gt_target).float(),
            )

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue
            #
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            torch.save(graph_input, osp.join(self.processed_dir, 'data_{}.pt'.format(file_id)))

    def __len__(self):
        return len(self.processed_file_names)

    def get(self, index: int):
        # [graph_data, gt] = torch.load(osp.join(self.processed_dir, self.processed_file_names[index]))
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[index]))
        # return graph_data, gt
        return data

    # def get(self, index: int):
    #     return self.data_list[index]


# dataset loader which loads data into memory
class ArgoverseInMem(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData """
        # counting the largest polyline id
        valid_len = []
        candidate_len = []
        print("[Argoverse]: Counting the valid length...")
        for raw_path in tqdm(self.raw_paths):
            raw_data = pd.read_pickle(raw_path)
            poly_feat = raw_data["POLYLINE_FEATURES"].values[0]
            cluster = poly_feat[:, -1].reshape(-1).astype(np.int32)
            valid_len.append(cluster.max())
            candidate_len.append(len(raw_data['CANDIDATES'].values[0]))
        index_to_pad = np.max(valid_len)
        candidate_len_max = np.max(candidate_len)
        # candidate_len_max = 702
        print("[Argoverse]: The longest valid length is {}.".format(index_to_pad))
        print("[Argoverse]: The mean of valid length is {}.".format(np.mean(valid_len)))

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        print("[Argoverse]: Transforming the data to GraphData...")
        data_list = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths)):
            # Read data from `raw_path`.
            raw_data = pd.read_pickle(raw_path)
            poly_feat = raw_data['POLYLINE_FEATURES'].values[0]
            add_len = raw_data['TARJ_LEN'].values[0]
            cluster = poly_feat[:, -1].reshape(-1).astype(np.int32)
            y = raw_data['GT'].values[0].reshape(-1).astype(np.float32)

            candidate = raw_data['CANDIDATES'].values[0]
            gt_candidate = raw_data['CANDIDATE_GT'].values[0]
            gt_offset = raw_data['OFFSET_GT'].values[0]
            gt_target = raw_data['TARGET_GT'].values[0]

            traj_mask, lane_mask = raw_data["TRAJ_ID_TO_MASK"].values[0], raw_data['LANE_ID_TO_MASK'].values[0]

            # rearrange x in the sequence of mask
            x_ls = []
            edge_index_ls = []
            edge_index_start = 0
            for id_, mask_ in traj_mask.items():
                data_ = poly_feat[mask_[0]:mask_[1]]
                edge_index_, edge_index_start = get_fc_edge_index(data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)

            for id_, mask_ in lane_mask.items():
                data_ = poly_feat[mask_[0]+add_len: mask_[1]+add_len]
                edge_index_, edge_index_start = get_fc_edge_index(data_.shape[0], edge_index_start)
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)
            edge_index = np.hstack(edge_index_ls)
            x = np.vstack(x_ls)

            # input data
            graph_input = GraphData(
                x=torch.from_numpy(x).float(),
                y=torch.from_numpy(y).float(),
                cluster=torch.from_numpy(cluster).short(),
                edge_index=torch.from_numpy(edge_index),
                # valid_len=torch.tensor([valid_len[ind]]),
                valid_len=torch.tensor([cluster.max()]),
                time_step_len=torch.tensor([index_to_pad + 1]),
                candidate_len_max=torch.tensor([candidate_len_max]).int(),
                candidate_mask=[],
                candidate=torch.from_numpy(candidate).float(),
                candidate_gt=torch.from_numpy(gt_candidate).float(),
                offset_gt=torch.from_numpy(gt_offset).float(),
                target_gt=torch.from_numpy(gt_target).float(),
            )
            data_list.append(graph_input)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(ArgoverseInMem, self).get(idx).clone()

        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item() - 1
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len+1, index_to_pad+1)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate, torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt, torch.zeros((num_cand_max - len(data.candidate_gt), 1))])

        return data


if __name__ == "__main__":

    # for folder in os.listdir("./data/interm_data"):
    # INTERMEDIATE_DATA_DIR = "../../dataset/interm_tnt_with_filter"
    INTERMEDIATE_DATA_DIR = "../../dataset/interm_tnt_n_s_0727"
    # INTERMEDIATE_DATA_DIR = "/media/Data/autonomous_driving/Argoverse/intermediate"

    for folder in ["train", "val"]:
    # for folder in ["val"]:
        dataset_input_path = os.path.join(
            # INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")

        # dataset = Argoverse(dataset_input_path)
        dataset = ArgoverseInMem(dataset_input_path)
        batch_iter = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=True, pin_memory=True)
        for k in range(3):
            for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
                pass

            # print("{}".format(i))
            candit_len = data.candidate_len_max[0]
            print(candit_len)
            # target_candite = data.candidate[candit_gt.squeeze(0).bool()]
            # try:
            #     # loss = torch.nn.functional.binary_cross_entropy(candit_gt, candit_gt)
            #     target_candite = data.candidate[candit_gt.bool()]
            # except:
            #     print(torch.argmax())
            #     print(candit_gt)
            # # print("type: {}".format(type(candit_gt)))
            # print("max: {}".format(candit_gt.max()))
            # print("min: {}".format(candit_gt.min()))


