import os
import sys
from tqdm import tqdm
from os.path import join as pjoin
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import argparse
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline

# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from core.dataloader.dataset import GraphDataset
# from core.dataloader.argoverse_loader import Argoverse, GraphData, ArgoverseInMem
# from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem
from core.dataloader.argoverse_loader_v3 import GraphData, ArgoverseInMem
from core.util.preprocessor.argoverse_preprocess_v3 import ArgoversePreprocessor, COLOR_DICT, plot_traj
from core.trainer.tnt_trainer import TNTTrainer

sys.path.append("core/dataloader")


def test(args):
    """
    script to test the tnt model
    "param args:
    :return:
    """
    # config
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = pjoin(args.save_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("The output folder does exists and is not empty! Check the folder.")
    else:
        os.makedirs(output_dir)

    # data loading
    root_dir = pjoin(args.data_root, "{}_intermediate".format(args.split))
    try:
        test_set = ArgoverseInMem(root_dir, angle_norm=True)
    except:
        raise Exception("Failed to load the data, please check the dataset!")

    # init trainer
    trainer = TNTTrainer(
        trainset=test_set,
        evalset=test_set,
        testset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    forcasted_trajs = trainer.test(miss_threshold=2.0,
                                   save_pred=False,
                                   convert_coordinate=True,
                                   compute_metric=True)

    for _, f in enumerate(tqdm(test_set.raw_paths)):
        raw_data = pd.read_pickle(f)
        seq_id = int(raw_data['seq_id'].values[0])

        # plot the predicted trajectories
        if seq_id in forcasted_trajs.keys():
            visualize_data(raw_data)
            
            for i, traj in enumerate(forcasted_trajs[seq_id]):
                plt.plot(traj[:, 0], traj[:, 1], "-", color="g", label="fut_" + str(i), linewidth=1)
                plt.plot(traj[0, 0], traj[0, 1], "o", color="g", linewidth=1)
                plt.plot(traj[-1, 0], traj[-1, 1], "x-", color="g", linewidth=1)
                plt.text(traj[-1, 0], traj[-1, 1], "fut_" + str(i))
            plt.show()


def visualize_data(data, show=False):
    """
    visualize the extracted data, and exam the data
    """
    fig = plt.figure(0, figsize=(8, 7))
    fig.clear()

    orig = data['orig'].values[0]

    # visualize the centerlines
    graph = data['graph'].values[0]
    lines_ctrs = graph['ctrs']
    lines_feats = graph['feats']
    lane_idcs = graph['lane_idcs']
    for i in np.unique(lane_idcs):
        line_ctr = lines_ctrs[lane_idcs == i]
        line_feat = lines_feats[lane_idcs == i]
        line_str = (2.0 * line_ctr - line_feat) / 2.0
        line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
        line = np.vstack([line_str, line_end.reshape(-1, 2)]) + orig
        visualize_centerline(line)

    # visualize the trajectory
    trajs = data['feats'].values[0][:, :, :2]
    has_obss = data['has_obss'].values[0]
    preds = data['gt_preds'].values[0]
    has_preds = data['has_preds'].values[0]
    for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
        plot_traj(traj[has_obs] + orig, pred[has_pred] + orig, COLOR_DICT, i)

    # visualize the target candidate
    candidates = data['tar_candts'].values[0] + orig
    candidate_gt = data['gt_candts'].values[0].astype(bool).reshape(-1)
    plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
    plt.plot(candidates[candidate_gt, 0], candidates[candidate_gt[:], 1], marker="o", c="r", alpha=1, zorder=15)

    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.axis("off")

    if show:
        # plt.show()
        plt.show(block=False)
        plt.pause(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default="dataset/interm_data_v3",
                        help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="val")

    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=16,
                        help="dataloader worker size")
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=0,
                        help="CUDA device ids")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        # default="/home/jb/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/run/tnt/08-30-13-49/checkpoint_iter39.ckpt",
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        default="/home/jb/Downloads/TNT/TNT/best_TNT.pth",
                        help="resume a model state for fine-tune")

    parser.add_argument("-d", "--save_dir", type=str, default="test_result")
    args = parser.parse_args()
    test(args)
