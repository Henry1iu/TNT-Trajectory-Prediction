import os
import sys
from os.path import join as pjoin
from datetime import datetime
from matplotlib import pyplot as plt

import argparse
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence

# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from core.dataloader.dataset import GraphDataset
# from core.dataloader.argoverse_loader import Argoverse, GraphData, ArgoverseInMem
from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem
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
    try:
        test_set = ArgoverseInMem(pjoin(args.data_root, "{}_intermediate".format(args.split)))
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

    forcasted_trajs = trainer.test(miss_threshold=2.0, save_pred=True, convert_coordinate=True, compute_metric=False)

    # root = "/home/jb/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/dataset"
    # split_dir = pjoin(root, "raw_data", args.split+"_obs" if args.split == "test" else args.split)
    #
    # loader = ArgoverseForecastingLoader(split_dir)
    #
    # for seq_id in forcasted_trajs.keys():
    #     trajs = forcasted_trajs[seq_id]
    #     f_path = os.path.join(split_dir, '{}.csv'.format(seq_id))
    #     seq = loader.get(f_path)
    #
    #     viz_sequence(seq.seq_df, show=False)
    #
    #     # plot the predicted trajectories
    #     for i, traj in enumerate(trajs):
    #         plt.plot(traj[:, 0], traj[:, 1], "-", color="g", label="fut_" + str(i), linewidth=1)
    #         plt.plot(traj[0, 0], traj[0, 1], "o", color="g", linewidth=1)
    #         plt.plot(traj[-1, 0], traj[-1, 1], "x-", color="g", linewidth=1)
    #         plt.text(traj[-1, 0], traj[-1, 1], "fut_" + str(i))
    #
    #     plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default="dataset/interm_data_2022",
                        help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="test")

    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=16,
                        help="dataloader worker size")
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=0,
                        help="CUDA device ids")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        # default="/home/jb/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/run/tnt/05-21-07-33/checkpoint_iter26.ckpt",
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        default="/home/jb/Downloads/TNT/TNT/best_TNT.pth",
                        help="resume a model state for fine-tune")

    parser.add_argument("-d", "--save_dir", type=str, default="test_result")
    args = parser.parse_args()
    test(args)
