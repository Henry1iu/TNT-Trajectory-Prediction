import os
import sys
from os.path import join as pjoin
from datetime import datetime

import argparse

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

    trainer.test(miss_threshold=2.0, save_pred=True, convert_coordinate=True)


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
