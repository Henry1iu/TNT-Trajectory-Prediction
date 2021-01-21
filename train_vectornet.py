import os
import sys
from os.path import join as pjoin
from datetime import datetime

import argparse
import numpy as np

# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader, DataListLoader

from core.dataloader.dataset import GraphDataset, GraphData
from core.vectornet_trainer import VectorNetTrainer

TEST = True

sys.path.append("core/dataloader")


def train(args):
    """
    script to train the vectornet
    :param args:
    :return:
    """
    # data loading
    train_set = GraphDataset(pjoin(args.data_root, "train_intermediate")).shuffle()
    eval_set = GraphDataset(pjoin(args.data_root, "val_intermediate"))
    test_set = GraphDataset(pjoin(args.data_root, "val_intermediate"))

    loader = DataLoader
    t_loader = loader(train_set[:10] if TEST else train_set,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True,
                      shuffle=True)
    e_loader = loader(eval_set[:2] if TEST else eval_set,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      pin_memory=True)
    ts_loader = loader(test_set[:1] if TEST else test_set,
                       batch_size=1,
                       num_workers=1,
                       pin_memory=True)

    # init output dir
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = pjoin(args.output_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("The output folder does exists and is not empty! Check the folder.")
    else:
        os.makedirs(output_dir)

    # init trainer
    trainer = VectorNetTrainer(
        train_loader=t_loader,
        eval_loader=e_loader,
        test_laoder=ts_loader,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        num_global_graph_layer=args.num_glayer,
        aux_loss=args.aux_loss,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        log_freq=args.log_freq
    )

    # resume checkpoint or model if available
    if hasattr(args, "resume_checkpoint") and args.resume_checkpoint:
        trainer.load(args.resume_checkpoint, 'c')
    elif hasattr(args, "resume_model") and args.resume_model:
        trainer.load(args.resume_model, 'm')
    min_eval_loss = trainer.min_eval_loss

    # training
    for iter_epoch in range(args.n_epoch):
        _ = trainer.train(iter_epoch)

        eval_loss = trainer.eval(iter_epoch)
        if not min_eval_loss:
            min_eval_loss = eval_loss
        elif eval_loss < min_eval_loss:
            # save the model when a lower eval_loss is found
            min_eval_loss = eval_loss
            trainer.save(iter_epoch, min_eval_loss)
            trainer.save_model()

    trainer.save_model("final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", required=False, type=str, default="dataset/interm_data",
                        help="root dir for datasets")
    parser.add_argument("-o", "--output_dir", required=False, type=str, default="run/vectornet/",
                        help="ex)dir to save checkpoint and model")

    parser.add_argument("-l", "--num_glayer", type=int, default=1,
                        help="number of global graph layers")
    parser.add_argument("-a", "--aux_loss", action="store_true", default=False,
                        help="Training with the auxiliary recovery loss")

    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="number of batch_size")
    parser.add_argument("-e", "--n_epoch", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8,
                        help="dataloader worker size")

    parser.add_argument("-c", "--with_cuda", type=bool, default=False,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=None,
                        help="CUDA device ids")
    # parser.add_argument("-cd", "--cuda_device", type=int, nargs='+', default=[],
    #                     help="CUDA device ids")
    parser.add_argument("--log_freq", type=int, default=2,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("-rc", "--resume_checkpoint", type=str, help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str, help="resume a model state for fine-tune")

    args = parser.parse_args()
    train(args)
