import argparse
import os
import random
from os.path import exists
from pathlib import Path

import numpy as np
import torch
from rtpt import RTPT

from main import get_args_parser, main


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # Create RTPT object
    rtpt = RTPT(name_initials="XX", experiment_name="Pix2Seq", max_iterations=100)

    # Start the RTPT tracking
    rtpt.start()

    parser = argparse.ArgumentParser(
        "Pix2Seq training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    args.resume = "./train_results/checkpoint_e299_ap370.pth"
    if args.device == "cuda":
        args.device = "cuda:0"
    args.epochs = 100
    args.lr = 3e-4
    args.lr_backbone = 3e-5
    args.batch_size = 16
    if args.seed == 42:
        print("need to specify a seed!")
        exit()

    i = args.seed
    print("Start evaluation for seed ", i)
    args.coco_path = "../data/pattern_free_clevr"
    args.resume = "./train_results/checkpoint_e299_ap370.pth"
    args.output_dir = "./train_results/clevr_multi_class/eval_" + str(i)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # train
    if exists(
        "./train_results/clevr_multi_class/eval_" + str(i) + "/checkpoint0099.pth"
    ):
        print("Already trained.")
        exit()
    if exists("./train_results/clevr_multi_class/eval_" + str(i) + "/checkpoint.pth"):
        print("Resume training")
        args.resume = (
            "./train_results/clevr_multi_class/eval_" + str(i) + "/checkpoint.pth"
        )

    main(args, rtpt)
