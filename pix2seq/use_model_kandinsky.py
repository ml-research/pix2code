import argparse
import datetime
import json
import os
import random
import time
from os.path import exists
from pathlib import Path

import datasets
import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import CocoDetection, make_coco_transforms
from engine import evaluate, train_one_epoch
from main import get_args_parser, main

# from models import build_model
from playground import build_all_model
from rtpt import RTPT
from timm.utils import NativeScaler
from torch.utils.data import DataLoader, DistributedSampler


def build_task_dataset(image_set, path, args):
    root = Path(path)
    assert root.exists(), f"provided path {root} does not exist"
    mode = "instances"
    PATHS = {
        "true": (root, root / f"{mode}.json"),
        "false": (root, root / f"{mode}.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=False,
        large_scale_jitter=args.large_scale_jitter,
        image_set=image_set,
    )
    return dataset


def get_id(file):
    if "json" in file:
        return int(file.split(".")[0])
    else:
        file_name = file.split(".")[0]
        return int(file_name.split("_")[-1])


def use_model(input_dir, output_dir, args):
    print("Start pix2seq model for detecting concept classes in Kandinsky Patterns...")

    # Create RTPT object
    rtpt = RTPT(name_initials="XX", experiment_name="Pix2SeqEval", max_iterations=250)

    # Start the RTPT tracking
    rtpt.start()

    args.coco_path = input_dir
    args.output_dir = output_dir
    args.device = "cuda"
    args.batch_size = 32
    args.eval = True
    args.distributed = True

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Input data
    if args.coco_path:
        print(f"Use data from {args.coco_path}")
    else:
        raise ValueError("No data provided")

    # Output dir
    if args.output_dir:
        print(f"Save results to {args.output_dir}")
    else:
        raise ValueError("No output dir provided")

    # Path to trained model
    if not args.resume:
        args.resume = "pix2seq/train_results/relkp/checkpoint_best.pth"

    # Set seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_all_model[args.model](args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = model.module

    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print("Pretrained model loaded")

    model.eval()

    # Get task directories
    task_dirs = [f.path for f in os.scandir(args.coco_path) if f.is_dir()]

    dirs_done = 0
    for data_dir in task_dirs:

        json_path = "data/kandinsky/test_task_names.json"
        test_task_names = json.load(open(json_path, "r"))

        task_name = data_dir.split("/")[-1]
        if task_name not in test_task_names:
            print(f"Task {task_name} not in test_task_names.")
            continue

        output_dir = args.output_dir + "/" + task_name

        # Create output dir if not exist yet
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir + "/true").mkdir(parents=True, exist_ok=True)
        Path(output_dir + "/false").mkdir(parents=True, exist_ok=True)

        # build datasets of positive and negative examples
        dataset_true = build_task_dataset(
            image_set="true", path=data_dir + "/true", args=args
        )
        dataset_false = build_task_dataset(
            image_set="false", path=data_dir + "/false", args=args
        )

        if args.distributed:
            sampler_true = DistributedSampler(dataset_true)
        else:
            sampler_true = torch.utils.data.SequentialSampler(dataset_true)
            sampler_false = torch.utils.data.SequentialSampler(dataset_false)

        data_loader_true = DataLoader(
            dataset_true,
            args.batch_size,
            sampler=sampler_true,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
        data_loader_false = DataLoader(
            dataset_false,
            args.batch_size,
            sampler=sampler_false,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

        base_ds_true = get_coco_api_from_dataset(dataset_true)
        base_ds_fasle = get_coco_api_from_dataset(dataset_false)

        metric_logger = utils.MetricLogger(delimiter="  ")

        iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())

        with torch.no_grad():
            for samples, targets in data_loader_true:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model([samples, targets])
                results = postprocessors["bbox"].get_output_seq(outputs, targets)

                for i, target in enumerate(targets):
                    image_id = int(target["image_id"])
                    root = Path(output_dir)
                    file = "%06d" % image_id + ".json"
                    path = root / "true" / file
                    with open(path, "w") as fp:
                        json.dump(results[i], fp, sort_keys=True, indent=4)
                    print(f"Wrote model result for image {image_id} to {path}.")

                rtpt.step()

        with torch.no_grad():
            for samples, targets in data_loader_false:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model([samples, targets])
                results = postprocessors["bbox"].get_output_seq(outputs, targets)

                for i, target in enumerate(targets):
                    image_id = int(target["image_id"])
                    root = Path(output_dir)
                    file = "%06d" % image_id + ".json"
                    path = root / "false" / file
                    with open(path, "w") as fp:
                        json.dump(results[i], fp, sort_keys=True, indent=4)
                    print(f"Wrote model result for image {image_id} to {path}.")

                rtpt.step()

    print("Finished retrieving model results for images.")


if __name__ == "__main__":

    # Process Arguments
    parser = argparse.ArgumentParser(
        "Pix2Seq training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    super_data_dir = args.coco_path
    super_output_dir = args.output_dir

    # support
    data_dir = super_data_dir + "/support/"
    output_dir = super_output_dir + "/support"

    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        output_dir_sub = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            print("Processing", subdir_path)
            use_model(subdir_path, output_dir_sub, args)

    # query
    data_dir = super_data_dir + "/query/"
    output_dir = super_output_dir + "/query"

    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        output_dir_sub = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            print("Processing", subdir_path)
            use_model(subdir_path, output_dir_sub, args)
