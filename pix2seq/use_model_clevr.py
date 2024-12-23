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

    try:
        # open the annotation file
        with open(ann_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(
            f"Error loading data from {PATHS[image_set][1]}, start creating dummy instances json..."
        )

        # create dummy annotation file
        annotations = {}
        annotations["annotations"] = []
        annotations["images"] = []

        # get all images in the directory
        files = [
            f
            for f in os.listdir(PATHS[image_set][0])
            if exists(os.path.join(PATHS[image_set][0], f))
        ]
        # get the ids of the files (name_name_id.png -> id)
        ids = [f.split("_")[-1].split(".")[0] for f in files]
        # ids to int
        ids = [int(i) for i in ids]
        # sort the ids
        ids.sort()
        files.sort()

        for file, id in zip(files, ids):
            annotations["images"].append({"file_name": file, "id": id})

        # save the annotations to the json file
        with open(ann_file, "w") as f:
            json.dump(annotations, f, indent=4)

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


if __name__ == "__main__":
    print("Start pix2seq model for detecting concept classes in CLEVR images...")

    # Create RTPT object
    rtpt = RTPT(name_initials="XX", experiment_name="Pix2SeqEval", max_iterations=250)

    # Start the RTPT tracking
    rtpt.start()

    # Process Arguments
    parser = argparse.ArgumentParser(
        "Pix2Seq training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    args.device = "cuda"
    args.batch_size = 64
    args.eval = True
    args.distributed = True

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Input data
    if args.coco_path:
        print(f"Use data from {args.coco_path}")
    else:
        raise ValueError("No data path provided")

    # Output dir
    if args.output_dir:
        print(f"Save results to {args.output_dir}")
    else:
        raise ValueError("No output dir provided")

    # Path to trained model
    if not args.resume:
        args.resume = "pix2seq/train_results/clevr/checkpoint_best.pth"

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

    # Get image directories
    task_dirs = [f.path for f in os.scandir(args.coco_path) if f.is_dir()]

    dirs_done = 0
    for data_dir in task_dirs:

        task_name = data_dir.split("/")[-1]

        output_dir = args.output_dir

        # Create output dir if not exist yet
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # build dataset
        dataset = build_task_dataset(image_set="true", path=data_dir, args=args)

        if args.distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        data_loader = DataLoader(
            dataset,
            args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

        base_ds = get_coco_api_from_dataset(dataset)

        metric_logger = utils.MetricLogger(delimiter="  ")

        iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())

        with torch.no_grad():
            for samples, targets in data_loader:
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model([samples, targets])
                results = postprocessors["bbox"].get_output_seq(outputs, targets)

                for i, target in enumerate(targets):
                    image_id = int(target["image_id"])
                    root = Path(output_dir)
                    file = "%06d" % image_id + ".json"
                    path = root / file
                    with open(path, "w") as fp:
                        json.dump(results[i], fp, sort_keys=True, indent=4)
                    print(f"Wrote model result for image {image_id} to {path}.")

                rtpt.step()

    print("Finished retrieving model results for images.")
