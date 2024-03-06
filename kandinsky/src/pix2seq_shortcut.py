import argparse
import datetime
import json
import os
import random
import time
from os.path import exists
from pathlib import Path

import numpy as np


def pix2seq_shortcut(input_dir, super_output_dir, mode="train"):

    with open(f"data/kandinsky/train_task_names.json", "r") as f:
        train_task_names = json.load(f)

    with open(f"data/kandinsky/test_task_names.json", "r") as f:
        test_task_names = json.load(f)

    # Get task directories
    super_task_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for super_task_dir in super_task_dirs:
        task_dirs = [f.path for f in os.scandir(super_task_dir) if f.is_dir()]

        for data_dir in task_dirs:
            task_name = data_dir.split("/")[-1]

            if mode == "train":
                if task_name not in train_task_names:
                    continue
            elif mode == "test":
                if task_name not in test_task_names:
                    continue

            output_dir = super_output_dir + "/" + task_name

            # Create output dir if not exist yet
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir + "/true").mkdir(parents=True, exist_ok=True)
            Path(output_dir + "/false").mkdir(parents=True, exist_ok=True)

            # parse data from instances.json
            with open(data_dir + "/true/instances.json") as f:
                data_true = json.load(f)

            with open(data_dir + "/false/instances.json") as f:
                data_false = json.load(f)

            true_results = []
            for item in data_true["annotations"]:
                image_id = item["image_id"]
                if len(true_results) <= image_id:
                    true_results.append({"boxes": [], "labels": []})

                coco_bbox = item["bbox"]
                bbox = [
                    coco_bbox[0],
                    coco_bbox[1],
                    coco_bbox[0] + coco_bbox[2],
                    coco_bbox[1] + coco_bbox[3],
                ]
                true_results[image_id]["boxes"].append(bbox)
                true_results[image_id]["labels"].append(item["category_id"])

            false_results = []
            for item in data_false["annotations"]:
                image_id = item["image_id"]
                if len(false_results) <= image_id:
                    false_results.append({"boxes": [], "labels": []})

                coco_bbox = item["bbox"]
                bbox = [
                    coco_bbox[0],
                    coco_bbox[1],
                    coco_bbox[0] + coco_bbox[2],
                    coco_bbox[1] + coco_bbox[3],
                ]
                false_results[image_id]["boxes"].append(bbox)
                false_results[image_id]["labels"].append(item["category_id"])

            for i in range(image_id + 1):
                file = "%06d" % i + ".json"
                path = output_dir + "/true/" + file

                if i < len(true_results):
                    with open(path, "w") as fp:
                        json.dump(true_results[i], fp, sort_keys=True, indent=4)
                print(f"Wrote model result for image {i} to {path}.")

                path = output_dir + "/false/" + file
                with open(path, "w") as fp:
                    json.dump(false_results[i], fp, sort_keys=True, indent=4)
                print(f"Wrote model result for image {i} to {path}.")

    print("Finished retrieving model results for images.")


if __name__ == "__main__":

    input_dir = "data/kandinsky_without_closeby/support"
    super_output_dir = "data/kandinsky_without_closeby_model_results/support"
    pix2seq_shortcut(input_dir, super_output_dir)

    input_dir = "data/kandinsky_without_closeby/query"
    super_output_dir = "data/kandinsky_without_closeby_model_results/query"
    pix2seq_shortcut(input_dir, super_output_dir)

    input_dir = "data/kandinsky_without_closeby_eval/support"
    super_output_dir = "data/kandinsky_without_closeby_model_results_eval/support"
    pix2seq_shortcut(input_dir, super_output_dir, mode="test")

    input_dir = "data/kandinsky_without_closeby_eval/query"
    super_output_dir = "data/kandinsky_without_closeby_model_results_eval/query"
    pix2seq_shortcut(input_dir, super_output_dir, mode="test")
