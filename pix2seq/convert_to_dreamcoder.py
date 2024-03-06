import argparse
import json
import os
import random
from pathlib import Path


def create_task_input(image_dict, domain="clevr"):

    if domain == "clevr":
        COLOR_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
        SHAPE_IDS = [8, 9, 10]
        SIZE_IDS = [11, 12]
        MATERIAL_IDS = [13, 14]
    elif domain == "kandinsky":
        COLOR_IDS = [3, 4, 5]
        SHAPE_IDS = [6, 7, 8]
        SIZE_IDS = [0, 1, 2]

    boxes = image_dict["boxes"]
    labels = image_dict["labels"]

    objects = []
    for i, box in enumerate(boxes):
        done = False
        for o in objects:
            if same_object(box, o["box"]):
                o["concepts"].append(labels[i])
                done = True
        if not done:
            o = {"box": [round(b) for b in box], "concepts": [labels[i]]}
            objects.append(o)

    input = []
    for obj in objects:
        seq = []
        seq += obj["box"]

        concepts = obj["concepts"]

        if domain == "clevr":
            if len(concepts) != 4:
                continue
            for c in concepts:
                if c in COLOR_IDS:
                    seq += [c]
                elif c in SHAPE_IDS:
                    seq += [c - 8]
                elif c in SIZE_IDS:
                    seq += [c - 11]
                elif c in MATERIAL_IDS:
                    seq += [c - 13]
                else:
                    continue
                    raise Exception(f"Concept id {c} is not known!")

        elif domain == "kandinsky":
            if len(concepts) != 3:
                continue
            for c in concepts:
                if c in COLOR_IDS:
                    seq += [c - 3]
                elif c in SHAPE_IDS:
                    seq += [c - 6]
                elif c in SIZE_IDS:
                    seq += [c]
                else:
                    continue
                    raise Exception(f"Concept id {c} is not known!")

        input.append(seq)

    return input


def same_object(box1, box2, tolerance=7):
    if (
        abs(box1[0] - box2[0]) < tolerance
        and abs(box1[1] - box2[1]) < tolerance
        and abs(box1[2] - box2[2]) < tolerance
        and abs(box1[3] - box2[3]) < tolerance
    ):
        return True


def main(input_path, output_path, domain):

    # iterate over folders in model_results
    super_tasks = [f.path for f in os.scandir(input_path) if f.is_dir()]
    for super_task in super_tasks:

        task_dirs = [f.path for f in os.scandir(super_task) if f.is_dir()]

        for task in task_dirs:
            examples = []
            task_name = task.split("/")[-1]

            # for each folder, look at true and false
            true_path = task + "/true"
            # True
            json_files = [
                f.path for f in os.scandir(true_path) if f.path.endswith(".json")
            ]

            for json_file in json_files:
                # iterate over each json in folder
                example = {}
                f = open(json_file)
                image_dict = json.load(f)
                example["input"] = create_task_input(
                    image_dict, shuffle_concepts=False, use_identifiers=False
                )
                example["output"] = True
                examples.append(example)

            # False
            false_path = task + "/false"
            json_files = [
                f.path for f in os.scandir(false_path) if f.path.endswith(".json")
            ]

            for json_file in json_files:
                example = {}
                f = open(json_file)
                image_dict = json.load(f)
                example["input"] = create_task_input(
                    image_dict, shuffle_concepts=False, use_identifiers=False
                )
                example["output"] = False
                examples.append(example)

        # save examples to folder data/dc_tasks/task_name/
        Path(output_path).mkdir(parents=True, exist_ok=True)

        with open(output_path + "/" + task_name + ".json", "w") as fp:
            json.dump(examples, fp, sort_keys=True, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="data input path")
    parser.add_argument("output_path", type=str, help="data output path")
    parser.add_argument(
        "domain",
        type=str,
        help="domain of the data",
        default="clevr",
        choices=["clevr", "kandinsky"],
    )

    # Parse the arguments
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    domain = args.domain

    if domain == "kandinsky":
        # process subfolders support and query
        input_path_support = input_path + "/support"
        output_path_support = output_path + "/support"
        # create output dir if not exist yet
        Path(output_path).mkdir(parents=True, exist_ok=True)

        main(input_path, output_path, domain)

        input_path = input_path + "/query"
        output_path = output_path + "/query"

        # create output dir if not exist yet
        Path(output_path).mkdir(parents=True, exist_ok=True)

        main(input_path, output_path, domain)
    elif domain == "clevr":
        # process folder of input path as it is
        # create output dir if not exist yet
        Path(output_path).mkdir(parents=True, exist_ok=True)
        main(input_path, output_path, domain)
