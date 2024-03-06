import pickle
from tqdm import tqdm
import json
from dataloaders.utils import DatasetFolderPathIndexing
from dataloaders.utils import clevr_json_loader
from dataloaders.vocabulary import ClevrJsonToTensor
import os
import argparse

SPLITS = [
    "color_boolean",
    "color_count",
    "color_location",
    "color_material",
    "color_sampling",
    "comp_sampling",
    "iid_sampling",
    "length_threshold",
    "shape_sampling",
]


def get_path(split, mode):
    map = {
        "iid": f"iid_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "color_boolean": f"color_boolean_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "color_count": f"color_count_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "color_location": f"color_location_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "color_material": f"color_material_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "color_sampling": f"color_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "comp_sampling": f"comp_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "iid_sampling": f"iid_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "length_threshold": f"length_threshold_10_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "shape_sampling": f"shape_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
    }
    return map[split]


SIZES = ["small", "large"]
COLORS = ["gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan"]
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["rubber", "metal"]


color_map = {0: 0, 1: 4, 2: 1, 3: 5, 4: 2, 5: 6, 6: 7, 7: 3}

COLOR_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
SHAPE_IDS = [8, 9, 10]
SIZE_IDS = [11, 12]
MATERIAL_IDS = [13, 14]


def create_task_input_clevr(image_dict):
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

        if len(concepts) != 4:
            continue

        concepts = [concepts[2], concepts[0], concepts[1], concepts[3]]

        for c in concepts:
            if c in COLOR_IDS:

                color = color_map[c]
                seq += [color]
            elif c in SHAPE_IDS:
                seq += [c - 8]
            elif c in SIZE_IDS:
                seq += [c - 11]
            elif c in MATERIAL_IDS:
                seq += [c - 13]
            else:
                continue

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


def get_loader_for_modality(
    image_path, json_path, modality, modality_to_transform_fn: dict
):
    if modality == "json":
        loader = DatasetFolderPathIndexing(
            json_path,
            clevr_json_loader,
            extensions=".json",
            transform=modality_to_transform_fn[modality],
        )
    return loader


def get_json_path(x):
    # get 6 digit number from x
    x = str(x)
    x = x.zfill(6)
    path = f"/workspace/data/curi_release_model_results/{x}.json"

    return path


def create_curi_tasks(target_folder, mode="support", single_support_samples=True):
    """Create DC tasks from CURI test set."""

    for split in SPLITS:
        if mode == "test":
            path_to_meta_dataset = (
                "/workspace/curi_release/hypotheses/v2_typed_simple_fol_depth_6_trials_2000000_ban_1_max_scene_id_200/"
                + get_path(split, "test")
            )
        elif mode == "train":
            path_to_meta_dataset = (
                "/workspace/curi_release/hypotheses/hypotheses_subset_dc/"
                + get_path(split, "train")
            )

        with open(path_to_meta_dataset, "rb") as f:
            meta_dataset_and_all_hypotheses = pickle.load(f)
            meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

        # load hyp to id json
        path_to_hyp_to_id = "/workspace/concept_data/hyp_to_id.json"
        with open(path_to_hyp_to_id, "r") as f:
            hyp_to_id = json.load(f)

        support_task_dict = {}
        query_task_dict = {}
        hyp_counter = [0] * 14929

        for i in tqdm(range(len(meta_dataset))):
            hypothesis = meta_dataset[i]["support"].hypothesis

            # get id for hypothesis
            hypothesis_id = hyp_to_id[hypothesis]

            # get support / query samples for hypothesis
            support_sample_ids = meta_dataset[i]["support"].raw_data_ids
            query_sample_ids = meta_dataset[i]["query"].raw_data_ids

            support_data_labels = meta_dataset[i]["support"].data_labels
            query_data_labels = meta_dataset[i]["query"].data_labels

            # get paths for support / query samples
            support_sample_paths = [get_json_path(x) for x in support_sample_ids]
            query_sample_paths = [get_json_path(x) for x in query_sample_ids]

            support_samples = [json.load(open(x, "r")) for x in support_sample_paths]
            query_samples = [json.load(open(x, "r")) for x in query_sample_paths]

            # convert model output to dc format
            support_samples = [create_task_input_clevr(x) for x in support_samples]
            query_samples = [create_task_input_clevr(x) for x in query_samples]

            # convert support to tasks
            support_examples = []
            for i, scene in enumerate(support_samples):
                example = {}
                example["input"] = scene
                example["output"] = True if support_data_labels[i] == 1 else False
                support_examples.append(example)

            # convert query to tasks
            query_examples = []
            for i, scene in enumerate(query_samples):

                example = {}
                example["input"] = scene
                example["output"] = True if query_data_labels[i] == 1 else False
                query_examples.append(example)

            if single_support_samples:
                hypothesis_id_n = (
                    str(hypothesis_id) + "_" + str(hyp_counter[hypothesis_id])
                )
                support_task_dict[hypothesis_id_n] = support_examples.copy()
                query_task_dict[hypothesis_id_n] = query_examples.copy()
                hyp_counter[hypothesis_id] += 1

            else:
                if hypothesis_id in support_task_dict:
                    query_task_dict[hypothesis_id] += query_examples
                else:
                    # only add one set of support examples
                    support_task_dict[hypothesis_id] = support_examples
                    query_task_dict[hypothesis_id] = query_examples

        # save task_dict
        for id in support_task_dict:
            path_to_task_dict = (
                f"/workspace/{target_folder}/{split}/support/task_{id}.json"
            )
            # if path does not exist, create it
            if not os.path.exists(os.path.dirname(path_to_task_dict)):
                os.makedirs(os.path.dirname(path_to_task_dict))

            with open(path_to_task_dict, "w") as f:
                json.dump(support_task_dict[id], f)

        for id in query_task_dict:
            path_to_task_dict = (
                f"/workspace/{target_folder}/{split}/query/task_{id}.json"
            )
            # if path does not exist, create it
            if not os.path.exists(os.path.dirname(path_to_task_dict)):
                os.makedirs(os.path.dirname(path_to_task_dict))

            with open(path_to_task_dict, "w") as f:
                json.dump(query_task_dict[id], f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("target_folder", type=str, help="data target folder")
    parser.add_argument(
        "mode",
        type=str,
        help="train or test",
        default="test",
        choices=["train", "test"],
    )

    # Parse the arguments
    args = parser.parse_args()

    target_folder = args.target_folder
    mode = args.mode

    create_curi_tasks(
        target_folder=target_folder,
        mode=mode,
        single_support_samples=False,
    )
