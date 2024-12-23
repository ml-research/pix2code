import argparse
import pickle
from tqdm import tqdm
import json
import os

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


def filter_meta_dataset_for_split(path, modes=["train"]):
    """Only keep one example per hypothesis."""

    for split in SPLITS:
        for mode in modes:

            # if filtered meta dataset already exists, skip
            path_to_filtered_meta_dataset = os.path.join(
                path,
                "hypotheses/filtered",
                get_path(split, mode),
            )

            try:
                with open(path_to_filtered_meta_dataset, "rb") as f:
                    print(
                        f"Filtered meta dataset for {split} and {mode} already exists."
                    )
                    continue

            except FileNotFoundError:
                pass

            path_to_meta_dataset = os.path.join(
                path,
                "hypotheses/v2_typed_simple_fol_depth_6_trials_2000000_ban_1_max_scene_id_200",
                get_path(split, mode),
            )

            with open(path_to_meta_dataset, "rb") as f:
                meta_dataset_and_all_hypotheses = pickle.load(f)
                meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

            hypotheses = set()
            filtered_meta_dataset = []

            for i in tqdm(range(len(meta_dataset))):
                hypothesis = meta_dataset[i]["support"].hypothesis

                if hypothesis in hypotheses:
                    continue

                hypotheses.add(hypothesis)
                filtered_meta_dataset.append(meta_dataset[i])

            print(f"Split {split} with mode {mode}:")
            print(f"Number of hypotheses: {len(hypotheses)}")
            print(f"Number of samples: {len(filtered_meta_dataset)}")

            # overwrite meta_dataset
            meta_dataset_and_all_hypotheses["meta_dataset"] = filtered_meta_dataset

            # create folder if not exists
            folder_path = path + "/hypotheses/filtered/"
            os.makedirs(folder_path, exist_ok=True)

            # save
            with open(path_to_filtered_meta_dataset, "wb") as f:
                pickle.dump(meta_dataset_and_all_hypotheses, f)


def create_hypotheses_subsets(
    path, modes=["train"], folder="filtered_subset", based_on_dict=False
):
    for split in SPLITS:
        for mode in modes:
            path_to_filtered_meta_dataset_subset = os.path.join(
                path,
                "hypotheses",
                folder,
                get_path(split, mode),
            )

            try:
                with open(path_to_filtered_meta_dataset_subset, "rb") as f:
                    print(
                        f"Filtered meta dataset subset for {split} and {mode} already exists."
                    )
                    continue
            except FileNotFoundError:
                pass

            path_to_filtered_meta_dataset = os.path.join(
                path,
                "hypotheses/filtered",
                get_path(split, mode).replace(".pkl", "_filtered.pkl"),
            )

            with open(path_to_filtered_meta_dataset, "rb") as f:
                meta_dataset_and_all_hypotheses = pickle.load(f)
                meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

            if based_on_dict and mode == "train":
                dict_path = os.path.join(
                    path,
                    "..",
                    "concept_data",
                    "task_id_dict_filtered.json",
                )
                with open(
                    dict_path,
                    "r",
                ) as f:
                    task_id_dict = json.load(f)
                task_ids = task_id_dict[f"{split}_{mode}"]

                # load hyp to id json
                path_to_hyp_to_id = os.path.join(
                    path, "..", "concept_data", "hyp_to_id.json"
                )
                with open(path_to_hyp_to_id, "r") as f:
                    hyp_to_id = json.load(f)

            hypotheses = set()
            meta_dataset_subset = []

            for i in tqdm(range(len(meta_dataset))):
                hypothesis = meta_dataset[i]["support"].hypothesis

                if len(list(hypotheses)) == 100 and mode == "train":
                    break

                if based_on_dict and mode == "train":
                    if hyp_to_id[hypothesis] in task_ids:
                        hypotheses.add(hypothesis)
                        meta_dataset_subset.append(meta_dataset[i])
                else:
                    hypotheses.add(hypothesis)
                    meta_dataset_subset.append(meta_dataset[i])

            print(f"Split {split} with mode {mode}:")
            print(f"Number of hypotheses: {len(hypotheses)}")
            print(f"Number of samples: {len(meta_dataset_subset)}")

            # overwrite meta_dataset
            meta_dataset_and_all_hypotheses["meta_dataset"] = meta_dataset_subset

            # create folder if not exists
            folder_path = os.path.join(path, "hypotheses", folder)
            os.makedirs(folder_path, exist_ok=True)

            # save
            with open(path_to_filtered_meta_dataset_subset, "wb") as f:
                pickle.dump(meta_dataset_and_all_hypotheses, f)


def create_task_id_list(path, folder, modes=["train"]):
    task_id_dict = {}

    for split in SPLITS:
        for mode in modes:
            if mode == "test":
                continue

            path_to_filtered_meta_dataset_subset = os.path.join(
                path,
                "hypotheses",
                folder,
                get_path(split, mode),
            )

            try:
                with open(path_to_filtered_meta_dataset_subset, "rb") as f:
                    meta_dataset_and_all_hypotheses = pickle.load(f)
                    meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

            except FileNotFoundError:
                print(f"For {split} and {mode} no file exists. Skip.")
                continue

            # load hyp to id json
            path_to_hyp_to_id = os.path.join(
                path, "..", "concept_data", "hyp_to_id.json"
            )

            with open(path_to_hyp_to_id, "r") as f:
                hyp_to_id = json.load(f)

            # create task id list
            task_id_list = []

            for i in tqdm(range(len(meta_dataset))):
                hypothesis = meta_dataset[i]["support"].hypothesis
                task_id = hyp_to_id[hypothesis]
                task_id_list.append(task_id)

            task_id_dict[f"{split}_{mode}"] = task_id_list
            print(f"Added {split}_{mode} to task_id_dict.")

    # save
    path_to_task_id_dict = os.path.join(
        path,
        "hypotheses",
        folder,
        "task_id_dict_filtered.json",
    )

    # create folder if not exists
    folder_path = os.path.join(path, "hypotheses", folder)
    os.makedirs(folder_path, exist_ok=True)

    # save task_id_dict
    with open(path_to_task_id_dict, "w") as f:
        json.dump(task_id_dict, f)


def filter_iid_test_data():
    split = "iid_sampling"
    mode = "test"
    folder = "hypotheses_subset_dc"

    path_to_filtered_meta_dataset_subset = (
        f"/workspace/curi_release/hypotheses/{folder}/" + get_path(split, mode)
    )

    try:
        with open(path_to_filtered_meta_dataset_subset, "rb") as f:
            print(
                f"Filtered meta dataset subset for {split} and {mode} already exists."
            )
            # continue
    except FileNotFoundError:
        pass

    path_to_filtered_meta_dataset = (
        "/workspace/curi_release/hypotheses/v2_typed_simple_fol_depth_6_trials_2000000_ban_1_max_scene_id_200/"
        + get_path(split, mode)
    )

    with open(path_to_filtered_meta_dataset, "rb") as f:
        meta_dataset_and_all_hypotheses = pickle.load(f)
        meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

    with open(
        f"/workspace/curi_release/hypotheses/hypotheses_subset_dc/task_id_dict_dc.json",
        "r",
    ) as f:
        task_id_dict = json.load(f)
    train_task_ids = task_id_dict[f"iid_train"]

    # load hyp to id json
    path_to_hyp_to_id = "/workspace/concept_data/hyp_to_id.json"
    with open(path_to_hyp_to_id, "r") as f:
        hyp_to_id = json.load(f)

    hypotheses = set()
    meta_dataset_subset = []

    for i in tqdm(range(len(meta_dataset))):
        hypothesis = meta_dataset[i]["support"].hypothesis

        if hyp_to_id[hypothesis] in train_task_ids:
            print(f"Skip hypothesis {hyp_to_id[hypothesis]}.")
            continue
        else:
            hypotheses.add(hypothesis)
            meta_dataset_subset.append(meta_dataset[i])

    print(f"Split {split} with mode {mode}:")
    print(f"Number of hypotheses: {len(hypotheses)}")
    print(f"Number of samples: {len(meta_dataset_subset)}")

    # overwrite meta_dataset
    meta_dataset_and_all_hypotheses["meta_dataset"] = meta_dataset_subset

    # save
    with open(path_to_filtered_meta_dataset_subset, "wb") as f:
        pickle.dump(meta_dataset_and_all_hypotheses, f)


if __name__ == "__main__":

    # parse arguments
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument(
        "--path_to_curi_data",
        type=str,
        default="pix2code/curi/curi_release",
        help="Path to curi data (curi_release folder).",
    )

    args = argsparser.parse_args()

    # filter meta dataset to only keep one example per hypothesis
    filter_meta_dataset_for_split(args.path_to_curi_data, modes=["train"])

    # create hypotheses subsets (100 for train, all for test)
    create_hypotheses_subsets(
        args.path_to_curi_data,
        modes=["train"],
        folder="filtered_subset",
        based_on_dict=True,
    )
