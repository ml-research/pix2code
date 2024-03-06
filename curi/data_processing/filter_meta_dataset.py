import pickle
from tqdm import tqdm
import json

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


def filter_meta_dataset_for_split(modes=["train", "val", "test"]):
    """Only keep one example per hypothesis."""

    for split in SPLITS:
        for mode in modes:

            # if filtered meta dataset already exists, skip
            path_to_filtered_meta_dataset = (
                "/workspace/curi_release/hypotheses/hypotheses_subset_dc/"
                + get_path(split, mode).replace(".pkl", "_filtered_subset.pkl")
            )

            try:
                with open(path_to_filtered_meta_dataset, "rb") as f:
                    print(
                        f"Filtered meta dataset for {split} and {mode} already exists."
                    )
                    # continue
            except FileNotFoundError:
                pass

            path_to_meta_dataset = (
                "/workspace/curi_release/hypotheses/v2_typed_simple_fol_depth_6_trials_2000000_ban_1_max_scene_id_200/"
                + get_path(split, mode)
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

            # save
            with open(path_to_filtered_meta_dataset, "wb") as f:
                pickle.dump(meta_dataset_and_all_hypotheses, f)


def create_hypotheses_subsets(
    modes=["train", "val", "test"], folder="hypotheses_subset", based_on_dc=False
):
    for split in SPLITS:
        for mode in modes:
            path_to_filtered_meta_dataset_subset = (
                f"/workspace/curi_release/hypotheses/{folder}/"
                + get_path(split, mode).replace(".pkl", "_filtered_subset.pkl")
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
                "/workspace/curi_release/hypotheses/filtered_hypotheses/"
                + get_path(split, mode).replace(".pkl", "_filtered.pkl")
            )

            with open(path_to_filtered_meta_dataset, "rb") as f:
                meta_dataset_and_all_hypotheses = pickle.load(f)
                meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

            if based_on_dc:
                with open(
                    f"/workspace/curi_release/hypotheses/hypotheses_subset_dc/task_id_dict_dc.json",
                    "r",
                ) as f:
                    task_id_dict = json.load(f)
                task_ids = task_id_dict[f"{split}_{mode}"]

                # load hyp to id json
                path_to_hyp_to_id = "/workspace/concept_data/hyp_to_id.json"
                with open(path_to_hyp_to_id, "r") as f:
                    hyp_to_id = json.load(f)

            hypotheses = set()
            meta_dataset_subset = []

            for i in tqdm(range(len(meta_dataset))):
                hypothesis = meta_dataset[i]["support"].hypothesis

                if len(list(hypotheses)) == 100 and mode == "train":
                    break

                if based_on_dc:
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

            # save
            with open(path_to_filtered_meta_dataset_subset, "wb") as f:
                pickle.dump(meta_dataset_and_all_hypotheses, f)


def create_task_id_list(folder):
    task_id_dict = {}
    modes = ["train", "val", "test"]

    for split in SPLITS:
        for mode in modes:
            if mode == "test":
                continue

            path_to_filtered_meta_dataset_subset = (
                f"/workspace/curi_release/hypotheses/{folder}/"
                + get_path(split, mode).replace(".pkl", "_filtered.pkl")
            )

            try:
                with open(path_to_filtered_meta_dataset_subset, "rb") as f:
                    meta_dataset_and_all_hypotheses = pickle.load(f)
                    meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

            except FileNotFoundError:
                print(f"For {split} and {mode} no file exists. Skip.")
                continue

            # load hyp to id json
            path_to_hyp_to_id = "/workspace/concept_data/hyp_to_id.json"
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
    path_to_task_id_dict = (
        f"/workspace/curi_release/hypotheses/{folder}/task_id_dict.json"
    )
    with open(path_to_task_id_dict, "w") as f:
        json.dump(task_id_dict, f)


def filter_iid_test_data():
    split = "iid_sampling"
    mode = "test"
    folder = "hypotheses_subset_dc"

    path_to_filtered_meta_dataset_subset = (
        f"/workspace/curi_release/hypotheses/{folder}/"
        + get_path(split, mode).replace(".pkl", "_filtered_subset.pkl")
    )

    try:
        with open(path_to_filtered_meta_dataset_subset, "rb") as f:
            print(
                f"Filtered meta dataset subset for {split} and {mode} already exists."
            )
            # continue
    except FileNotFoundError:
        pass

    path_to_filtered_meta_dataset = "/workspace/curi_release/hypotheses/v2_typed_simple_fol_depth_6_trials_2000000_ban_1_max_scene_id_200/" + get_path(
        split, mode
    ).replace(
        ".pkl", ".pkl"
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

    filter_meta_dataset_for_split(modes=["train"])
    create_hypotheses_subsets(
       modes=["train"], folder="hypotheses_subset_dc", based_on_dc=False
    )
    create_task_id_list("filtered_hypotheses")
    filter_iid_test_data()
