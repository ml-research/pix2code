import os
import pickle

from hypothesis_generation.hypothesis_utils import MetaDatasetExample

RAW_DATA_IDS_ALL_CUBES = [
    36300,
    959659,
    900335,
    282149,
    498224,
    168863,
    366851,
    159249,
    305725,
    574765,
    701044,
    696779,
    458496,
    179557,
    198599,
    514778,
    141112,
    564071,
    396703,
    137156,
    997772,
    900538,
    347267,
    505921,
    832114,
]
RAW_DATA_IDS_ALL_METAL = [
    192397,
    571973,
    832944,
    371125,
    299218,
    550928,
    307230,
    813453,
    540274,
    929485,
    798569,
    650086,
    369413,
    669194,
    29089,
    336878,
    126334,
    130855,
    347536,
    43598,
    794342,
    922908,
    330685,
    307237,
    472241,
]

DATA_LABELS = [1] * 5 + [0] * 20


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


def rename_jsons(path, or_concept=False):
    """
    Rename all json files in a directory to a format that CURI can read.
    """
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            old_name = filename
            end = filename.split("_")[-1]
            id = end.split(".")[0]
            class_value = filename.split("_")[-2]
            if or_concept:
                if class_value == "2":
                    id = int(id) + 100
                elif class_value == "1":
                    id = int(id) + 50
            else:
                if class_value == "1":
                    id = int(id) + 100

            # id as 8 digit string
            id = str(id).zfill(8)
            new_name = "ADHOC_train_" + id + ".json"
            os.rename(os.path.join(path, old_name), os.path.join(path, new_name))
            print("Renamed {} to {}".format(old_name, new_name))


def copy_raw_data_ids_for_support_set(folder, ids):
    assert len(ids) == 25

    target_folder = f"{folder}/test/scenes/"
    # get all json files in scenes
    all_file_dirs = os.listdir("curi_release/scenes")
    for dir in all_file_dirs:
        # get all json files in dir
        all_files = os.listdir("curi_release/scenes/" + dir)

        for file in all_files:
            if file.endswith(".json"):
                id = file.split("_")[-1].split(".")[0]
                raw_data_id = int(id)
                if raw_data_id in ids:
                    # copy it to support set
                    os.system(
                        "cp curi_release/scenes/"
                        + dir
                        + "/"
                        + file
                        + " "
                        + target_folder
                        + file
                    )


def get_ids(query_examples):
    """
    Get the ids of the query examples.
    """
    ids = []
    for query_example in query_examples:
        id = query_example.split("_")[-1].split(".")[0]
        ids.append(int(id))
    return ids


def create_clevr_meta_dataset(
    folder, hypothesis, hypothesis_idx_within_split, raw_data_ids_support
):
    split = "iid_sampling"
    mode = "test"

    path_to_filtered_meta_dataset = "/workspace/curi_release/hypotheses/v2_typed_simple_fol_depth_6_trials_2000000_ban_1_max_scene_id_200/" + get_path(
        split, mode
    ).replace(
        ".pkl", ".pkl"
    )

    with open(path_to_filtered_meta_dataset, "rb") as f:
        meta_dataset_and_all_hypotheses = pickle.load(f)
        meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

    clevr_meta_dataset_and_all_hypotheses = {
        "all_hypotheses_across_splits": meta_dataset_and_all_hypotheses[
            "all_hypotheses_across_splits"
        ],
        "split_name_to_all_hypothesis_idx": meta_dataset_and_all_hypotheses[
            "split_name_to_all_hypothesis_idx"
        ],
    }

    clevr_meta_dataset = []

    all_files = os.listdir(f"{folder}/test/scenes/")
    hypothesis_length = 7

    all_files.sort()

    for i in range(int(len(all_files) / 25) - 1):
        query_files = all_files[i * 25 : (i + 1) * 25]
        raw_data_ids = get_ids(query_files)
        data_labels = [1 if id < 100 else 0 for id in raw_data_ids]

        # One fix support set for all
        support_example = MetaDatasetExample(
            index=i,
            hypothesis=hypothesis,
            hypothesis_idx_within_split=hypothesis_idx_within_split,
            hypothesis_length=hypothesis_length,
            raw_data_ids=raw_data_ids_support,
            data_labels=DATA_LABELS,
            optimistic_data_labels=DATA_LABELS,
            all_valid_hypotheses=[],
            posterior_logprobs=[],
            alternate_hypotheses_for_positives=[],
            prior_logprobs=[],
        )

        query_example = MetaDatasetExample(
            index=i,
            hypothesis=hypothesis,
            hypothesis_idx_within_split=hypothesis_idx_within_split,
            hypothesis_length=hypothesis_length,
            raw_data_ids=raw_data_ids,
            data_labels=data_labels,
            optimistic_data_labels=data_labels,
            all_valid_hypotheses=[],
            posterior_logprobs=[],
            alternate_hypotheses_for_positives=[],
            prior_logprobs=[],
        )

        item = {"support": support_example, "query": query_example}
        clevr_meta_dataset.append(item.copy())

    clevr_meta_dataset_and_all_hypotheses["meta_dataset"] = clevr_meta_dataset

    # create path if not exists
    os.makedirs(f"{folder}/test/hypotheses/hypotheses_clevr", exist_ok=True)

    # save the meta dataset
    with open(
        f"{folder}/test/hypotheses/hypotheses_clevr/meta_dataset.pkl",
        "wb",
    ) as f:
        pickle.dump(clevr_meta_dataset_and_all_hypotheses, f)


def main():

    # all cubes
    hypothesis = hypothesis = "cube x shape? = lambda S. for-all="
    for n in [5, 8, 10]:
        folder = f"all-cubes/CLEVR-{n}-all-cubes"
        # rename json files
        path = f"{folder}/test/scenes/"
        rename_jsons(path, or_concept=False)
        # copy json files for support set
        copy_raw_data_ids_for_support_set(folder, ids=RAW_DATA_IDS_ALL_CUBES)
        # create clevr meta dataset
        create_clevr_meta_dataset(
            folder=folder,
            hypothesis=hypothesis,
            hypothesis_idx_within_split=7043,
            raw_data_ids_support=RAW_DATA_IDS_ALL_CUBES,
        )

    # all metal one gray
    hypothesis = "S color? gray exists= x material? metal = and lambda S. for-all="

    for n in [5, 8, 10]:
        folder = f"all-metal-one-gray-X/CLEVR-{n}-all-metal-one-gray"
        # rename json files
        path = f"{folder}/test/scenes/"
        rename_jsons(path, or_concept=False)
        # copy json files for support set
        copy_raw_data_ids_for_support_set(folder, ids=RAW_DATA_IDS_ALL_METAL)
        # create clevr meta dataset
        create_clevr_meta_dataset(
            folder=folder,
            hypothesis=hypothesis,
            hypothesis_idx_within_split=13684,
            raw_data_ids_support=RAW_DATA_IDS_ALL_METAL,
        )


if __name__ == "__main__":
    main()
