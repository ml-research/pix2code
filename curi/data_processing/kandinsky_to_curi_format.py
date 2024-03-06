import os
import pickle
import json

from hypothesis_generation.hypothesis_utils import MetaDatasetExample, HypothesisEval


def rename_file(path, current_name, new_id, extension=".png"):
    """
    Rename a file to a new id.
    """
    new_id = str(new_id).zfill(8)
    new_name = "ADHOC_train_" + new_id + extension

    os.rename(os.path.join(path, current_name), os.path.join(path, new_name))

    return new_name


def get_all_kandinsky_hypotheses(folder, folder_eval):
    all_hypotheses = []
    hypothesis_indices_train = []
    hypothesis_indices_test = []
    map_name_to_id = {}

    idx = 0

    # load all hypotheses from json file
    with open(f"{folder}/train_task_names.json", "r") as f:
        train_task_names = json.load(f)

    with open(f"{folder}/test_task_names.json", "r") as f:
        test_task_names = json.load(f)

    for task in train_task_names:
        hypothesis = task
        all_hypotheses.append(hypothesis)
        hypothesis_indices_train.append(idx)
        map_name_to_id[hypothesis] = idx
        idx += 1

    for task in test_task_names:
        hypothesis = task
        all_hypotheses.append(hypothesis)
        hypothesis_indices_test.append(idx)
        map_name_to_id[hypothesis] = idx
        idx += 1

    split_name_to_all_hypothesis_idx = {
        "train": hypothesis_indices_train,
        "val": hypothesis_indices_test,
        "test": hypothesis_indices_test,
    }

    return all_hypotheses, split_name_to_all_hypothesis_idx, map_name_to_id


def create_meta_dataset_and_all_hypotheses():
    (
        hypotheses,
        split_name_to_all_hypothesis_idx,
        map_name_to_id,
    ) = get_all_kandinsky_hypotheses(f"kandinsky_copy", f"kandinsky_eval_copy")

    hypotheses_eval = HypothesisEval(
        hypothesis=hypotheses,
        image_id_list=[[0]] * len(hypotheses),
        length=[],
        logprob=[],
    )

    meta_dataset_and_all_hypotheses = {
        "all_hypotheses_across_splits": hypotheses_eval,
        "split_name_to_all_hypothesis_idx": split_name_to_all_hypothesis_idx,
    }

    return meta_dataset_and_all_hypotheses, map_name_to_id


def create_kandinsky_meta_data_set(
    folder,
    target_folder,
    mode,
    meta_dataset_and_all_hypotheses,
    map_name_to_id,
    id_offset=0,
    extension=".png",
):
    kandinksy_meta_dataset = []

    data_id_counter = 0 + id_offset
    hypothesis_length = 7
    example_id = 0

    with open(f"kandinsky_copy/train_task_names.json", "r") as f:
        train_task_names = json.load(f)

    with open(f"kandinsky_copy/test_task_names.json", "r") as f:
        test_task_names = json.load(f)

    if mode == "train":
        task_names = train_task_names
    elif mode == "test":
        task_names = test_task_names

    print(task_names)

    if extension == ".png":
        ignore = "json"
    else:
        ignore = "png"

    test_counter = 0
    # get all folders in folder
    for group in os.listdir(f"{folder}/support"):
        # all folders in group
        for task in os.listdir(f"{folder}/support/{group}"):
            hypothesis = task

            test_counter += 1
            if not task in task_names:
                continue

            support_true = [
                x
                for x in os.listdir(f"{folder}/support/{group}/{task}/true")
                if ignore not in x and "instances" not in x
            ]
            support_false = [
                x
                for x in os.listdir(f"{folder}/support/{group}/{task}/false")
                if ignore not in x and "instances" not in x
            ]
            query_true = [
                x
                for x in os.listdir(f"{folder}/query/{group}/{task}/true")
                if ignore not in x and "instances" not in x
            ]
            query_false = [
                x
                for x in os.listdir(f"{folder}/query/{group}/{task}/false")
                if ignore not in x and "instances" not in x
            ]

            support_data_ids = []
            support_labels = []

            for file in support_true:
                # rename file
                new_name = rename_file(
                    f"{folder}/support/{group}/{task}/true", file, data_id_counter
                )
                support_data_ids.append(data_id_counter)
                support_labels.append(1)
                data_id_counter += 1

            for file in support_false:
                # rename file
                new_name = rename_file(
                    f"{folder}/support/{group}/{task}/false", file, data_id_counter
                )
                support_data_ids.append(data_id_counter)
                support_labels.append(0)
                data_id_counter += 1

            # number of episodes (more query than support examples?)
            num_episodes = len(query_true) // 5

            episodes = []
            for i in range(num_episodes):
                episodes.append(
                    [
                        query_true[i * 5 : (i + 1) * 5],
                        query_false[i * 20 : (i + 1) * 20],
                    ]
                )

            for episode in episodes:
                query_true_files = episode[0]
                query_false_files = episode[1]

                query_data_ids = []
                query_labels = []

                for file in query_true_files:
                    # rename file
                    new_name = rename_file(
                        f"{folder}/query/{group}/{task}/true", file, data_id_counter
                    )
                    query_data_ids.append(data_id_counter)
                    query_labels.append(1)
                    data_id_counter += 1

                for file in query_false_files:
                    # rename file
                    new_name = rename_file(
                        f"{folder}/query/{group}/{task}/false", file, data_id_counter
                    )
                    query_data_ids.append(data_id_counter)
                    query_labels.append(0)
                    data_id_counter += 1

                # support example for task
                support_example = MetaDatasetExample(
                    index=example_id,
                    hypothesis=hypothesis,
                    hypothesis_idx_within_split=map_name_to_id[hypothesis],
                    hypothesis_length=hypothesis_length,
                    raw_data_ids=support_data_ids,
                    data_labels=support_labels,
                    optimistic_data_labels=support_labels,
                    all_valid_hypotheses=[],
                    posterior_logprobs=[],
                    alternate_hypotheses_for_positives=[],
                    prior_logprobs=[],
                )

                # query example for task
                query_example = MetaDatasetExample(
                    index=example_id,
                    hypothesis=hypothesis,
                    hypothesis_idx_within_split=map_name_to_id[hypothesis],
                    hypothesis_length=hypothesis_length,
                    raw_data_ids=query_data_ids,
                    data_labels=query_labels,
                    optimistic_data_labels=query_labels,
                    all_valid_hypotheses=[],
                    posterior_logprobs=[],
                    alternate_hypotheses_for_positives=[],
                    prior_logprobs=[],
                )

                example_id += 1

                item = {"support": support_example, "query": query_example}
                kandinksy_meta_dataset.append(item.copy())

    meta_dataset_and_all_hypotheses["meta_dataset"] = kandinksy_meta_dataset

    # create path if not exists
    os.makedirs(f"{target_folder}/hypotheses/hypotheses_kandinsky", exist_ok=True)

    # save the meta dataset
    with open(
        f"{target_folder}/hypotheses/hypotheses_kandinsky/meta_dataset_image_{mode}.pkl",
        "wb",
    ) as f:
        pickle.dump(meta_dataset_and_all_hypotheses, f)


def collect_files(folder, mode, extension=".png"):
    if extension == ".png":
        target_folder = f"kandinsky/images/"
    else:
        target_folder = f"kandinsky/scenes/"

    with open(f"kandinsky_copy/train_task_names.json", "r") as f:
        train_task_names = json.load(f)

    with open(f"kandinsky_copy/test_task_names.json", "r") as f:
        test_task_names = json.load(f)

    # create path if not exists
    os.makedirs(target_folder, exist_ok=True)

    for group in os.listdir(f"{folder}/support"):
        # all folders in group
        for task in os.listdir(f"{folder}/support/{group}"):
            if mode == "train" and not task in train_task_names:
                continue
            elif mode == "test" and not task in test_task_names:
                continue

            for file in os.listdir(f"{folder}/support/{group}/{task}/true"):
                if not extension in file or "instances" in file:
                    continue
                # copy file to target folder
                os.system(
                    f"cp {folder}/support/{group}/{task}/true/{file} {target_folder}"
                )

            for file in os.listdir(f"{folder}/support/{group}/{task}/false"):
                if not extension in file or "instances" in file:
                    continue
                # copy file to target folder
                os.system(
                    f"cp {folder}/support/{group}/{task}/false/{file} {target_folder}"
                )

            for file in os.listdir(f"{folder}/query/{group}/{task}/true"):
                if not extension in file or "instances" in file:
                    continue
                # copy file to target folder
                os.system(
                    f"cp {folder}/query/{group}/{task}/true/{file} {target_folder}"
                )

            for file in os.listdir(f"{folder}/query/{group}/{task}/false"):
                if not extension in file or "instances" in file:
                    continue
                # copy file to target folder
                os.system(
                    f"cp {folder}/query/{group}/{task}/false/{file} {target_folder}"
                )

    print("Done copying files.")


def main():
    (
        meta_dataset_and_all_hypotheses,
        map_name_to_id,
    ) = create_meta_dataset_and_all_hypotheses()

    folder = f"kandinsky_train"
    # collect json files
    create_kandinsky_meta_data_set(
        folder, "kandinsky", "train", meta_dataset_and_all_hypotheses, map_name_to_id
    )

    collect_files(folder, "train", extension=".png")

    folder = "kandinsky_eval"
    create_kandinsky_meta_data_set(
        folder,
        "kandinsky",
        "test",
        meta_dataset_and_all_hypotheses,
        map_name_to_id,
        id_offset=50000,
    )

    collect_files(folder, "test", extension=".png")


if __name__ == "__main__":
    main()
