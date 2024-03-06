import pickle
from tqdm import tqdm
import json
from dataloaders.utils import DatasetFolderPathIndexing
from dataloaders.utils import clevr_json_loader
from dataloaders.vocabulary import ClevrJsonToTensor
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
    # "all_cubes_5",
    # "all_cubes_8",
    # "all_cubes_10",
    # "all_metal_one_gray_5",
    # "all_metal_one_gray_8",
    # "all_metal_one_gray_10",
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
        "iid_sampling": f"iid_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42_filtered_subset.pkl",
        "length_threshold": f"length_threshold_10_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
        "shape_sampling": f"shape_sampling_log_linear_{mode}_threshold_0.10_pos_im_5_neg_im_20_train_examples_500000_neg_type_alternate_hypotheses_alternate_hypo_1_random_seed_42.pkl",
    }
    return map[split]


SIZES = ["small", "large"]
COLORS = ["gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan"]
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["rubber", "metal"]


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


def extract_bounding_boxes(scene):
    # Code from https://github.com/larchen/clevr-vqa/blob/a224099addec82cf25f21d1fcbe11b15d3c02355/bounding_box.py 
    objs = scene["objects"]
    rotation = scene["directions"]["right"]

    for i, obj in enumerate(objs):
        [x, y, z] = obj["pixel_coords"]

        [x1, y1, z1] = obj["3d_coords"]

        cos_theta, sin_theta, _ = rotation

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta

        height_d = 6.9 * z1 * (15 - y1) / 2.0
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj["shape"] == "cylinder":
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
            height_d = height_u * (h - s + d) / (h + s + d)

            width_l *= 11 / (10 + y1)
            width_r = width_l

        if obj["shape"] == "cube":
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        ymin = y - height_d  # / 320.0
        ymax = y + height_u  # / 320.0
        xmin = x - width_l  # / 480.0
        xmax = x + width_r  # / 480.0

        width = xmax - xmin
        height = ymax - ymin

        xmin = xmin + width / 4
        ymin = ymin + height / 4

        width = width / 2
        height = height / 2

        obj["bbox"] = [xmin, ymin, width, height]

    return scene


def create_task_input(image_dict, shuffle_concepts=True):
    objects = image_dict["objects"]

    input = []
    for obj in objects:
        seq = []
        xmin, ymin, width, height = [int(x) for x in obj["bbox"]]
        xmax = xmin + width
        ymax = ymin + height
        seq += [xmin, ymin, xmax, ymax]

        seq += [SIZES.index(obj["size"])]
        seq += [COLORS.index(obj["color"])]
        seq += [SHAPES.index(obj["shape"])]
        seq += [MATERIALS.index(obj["material"])]

        input.append(seq)

    return input


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
        elif mode == "all_cubes":
            if split == "all_cubes_5":
                path_to_meta_dataset = "/workspace/all-cubes-X/CLEVR-5-all-cube/test/hypotheses/hypotheses_clevr/meta_dataset.pkl"
                num_objects = 5
            elif split == "all_cubes_8":
                path_to_meta_dataset = "/workspace/all-cubes-X/CLEVR-8-all-cube/test/hypotheses/hypotheses_clevr/meta_dataset.pkl"
                num_objects = 8
            elif split == "all_cubes_10":
                path_to_meta_dataset = "/workspace/all-cubes-X/CLEVR-10-all-cube/test/hypotheses/hypotheses_clevr/meta_dataset.pkl"
                num_objects = 10

        elif mode == "all_metal_one_gray":
            if split == "all_metal_one_gray_5":
                path_to_meta_dataset = "/workspace/all-metal-one-gray-X/CLEVR-5-all-metal-one-gray/test/hypotheses/hypotheses_clevr/meta_dataset.pkl"
                num_objects = 5
                json_loader_path = "/workspace/all-metal-one-gray-X/CLEVR-5-all-metal-one-gray/test/scenes"
            elif split == "all_metal_one_gray_8":
                path_to_meta_dataset = "/workspace/all-metal-one-gray-X/CLEVR-8-all-metal-one-gray/test/hypotheses/hypotheses_clevr/meta_dataset.pkl"
                num_objects = 8
                json_loader_path = "/workspace/all-metal-one-gray-X/CLEVR-8-all-metal-one-gray/test/scenes"
            elif split == "all_metal_one_gray_10":
                path_to_meta_dataset = "/workspace/all-metal-one-gray-X/CLEVR-10-all-metal-one-gray/test/hypotheses/hypotheses_clevr/meta_dataset.pkl"
                num_objects = 10
                json_loader_path = "/workspace/all-metal-one-gray-X/CLEVR-10-all-metal-one-gray/test/scenes"

        with open(path_to_meta_dataset, "rb") as f:
            meta_dataset_and_all_hypotheses = pickle.load(f)
            meta_dataset = meta_dataset_and_all_hypotheses["meta_dataset"]

        # load hyp to id json
        path_to_hyp_to_id = "/workspace/concept_data/hyp_to_id.json"
        with open(path_to_hyp_to_id, "r") as f:
            hyp_to_id = json.load(f)

        modality_to_transform_fn = {
            "json": ClevrJsonToTensor("concept_data/clevr_typed_fol_properties.json"),
        }
        # json loader
        loader = get_loader_for_modality(
            "curi_release/images",
            json_loader_path,
            "json",
            modality_to_transform_fn=modality_to_transform_fn,
        )

        support_task_dict = {}
        query_task_dict = {}
        hyp_counter = [0] * 14929

        for i in tqdm(range(len(meta_dataset))):
            hypothesis = meta_dataset[i]["support"].hypothesis

            # get id for hypothesis
            hypothesis_id = hyp_to_id[hypothesis]

            if mode == "query" or mode == "test":
                key = "query"
            else:
                key = "support"

            # get support / query samples for hypothesis
            support_sample_ids = meta_dataset[i]["support"].raw_data_ids
            query_sample_ids = meta_dataset[i]["query"].raw_data_ids

            support_data_labels = meta_dataset[i]["support"].data_labels
            query_data_labels = meta_dataset[i]["query"].data_labels

            support_sample_paths = [
                x["path"] for x in loader.get_item_list(support_sample_ids)
            ]
            query_sample_paths = [
                x["path"] for x in loader.get_item_list(query_sample_ids)
            ]

            support_samples = [json.load(open(x, "r")) for x in support_sample_paths]
            query_samples = [json.load(open(x, "r")) for x in query_sample_paths]

            # convert support to tasks
            support_examples = []
            for i, scene in enumerate(support_samples):
                scene = extract_bounding_boxes(scene)

                example = {}
                example["input"] = create_task_input(scene)
                example["output"] = True if support_data_labels[i] == 1 else False
                support_examples.append(example)

            # convert query to tasks
            query_examples = []
            for i, scene in enumerate(query_samples):
                scene = extract_bounding_boxes(scene)

                example = {}
                example["input"] = create_task_input(scene)
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

    create_curi_tasks(
        target_folder="curi",
        mode=f"train",
        single_support_samples=True,
    )

    create_curi_tasks(
        target_folder="curi_dc_test_tasks",
        mode=f"test",
        single_support_samples=False,
    )

    # create_curi_tasks(
    #     target_folder="curi_dc_test_tasks",
    #     mode=f"all_cubes",
    #     single_support_samples=False,
    # )

    # create_curi_tasks(
    #     target_folder="curi_dc_test_tasks",
    #     mode=f"all_metal_one_gray",
    #     single_support_samples=False,
    # )
