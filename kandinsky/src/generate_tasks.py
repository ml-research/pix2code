import json
import os
import random
import shutil
from kp.generate_task_clauses import Clause, generate_clauses
from kp.ClauseBasedKandinskyFigure import ClauseBasedKandinskyFigure

from kp import (
    KandinskyCaptions,
    KandinskyUniverse,
    RandomKandinskyFigure,
    SameConcept,
    SameColorSameShape,
)
from kp.KandinskyUniverse import (
    get_area,
    get_bounding_box,
    get_coco_bbox_from_bbox,
    get_coco_bounding_box,
    get_relation_area,
    kandinskyFigureAsFact,
)
from map.class_combinations import (
    generate_class_mapping,
    generate_concept_mapping,
    generate_concept_relation_mapping,
    get_class_categories_dict,
    get_concept_categories_dict,
    get_concept_class_id,
    get_concept_relation_categories_dict,
    get_concept_relation_class_id,
)


from kp.KandinskyUniverse import (
    get_area,
    get_bounding_box,
    get_coco_bbox_from_bbox,
    get_coco_bounding_box,
)

u = KandinskyUniverse.SimpleUniverse()
cg = KandinskyCaptions.CaptionGenerator(u)


def get_curi_dict(obj_dict):
    curi_dict = {
        "pixel_coords": obj_dict["pos"],
        "size": obj_dict["size_cls"],
        "shape": obj_dict["shape"],
        "3d_coords": obj_dict["pos"],
        "color": obj_dict["color"],
    }
    return curi_dict


def generate_task_examples(basedir, kfgen, n=50, width=200, curi_annotations=True):
    os.makedirs(basedir, exist_ok=True)

    instances = dict()
    images = []
    annotations = []
    ann_id = 0

    if n == 25:
        true_n = 5
        false_n = 20
    else:
        i = int(n / 25)
        true_n = i * 5
        false_n = i * 20
    # build instances json
    kfs = kfgen.true_kf(true_n)
    if kfs is None:
        shutil.rmtree(basedir)
        return
    for i, kf in enumerate(kfs):
        if n > 20 and i % 10 == 0:
            print(f"Generating {i}th image")
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/true/%06d" % i + ".png")

        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})

        # annotations
        curi_image_annotation = {"objects": []}
        for j, obj in enumerate(kf):
            obj_dic = kf[j].__dict__
            obj_dic["object_id"] = j

            img_dic = {"img_id": i, "scene": []}
            img_dic["scene"].append(obj_dic)

            curi_image_annotation["objects"].append(get_curi_dict(obj_dic))

            for concept in ["size_cls", "color", "shape"]:
                annotation_dict = {
                    "image_id": i,
                    "pos": obj_dic["pos"],
                    "bbox": get_coco_bounding_box(kf[j], width=width),
                    "iscrowd": 0,
                    "category_id": get_concept_class_id(obj_dic[concept]),
                    "object_id": j,
                    "id": ann_id,
                    "area": get_area(kf[j], width=width),
                }

                annotations.append(annotation_dict)
                ann_id += 1

        # save annotation_dict
        with open(basedir + "/true/%06d" % i + ".json", "w") as f:
            json.dump(curi_image_annotation, f, sort_keys=True, indent=4)

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_concept_categories_dict()

    with open(basedir + "/true/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)

    # reset annotations
    instances = dict()
    images = []
    annotations = []
    ann_id = 0

    # build negative instances json
    kfs = kfgen.false_kf(false_n)
    if kfs is None:
        shutil.rmtree(basedir)
        return
    for i, kf in enumerate(kfs):
        if n > 20 and i % 10 == 0:
            print(f"Generating {i}th image")
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/false/%06d" % i + ".png")

        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})

        curi_image_annotation = {"objects": []}
        # annotations
        for j, obj in enumerate(kf):
            obj_dic = kf[j].__dict__
            obj_dic["object_id"] = j

            img_dic = {"img_id": i, "scene": []}
            img_dic["scene"].append(obj_dic)

            curi_image_annotation["objects"].append(get_curi_dict(obj_dic))

            for concept in ["size_cls", "color", "shape"]:
                annotation_dict = {
                    "image_id": i,
                    "pos": obj_dic["pos"],
                    "bbox": get_coco_bounding_box(kf[j], width=width),
                    "iscrowd": 0,
                    "category_id": get_concept_class_id(obj_dic[concept]),
                    "object_id": j,
                    "id": ann_id,
                    "area": get_area(kf[j], width=width),
                }
                # save annotation_dict
                with open(basedir + "/false/%06d" % i + ".json", "w") as f:
                    json.dump(annotation_dict, f, sort_keys=True, indent=4)

                annotations.append(annotation_dict)
                ann_id += 1

        # save annotation_dict
        with open(basedir + "/false/%06d" % i + ".json", "w") as f:
            json.dump(curi_image_annotation, f, sort_keys=True, indent=4)

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_concept_categories_dict()

    with open(basedir + "/false/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)

    task_name = basedir.split("/")[-1]
    print(f"Created data for task {task_name}.\n")


def generate_task_validation_examples(
    path, num_examples=20, eval=False, parse_support=False
):

    test_task_names = None
    train_task_names = None

    # load train task names if available
    try:
        with open(f"data/kandinsky/train_task_names.json", "r") as f:
            train_task_names = json.load(f)
    except:
        print("No train task names available")

    # load test task names if available
    try:
        with open(f"data/kandinsky/test_task_names.json", "r") as f:
            test_task_names = json.load(f)
    except:
        print("No test task names available")

    data_dir = path

    random.seed(1244)

    no_pair_1, no_pair_2, pair_clauses = generate_clauses()

    if num_examples == 20:
        max_true_images = 5
        max_false_images = 20
    else:
        sets = int(num_examples / 25)
        max_true_images = 5 * sets
        max_false_images = 20 * sets

    for i in range(2, 7):
        super_task_name = str(i) + "_no_pairs_1"

        clauses_1 = no_pair_1
        clauses_2 = no_pair_2
        random.shuffle(clauses_1)
        random.shuffle(clauses_2)

        for c in clauses_1:

            task_name = str(c).replace(" ", "")
            cur_task_name = str(i) + "_" + task_name

            if eval:
                if test_task_names:
                    if cur_task_name not in test_task_names:
                        continue
                if train_task_names:
                    if cur_task_name in train_task_names:
                        continue

            if not eval:
                if train_task_names:
                    if cur_task_name not in train_task_names:
                        continue
                if test_task_names:
                    if cur_task_name in test_task_names:
                        continue

            if cur_task_name in unsolvable:
                continue

            if super_task_name in ["2_no_pairs_1", "3_no_pairs_1"]:
                n = 10
            else:
                n = 6

            # add folder if not exists
            if not os.path.exists(data_dir + super_task_name):
                os.makedirs(data_dir + super_task_name)

            # check if already 6 tasks exist
            n_tasks = [f for f in os.listdir(data_dir + super_task_name)]
            if len(n_tasks) >= n:
                print(
                    f"Already {n} tasks in {data_dir + super_task_name}. Skipping. \n"
                )
                break

            print(f"Start creating data for {cur_task_name}")
            path = data_dir + super_task_name + "/" + cur_task_name
            # check if path already exists
            if os.path.exists(path):
                true_path = path + "/true"
                false_path = path + "/false"
                # count files in path
                num_true_images = len(
                    [
                        f
                        for f in os.listdir(true_path)
                        if os.path.isfile(os.path.join(true_path, f))
                        and f.endswith(".png")
                    ]
                )
                num_false_images = len(
                    [
                        f
                        for f in os.listdir(false_path)
                        if os.path.isfile(os.path.join(false_path, f))
                        and f.endswith(".png")
                    ]
                )

                if (
                    num_true_images == max_true_images
                    and num_false_images == max_false_images
                ):
                    print(f"Task {path} already exists correctly. Skipping. \n")
                    # task_set.discard(cur_task_name)
                    continue

            true_path = path + "/true"
            false_path = path + "/false"
            os.makedirs(true_path, exist_ok=True)
            os.makedirs(false_path, exist_ok=True)
            kf = ClauseBasedKandinskyFigure(u, i, i, clause=c)
            generate_task_examples(path, kf, n=num_examples, width=640)

        for c in clauses_2:
            if i == 2 and (
                "online" in str(c.predicates[0]) or "online" in str(c.predicates[1])
            ):
                continue

            if "closeby" in str(c.predicates[0]):
                continue

            task_name = str(c).replace(" ", "")
            cur_task_name = str(i) + "_" + task_name
            super_task_name = str(i) + "_no_pairs_2"

            if cur_task_name in unsolvable:
                continue

            if eval:
                if test_task_names:
                    if cur_task_name not in test_task_names:
                        continue
                if train_task_names:
                    if cur_task_name in train_task_names:
                        continue

            if not eval:
                if train_task_names:
                    if cur_task_name not in train_task_names:
                        continue
                if test_task_names:
                    if cur_task_name in test_task_names:
                        continue

            # add folder if not exists
            if not os.path.exists(data_dir + super_task_name):
                os.makedirs(data_dir + super_task_name)

            # check if already 10 tasks exist
            n_tasks = [f for f in os.listdir(data_dir + super_task_name)]
            if len(n_tasks) >= 10:
                print(f"Already 10 tasks in {data_dir + super_task_name}. Skipping. \n")
                break

            print(f"Start creating data for {cur_task_name}")
            path = data_dir + super_task_name + "/" + cur_task_name
            # check if path already exists
            if os.path.exists(path):
                true_path = path + "/true"
                false_path = path + "/false"
                # count files in path
                num_true_images = len(
                    [
                        f
                        for f in os.listdir(true_path)
                        if os.path.isfile(os.path.join(true_path, f))
                        and f.endswith(".png")
                    ]
                )
                num_false_images = len(
                    [
                        f
                        for f in os.listdir(false_path)
                        if os.path.isfile(os.path.join(false_path, f))
                        and f.endswith(".png")
                    ]
                )

                if (
                    num_true_images == max_true_images
                    and num_false_images == max_false_images
                ):
                    print(f"Task {path} already exists correctly. Skipping. \n")
                    # task_set.discard(cur_task_name)
                    continue

            true_path = path + "/true"
            false_path = path + "/false"
            os.makedirs(true_path, exist_ok=True)
            os.makedirs(false_path, exist_ok=True)
            kf = ClauseBasedKandinskyFigure(u, i, i, clause=c)

            generate_task_examples(path, kf, n=num_examples, width=640)

    for key in pair_clauses:
        clauses = pair_clauses[key]
        random.shuffle(clauses)

        super_task_name = key
        for c in clauses:
            cur_task_name = str(c).replace(" ", "")

            if "closeby" in str(c.predicates[0]):
                continue

            if eval:
                if test_task_names:
                    if cur_task_name not in test_task_names:
                        continue
                if train_task_names:
                    if cur_task_name in train_task_names:
                        continue

            if not eval:
                if train_task_names:
                    if cur_task_name not in train_task_names:
                        continue
                if test_task_names:
                    if cur_task_name in test_task_names:
                        continue

            if cur_task_name in unsolvable:
                continue

            if super_task_name == "4_pair_1_1" or super_task_name == "6_pair_1_1":
                n = 6
            else:
                n = 10

            # add folder if not exists
            if not os.path.exists(data_dir + super_task_name):
                os.makedirs(data_dir + super_task_name)

            # check if already n tasks exist
            n_tasks = [f for f in os.listdir(data_dir + super_task_name)]
            if len(n_tasks) >= n:
                print(
                    f"Already {n} tasks in {data_dir + super_task_name}. Continue. \n"
                )
                break

            print(f"Start creating data for {cur_task_name}")
            path = data_dir + super_task_name + "/" + cur_task_name
            # check if path already exists
            if os.path.exists(path):
                true_path = path + "/true"
                false_path = path + "/false"
                # count files in path
                num_true_images = len(
                    [
                        f
                        for f in os.listdir(true_path)
                        if os.path.isfile(os.path.join(true_path, f))
                        and f.endswith(".png")
                    ]
                )
                num_false_images = len(
                    [
                        f
                        for f in os.listdir(false_path)
                        if os.path.isfile(os.path.join(false_path, f))
                        and f.endswith(".png")
                    ]
                )

                if (
                    num_true_images == max_true_images
                    and num_false_images == max_false_images
                ):
                    print(f"Task {path} already exists correctly. Skipping. \n")
                    continue

            object_number = c.num_pairs * 2
            true_path = path + "/true"
            false_path = path + "/false"
            os.makedirs(true_path, exist_ok=True)
            os.makedirs(false_path, exist_ok=True)
            kf = ClauseBasedKandinskyFigure(u, object_number, object_number, clause=c)
            generate_task_examples(path, kf, n=num_examples, width=640)


if __name__ == "__main__":
    print("Welcome to the Kandinsky Task Generator")

    generate_concept_mapping()

    path = "data/kandinsky/support/"
    generate_task_validation_examples(path=path, num_examples=25, eval=False)
    path = "data/kandinsky/query/"
    generate_task_validation_examples(path=path, num_examples=25, eval=False)

    path = "data/kandinsky_eval/support/"
    generate_task_validation_examples(path=path, num_examples=25, eval=True)
    path = "data/kandinsky_eval/query/"
    generate_task_validation_examples(path=path, num_examples=200, eval=True)
