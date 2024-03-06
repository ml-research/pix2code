import json
import os

from kp import KandinskyCaptions, KandinskyUniverse, RandomKandinskyFigure
from kp.KandinskyUniverse import (
    get_area,
    get_bounding_box,
    get_coco_bbox_from_bbox,
    get_coco_bounding_box,
    get_relation_area,
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


def generateImagesAndCaptions(basedir, kfgen, n=50, width=200):
    os.makedirs(basedir, exist_ok=True)
    capt_color_shape_size_file = open(basedir + "/color_shape_size.cap", "w")
    capt_numbers = open(basedir + "/numbers.cap", "w")
    for i, kf in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/%06d" % i + ".png")
        capt_color_shape_size_file.write(
            str(i) + "\t" + cg.colorShapesSize(kf, "one ") + "\n"
        )
        capt_numbers.write(str(i) + "\t" + cg.numbers(kf) + "\n")
    capt_color_shape_size_file.close()
    capt_numbers.close()


def generateSimpleNumbersCaptions(basedir, kfgen, n=50, width=200):
    os.makedirs(basedir, exist_ok=True)
    capt_numbers_file = open(basedir + "/numbers.cap", "w")
    for i, kf in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/%06d" % i + ".png")
        capt_numbers_file.write(str(i) + "\t" + cg.simpleNumbers(kf) + "\n")
    capt_numbers_file.close()


def generateClasses(basedir, kfgen, n=50, width=200, counterfactual=False):
    os.makedirs(basedir + "/true", exist_ok=True)
    os.makedirs(basedir + "/false", exist_ok=True)
    for i, kf in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/true/%06d" % i + ".png")

    for i, kf in enumerate(kfgen.false_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/false/%06d" % i + ".png")
    if counterfactual:
        os.makedirs(basedir + "/counterfactual", exist_ok=True)
        for i, kf in enumerate(kfgen.almost_true_kf(n)):
            image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
            image.save(basedir + "/counterfactual/%06d" % i + ".png")


def generateWithJson(basedir, kfgen, n=50, width=200):
    os.makedirs(basedir, exist_ok=True)
    capt_numbers_file = open(basedir + "/numbers.cap", "w")

    instances = dict()
    images = []
    annotations = []
    ann_id = 0

    # build instances json
    for i, kf in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/%06d" % i + ".png")

        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})

        # annotations
        for j, obj in enumerate(kf):
            obj_dic = kf[j].__dict__
            obj_dic["object_id"] = j

            img_dic = {"img_id": i, "scene": []}
            img_dic["scene"].append(obj_dic)

            annotation_dict = {
                "image_id": i,
                "pos": obj_dic["pos"],
                "bbox": get_coco_bounding_box(kf[j], width=width),
                "iscrowd": 0,
                "category_id": obj_dic["category_id"],
                "object_id": j,
                "id": ann_id,
                "area": get_area(kf[j], width=width),
            }
            annotations.append(annotation_dict)
            ann_id += 1

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_class_categories_dict()

    with open(basedir + "/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)

    capt_numbers_file.close()


def generateWithJsonConcepts(basedir, kfgen, n=50, width=200):
    os.makedirs(basedir, exist_ok=True)
    capt_numbers_file = open(basedir + "/numbers.cap", "w")

    instances = dict()
    instances_without_color = dict()
    images = []
    annotations = []
    annotations_without_color = []
    ann_id = 0

    # build instances json
    for i, kf in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/%06d" % i + ".png")

        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})

        # annotations
        for j, obj in enumerate(kf):
            obj_dic = kf[j].__dict__
            obj_dic["object_id"] = j

            img_dic = {"img_id": i, "scene": []}
            img_dic["scene"].append(obj_dic)

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
                if concept != "color":
                    annotations_without_color.append(annotation_dict)
                ann_id += 1

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_concept_categories_dict()

    instances_without_color["images"] = images
    instances_without_color["annotations"] = annotations_without_color
    instances_without_color["categories"] = get_concept_categories_dict()

    with open(basedir + "/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)

    with open(basedir + "/instances_without_color.json", "w") as f:
        json.dump(instances_without_color, f, sort_keys=True, indent=4)

    capt_numbers_file.close()


def generateWithJsonConceptsRelations(basedir, kfgen, n=50, width=200):
    os.makedirs(basedir, exist_ok=True)
    capt_numbers_file = open(basedir + "/numbers.cap", "w")

    instances = dict()
    images = []
    annotations = []
    ann_id = 0

    # build instances json
    for i, kf in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        image.save(basedir + "/%06d" % i + ".png")

        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})

        # annotations
        for j, obj in enumerate(kf):
            obj_dic = kf[j].__dict__

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

        for j in range(len(kf)):
            obj1 = kf[j].__dict__
            for k in range(j + 1, len(kf)):
                obj2 = kf[k].__dict__
                if obj1["shape"] == obj2["shape"]:
                    bbox1 = get_bounding_box(kf[j], width=width)
                    bbox2 = get_bounding_box(kf[k], width=width)
                    bbox_joined = [
                        min(bbox1[0], bbox2[0]),
                        min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]),
                        max(bbox1[3], bbox2[3]),
                    ]
                    cbbox_joined = get_coco_bbox_from_bbox(bbox_joined)
                    annotation_dict = {
                        "image_id": i,
                        "bbox": cbbox_joined,
                        "iscrowd": 0,
                        "category_id": get_concept_relation_class_id("same_shape"),
                        "id": ann_id,
                        "area": get_relation_area(cbbox_joined),
                    }
                    annotations.append(annotation_dict)
                    ann_id += 1
                if obj1["size_cls"] == obj2["size_cls"]:
                    bbox1 = get_bounding_box(kf[j], width=width)
                    bbox2 = get_bounding_box(kf[k], width=width)
                    bbox_joined = [
                        min(bbox1[0], bbox2[0]),
                        min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]),
                        max(bbox1[3], bbox2[3]),
                    ]
                    cbbox_joined = get_coco_bbox_from_bbox(bbox_joined)
                    annotation_dict = {
                        "image_id": i,
                        "bbox": cbbox_joined,
                        "iscrowd": 0,
                        "category_id": get_concept_relation_class_id("same_size"),
                        "id": ann_id,
                        "area": get_relation_area(cbbox_joined),
                    }
                    annotations.append(annotation_dict)
                    ann_id += 1
                if obj1["color"] == obj2["color"]:
                    bbox1 = get_bounding_box(kf[j], width=width)
                    bbox2 = get_bounding_box(kf[k], width=width)
                    bbox_joined = [
                        min(bbox1[0], bbox2[0]),
                        min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]),
                        max(bbox1[3], bbox2[3]),
                    ]
                    cbbox_joined = get_coco_bbox_from_bbox(bbox_joined)
                    annotation_dict = {
                        "image_id": i,
                        "bbox": cbbox_joined,
                        "iscrowd": 0,
                        "category_id": get_concept_relation_class_id("same_color"),
                        "id": ann_id,
                        "area": get_relation_area(cbbox_joined),
                    }
                    annotations.append(annotation_dict)
                    ann_id += 1

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_concept_relation_categories_dict()

    with open(basedir + "/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)

    capt_numbers_file.close()


if __name__ == "__main__":
    print("Welcome to the Kandinsky Figure Generator")

    train_path = "data/pattern_free_kandinsky/train"
    val_path = "data/pattern_free_kandinsky/val"
    test_path = "../data/pattern_kandinsky/test"

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # generate kandinsky figures
    # n is the number of figures to be generated
    generate_concept_relation_mapping()

    randomkf = RandomKandinskyFigure.Random(u, 2, 7)
    generateWithJsonConcepts(train_path, randomkf, n=2000, width=640)
    generateWithJsonConcepts(val_path, randomkf, n=750, width=640)
    generateWithJsonConcepts(test_path, randomkf, n=750, width=640)
