import json


color_map = {
    "gray": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "brown": 4,
    "purple": 5,
    "cyan": 6,
    "yellow": 7,
}
shape_map = {
    "cube": 8,
    "sphere": 9,
    "cylinder": 10,
}

size_map = {
    "small": 11,
    "large": 12,
}

material_map = {
    "rubber": 13,
    "metal": 14,
}

combi_map = {
    "graycubesmallmetal": 0,
    "graycubesmallrubber": 1,
    "graycubelargemetal": 2,
    "graycubelargerubber": 3,
    "graycylindersmallmetal": 4,
    "graycylindersmallrubber": 5,
    "graycylinderlargemetal": 6,
    "graycylinderlargerubber": 7,
    "grayspheresmallmetal": 8,
    "grayspheresmallrubber": 9,
    "grayspherelargemetal": 10,
    "grayspherelargerubber": 11,
    "redcubesmallmetal": 12,
    "redcubesmallrubber": 13,
    "redcubelargemetal": 14,
    "redcubelargerubber": 15,
    "redcylindersmallmetal": 16,
    "redcylindersmallrubber": 17,
    "redcylinderlargemetal": 18,
    "redcylinderlargerubber": 19,
    "redspheresmallmetal": 20,
    "redspheresmallrubber": 21,
    "redspherelargemetal": 22,
    "redspherelargerubber": 23,
    "bluecubesmallmetal": 24,
    "bluecubesmallrubber": 25,
    "bluecubelargemetal": 26,
    "bluecubelargerubber": 27,
    "bluecylindersmallmetal": 28,
    "bluecylindersmallrubber": 29,
    "bluecylinderlargemetal": 30,
    "bluecylinderlargerubber": 31,
    "bluespheresmallmetal": 32,
    "bluespheresmallrubber": 33,
    "bluespherelargemetal": 34,
    "bluespherelargerubber": 35,
    "greencubesmallmetal": 36,
    "greencubesmallrubber": 37,
    "greencubelargemetal": 38,
    "greencubelargerubber": 39,
    "greencylindersmallmetal": 40,
    "greencylindersmallrubber": 41,
    "greencylinderlargemetal": 42,
    "greencylinderlargerubber": 43,
    "greenspheresmallmetal": 44,
    "greenspheresmallrubber": 45,
    "greenspherelargemetal": 46,
    "greenspherelargerubber": 47,
    "browncubesmallmetal": 48,
    "browncubesmallrubber": 49,
    "browncubelargemetal": 50,
    "browncubelargerubber": 51,
    "browncylindersmallmetal": 52,
    "browncylindersmallrubber": 53,
    "browncylinderlargemetal": 54,
    "browncylinderlargerubber": 55,
    "brownspheresmallmetal": 56,
    "brownspheresmallrubber": 57,
    "brownspherelargemetal": 58,
    "brownspherelargerubber": 59,
    "purplecubesmallmetal": 60,
    "purplecubesmallrubber": 61,
    "purplecubelargemetal": 62,
    "purplecubelargerubber": 63,
    "purplecylindersmallmetal": 64,
    "purplecylindersmallrubber": 65,
    "purplecylinderlargemetal": 66,
    "purplecylinderlargerubber": 67,
    "purplespheresmallmetal": 68,
    "purplespheresmallrubber": 69,
    "purplespherelargemetal": 70,
    "purplespherelargerubber": 71,
    "cyancubesmallmetal": 72,
    "cyancubesmallrubber": 73,
    "cyancubelargemetal": 74,
    "cyancubelargerubber": 75,
    "cyancylindersmallmetal": 76,
    "cyancylindersmallrubber": 77,
    "cyancylinderlargemetal": 78,
    "cyancylinderlargerubber": 79,
    "cyanspheresmallmetal": 80,
    "cyanspheresmallrubber": 81,
    "cyanspherelargemetal": 82,
    "cyanspherelargerubber": 83,
    "yellowcubesmallmetal": 84,
    "yellowcubesmallrubber": 85,
    "yellowcubelargemetal": 86,
    "yellowcubelargerubber": 87,
    "yellowcylindersmallmetal": 88,
    "yellowcylindersmallrubber": 89,
    "yellowcylinderlargemetal": 90,
    "yellowcylinderlargerubber": 91,
    "yellowspheresmallmetal": 92,
    "yellowspheresmallrubber": 93,
    "yellowspherelargemetal": 94,
    "yellowspherelargerubber": 95,
}


def get_categories(mode="combi"):
    if mode == "combi":
        categories = []
        for c in combi_map:
            categories.append({"id": combi_map[c], "name": c})
        return categories
    elif mode == "single":
        categories = [
            {"id": 0, "name": "gray"},
            {"id": 1, "name": "red"},
            {"id": 2, "name": "blue"},
            {"id": 3, "name": "green"},
            {"id": 4, "name": "brown"},
            {"id": 5, "name": "purple"},
            {"id": 6, "name": "cyan"},
            {"id": 7, "name": "yellow"},
            {"id": 8, "name": "cube"},
            {"id": 9, "name": "sphere"},
            {"id": 10, "name": "cylinder"},
            {"id": 11, "name": "small"},
            {"id": 12, "name": "large"},
            {"id": 13, "name": "rubber"},
            {"id": 14, "name": "metal"},
        ]
        return categories


def create_annotations(source, destination, num_images, single_concepts=True):
    # open clevr annotations
    with open(source) as f:
        instances = json.load(f)

    images = []

    for i in range(num_images):
        images.append(
            {
                "file_name": f"CLEVR_Hans_classid_0_{i:06d}.png",
                "id": i,
            }
        )

    # edit train annotations
    instances["images"] = images

    annotations = []
    id = 0
    for scene in instances["scenes"]:
        for o_id, object in enumerate(scene["objects"]):
            if single_concepts:
                # add single concepts
                # color
                annotations.append(
                    {
                        "id": id,
                        "image_id": scene["image_index"],
                        "object_id": o_id,
                        "category_id": color_map[object["color"]],
                        "bbox": object["bbox"],
                        "area": object["bbox"][2] * object["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                id += 1
                # shape
                annotations.append(
                    {
                        "id": id,
                        "image_id": scene["image_index"],
                        "object_id": o_id,
                        "category_id": shape_map[object["shape"]],
                        "bbox": object["bbox"],
                        "area": object["bbox"][2] * object["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                id += 1
                # size
                annotations.append(
                    {
                        "id": id,
                        "image_id": scene["image_index"],
                        "object_id": o_id,
                        "category_id": size_map[object["size"]],
                        "bbox": object["bbox"],
                        "area": object["bbox"][2] * object["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                id += 1
                # material
                annotations.append(
                    {
                        "id": id,
                        "image_id": scene["image_index"],
                        "object_id": o_id,
                        "category_id": material_map[object["material"]],
                        "bbox": object["bbox"],
                        "area": object["bbox"][2] * object["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                id += 1
            else:
                annotations.append(
                    {
                        "id": id,
                        "image_id": scene["image_index"],
                        "object_id": o_id,
                        "category_id": color_map[object["color"]],
                        "bbox": object["bbox"],
                        "area": object["bbox"][2] * object["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                id += 1

            instances["annotations"] = annotations

    # add categories
    if single_concepts:
        instances["categories"] = get_categories(mode="single")
    else:
        instances["categories"] = get_categories(mode="combi")

    # save annotations
    with open(destination, "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    create_annotations(
        "data/pattern_free_clevr_combis/train/instances copy.json",
        "data/pattern_free_clevr_combis/train/instances.json",
        2000,
        single_concepts=False,
    )

    create_annotations(
        "data/pattern_free_clevr_combis/test/CLEVR_HANS_scenes_test_classid_0.json",
        "data/pattern_free_clevr_combis/test/instances.json",
        1000,
        single_concepts=False,
    )
