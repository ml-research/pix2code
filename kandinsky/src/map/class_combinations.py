import json


def generate_class_mapping():
    sizes = ["small", "medium", "big"]
    colors = ["red", "blue", "yellow"]
    shapes = ["triangle", "square", "circle"]

    id = 0
    class_name_to_id = dict()
    class_id_to_name = dict()
    categories = []

    for size in sizes:
        for color in colors:
            for shape in shapes:
                class_name = size + color + shape
                class_name_to_id[class_name] = id
                class_id_to_name[id] = class_name
                categories.append({"id": id, "name": class_name})
                id += 1

    json.dump(
        class_id_to_name,
        open("kandinsky/src/map/class_id_to_name.json", "w"),
        sort_keys=True,
        indent=4,
    )
    json.dump(
        class_name_to_id,
        open("kandinsky/map/class_name_to_id.json", "w"),
        sort_keys=True,
        indent=4,
    )
    json.dump(categories, open("kandinsky/src/map/categories.json", "w"), sort_keys=True, indent=4)


def get_class_id(class_name):
    map = json.load(open("kandinsky/src/map/class_name_to_id.json"))
    return map[class_name]


def get_class_name(class_id):
    map = json.load(open("kandinsky/src/map/class_id_to_name.json"))
    return map[class_id]


def get_class_categories_dict():
    categories = json.load(open("kandinsky/src/map/categories.json"))
    return categories


def generate_concept_mapping():
    sizes = ["small", "medium", "big"]
    colors = ["red", "blue", "yellow"]
    shapes = ["triangle", "square", "circle"]

    id = 0
    categories = []

    for size in sizes:
        class_name = size
        categories.append({"id": id, "name": class_name})
        id += 1
    for color in colors:
        class_name = color
        categories.append({"id": id, "name": class_name})
        id += 1
    for shape in shapes:
        class_name = shape
        categories.append({"id": id, "name": class_name})
        id += 1

    json.dump(categories, open("kandinsky/src/map/concept_categories.json", "w"), sort_keys=True, indent=4)


def get_concept_class_id(name):
    concept_dict = get_concept_categories_dict()
    for cd in concept_dict:
        if cd["name"] == name:
            return cd["id"]
    return None


def get_concept_categories_dict():
    categories = json.load(open("kandinsky/src/map/concept_categories.json"))
    return categories


def generate_concept_relation_mapping():
    sizes = ["small", "medium", "big"]
    colors = ["red", "blue", "yellow"]
    shapes = ["triangle", "square", "circle"]

    id = 0
    categories = []

    for size in sizes:
        class_name = size
        categories.append({"id": id, "name": class_name})
        id += 1
    for color in colors:
        class_name = color
        categories.append({"id": id, "name": class_name})
        id += 1
    for shape in shapes:
        class_name = shape
        categories.append({"id": id, "name": class_name})
        id += 1

    class_name = "same_size"
    categories.append({"id": id, "name": class_name})
    id += 1
    class_name = "same_color"
    categories.append({"id": id, "name": class_name})
    id += 1
    class_name = "same_shape"
    categories.append({"id": id, "name": class_name})
    id += 1

    json.dump(
        categories,
        open("kandinsky/src/map/concept_relation_categories.json", "w"),
        sort_keys=True,
        indent=4,
    )


def get_concept_relation_class_id(name):
    concept_dict = get_concept_relation_categories_dict()
    for cd in concept_dict:
        if cd["name"] == name:
            return cd["id"]
    return None


def get_concept_relation_categories_dict():
    categories = json.load(open("kandinsky/src/map/concept_relation_categories.json"))
    return categories
