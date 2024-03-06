import math

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw

from map.class_combinations import get_class_id


class kandinskyShape:
    def __init__(self):
        self.shape = "circle"
        self.color = "red"
        self.x = 0.5
        self.y = 0.5
        self.size = 0.5

    def __str__(self):
        return (
            self.color
            + " "
            + self.shape
            + " ("
            + str(self.size)
            + ","
            + str(self.x)
            + ","
            + str(self.y)
            + ")"
        )


class SimpleUniverse:
    kandinsky_colors = ["red", "yellow", "blue"]
    kandinsky_shapes = ["square", "circle", "triangle"]


class ExtendedUniverse:
    # still have to add drawing functions below
    kandinsky_colors = ["red", "yellow", "blue", "green", "orange"]
    kandinsky_shapes = ["square", "circle", "triangle", "star"]


def square(d, cx, cy, s, f):
    s = 0.6 * s
    d.rectangle(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def circle(d, cx, cy, s, f):
    # correct the size to  the same area as an square
    s = 0.6 * math.sqrt(4 * s * s / math.pi)
    d.ellipse(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def triangle(d, cx, cy, s, f):
    r = math.radians(30)
    # correct the size to  the same area as an square
    s = 0.6 * math.sqrt(4 * s * s / math.sqrt(3))
    s = math.sqrt(3) * s / 3
    dx = s * math.cos(r)
    dy = s * math.sin(r)
    d.polygon([(cx, cy - s), (cx + dx, cy + dy), (cx - dx, cy + dy)], fill=f)


def kandinskyFigureAsImagePIL(shapes, width=600, subsampling=4):
    image = Image.new(
        "RGBA", (subsampling * width, subsampling * width), (215, 215, 215, 255)
    )
    d = ImageDraw.Draw(image)
    w = subsampling * width

    for s in shapes:
        globals()[s.shape](d, w * s.x, w * s.y, w * s.size, s.color)
    if subsampling > 1:
        image.thumbnail((width, width), Image.ANTIALIAS)

    return image


def get_rgb_pastel(color):
    # load clevr colormap
    if color == "red":
        return [173, 35, 35]
    if color == "yellow":
        return [255, 238, 51]
    if color == "blue":
        return [42, 75, 215]


def kandinskyFigureAsImage(shapes, width=600, subsampling=4, bbox=False):
    w = subsampling * width
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [215, 215, 215]

    for s in shapes:
        # not sure if this is the right color for openCV
        # rgbcolorvalue = ImageColor.getrgb(s.color)
        # use pastel colors
        rgbcolorvalue = get_rgb_pastel(s.color)

        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
            cx = round(w * s.x)
            cy = round(w * s.y)
            cv2.circle(img, (cx, cy), round(size), rgbcolorvalue, -1)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (round(w * s.x), round(w * s.y - size))
            p2 = (round(w * s.x + dx), round(w * s.y + dy))
            p3 = (round(w * s.x - dx), round(w * s.y + dy))
            points = np.array([p1, p2, p3])
            cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)

        if s.shape == "square":
            size = 0.5 * 0.6 * w * s.size
            xs = round(w * s.x - size)
            ys = round(w * s.y - size)
            xe = round(w * s.x + size)
            ye = round(w * s.y + size)
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)

        # bounding box
        if bbox:
            bbox = get_coco_bounding_box(s, width=width, subsampling=subsampling)
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), get_rgb_pastel("red"), 3
            )

    img_resampled = cv2.resize(img, (width, width), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(img_resampled)

    return image

def kandinskyFigureAsFact(shapes, image_id, width=128, subsampling=1):
    facts = []

    for j, obj in enumerate(shapes):
        obj_dic = shapes[j].__dict__
        obj_id = "obj" + str(j)
        color = "color(" + obj_id + ", " + obj_dic['color'] + ")"
        facts.append(color)
        shape =  "shape(" + obj_id + ", " + obj_dic['shape'] + ")"
        facts.append(shape)
        # TODO
        # size = "size(" + obj_id + ", " + obj_dic['size'] + ")"
        # facts.append(size)
        # bbox

    print(facts)
    return facts


def kandinskyFigureAsAnnotation(shapes, image_id, width=128, subsampling=1):
    annotations = []

    w = subsampling * width
    b = subsampling
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [215, 215, 215]

    eps = 3
    for si, s in enumerate(shapes):
        annotation = {
            "segmentation": [],
            "area": 0,  # to be filled
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],  # to be filled
            "pos": [],
            "category_id": get_class_id(s.size_cls + s.color + s.shape),
            "id": si,
        }

        # rescaling for annotations
        #  [top left x position, top left y position, width, height].
        cx = round(w * s.x)
        cy = round(w * s.y)
        cx_ = cx / b
        cy_ = cy / b
        # print('s.x: ', s.x)
        # print('cx_: ', cx_)

        # not sure if this is the right color for openCV
        rgbcolorvalue = get_rgb_pastel(s.color)
        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
            cx = round(w * s.x)
            cy = round(w * s.y)
            # cv2.circle(img, (cx, cy), round(size), rgbcolorvalue, -1)

            bbox = [
                round(cx - size),
                round(cy - size),
                round(cx + size),
                round(cy + size),
            ]
            area = 3.14 * (size / b) * (size / b)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (round(w * s.x), round(w * s.y - size))
            p2 = (round(w * s.x + dx), round(w * s.y + dy))
            p3 = (round(w * s.x - dx), round(w * s.y + dy))
            points = np.array([p1, p2, p3])
            # cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
            eps_h = size / 4.5
            eps_l = size / 6
            bbox = [
                round(w * s.x - dx),
                round(w * s.y - dy),
                round(w * s.x + dx),
                round(w * s.y + dy),
            ]
            # area = 2 * (dx / 4) * (dy / 4)
            area = 1 / 2 * 2 * dx * 2 * dy

        if s.shape == "square":
            size = 0.5 * 0.6 * w * s.size
            xs = round(w * s.x - size)
            ys = round(w * s.y - size)
            xe = round(w * s.x + size)
            ye = round(w * s.y + size)
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)
            bbox = [xs, ys, xe, ye]
            area = 2 * size * 2 * size
        annotation["bbox"] = bbox
        annotation["area"] = area
        annotations.append(annotation)
    return annotations


def get_bounding_box(shape, width=128, subsampling=1):
    s = shape
    w = subsampling * width
    b = subsampling

    # rescaling
    cx = round(w * s.x)
    cy = round(w * s.y)

    if s.shape == "circle":
        size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
        bbox = [round(cx - size), round(cy - size), round(cx + size), round(cy + size)]
    if s.shape == "triangle":
        r = math.radians(30)
        size = 0.7 * math.sqrt(3) * w * s.size / 3
        dx = size * math.cos(r)
        dy = size * math.sin(r)
        bbox = [
            round(w * s.x - dx),
            round(w * s.y - size),
            round(w * s.x + dx),
            round(w * s.y + dy),
        ]
    if s.shape == "square":
        size = 0.5 * 0.6 * w * s.size
        bbox = [
            round(w * s.x - size),
            round(w * s.y - size),
            round(w * s.x + size),
            round(w * s.y + size),
        ]

    return bbox


def get_coco_bounding_box(shape, width=640, subsampling=1):
    s = shape
    w = subsampling * width
    b = subsampling

    # rescaling
    cx = round(w * s.x)
    cy = round(w * s.y)

    if s.shape == "circle":
        size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
        bbox = [
            round(cx - size - 2),
            round(cy - size - 3),
            round(2 * size + 1),
            round(2 * size + 1),
        ]
    if s.shape == "triangle":
        r = math.radians(30)
        size = 0.7 * math.sqrt(3) * w * s.size / 3
        dx = size * math.cos(r)
        dy = size * math.sin(r)
        bbox = [
            round(w * s.x - dx - 1),
            round(w * s.y - size - 2),
            round(2 * dx),
            round(size + dy + 1),
        ]
    if s.shape == "square":
        size = 0.5 * 0.6 * w * s.size
        bbox = [
            round(w * s.x - size - 1),
            round(w * s.y - size - 2),
            round(2 * size),
            round(2 * size),
        ]

    return bbox


def get_area(shape, width=128, subsampling=1):
    s = shape
    w = subsampling * width
    b = subsampling
    area = 0
    if s.shape == "circle":
        size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
        area = 3.14 * (size / b) * (size / b)
    if s.shape == "triangle":
        r = math.radians(30)
        size = 0.7 * math.sqrt(3) * w * s.size / 3
        dx = size * math.cos(r)
        dy = size * math.sin(r)
        # todo: verify area
        area = 1 / 2 * 2 * dx * 2 * dy
        area2 = 1 / 2 * 2 * dx * dy + size  # ?
    if s.shape == "square":
        size = 0.5 * 0.6 * w * s.size
        area = 2 * size * 2 * size

    return area

def get_relation_area(cbbox):
    return cbbox[2] * cbbox[3]

def get_coco_bbox_from_bbox(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return [bbox[0], bbox[1], width, height]


def kandinskyFigureAsYOLOText(shapes, image_id, category_ids, width=128, subsampling=1):
    annotations = []
    label_texts = []

    w = subsampling * width
    b = subsampling
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [215, 215, 215]

    eps = 3
    print(category_ids)
    for si, s in enumerate(shapes):
        annotation = {
            "segmentation": [],
            "area": 0,  # to be filled
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],  # to be filled
            "category_id": category_ids[si],
            "id": si,
        }

        # rescaling for annotations
        #  [top left x position, top left y position, width, height].
        cx = round(w * s.x)
        cy = round(w * s.y)
        cx_ = cx / b
        cy_ = cy / b
        # print('s.x: ', s.x)
        # print('cx_: ', cx_)

        # not sure if this is the right color for openCV
        rgbcolorvalue = ImageColor.getrgb(s.color)
        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * w * s.size * w * s.size / math.pi)
            cx = round(w * s.x)
            cy = round(w * s.y)
            cv2.circle(img, (cx, cy), round(size), rgbcolorvalue, -1)

            bbox = (
                round(w * s.x) - 1,
                round(w * s.y) - 1,
                2 * size / b + eps,
                2 * size / b + eps,
            )
            area = 3.14 * (size / b) * (size / b)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * w * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (round(w * s.x), round(w * s.y - size))
            p2 = (round(w * s.x + dx), round(w * s.y + dy))
            p3 = (round(w * s.x - dx), round(w * s.y + dy))
            points = np.array([p1, p2, p3])
            cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
            eps_h = size / 4.5
            eps_l = size / 6
            bbox = (
                round(w * s.x) - 1,
                round(w * s.y) - 1,
                2 * size / b - eps_l,
                2 * size / b - eps_h,
            )
            area = 2 * (dx / 4) * (dy / 4)

        if s.shape == "square":
            size = 0.5 * 0.6 * w * s.size
            xs = round(w * s.x - size)
            ys = round(w * s.y - size)
            xe = round(w * s.x + size)
            ye = round(w * s.y + size)
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)
            bbox = (w * s.x - 1, w * s.y - 1, 2 * size / b + eps, 2 * size / b + eps)
            area = 4 * (size / b) * (size / b)
        annotation["bbox"] = bbox
        annotation["area"] = area
        annotations.append(annotation)
        label_text = (
            str(category_ids[si])
            + " "
            + str(bbox[0] / w)
            + " "
            + str(bbox[1] / w)
            + " "
            + str(bbox[2] / w)
            + " "
            + str(bbox[3] / w)
        )
        print(label_text)
        label_texts.append(label_text)
    return label_texts


def __kandinskyFigureAsYOLOText(
    shapes, image_id, category_ids, width=128, subsampling=1
):
    annotations = []
    label_texts = []
    w = subsampling * width
    b = subsampling
    img = np.zeros((w, w, 3), np.uint8)
    img[:, :] = [150, 150, 150]

    eps = 3
    print(category_ids)
    for si, s in enumerate(shapes):
        annotation = {
            "segmentation": [],
            "area": 0,  # to be filled
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [],  # to be filled
            "category_id": category_ids[si],
            "id": si,
        }

        # rescaling for annotations
        #  [top left x position, top left y position, width, height].
        # print('s.x: ', s.x)
        # print('cx_: ', cx_)

        # not sure if this is the right color for openCV
        rgbcolorvalue = ImageColor.getrgb(s.color)
        if s.shape == "circle":
            size = 0.5 * 0.6 * math.sqrt(4 * s.size * s.size / math.pi)
            cv2.circle(img, (s.x, s.y), round(size), rgbcolorvalue, -1)
            # label_idx x_center y_center width height
            label_text = (
                str(category_id)
                + " "
                + str(s.x - size)
                + " "
                + str(y.x - size)
                + " "
                + str(2 * size)
                + " "
                + str(2 * size)
            )
            label_texts.append(label_text)

        if s.shape == "triangle":
            r = math.radians(30)
            size = 0.7 * math.sqrt(3) * s.size / 3
            dx = size * math.cos(r)
            dy = size * math.sin(r)
            p1 = (s.x, s.y - size)
            p2 = (s.x + dx, s.y + dy)
            p3 = (s.x - dx, s.y + dy)
            points = np.array([p1, p2, p3])
            cv2.fillConvexPoly(img, points, rgbcolorvalue, 1)
            label_text = (
                str(category_id)
                + " "
                + str(s.x - dx)
                + " "
                + str(s.y - dy)
                + " "
                + str(2 * size)
                + " "
                + str(2 * size)
            )
            label_texts.append(label_text)

        if s.shape == "square":
            size = 0.5 * 0.6 * s.size
            xs = s.x - size
            ys = s.y - size
            xe = s.x + size
            ye = s.y + size
            cv2.rectangle(img, (xs, ys), (xe, ye), rgbcolorvalue, -1)
            label_text = (
                str(category_id)
                + " "
                + str(xs)
                + " "
                + str(ys)
                + " "
                + str(2 * size)
                + " "
                + str(2 * size)
            )
            label_texts.append(label_text)

    return label_texts


def overlaps(shapes, width=1024):
    image = Image.new("L", (width, width), 0)
    sumarray = np.array(image)
    d = ImageDraw.Draw(image)
    w = width

    for s in shapes:
        image = Image.new("L", (width, width), 0)
        d = ImageDraw.Draw(image)
        globals()[s.shape](d, w * s.x, w * s.y, w * s.size, 10)
        sumarray = sumarray + np.array(image)

    sumimage = Image.fromarray(sumarray)
    return sumimage.getextrema()[1] > 10
