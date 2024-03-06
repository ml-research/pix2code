import math
import os
import random
import sys
import json

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont
from src.map.class_combinations import get_concept_categories_dict, get_concept_class_id

import src.kp.SimpleObjectAndShape
from src.kp import (KandinskyCaptions, KandinskyUniverse,
                    NumbersKandinskyFigure, RandomKandinskyFigure,
                    ShapeOnShapes, SimpleObjectAndShape)

u = KandinskyUniverse.SimpleUniverse()


###
# Parameters for images: object sizes, object colors, back ground colors
###
WIDTH = 640

MINSIZE = 10 * 5
MAXSIZE = 24 * 5
# pastel
# kandinsky_colors = [(255, 179, 186), (255, 255, 186), (186, 225, 255)]
# background = (140, 140, 140, 255)
# a bit dark
# kandinsky_colors = [(215, 139, 136), (215, 215, 146), (146, 185, 215)]

# clevr
# kandinsky_colors = [(173, 35, 35), (255, 238, 51), (42, 75, 215)]
kandinsky_colors = [(173, 35, 35), (255, 238, 51), (42, 75, 215)]

# a bit lighter
# kandinsky_colors = [(193, 85, 85), (255, 238, 101), (90, 135, 235)]
background = (215, 215, 215, 255)
###
###
###


def square(d, cx, cy, s, f):
    s = 0.7 * s
    d.rectangle(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def circle(d, cx, cy, s, f):
    # correct the size to  the same area as an square
    s = 0.7 * s * 4 / math.pi
    d.ellipse(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)


def triangle(d, cx, cy, s, f):
    r = math.radians(30)
    # correct the size to  the same area as an square
    s = 0.7 * s * 3 * math.sqrt(3) / 4
    dx = s * math.cos(r) / 2
    dy = s * math.sin(r) / 2
    d.polygon([(cx, cy - s / 2), (cx + dx, cy + dy), (cx - dx, cy + dy)], fill=f)


kandinsky_shapes = [square, circle, triangle]


def kandinskyFigure(shapes, subsampling=1):
    image = Image.new("RGBA", (subsampling * WIDTH, subsampling * WIDTH), background)
    d = ImageDraw.Draw(image)
    for s in shapes:
        s["shape"](
            d,
            subsampling * s["cx"],
            subsampling * s["cy"],
            subsampling * s["size"],
            s["color"],
        )
    if subsampling > 1:
        image = image.resize((WIDTH, WIDTH), Image.BICUBIC)
    return image


def overlaps(shapes):
    image = Image.new("L", (WIDTH, WIDTH), 0)
    sumarray = np.array(image)
    d = ImageDraw.Draw(image)

    for s in shapes:
        image = Image.new("L", (WIDTH, WIDTH), 0)
        d = ImageDraw.Draw(image)
        s["shape"](d, s["cx"], s["cy"], s["size"], 10)
        sumarray = sumarray + np.array(image)

    sumimage = Image.fromarray(sumarray)
    return sumimage.getextrema()[1] > 10


def combineFigures(n, f):
    images = []
    for i in range(n):
        shapes = f()
        while overlaps(shapes):
            shapes = f()
        image = kandinskyFigure(shapes, 4)
        images.append(image)

    allimages = Image.new(
        "RGBA", (WIDTH * n + 20 * (n - 1), WIDTH), (255, 255, 255, 255)
    )
    for i in range(n):
        allimages.paste(images[i], (WIDTH * i + 20 * (i), 0))
    return allimages


def listFigures(n, f):
    images = []
    for i in range(n):
        shapes = f()
        while overlaps(shapes):
            shapes = f()
        image = kandinskyFigure(shapes, 4)
        images.append(image)

    return images

def listShapes(n, f):
    images = []
    for i in range(n):
        shapes = f()
        while overlaps(shapes):
            shapes = f()
        images.append(shapes)
    
    return images


def randomShapes(min, max):
    nshapes = random.randint(min, max)
    shapes = []
    for i in range(nshapes):
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        sha = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def onlyCircles(min, max):
    nshapes = random.randint(min, max)
    shapes = []
    for i in range(nshapes):
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        shape = {
            "shape": circle,
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def randomSmallShapes(min, max):
    MINSIZE = 10 * 5
    MAXSIZE = 24 * 5
    nshapes = random.randint(min, max)
    shapes = []
    for i in range(nshapes):
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        sha = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def shapesOnLine(min, max):
    MINSIZE = 8 * 5
    MAXSIZE = 15 * 5
    nshapes = random.randint(min, max)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    sx = WIDTH / 2 - dx
    sy = WIDTH / 2 + dy
    ex = WIDTH / 2 + dx
    ey = WIDTH / 2 - dy
    dx = ex - sx
    dy = ey - sy
    shapes = []
    for i in range(nshapes):
        r = random.random()
        cx = sx + r * dx
        cy = sy + r * dy
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        sha = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)

    return shapes


def shapesOnLinePair(min, max):
    # MINSIZE = 10*5
    # MAXSIZE = 24*5
    MINSIZE = 10 * 5
    MAXSIZE = 24 * 5
    nshapes = random.randint(min, max)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    sx = WIDTH / 2 - dx
    sy = WIDTH / 2 + dy
    ex = WIDTH / 2 + dx
    ey = WIDTH / 2 - dy
    dx = ex - sx
    dy = ey - sy

    shapes = []
    color_ids = []
    shape_ids = []
    is_reject = True
    while is_reject or len(shapes) < nshapes:
        shapes = []
        color_ids = []
        shape_ids = []
        for i in range(nshapes):
            r = random.random()
            cx = sx + r * dx
            cy = sy + r * dy
            size = random.randint(MINSIZE, MAXSIZE)
            col = random.randint(0, 2)
            sha = random.randint(0, 2)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
            color_ids.append(col)
            shape_ids.append(sha)

            pairs = [(color_ids[i], shape_ids[i]) for i in range(len(color_ids))]
            if len(set(pairs)) < len(pairs):
                is_reject = False
    return shapes


def shapesOnLineWOPair(min, max):
    MINSIZE = 10 * 5
    MAXSIZE = 24 * 5

    nshapes = random.randint(min, max)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    sx = WIDTH / 2 - dx
    sy = WIDTH / 2 + dy
    ex = WIDTH / 2 + dx
    ey = WIDTH / 2 - dy
    dx = ex - sx
    dy = ey - sy

    shapes = []
    color_ids = []
    shape_ids = []
    is_reject = True
    while is_reject or len(shapes) < nshapes:
        shapes = []
        color_ids = []
        shape_ids = []
        for i in range(nshapes):
            r = random.random()
            cx = sx + r * dx
            cy = sy + r * dy
            size = random.randint(MINSIZE, MAXSIZE)
            col = random.randint(0, 2)
            sha = random.randint(0, 2)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
            color_ids.append(col)
            shape_ids.append(sha)

            pairs = [(color_ids[i], shape_ids[i]) for i in range(len(color_ids))]
            if len(shapes) == nshapes and len(set(pairs)) == len(pairs):
                is_reject = False
    return shapes


def shapesWithEqualArea(min, max):
    nshapes = random.randint(min, max)
    shapes = []
    size = random.randint(MINSIZE, MAXSIZE)
    for i in range(nshapes):
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        col = random.randint(0, 2)
        sha = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def twoPairsOnlyOneWithSameColor(n=4):
    shapes = []
    size = random.randint(MINSIZE, MAXSIZE)

    sha = random.randint(0, 2)
    col = random.randint(0, 2)
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)

    colOld = col
    while col == colOld:
        col = random.randint(0, 2)
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    size = random.randint(MINSIZE, MAXSIZE)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)

    shaOld = sha
    while sha == shaOld:
        sha = random.randint(0, 2)

    col = random.randint(0, 2)
    for i in range(2):
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        size = random.randint(MINSIZE, MAXSIZE)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def twoPairsMultiOnlyOneWithSameColor(n=6):
    # three pairs: each pair for each shape
    # one: same color
    # two: diff color
    shapes = []
    shas = []
    # first object
    size = random.randint(MINSIZE, MAXSIZE)
    sha = random.randint(0, 2)
    col = random.randint(0, 2)
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    shas.append(sha)

    # add same color same shape object
    size = random.randint(MINSIZE, MAXSIZE)
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)

    # add a diff-color same-shape pair
    # add one
    while sha in shas:
        sha = random.randint(0, 2)
    size = random.randint(MINSIZE, MAXSIZE)
    col = random.randint(0, 2)
    old_col = col
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    shas.append(sha)
    # add another
    while col == old_col:
        col = random.randint(0, 2)
    size = random.randint(MINSIZE, MAXSIZE)
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    # add a diff-color same-shape pair
    # add one
    while sha in shas:
        sha = random.randint(0, 2)
    size = random.randint(MINSIZE, MAXSIZE)
    col = random.randint(0, 2)
    old_col = col
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    shas.append(sha)
    # add another
    while col == old_col:
        col = random.randint(0, 2)
    size = random.randint(MINSIZE, MAXSIZE)
    cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    return shapes


def nottwoPairsOnlyOneWithSameColor(n=4):
    nshapes = n
    shapes = []
    flag = True

    def random_shapes():
        shapes = []
        for i in range(n):
            cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            size = random.randint(MINSIZE, MAXSIZE)
            col = random.randint(0, 2)
            sha = random.randint(0, 2)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
        return shapes

    def check(shapes):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if (
                            shapes[i]["shape"] == shapes[j]["shape"]
                            and shapes[k]["shape"] == shapes[l]["shape"]
                        ):
                            if (
                                shapes[i]["color"] == shapes[j]["color"]
                                and shapes[k]["color"] != shapes[l]["color"]
                            ):
                                return False
                            if (
                                shapes[i]["color"] != shapes[j]["color"]
                                and shapes[k]["color"] == shapes[l]["color"]
                            ):
                                return False
        return True

    while flag:
        shapes = random_shapes()
        if check(shapes):
            flag = False

    return shapes


def shapesNear(min, max):
    nshapes = random.randint(min, max)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    base = random.uniform(0.3, 0.7) * WIDTH

    sx = base - dx
    sy = base + dy
    ex = base + dx
    ey = base - dy
    dx = ex - sx
    dy = ey - sy
    shapes = []

    for i in range(nshapes):
        sha = random.randint(0, 2)
        r = random.uniform(0.5, 0.8)
        cx = sx + r * dx
        cy = sy + r * dy
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def shapesRedTriangle(n):
    MINSIZE = 12 * 5
    MAXSIZE = 12 * 5
    nshapes = n

    cx = random.uniform(0.2, 0.8) * WIDTH
    cy = random.uniform(0.2, 0.8) * WIDTH

    # red triangle coord
    rt_cx = cx
    rt_cy = cy

    shapes = []
    coords = []

    # add red triangle
    coord = np.array([cx, cy])
    size = random.randint(MINSIZE, MAXSIZE)
    col = 0
    sha = 2
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    coords.append(np.array([cx, cy]))

    # add (attacked) target object
    col = random.randint(1, 2)  # no red target object
    sha = random.randint(0, 1)  # no triangle as a target

    dx = 0.75 * size
    dy = 0.75 * size

    # scale = random.uniform(0.7, 1.2)
    # dx = scale * size
    # dy = scale * size

    pos = random.randint(0, 3)
    if pos == 0:
        cx = rt_cx + dx
        cy = rt_cy + dy
    elif pos == 1:
        cx = rt_cx - dx
        cy = rt_cy + dy
    elif pos == 2:
        cx = rt_cx + dx
        cy = rt_cy - dy
    else:
        cx = rt_cx - dx
        cy = rt_cy - dy

    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    coords.append(np.array([cx, cy]))

    # add others
    th = 150
    while len(coords) < 6:
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        coord = np.array([cx, cy])
        size = random.randint(MINSIZE, MAXSIZE)
        sha = random.randint(0, 2)
        col = random.randint(0, 2)
        distant_flag = True
        for c in coords:
            if np.linalg.norm(c[0] - coord) < th:
                distant_flag = False
        if distant_flag:
            coords.append(coord)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
    return shapes


def shapesNotRedTriangle(n):
    MINSIZE = 12 * 5
    MAXSIZE = 12 * 5
    # no red triagnle attacking other objects
    shapes = []
    flag = True

    shape_ids = []

    def random_shapes(n):
        shapes = []
        counter = 0

        # for i in range(n):
        while counter < n:
            cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            size = random.randint(MINSIZE, MAXSIZE)
            if counter == 0:
                # place a red triangle first
                col = 0
                sha = 2
            else:
                col = random.randint(0, 2)
                sha = random.randint(0, 2)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            if add_check(shapes, shape):
                shapes.append(shape)
                shape_ids.append(sha)
                counter += 1
            else:
                0
                # print("rejected ", shape)
        return shapes, shape_ids

    def add_check(shapes, new_shape):
        flag = True
        th = 200
        # if new shape is triangle and red
        if (
            new_shape["shape"] == kandinsky_shapes[2]
            and new_shape["color"] == kandinsky_colors[0]
        ):
            for shape in shapes:
                # not too close or a triangle
                xy1 = np.array([new_shape["cx"], new_shape["cy"]])
                xy2 = np.array([shape["cx"], shape["cy"]])
                # not too close or both are triangle
                # distant or same triangle or same red
                flag = flag and (
                    np.linalg.norm(xy1 - xy2) > th
                    or shape["shape"] == kandinsky_shapes[2]
                    or shape["color"] == kandinsky_colors[0]
                )
        # if new shape is not red triangle
        else:
            for shape in shapes:
                # if red triangle already exists
                if (
                    shape["shape"] == kandinsky_shapes[2]
                    and shape["color"] == kandinsky_colors[0]
                ):
                    xy1 = np.array([new_shape["cx"], new_shape["cy"]])
                    xy2 = np.array([shape["cx"], shape["cy"]])
                    flag = flag and (
                        np.linalg.norm(xy1 - xy2) > th
                        or new_shape["shape"] == kandinsky_shapes[2]
                        or new_shape["color"] == kandinsky_colors[0]
                    )
                    if flag:
                        print(np.linalg.norm(xy1 - xy2), shape, new_shape)
        return flag

    def check(shapes, shape_ids):
        th = 200
        n = len(shapes)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # if triangle and red
                    if shape_ids[i] == 2 and shapes[i]["color"] == kandinsky_colors[0]:
                        if shape_ids[j] in [0, 1] and shapes[j]["color"] in [
                            kandinsky_colors[1],
                            kandinsky_colors[2],
                        ]:
                            # if close by
                            xy1 = np.array([shapes[i]["cx"], shapes[i]["cy"]])
                            xy2 = np.array([shapes[j]["cx"], shapes[j]["cy"]])
                            if np.linalg.norm(xy1 - xy2) < th:
                                # print(np.linalg.norm(xy1-xy2), xy1, xy2)
                                # print('rejected')
                                return False
        return True

    # while flag:
    ###    shapes, shape_ids = random_shapes(n)
    #    if check(shapes, shape_ids):
    #        flag = False

    # while True:
    shapes, shape_ids = random_shapes(n)
    return shapes
    # if check(shapes, shape_ids):
    #        return shapes

    # return shapes


def shapesNearShapeWithOthers(min, max):
    MINSIZE = 16 * 5
    MAXSIZE = 16 * 5
    nshapes = random.randint(min, max)

    # dx = math.cos(random.random() * math.pi * 2) * (WIDTH/2-MAXSIZE/2)
    # dy = math.sin(random.random() * math.pi * 2) * (WIDTH/2-MAXSIZE/2)
    base = random.uniform(0.2, 0.8) * WIDTH

    dx = MINSIZE * 1.2
    dy = MINSIZE * 1.2

    sx = base - dx
    sy = base + dy
    # ex = base + dx
    # ey = base - dy
    # dx = ex-sx
    # dy = ey-sy
    shapes = []
    coords = []

    th = 150
    # add 2 closeby pair
    sha = random.randint(0, 2)
    for i in range(2):
        r = random.uniform(0.5, 0.8)
        cx = sx + r * dx
        cy = sy + r * dy
        coord = np.array([cx, cy])
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
        coords.append(coord)

    while len(coords) < 6:
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        coord = np.array([cx, cy])
        size = random.randint(MINSIZE, MAXSIZE)
        sha = random.randint(0, 2)
        col = random.randint(0, 2)
        distant_flag = True
        for c in coords:
            if np.linalg.norm(c[0] - coord) < th:
                distant_flag = False
        if distant_flag:
            coords.append(coord)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)

    return shapes


def shapesNearShape(min, max):
    nshapes = random.randint(min, max)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    base = random.uniform(0.3, 0.7) * WIDTH

    sx = base - dx
    sy = base + dy
    ex = base + dx
    ey = base - dy
    dx = ex - sx
    dy = ey - sy
    shapes = []
    sha = random.randint(0, 2)

    for i in range(nshapes):
        r = random.uniform(0.5, 0.8)
        cx = sx + r * dx
        cy = sy + r * dy
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
    return shapes


def shapesNearCF(min, max):
    nshapes = random.randint(min, max)

    dx = math.cos(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    dy = math.sin(random.random() * math.pi * 2) * (WIDTH / 2 - MAXSIZE / 2)
    sx = WIDTH / 2 - dx
    sy = WIDTH / 2 + dy
    ex = WIDTH / 2 + dx
    ey = WIDTH / 2 - dy
    dx = ex - sx
    dy = ey - sy
    shapes = []
    sha_ids = []
    for i in range(nshapes):
        r = random.uniform(0.5, 0.8)
        cx = sx + r * dx
        cy = sy + r * dy
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        sha = random.randint(0, 2)
        while sha in sha_ids:
            sha = random.randint(0, 2)
        shape = {
            "shape": kandinsky_shapes[sha],
            "cx": cx,
            "cy": cy,
            "size": size,
            "color": kandinsky_colors[col],
        }
        shapes.append(shape)
        sha_ids.append(sha)

    return shapes


def __randomDistantShapes(min, max):
    MINSIZE = 16 * 5
    MAXSIZE = 16 * 5
    nshapes = random.randint(min, max)
    shapes = []
    coords = []
    for i in range(nshapes):
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        size = random.randint(MINSIZE, MAXSIZE)
        col = random.randint(0, 2)
        sha = random.randint(0, 2)
        coord = np.array([cx, cy])

        if i == 0:
            coords.append(coord)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
        if i == 1:
            while len(coords) < max:
                cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
                cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
                size = random.randint(MINSIZE, MAXSIZE)
                coord = np.array([cx, cy])
                if np.linalg.norm(coords[0] - coord) > 250:
                    coords.append(coord)
                    shape = {
                        "shape": kandinsky_shapes[sha],
                        "cx": cx,
                        "cy": cy,
                        "size": size,
                        "color": kandinsky_colors[col],
                    }
                    shapes.append(shape)
    return shapes


def randomClosebyDistantShapes(num_near=2, num_distant=2):
    MINSIZE = 12 * 5
    MAXSIZE = 12 * 5
    nshapes = num_distant

    cx = random.uniform(0.2, 0.8) * WIDTH
    cy = random.uniform(0.2, 0.8) * WIDTH

    # red triangle coord
    rt_cx = cx
    rt_cy = cy

    shapes = []
    coords = []

    # add red triangle
    coord = np.array([cx, cy])
    size = random.randint(MINSIZE, MAXSIZE)
    col = random.randint(0, 2)
    sha = random.randint(0, 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    coords.append(np.array([cx, cy]))

    # add (attacked) target object
    col = random.randint(0, 2)  # no red target object
    sha = random.randint(0, 2)  # no triangle as a target

    dx = 0.75 * size
    dy = 0.75 * size

    pos = random.randint(0, 3)
    if pos == 0:
        cx = rt_cx + dx
        cy = rt_cy + dy
    elif pos == 1:
        cx = rt_cx - dx
        cy = rt_cy + dy
    elif pos == 2:
        cx = rt_cx + dx
        cy = rt_cy - dy
    else:
        cx = rt_cx - dx
        cy = rt_cy - dy

    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    coords.append(np.array([cx, cy]))

    # add others
    th = 30
    while len(coords) < 4:
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        coord = np.array([cx, cy])
        size = random.randint(MINSIZE, MAXSIZE)
        sha = random.randint(0, 2)
        col = random.randint(0, 2)
        distant_flag = True
        for c in coords:
            if np.linalg.norm(c - coord) < th:
                distant_flag = False
        if distant_flag:
            coords.append(coord)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
    return shapes


def randomClosebyShapes(n=2):
    MINSIZE = 12 * 5
    MAXSIZE = 12 * 5
    nshapes = n

    cx = random.uniform(0.2, 0.8) * WIDTH
    cy = random.uniform(0.2, 0.8) * WIDTH

    # red triangle coord
    rt_cx = cx
    rt_cy = cy

    shapes = []
    coords = []

    # add first object
    coord = np.array([cx, cy])
    size = random.randint(MINSIZE, MAXSIZE)
    col = random.randint(0, 2)
    sha = random.randint(0, 2)
    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    coords.append(np.array([cx, cy]))

    # add another object
    col = random.randint(0, 2)
    sha = random.randint(0, 2)

    scale = random.uniform(0.7, 1.2)
    dx = scale * size
    dy = scale * size

    pos = random.randint(0, 3)
    if pos == 0:
        cx = rt_cx + dx
        cy = rt_cy + dy
    elif pos == 1:
        cx = rt_cx - dx
        cy = rt_cy + dy
    elif pos == 2:
        cx = rt_cx + dx
        cy = rt_cy - dy
    else:
        cx = rt_cx - dx
        cy = rt_cy - dy

    shape = {
        "shape": kandinsky_shapes[sha],
        "cx": cx,
        "cy": cy,
        "size": size,
        "color": kandinsky_colors[col],
    }
    shapes.append(shape)
    coords.append(np.array([cx, cy]))

    return shapes


def randomDistantShapes(num_distant=2):
    MINSIZE = 12 * 5
    MAXSIZE = 12 * 5
    nshapes = num_distant

    cx = random.uniform(0.2, 0.8) * WIDTH
    cy = random.uniform(0.2, 0.8) * WIDTH

    # red triangle coord
    rt_cx = cx
    rt_cy = cy

    shapes = []
    coords = []

    # add others
    th = 120
    while len(coords) < nshapes:
        cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
        coord = np.array([cx, cy])
        size = random.randint(MINSIZE, MAXSIZE)
        sha = random.randint(0, 2)
        col = random.randint(0, 2)
        distant_flag = True
        dist = 0
        for c in coords:
            dist = np.linalg.norm(c - coord)
            if dist < th:
                distant_flag = False
                print("skipped, ", dist)
        if distant_flag:
            print("accepted: ", dist)
            coords.append(coord)
            shape = {
                "shape": kandinsky_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": kandinsky_colors[col],
            }
            shapes.append(shape)
            print(coords)
    return shapes


def to_images(kfgen, n=50, width=640):
    pos_imgs = []
    neg_imgs = []
    for (i, kf) in enumerate(kfgen.true_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        pos_imgs.append(image)

    for (i, kf) in enumerate(kfgen.false_kf(n)):
        image = KandinskyUniverse.kandinskyFigureAsImage(kf, width)
        neg_imgs.append(image)
    return pos_imgs, neg_imgs


def generate_images(dataset, mode="train", n=1):
    print("Generating dataset ", dataset, mode)
    if dataset == "twopairs":
        #pos_imgs = listShapes(n, lambda: twoPairsOnlyOneWithSameColor(4))
        pos_imgs = listFigures(n, lambda: twoPairsOnlyOneWithSameColor(4))
        neg_imgs = listFigures(n, lambda: nottwoPairsOnlyOneWithSameColor(4))
        #neg_imgs = listShapes(n, lambda: nottwoPairsOnlyOneWithSameColor(4))
    elif dataset == "threepairs":
        pos_imgs = listFigures(n, lambda: twoPairsMultiOnlyOneWithSameColor(6))
        neg_imgs = listFigures(n, lambda: nottwoPairsOnlyOneWithSameColor(6))
    # elif dataset == 'closeby':
    #    pos_imgs = listFigures(n, lambda: shapesNear(2, 2))
    #    neg_imgs = listFigures(n, lambda: randomDistantShapes(2, 2))
    elif dataset == "closeby_pretrain":
        pos_imgs = listFigures(n, lambda: randomClosebyShapes(2))
        neg_imgs = listFigures(n, lambda: randomDistantShapes(2))
    elif dataset == "closeby":
        pos_imgs = listFigures(n, lambda: randomClosebyDistantShapes(2, 2))
        neg_imgs = listFigures(n, lambda: randomDistantShapes(4))
    elif dataset == "closeby-multi":
        pos_imgs = listFigures(n, lambda: shapesNearShapeWithOthers(4, 4))
        neg_imgs = []
        # neg_imgs = listFigures(int(n/2), lambda: shapesNearCF(
        #    2, 2)) + listFigures(int(n/2), lambda: randomDistantShapes(4, 4))
    elif dataset == "red-triangle":
        pos_imgs = listFigures(n, lambda: shapesRedTriangle(6))
        neg_imgs = listFigures(n, lambda: shapesNotRedTriangle(6))
    elif dataset == "online_pretrain":
        pos_imgs = listFigures(n, lambda: shapesOnLine(5, 5))
        neg_imgs = listFigures(n, lambda: randomSmallShapes(5, 5))
    elif dataset == "online-pair":
        pos_imgs = listFigures(n, lambda: shapesOnLinePair(5, 5))
        neg_imgs = listFigures(
            int(n / 2), lambda: shapesOnLineWOPair(5, 5)
        ) + listFigures(int(n / 2), lambda: randomShapes(5, 5))
    elif dataset == "online-7":
        pos_imgs = listFigures(n, lambda: shapesOnLine(7, 7))
        neg_imgs = listFigures(int(n / 2), lambda: randomShapes(7, 7))
    else:
        assert False, "Invalid dataset: " + str(dataset)

    # save iamges into true/false folders
    base_path = "data/task/nsfr/" + dataset + "/"
    true_path = base_path + mode + "/true/"
    false_path = base_path + mode + "/false/"

    os.makedirs(true_path, exist_ok=True)
    os.makedirs(false_path, exist_ok=True)

    instances = dict()
    images = []
    annotations = []
    ann_id = 0

    for i, img in enumerate(pos_imgs):
        # img = kandinskyFigure(kf)
        img.save(true_path + "/%06d" % i + ".png")

        """
        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})

        # annotations
        for j, obj in enumerate(kf):
            obj_dic = obj.__dict__
            obj_dic["object_id"] = j

            img_dic = {"img_id": i, "scene": []}
            img_dic["scene"].append(obj_dic)

            for concept in ["size_cls", "color", "shape"]:
                annotation_dict = {
                    "image_id": i,
                    "pos": obj_dic["pos"],
                    "bbox": KandinskyUniverse.get_coco_bounding_box(obj, width=640),
                    "iscrowd": 0,
                    "category_id": get_concept_class_id(obj_dic[concept]),
                    "object_id": j,
                    "id": ann_id,
                    "area": KandinskyUniverse.get_area(obj, width=640),
                }
                annotations.append(annotation_dict)
                ann_id += 1

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_concept_categories_dict()
    
    with open(base_path + "true/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)
    """
    # reset annotations
    instances = dict()
    images = []
    annotations = []
    ann_id = 0

    for i, img in enumerate(neg_imgs):
        # img = kandinskyFigure(kf)
        img.save(false_path + "/%06d" % i + ".png")

        """
        # images
        images.append({"file_name": "%06d" % i + ".png", "id": i})
        
        # annotations
        for j, obj in enumerate(kf):
            obj_dic = obj.__dict__
            obj_dic["object_id"] = j

            img_dic = {"img_id": i, "scene": []}
            img_dic["scene"].append(obj_dic)

            for concept in ["size_cls", "color", "shape"]:
                annotation_dict = {
                    "image_id": i,
                    "pos": obj_dic["pos"],
                    "bbox": KandinskyUniverse.get_coco_bounding_box(obj, width=640),
                    "iscrowd": 0,
                    "category_id": get_concept_class_id(obj_dic[concept]),
                    "object_id": j,
                    "id": ann_id,
                    "area": KandinskyUniverse.get_area(obj, width=640),
                }
                annotations.append(annotation_dict)
                ann_id += 1

    instances["images"] = images
    instances["annotations"] = annotations
    instances["categories"] = get_concept_categories_dict()
    
    with open(base_path + "false/instances.json", "w") as f:
        json.dump(instances, f, sort_keys=True, indent=4)
    """
        
if __name__ == "__main__":
    modes = ["train", "val"] # , "val", "test"]
    for mode in modes:
        # n: number of examples for each class
        if mode == "train":
            #generate_images("twopairs", mode, n=15)
            #generate_images("threepairs", mode, n=15)
            #generate_images("closeby", mode, n=15)
            # generate_images("closeby-multi", mode, n=15)
            #generate_images("red-triangle", mode, n=15)
            # generate_images("online_pretrain", mode, n=15)
            generate_images("online-pair", mode, n=16)
            # generate_images("online-7", mode, n=15)
            # generate_images(sys.argv[1], mode, n=10000)
            # generate_images(sys.argv[1], mode, n=5000)
        else:
            #generate_images("twopairs", mode, n=2000)
            #generate_images("threepairs", mode, n=2000)
            generate_images("closeby", mode, n=2000)
            # generate_images("closeby-multi", mode, n=15)
            #generate_images("red-triangle", mode, n=2000)
            # generate_images("online_pretrain", mode, n=15)
            #generate_images("online-pair", mode, n=2000)

