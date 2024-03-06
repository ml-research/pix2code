import random
import math
import PIL
import numpy as np
from kp.generate_task_clauses import Clause, ClauseForPairs

from kp.generate_task_clauses import Object

from .KandinskyTruth import KandinskyTruthInterfce
from .KandinskyUniverse import get_bounding_box, kandinskyShape, overlaps


MAX_TRY = 10000

class ClauseBasedKandinskyFigure(KandinskyTruthInterfce):

    def __init__(self, universe, min=4, max=4, clause=None, color=None, shape=None, size=None):
        super().__init__(universe, min, max)
        assert clause != None
        self.clause = clause
        self.color = color
        self.shape = shape
        self.size = size

    def humanDescription(self):
        return f"Kandinsky figure based on clause {str(self.clause)}."

    def true_kf(self, n=1):
        kfs = []

        for i in range(n):
            clause_true = False
            counter = 0

            while not clause_true and counter < MAX_TRY:
                clause_name = str(self.clause)

                # targeted generation of KFs
                if "online" in clause_name and not "not_online" in clause_name:
                    if random.choice(list(range(10))) == 1:
                        kf = self._onlinekf(self.min, self.max)
                    else:
                        kf = self._randomkf(self.min, self.max)
                elif "closeby" in str(self.clause) and not "not_closeby" in str(self.clause):
                    if random.choice(list(range(10))) == 1:
                        kf = self._closebykf(self.min, self.max)
                    else:
                        kf = self._randomkf(self.min, self.max)
                else:
                    kf = self._randomkf(self.min, self.max)

                if type(self.clause) == Clause:
                    # set concepts if all should be the same
                    if "same_color" in clause_name and "not_same_color" not in clause_name:
                        if random.choice(list(range(10))) == 1:  # TODO: remove this?
                            color = random.choice(self.u.kandinsky_colors)
                            for o in kf:
                                o.color = color
                    if "same_shape" in clause_name and "not_same_shape" not in clause_name:
                        if random.choice(list(range(10))) == 1:
                            shape = random.choice(self.u.kandinsky_shapes)
                            for o in kf:
                                o.shape = shape

                objects = []
                for obj in kf:
                    new_obj = Object(obj.color, obj.shape, obj.size_cls, get_bounding_box(obj, width=640), obj.pos) # TODO: coco or normal bbox?
                    objects.append(new_obj)
                # eval kf based on clause
                clause_true = self.clause.eval(objects)
                counter += 1
            if counter == MAX_TRY:
                print(f"WARNING: could not generate true KF for image {i} \n")
                return None
            else:
                kfs.append(kf)
        return kfs

    def false_kf(self, n=1):
        kfs = []
        for i in range(n):
            if i % 100 == 0:
                print(f"Generating false KF {i} of {n} \n")
            clause_false = False
            counter = 0
            while not clause_false and counter < MAX_TRY:
                    kf = self._randomkf(self.min, self.max)

                if type(self.clause) == ClauseForPairs or type(self.clause) == Clause: # todo remove second part
                    # set concepts if all should be the same
                    if "not_same_color" in clause_name:
                        if random.choice([0,1,2,3,4,5,6,7,8,9]) == 1:
                            color = random.choice(self.u.kandinsky_colors)
                            for o in kf:
                                o.color = color
                    if "not_same_shape" in clause_name:
                        if random.choice([0,1,2,3,4,5,6,7,8,9]) == 1:
                            shape = random.choice(self.u.kandinsky_shapes)
                            for o in kf:
                                o.shape = shape
                    if "not_same_size" in clause_name:
                        if random.choice([0,1]) == 1:
                            minsize, maxsize = self._get_min_max_size(4)
                            size = minsize + (maxsize - minsize) * random.random()
                            if size > 2 / 3:
                                size_cls = "big"
                            elif size > 1 / 3:
                                size_cls = "medium"
                            else:
                                size_cls = "small"
                            for o in kf:
                                o.size = size
                                o.size_cls = size_cls

                objects = []
                for obj in kf:
                    new_obj = Object(obj.color, obj.shape, obj.size_cls, get_bounding_box(obj), obj.pos)
                    objects.append(new_obj)
                # TODO eval kf based on clause
                clause_false = not self.clause.eval(objects)
                counter += 1
            if counter == MAX_TRY:
                print(f"WARNING: could not generate false KF for image {i} \n")
                return None
            else:
                kfs.append(kf)
        return kfs

    def _generateobject(self, minsize=0.1, maxsize=0.5):
        o = kandinskyShape()
        o.color = random.choice(self.u.kandinsky_colors)
        o.shape = random.choice(self.u.kandinsky_shapes)
        o.size = minsize + (maxsize - minsize) * random.random()
        # discretize size
        if o.size > 2 / 3:
            size = "big"
        elif o.size > 1 / 3:
            size = "medium"
        else:
            size = "small"
        o.size_cls = size
        o.x = o.size / 2 + random.random() * (1 - o.size)
        o.y = o.size / 2 + random.random() * (1 - o.size)
        o.pos = [o.x, o.y]
        return o

    def _randomkf(self, min, max):
        kf = []
        kftemp = []
        n = random.randint(min, max)

        minsize, maxsize = self._get_min_max_size(n)

        i = 0
        maxtry = 30
        while i < n:
            kftemp = kf
            t = 0
            o = self._generateobject(minsize, maxsize)
            kftemp = kf[:]
            kftemp.append(o)
            while overlaps(kftemp) and (t < maxtry):
                o = self._generateobject(minsize, maxsize)
                kftemp = kf[:]
                kftemp.append(o)
                t = t + 1
            if t < maxtry:
                kf = kftemp[:]
                i = i + 1
            else:
                maxsize = maxsize * 0.95
                minsize = minsize * 0.95
        return kf

    def _get_sx_sy_dx_dy(self, minsize, maxsize):

        minsize = minsize * 0.8
        maxsize = maxsize * 0.8

        dx = math.cos(random.random() * math.pi * 2) * (1/2 - maxsize / 2)
        dy = math.sin(random.random() * math.pi * 2) * (1/2 - maxsize / 2)

        sx = 1/2 - dx
        sy = 1/2 + dy

        ex =  1/2 + dx
        ey = 1/2 - dy

        dx = ex - sx
        dy = ey - sy

        return sx, sy, dx, dy

    def _get_min_max_size(self, n):
        minsize = 0.03
        if n < 7:
            minsize = 0.2
        if n == 3:
            minsize = 0.2
        if n == 2:
            minsize = 0.3
        if n == 1:
            minsize = 0.4

        maxsize = 0.6
        if n == 5:
            maxsize = 0.5
        if n == 6:
            maxsize = 0.4
        if n == 7:
            maxsize = 0.3
        if n > 7:
            m = n - 7
            maxsize = 0.2 - m * (0.2) / 70.0

        return minsize, maxsize
