
import random

import PIL

from .KandinskyTruth import KandinskyTruthInterfce
from .KandinskyUniverse import kandinskyShape, overlaps


class SameConcept(KandinskyTruthInterfce):

    def __init__(self, universe, min=4, max=4, color=None, shape=None, size=None):
        super().__init__(universe, min, max)
        self.color = color
        self.shape = shape
        self.size = size

    def humanDescription(self):
        return "Kandinsky figure with n objects that have at least one concept in common"
    
    def true_kf(self, n=1):
        kfs = []
        
        for i in range(n):
            kf = self._randomkf(self.min, self.max)
            # set same concept for image
            if type(self.color) == bool:
                color = random.choice(self.u.kandinsky_colors)
            if type(self.shape) == bool:
                shape = random.choice(self.u.kandinsky_shapes)
            for s in kf:
                if self.color:
                    if type(self.color) == bool:
                        s.color = color
                    else:
                        s.color = self.color
                if self.shape:
                    if type(self.shape) == bool:
                        s.shape = shape
                    else:
                        s.shape = self.shape
            kfs.append(kf)
        return kfs

    def false_kf(self, n=1):
        kfs = []
        for i in range(n):
            kf = self._randomkf(self.min, self.max)
            if type(self.color) == bool:
                color = random.choice(self.u.kandinsky_colors)
            else:
                color = self.color
            if type(self.shape) == bool:
                shape = random.choice(self.u.kandinsky_shapes)
            else:
                shape = self.shape
            for i, s in enumerate(kf):
                if i == 0:
                    if self.color:
                        s.color = random.choice([c for c in self.u.kandinsky_colors if c != color])
                    if self.shape:
                        s.shape = random.choice([c for c in self.u.kandinsky_shapes if c != shape])
                else:
                    if self.color:
                        s.color = color
                    if self.shape:
                        s.shape = shape
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
        maxtry = 20
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



    
    

