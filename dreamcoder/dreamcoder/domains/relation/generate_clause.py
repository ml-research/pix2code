import random
from enum import Enum
from scipy.spatial import distance


# Modes
FILTER = 0
CONJ = 1
NOT = 2
PRED = 3

# predicates
def _same_color(c1, c2): return c1 == c2
def _same_shape(s1, s2): return s1 == s2
def _same_size(s1, s2): return s1 == s2

def _closeby(bbox1, bbox2, threshold=10):
    corners1 = [
        (bbox1[0], bbox1[1]),
        (bbox1[0], bbox1[3]),
        (bbox1[2], bbox1[1]),
        (bbox1[2], bbox1[3]),
    ]
    corners2 = [
        (bbox2[0], bbox2[1]),
        (bbox2[0], bbox2[3]),
        (bbox2[2], bbox2[1]),
        (bbox2[2], bbox2[3]),
    ]
    output = False
    for c1 in corners1:
        for c2 in corners2:
            if distance.euclidean(c1, c2) < threshold:
                output = True

    return output

# predicates (2 objects only)
def same_color(colors, shapes, sizes, bboxes): 
    print(colors)
    return colors[0] == colors[1]

def same_shape(colors, shapes, sizes, bboxes): 
    print(shapes)
    return shapes[0] == shapes[1]

def same_size(colors, shapes, sizes, bboxes): 
    print(sizes)
    return sizes[0] == sizes[1]

def closeby(colors, shapes, sizes, bboxes):
    return _closeby(bboxes[0], bboxes[1])

# conjunctors
def _and(x,y): return x and y 
def _or(x,y): return x or y

# not
def _not(c): return not c

predicates = [same_color, same_shape, same_size, closeby]
conjunctors = [_and, _or]


def get_clause(objects, d=0):
    """ Creates a clause function based on predicates and conjunctors and a text description of it."""
    print(f"depth: {d}")
    num_objects = len(objects)
    # randomly select next step
    mode = random.choice(range(4))

    if d == 0 or (mode == CONJ and d < 3):
        # choose conjunctor
        conjuctor = random.choice(conjuctors)

        # get two clauses to combine
        c1, ex1 = get_clause(objects, d+1)
        c2, ex2 = get_clause(objects, d+1)

        description = f"({ex1} {conjuctor.__name__} {ex2})"

        f = lambda c,s,sz,b: conjuctor(c1(c,s,sz,b), c2(c,s,sz,b))
        return f, description

    if mode == FILTER and len(objects) > 2:
        # sample 2 items for which the further clause should apply
        indices = random.sample(range(num_objects), 2)
        indices.sort()
        c, ex = get_clause(indices, d+1)
        f = lambda colors,shapes,sizes,bboxes: c(
            [c for idx, c in colors if idx in indices],
            [s for idx, s in shapes if idx in indices],
            [sz for idx, sz in sizes if idx in indices],
            [b for idx, b in bboxes if idx in indices])
        
        return f, ex

    if mode == NOT and d < 3:
        c, ex = get_clause(objects, d+1)
        f = lambda c,s,sz,b: _not(c(c,s,sz,b))
        return f, f"not({ex})"  

    else:
        # end recursion and randomly select predicate
        predicate = random.choice(predicates)

        # predicate for objects 
        os = ""
        for o in objects:
            os += "obj" + str(o) + ","
        description = f"{predicate.__name__}( {os} )"
        
        return lambda c,s,sz,b: predicate(c,s,sz,b), description


            
import random

rules = []

def get_clauses(objects, d=0):
    """ Creates clause functions based on predicates and conjunctors and gives a text description of them."""
    print(f"depth: {d}")

    num_objects = len(objects)
    new_clauses = []

    # for predicate in predicates:

    # predicate for objects 
    # TODO iterate over combinations of objects
    for o1 in objects:
        for o2 in range(o1 + 1, len(objects)):

            os = ""
            for o in [o1,o2]:
                os += "obj" + str(o) + ","

            dict = {}
            dict["desc"] = f"{same_color.__name__}( {os} )"
            dict["func"] = lambda colors,shapes,sizes,bboxes: same_color(colors, shapes, sizes, bboxes)

            new_clauses.append(dict)

            dict = {}
            dict["desc"] = f"{same_shape.__name__}( {os} )"
            dict["func"] = lambda colors,shapes,sizes,bboxes: same_shape(colors, shapes, sizes, bboxes)

            new_clauses.append(dict)

    # not 
    if d < 2:
        clauses = get_clauses(objects, d+1)

        for clause in clauses:
            dict = {}
            dict["desc"] = f"not({clause['desc']})"  
            dict["func"] = lambda c,s,sz,b: _not(clause['func'](c,s,sz,b))
            new_clauses.append(dict)


    if d < 1:
        # choose conjunctor if more than two predicates are left
        # TODO track predicates
        for conjunctor in conjunctors:

            # get two clauses to combine
            clauses1 = get_clauses(objects, d+1)
            clauses2 = clauses1

            for c1 in clauses1:
                for c2 in clauses2:
                    # only combine if clauses are not the same
                    if not (c1["desc"] in c2["desc"] or c2["desc"] in c1["desc"]):
                        dict = {}
                        dict["desc"] = f"({c1['desc']} {conjunctor.__name__} {c2['desc']})"
                        dict["func"] = lambda c,s,sz,b: conjunctor(c1["func"](c,s,sz,b), c2["func"](c,s,sz,b))

                        new_clauses.append(dict)

    return new_clauses

    """
    if mode == Mode.FILTER:
        # sample 2 items for which the further clause should apply
        indices = random.sample(range(num_objects), 2)
        indices.sort()
        c, ex = get_clause(indices, h=h-1)
        f = lambda colors,shapes,sizes,bboxes: c(
            [c for idx, c in colors if idx in indices],
            [s for idx, s in shapes if idx in indices],
            [sz for idx, sz in sizes if idx in indices],
            [b for idx, b in bboxes if idx in indices])
        
        return f, ex
    """
            

if __name__ == "__main__":
    clauses = get_clauses([0,1])
    for c in clauses:
        print(c["desc"])
    print(len(clauses))
    
    print(clauses[0]["func"](["red", "red"], ["triangle", "square"], [1, 1], [[1,2,3,4],[2,3,4,5]]))