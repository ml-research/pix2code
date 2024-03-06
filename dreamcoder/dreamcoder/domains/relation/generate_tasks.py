import dill
import random
from scipy.spatial import distance



class Object():
    def __init__(self, color, shape, size, bbox):
        self.color = color
        self.shape = shape
        self.size = size
        self.bbox = bbox

"""
Predicates
"""
def same_color(objects):
    result = True
    for i in range(len(objects) - 1):
        if objects[i].color != objects[i+1].color:
            result = False
    return result

def same_shape(objects):
    result = True
    for i in range(len(objects) - 1):
        if objects[i].shape != objects[i+1].shape:
            result = False
    return result

def same_size(objects):
    result = True
    for i in range(len(objects) - 1):
        if objects[i].size != objects[i+1].size:
            result = False
    return result

def _closeby(bbox1, bbox2, threshold=10):
    bbox1 = [int(b) for b in bbox1]
    bbox2 = [int(b) for b in bbox2]

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

def closeby(objects):
    result = True
    for i in range(len(objects) - 1):
        if not _closeby(objects[i].bbox, objects[i+1].bbox):
            result = False
    return result


class Clause():

    def __init__(self, predicates, conjunctors=[], nots=[]):
        self.predicates = predicates
        self.conjunctors = conjunctors
        self.nots = nots

    def eval(self, objects):

        # get evaluations of single predicates
        evals = [pred(objects) for pred in self.predicates]

        # apply nots
        for i, n in enumerate(self.nots):
            if n:
                evals[i] = not(evals[i])

        result = evals[0]       
        for i, conj in enumerate(self.conjunctors):

            if conj == "and":
                result = result and evals[i+1]
            elif conj == "or":
                result = result or evals[i+1]

        return result


    def __str__(self) -> str:
        description = ""
        for i in range(len(self.predicates)):
            if self.nots[i]:
                description += "not "
            description += f"{self.predicates[i].__name__} "
            if len(self.conjunctors) > i:
                description += f"{self.conjunctors[i]} "

        return description


def generate_tasks_1(predicates):
    clauses = []
    for pred in predicates:
        clauses.append(Clause([pred], [], [False]))
        clauses.append(Clause([pred], [], [True]))

    return clauses

def generate_tasks_2(predicates, conjunctors):
    clauses = []
    for i, pred in enumerate(predicates):
        for pred2 in predicates[i+1:]:
            for conj in conjunctors:

                clauses.append(Clause([pred, pred2], [conj], [False, False]))
                clauses.append(Clause([pred, pred2], [conj], [True, False]))
                clauses.append(Clause([pred, pred2], [conj], [False, True]))
                clauses.append(Clause([pred, pred2], [conj], [True, True]))

    return clauses


def get_objects_from_seq(seq):
    num_objects = seq.count("bbox")
    objects = []
    for i in range(num_objects):
        s = i * 15
        objects.append(Object(color=seq[s + 8], shape=seq[s + 11], size=seq[s + 14], bbox=seq[s + 2 : s + 6]))

    return objects

def generate_tasks(examples_per_task=20):

    predicates = [same_shape, same_color, same_size, closeby]
    conjunctors = ["and", "or"]

    clauses = generate_tasks_1(predicates)
    clauses += generate_tasks_2(predicates, conjunctors)

    for c in clauses:
        print(c)

    print(f"Generated {len(clauses)} clauses.")

    task_examples = []

    for c in clauses:
        positives = []
        negatives = []
        iteration = 0
        while (len(positives) < 10 or len(negatives) < 10):
            seq = get_random_pix2seq_seq()
            objects = get_objects_from_seq(seq)
            eval = c.eval(objects)
            if eval:
                positives.append({"i": seq, "o": eval})
            else:
                negatives.append({"i": seq, "o": eval})
            iteration += 1

        examples = positives + negatives
        task_examples.append({
            "name": str(c),
            "examples": examples
        })

    return task_examples

def get_random_pix2seq_seq(objects=2, colors=["blue", "red", "yellow"], all_str=True):
    """Creates a random pix2seq sequence with variable objects"""
    seq = []
    for o in range(objects):
        seq = seq + ["bbox", "obj" + str(o)]
        # bbox
        x = random.choice(range(400))
        y = random.choice(range(400))
        w = random.choice(range(10, 100))
        if all_str:
            seq = seq + [str(x), str(y), str(x + w), str(y + w)]
        else:
            seq = seq + [x, y, x + w, y + w]
        # color
        seq = seq + ["color", "obj" + str(o)]
        seq = seq + [random.choice(colors)]
        # shape
        seq = seq + ["shape", "obj" + str(o)]
        seq = seq + [random.choice(["square", "triangle", "circle"])]
        # size
        # TODO: set size based on bbox
        seq = seq + ["size", "obj" + str(o)]
        if all_str:
            seq = seq + [str(random.choice([1, 2, 3]))]
        else:
            seq = seq + [random.choice([1, 2, 3])]
    return seq


if __name__ == "__main__":

    objects = [Object("blue", "triangle", 1, [1,1,1,1]), Object("red", "triangle", 1 , [2,2,2,2])]
    predicates = [same_shape, same_color, same_size, closeby]
    conjunctors = ["and", "or"] # ["and"]
    nots = [False, True]
    c = Clause(predicates, conjunctors, nots)
    print(c.eval(objects))

    generate_tasks(predicates, conjunctors)
    


