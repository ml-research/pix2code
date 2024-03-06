import datetime
import os
import random

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.text.makeTextTasks import delimiters
from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import *
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs


# Primitives
def _incr(x):
    return x + 1


def _incr2(x):
    return x + 2


def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}


def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


if __name__ == "__main__":
    args = commandlineArguments(
        enumerationTimeout=10,
        activation="tanh",
        iterations=10,
        recognitionTimeout=3600,
        a=3,
        maximumFrontier=10,
        topK=2,
        pseudoCounts=30.0,
        helmholtzRatio=0.5,
        structurePenalty=1.0,
        CPUs=numberOfCPUs(),
    )

    timestamp = datetime.datetime.now().isoformat()
    outdir = "experimentOutputs/demo/"
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create list of primitives
    primitives = [
        Primitive("incr", arrow(tint, tint), _incr),
        Primitive("incr2", arrow(tint, tint), _incr2),
    ]

    # Create grammar
    grammar = Grammar.uniform(primitives)

    # Training data
    def add1():
        return addN(1)

    def add2():
        return addN(2)

    def add3():
        return addN(3)

    training_examples = [
        {"name": "add1", "examples": [add1() for _ in range(5000)]},
        {"name": "add2", "examples": [add2() for _ in range(5000)]},
        {"name": "add3", "examples": [add3() for _ in range(5000)]},
    ]

    training = [get_tint_task(item) for item in training_examples]

    # Testing data
    def add4():
        return addN(4)

    testing_examples = [
        {"name": "add4", "examples": [add4() for _ in range(500)]},
    ]
    testing = [get_tint_task(item) for item in testing_examples]

    # EC iterate
    generator = ecIterator(grammar, training, testingTasks=testing, **args)
    for i, _ in enumerate(generator):
        print("ecIterator count {}".format(i))
