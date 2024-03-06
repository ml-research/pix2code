try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import datetime
import os
import random

import torch

from dreamcoder.domains.clevr_revised_color_count.main import main
from dreamcoder.domains.list.main import LearnedFeatureExtractor
from dreamcoder.domains.text.makeTextTasks import delimiters, guessConstantStrings
from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import *
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs


if __name__ == "__main__":
    # Set recursion limit higher for pickle
    sys.setrecursionlimit(10000)

    args = commandlineArguments(
        enumerationTimeout=720,
        activation="tanh",
        iterations=16,
        recognitionTimeout=3600,
        a=3,
        maximumFrontier=5,
        topK=2,
        pseudoCounts=30.0,
        helmholtzRatio=0.5,
        structurePenalty=1.5,
        CPUs=96,
        featureExtractor=LearnedFeatureExtractor,
        biasOptimal=True,
        contextual=True,
        auxiliary=True,
        taskReranker="unsolved",
    )

    # start timing
    start_time = datetime.datetime.now()
    eval = args.pop("eval")
    print("EVAL: ", eval)
    if eval == 0:
        _ = args.pop("test_idx")
        main(args, eval=False)
    else:
        main(args, eval=True)
    # end timing
    end_time = datetime.datetime.now()
    print("Total time taken: {}".format(end_time - start_time))
