import datetime
import os
import random
import numpy as np
from rtpt.rtpt import RTPT

from dreamcoder.domains.list.listPrimitives import bootstrapTarget
from dreamcoder.domains.clevr.parse_clevr_tasks import parse_relation_tasks
from dreamcoder.domains.relation.relation_primitives import (
    get_baseline_primitives,
    get_less_primitives,
    get_only_used_primitives,
    get_primitives,
    get_less_primitives_with_plus,
    get_clevr_primitives,
)
from dreamcoder.domains.text.main import (
    ConstantInstantiateVisitor,
    LearnedFeatureExtractor,
)
from dreamcoder.domains.text.makeTextTasks import (
    delimiters,
    guessConstantStrings,
    loadPBETasks,
    makeTasks,
)
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import *
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import eprint, numberOfCPUs
from sklearn.model_selection import KFold


def main(args, eval=False):

    # Variables for storing results
    timestamp = datetime.datetime.now().isoformat()

    split = args.pop("task_folder")
    n_tasks = args.pop("number_tasks")
    seed = args.pop("seed")

    outdir = f"experimentOutputs/clevr/{split}/100/{seed}/"
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # check if checkpoint exists
    if len(os.listdir(outdir)) > 0:
        # get most recent file
        files = os.listdir(outdir)
        files.sort()
        # remove pdf files
        files = [f for f in files if ".pdf" not in f]
        most_recent_file = None
        for i in [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
            for file in files:
                if f"it={i}" in file:
                    most_recent_file = file
                    break
            if most_recent_file is not None:
                break

        if most_recent_file is not None:
            args.update({"resume": f"{outdir}{most_recent_file}"})

    # Create grammar
    baseGrammar = Grammar.uniform(get_clevr_primitives())

    # Parse tasks
    all_train = parse_relation_tasks(path=f"data/curi/{split}/train")
    print("Number of train tasks:", len(all_train))
    # shuffle tasks
    random.shuffle(all_train)

    if eval:
        test_idx = args.pop("test_idx")
        test = parse_relation_tasks(
            path=f"data/curi_image_dc_test_tasks/{split}/support"
        )
        print("Number of test tasks:", len(test))
        idx_start = n_tasks * test_idx
        idx_end = n_tasks * (test_idx + 1)
        print("Evaluate test tasks from %d to %d" % (idx_start, idx_end))
        test = test[idx_start:idx_end]
    else:
        test = []

    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    # set seed for model
    random.seed(seed)
    np.random.seed(seed)
    args.update({"seed": seed})
    print("Seed: ", seed)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="XX",
        experiment_name=f"DreamCoder_{split}_{seed}",
        max_iterations=100,
    )
    # Start the RTPT tracking
    rtpt.start()
    rtpt.step()

    if eval:
        print("Start evaluating...")
        train = []
        args.update({"testingTimeout": 600})

    # EC iterate
    generator = ecIterator(baseGrammar, train, testingTasks=test, **args)
    for i, _ in enumerate(generator):
        rtpt.step()
        print("ecIterator count {}".format(i))
