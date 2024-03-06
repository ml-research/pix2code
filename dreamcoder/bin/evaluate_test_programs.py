try:
    import binutil
except ModuleNotFoundError:
    import bin.binutil

import pandas as pd
import numpy as np

from dreamcoder.program import *
from dreamcoder.domains.relation import *
from dreamcoder.domains.relation.relation_primitives import *

get_baseline_primitives()
get_clevr_primitives()
get_clevr_primitives_unconfounded()

SPLITS = [
    "iid_sampling",
    "color_boolean",
    "color_count",
    "color_location",
    "color_material",
    "color_sampling",
    "comp_sampling",
    "length_threshold",
    "shape_sampling",
]

SPLITS_NAMES = [
    "IID",
    "Boolean",
    "Counting",
    "Extrinsic",
    "Intrinsic",
    "Binding(color)",
    "Compositional",
    "Complexity",
    "Binding(shape)",
]


PROGRAMS_PER_SPLIT = {
    "iid_sampling": 8389,
    "color_boolean": 2565,
    "color_count": 47,
    "color_location": 750,
    "color_material": 283,
    "color_sampling": 2590,
    "comp_sampling": 2402,
    "length_threshold": 8363,
    "shape_sampling": 1484,
}


def get_accuracies_for_concept():
    task = "task_4919"

    # create df for solved programs
    SPLITS = ["iid_sampling"]
    df_solved_programs = pd.DataFrame(
        columns=["0", "1", "2", "avg", "total tasks"], index=SPLITS
    )

    for split in SPLITS:
        results = []
        for seed in range(3):
            results_df = pd.read_csv(
                f"/workspace/experimentOutputs/clevr/{split}_test_{seed}.csv"
            )
            results.append(results_df)

        task_cbas = []
        for seed in range(3):
            task_cbas.append(
                float(results[seed].loc[results[seed]["task_name"] == task]["CBA"])
            )
        # get mean and std
        print(f"{split} CBA: {task_cbas}")
        mean_cba = np.mean(task_cbas)
        std_cba = np.std(task_cbas)
        print(f"{split} mean CBA: {mean_cba}, std: {std_cba}")


def main(domain, mode, modality):

    get_baseline_primitives()
    get_clevr_primitives()

    if modality == "image":
        modality = "image_"
    else:
        modality = ""

    # create df for solved programs

    if domain == "kandinsky":
        SPLITS = ["kandinsky"]

    df_solved_programs = pd.DataFrame(
        columns=["0", "1", "2", "avg", "total tasks"], index=SPLITS
    )

    for split in SPLITS:

        results = []
        for seed in range(3):
            results_df = pd.read_csv(
                f"/workspace/experimentOutputs/{domain}/{split}_{mode}_{modality}{seed}.csv"
            )
            results.append(results_df)

            df_solved_programs.loc[split, str(seed)] = len(results_df)
        # average number of solved tasks

        avg_solved = round(np.mean([len(df) for df in results]))
        print(f"{split} avg solved: {avg_solved}")

        df_solved_programs.loc[split, "avg"] = avg_solved
        df_solved_programs.loc[split, "total tasks"] = PROGRAMS_PER_SPLIT[split]

        # mean_accs = [df["CBA"].mean() for df in results]
        # print(mean_accs)

        # cba_all = [0, 0, 0]
        # for seed in range(3):
        #     cba_all[seed] = mean_accs[seed] * (
        #         len(results[seed]) / NUMBER_TEST_TASKS
        #     ) + 0.5 * (1 - (len(results[seed]) / NUMBER_TEST_TASKS))

        # mean_accs = np.array(cba_all) * 100
        # print(mean_accs)
        # mean_acc = np.mean(mean_accs)
        # std = np.std(mean_accs)
        # print(round(mean_acc, 2), round(std, 2))

    # rename index
    df_solved_programs = df_solved_programs.rename(
        index=dict(zip(SPLITS, SPLITS_NAMES))
    )
    print(df_solved_programs)
    print(df_solved_programs.to_latex())


if __name__ == "__main__":
    # main("kandinsky", "test", modality="image")

    get_accuracies_for_concept()
