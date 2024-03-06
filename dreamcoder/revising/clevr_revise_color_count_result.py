import dill

try:
    import binutil
except ModuleNotFoundError:
    import bin.binutil

from dreamcoder.program import *
from dreamcoder.domains.relation import *
from dreamcoder.domains.relation.relation_primitives import *

get_baseline_primitives()
get_clevr_primitives()


def revise_color_count():

    path = "experimentOutputs/clevr/color_count/100/0/2023-11-04T14:09:18.846576_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=15_MF=5_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False_graph=True.pickle"
    with open(path, "rb") as handle:
        result = dill.load(handle)

    # current grammar
    grammar = result.grammars[-1]
    print(result.grammars[-1])

    invented_programs_string = [
        "(lambda (lambda (count (map (lambda (index 4 $0)) $1) $0)))",
        "(lambda (lambda (count (map (lambda (index 5 $0)) $1) $0)))",
        "(lambda (lambda (count (map (lambda (index 6 $0)) $1) $0)))",
        "(lambda (lambda (count (map (lambda (index 7 $0)) $1) $0)))",
    ]

    # parse to programs
    invented_programs = [Invented(Program.parse(s)) for s in invented_programs_string]

    # set parameters
    likelihood = -0.3

    example_list = [[[[[1, 2], [2, 3]], 1], 1], [[[[3, 4], [5, 6]], 1], 1]]
    function_type = guess_arrow_type(example_list)

    for invented_program in invented_programs:
        invented_program.tp = function_type

    # revise grammar
    e2l = grammar.expression2likelihood
    for p in invented_programs:
        e2l[p] = likelihood
        grammar.primitives.append(p)
        grammar.productions.append((likelihood, function_type, p))

    grammar.expression2likelihood = e2l

    if False:
        # reset recognition task metrics
        result.recognitionTaskMetrics = {}

        # frontiersOverTime
        frontiers_over_time = result.frontiersOverTime

        for task, frontiers in frontiers_over_time.items():
            for frontier in frontiers:
                frontier.entries = []

        result.frontiersOverTime = frontiers_over_time

        # taskSolutions
        task_solutions = result.taskSolutions
        for task, frontier in task_solutions.items():
            frontier.entries = []

        result.taskSolutions = task_solutions

        # allFrontiers
        all_frontiers = result.allFrontiers
        for task, frontier in all_frontiers.items():
            frontier.entries = []

        result.allFrontiers = all_frontiers

    # save checkpoint
    target_path = "/workspace/experimentOutputs/clevr_revised/color_count/0/only_added/2023-11-04T14:09:18.846576_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=15_MF=5_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False.pickle"
    result.grammars[-1] = grammar
    with open(target_path, "wb") as handle:
        dill.dump(result, handle)


if __name__ == "__main__":
    revise_color_count()
