import itertools
import dill
import random
from scipy.spatial import distance
import numpy as np


class Object:
    def __init__(self, color, shape, size, bbox, pos):
        self.color = color
        self.shape = shape
        self.size = size
        self.bbox = bbox
        self.pos = pos


"""
Predicates
"""


def same_color(objects):
    result = True
    for i in range(len(objects) - 1):
        if objects[i].color != objects[i + 1].color:
            result = False
    return result


def same_shape(objects):
    result = True
    for i in range(len(objects) - 1):
        if objects[i].shape != objects[i + 1].shape:
            result = False
    return result


def same_size(objects):
    result = True
    for i in range(len(objects) - 1):
        if objects[i].size != objects[i + 1].size:
            result = False
    return result


def one_is_red_triangle(objects):
    eval = False
    red_triangle = False
    for i in range(len(objects)):
        if objects[i].color == "red" and objects[i].shape == "triangle":
            if red_triangle == True:
                return False
            else:
                red_triangle = True
                eval = True
        elif objects[i].color == "red" or objects[i].shape == "triangle":
            return False
    return eval


class Clause:
    """Class for a clause of predicates"""

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
                evals[i] = not (evals[i])

        result = evals[0]
        for i, conj in enumerate(self.conjunctors):

            if conj == "and":
                result = result and evals[i + 1]
            elif conj == "or":
                result = result or evals[i + 1]

        return result

    def __str__(self) -> str:
        description = ""
        for i in range(len(self.predicates)):
            if self.nots[i]:
                description += "not_"
            description += f"{self.predicates[i].__name__}"
            if len(self.conjunctors) > i:
                description += f"_{self.conjunctors[i]}_"

        return description


class ClauseForPairs:
    """Class for a clause of predicates for pairs of objects"""

    def __init__(self, num_pairs, predicates, conjunctors=[], nots=[]):
        self.num_pairs = num_pairs
        self.evals = [False] * num_pairs
        self.predicates = predicates
        self.conjunctors = conjunctors
        self.nots = nots

        assert len(predicates) == len(conjunctors)
        for i in range(num_pairs):
            assert len(predicates[i]) >= len(nots[i])
            if len(conjunctors) > 0:
                if len(self.predicates[i]) > 1:
                    assert len(self.predicates[i]) == len(self.conjunctors[i]) + 1

            self.evals[i] = [False] * len(predicates)

    def __str__(self) -> str:
        description = str(self.num_pairs * 2)
        for i, preds in enumerate(self.predicates):
            if len(preds) > 0:
                description += "_pair" + str(i + 1) + "_"
                for j in range(len(preds)):
                    if self.nots[i][j]:
                        description += "not_"
                    description += f"{preds[j].__name__}"
                    if len(self.conjunctors[i]) > j:
                        description += f"_{self.conjunctors[i][j]}_"

        return description

    def _divide_into_pairs(self, list_of_pairs):
        result = []
        for lst in list_of_pairs:
            pairs = []
            for i in range(0, len(lst), 2):
                if i + 1 < len(lst):
                    pairs.append((lst[i], lst[i + 1]))
                else:
                    raise ValueError("List of pairs must have even length")
            result.append(pairs)
        return result

    def _get_all_pairs(self, objects):
        assert len(objects) % 2 == 0
        all_pairs = []

        for subset in itertools.permutations(objects, len(objects)):
            all_pairs.append(subset)

        # reformat
        all_pairs = self._divide_into_pairs(all_pairs)

        # remove duplicate pairs
        for i, pairs in enumerate(all_pairs):
            for pairs2 in all_pairs[i + 1 :]:
                if all(set(a) == set(b) for a, b in zip(pairs, pairs2)):
                    all_pairs.remove(pairs2)

        return all_pairs

    def eval(self, objects):

        pair_combinations = self._get_all_pairs(objects)
        for pairs in pair_combinations:

            evals = [False] * self.num_pairs
            for i in range(self.num_pairs):
                evals[i] = [pred(pairs[i]) for pred in self.predicates[i]]

                # apply nots
                for j, n in enumerate(self.nots[i]):
                    if n:
                        evals[i][j] = not (evals[i][j])

                # check if there are predicates for this pair
                if evals[i] == []:
                    # no predicates, so this pair is always true
                    evals[i] = True
                    continue

                # apply conjunctors
                result = evals[i][0]
                for j, conj in enumerate(self.conjunctors[i]):
                    if conj == "and":
                        result = result and evals[i][j + 1]
                    elif conj == "or":
                        result = result or evals[i][j + 1]

                # if evaluation of first pair fails, break and try other combination
                if not result:
                    evals[i] = result
                    break
                else:
                    evals[i] = result

            # if all evaluations of pairs are true, return true
            if all(evals):
                return True

        # no pair matched the predicates
        return False


NOT_COMBINATIONS_2_2 = [
    [[False, False], [False, False]],
    [[False, False], [False, True]],
    [[False, False], [True, False]],
    [[False, False], [True, True]],
    [[False, True], [False, True]],
    [[False, True], [True, False]],
    [[False, True], [True, True]],
    [[True, False], [True, False]],
    [[True, False], [True, True]],
    [[True, True], [True, True]],
]

NOT_COMBINATIONS_3_1 = [
    [[False], [False], [False]],
    [[False], [False], [True]],
    [[False], [True], [True]],
    [[True], [True], [True]],
]

NOT_COMBINATIONS_3_2 = [
    [[False, False], [False, False], [False, False]],
    [[False, False], [False, False], [False, True]],
    [[False, False], [False, False], [True, False]],
    [[False, False], [False, False], [True, True]],
    [[False, False], [False, True], [False, True]],
    [[False, False], [False, True], [True, False]],
    [[False, False], [False, True], [True, True]],
    [[False, False], [True, False], [True, False]],
    [[False, False], [True, False], [True, True]],
    [[False, False], [True, True], [True, True]],
    [[False, True], [False, True], [False, True]],
    [[False, True], [False, True], [True, False]],
    [[False, True], [False, True], [True, True]],
    [[False, True], [True, False], [True, False]],
    [[False, True], [True, False], [True, True]],
    [[False, True], [True, True], [True, True]],
    [[True, False], [True, False], [True, False]],
    [[True, False], [True, False], [True, True]],
    [[True, False], [True, True], [True, True]],
    [[True, True], [True, True], [True, True]],
]


def generate_tasks_1(predicates):
    preds = predicates.copy()
    # preds.append(online)
    clauses = []
    for pred in preds:
        clauses.append(Clause([pred], [], [False]))
        clauses.append(Clause([pred], [], [True]))

    return clauses


def generate_tasks_2(predicates, conjunctors):
    preds = predicates.copy()
    # preds.append(online)
    clauses = []
    for i, pred in enumerate(preds):
        for pred2 in preds[i + 1 :]:
            for conj in conjunctors:

                clauses.append(Clause([pred, pred2], [conj], [False, False]))
                clauses.append(Clause([pred, pred2], [conj], [True, False]))
                clauses.append(Clause([pred, pred2], [conj], [False, True]))
                clauses.append(Clause([pred, pred2], [conj], [True, True]))

    print(len(clauses))
    return clauses


def generate_tasks_for_two_pairs_1(predicates, conjunctors, consider_second_pair=False):
    clauses = []

    for pred in predicates:

        first_pair_predicates = [pred]
        first_pair_conjunctors = []

        if consider_second_pair:
            for pred in predicates:

                second_pair_predicates = [pred]
                second_pair_conjunctors = []

                clauses.append(
                    ClauseForPairs(
                        2,
                        [first_pair_predicates, second_pair_predicates],
                        [first_pair_conjunctors, second_pair_conjunctors],
                        [[False], [False]],
                    )
                )
                clauses.append(
                    ClauseForPairs(
                        2,
                        [first_pair_predicates, second_pair_predicates],
                        [first_pair_conjunctors, second_pair_conjunctors],
                        [[False], [True]],
                    )
                )
                clauses.append(
                    ClauseForPairs(
                        2,
                        [first_pair_predicates, second_pair_predicates],
                        [first_pair_conjunctors, second_pair_conjunctors],
                        [[True], [False]],
                    )
                )
                clauses.append(
                    ClauseForPairs(
                        2,
                        [first_pair_predicates, second_pair_predicates],
                        [first_pair_conjunctors, second_pair_conjunctors],
                        [[True], [True]],
                    )
                )

        else:
            second_pair_predicates = []
            second_pair_conjunctors = []

            clauses.append(
                ClauseForPairs(
                    2,
                    [first_pair_predicates, second_pair_predicates],
                    [first_pair_conjunctors, second_pair_conjunctors],
                    [[False], []],
                )
            )
            clauses.append(
                ClauseForPairs(
                    2,
                    [first_pair_predicates, second_pair_predicates],
                    [first_pair_conjunctors, second_pair_conjunctors],
                    [[True], []],
                )
            )

    print(len(clauses))
    return clauses


def generate_tasks_for_two_pairs_2(predicates, conjunctors, consider_second_pair=False):
    clauses = []

    for i, pred in enumerate(predicates):
        for pred2 in predicates[i + 1 :]:
            for conj in conjunctors:

                first_pair_predicates = [pred, pred2]
                first_pair_conjunctors = [conj]

                if consider_second_pair:
                    for j, pred3 in enumerate(predicates):
                        for pred4 in predicates[j + 1 :]:
                            for conj in conjunctors:

                                second_pair_predicates = [pred3, pred4]
                                second_pair_conjunctors = [conj]

                                for not_values in NOT_COMBINATIONS_2_2:
                                    clauses.append(
                                        ClauseForPairs(
                                            2,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                            ],
                                            not_values,
                                        )
                                    )

                else:
                    second_pair_predicates = []
                    second_pair_conjunctors = []

                    clauses.append(
                        ClauseForPairs(
                            2,
                            [first_pair_predicates, second_pair_predicates],
                            [first_pair_conjunctors, second_pair_conjunctors],
                            [[False, False], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            2,
                            [first_pair_predicates, second_pair_predicates],
                            [first_pair_conjunctors, second_pair_conjunctors],
                            [[True, False], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            2,
                            [first_pair_predicates, second_pair_predicates],
                            [first_pair_conjunctors, second_pair_conjunctors],
                            [[False, True], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            2,
                            [first_pair_predicates, second_pair_predicates],
                            [first_pair_conjunctors, second_pair_conjunctors],
                            [[True, True], []],
                        )
                    )

    print(len(clauses))
    return clauses


def generate_tasks_for_three_pairs_1(
    predicates, conjunctors, consider_second_pair=False, consider_third_pair=False
):
    clauses = []

    for pred in predicates:

        first_pair_predicates = [pred]
        first_pair_conjunctors = []

        if consider_second_pair:
            for pred2 in predicates:

                second_pair_predicates = [pred2]
                second_pair_conjunctors = []

                if consider_third_pair:
                    for pred3 in predicates:
                        third_pair_predicates = [pred3]
                        third_pair_conjunctors = []

                        for not_values in NOT_COMBINATIONS_3_1:
                            clauses.append(
                                ClauseForPairs(
                                    3,
                                    [
                                        first_pair_predicates,
                                        second_pair_predicates,
                                        third_pair_predicates,
                                    ],
                                    [
                                        first_pair_conjunctors,
                                        second_pair_conjunctors,
                                        third_pair_conjunctors,
                                    ],
                                    not_values,
                                )
                            )

                else:
                    third_pair_predicates = []
                    third_pair_conjunctors = []
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[False], [False], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[False], [False], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[False], [True], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[False], [True], []],
                        )
                    )

        else:
            second_pair_predicates = []
            second_pair_conjunctors = []
            third_pair_predicates = []
            third_pair_conjunctors = []

            clauses.append(
                ClauseForPairs(
                    3,
                    [
                        first_pair_predicates,
                        second_pair_predicates,
                        third_pair_predicates,
                    ],
                    [
                        first_pair_conjunctors,
                        second_pair_conjunctors,
                        third_pair_conjunctors,
                    ],
                    [[False], [], []],
                )
            )
            clauses.append(
                ClauseForPairs(
                    3,
                    [
                        first_pair_predicates,
                        second_pair_predicates,
                        third_pair_predicates,
                    ],
                    [
                        first_pair_conjunctors,
                        second_pair_conjunctors,
                        third_pair_conjunctors,
                    ],
                    [[True], [], []],
                )
            )

    print(len(clauses))
    return clauses


def generate_tasks_for_three_pairs_2(
    predicates, conjunctors, consider_second_pair=False, consider_third_pair=False
):
    clauses = []

    for i, pred in enumerate(predicates):
        for pred2 in predicates[i + 1 :]:
            for conj in conjunctors:

                first_pair_predicates = [pred, pred2]
                first_pair_conjunctors = [conj]

                if consider_second_pair:
                    for j, pred3 in enumerate(predicates):
                        for pred4 in predicates[j + 1 :]:
                            for conj in conjunctors:

                                second_pair_predicates = [pred3, pred4]
                                second_pair_conjunctors = [conj]

                                if consider_third_pair:
                                    for k, pred5 in enumerate(predicates):
                                        for pred6 in predicates[k + 1 :]:
                                            for conj in conjunctors:

                                                third_pair_predicates = [pred5, pred6]
                                                third_pair_conjunctors = [conj]

                                                for not_values in NOT_COMBINATIONS_3_2:
                                                    clauses.append(
                                                        ClauseForPairs(
                                                            3,
                                                            [
                                                                first_pair_predicates,
                                                                second_pair_predicates,
                                                                third_pair_predicates,
                                                            ],
                                                            [
                                                                first_pair_conjunctors,
                                                                second_pair_conjunctors,
                                                                third_pair_conjunctors,
                                                            ],
                                                            not_values,
                                                        )
                                                    )

                                else:

                                    third_pair_predicates = [[]]
                                    third_pair_conjunctors = [[]]

                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[False, False], [False, False], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[False, True], [False, False], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[False, True], [False, True], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[False, True], [True, True], []],
                                        )
                                    )

                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[True, False], [False, False], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[True, False], [False, True], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[True, False], [True, False], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[True, False], [True, True], []],
                                        )
                                    )

                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[True, True], [True, True], []],
                                        )
                                    )
                                    clauses.append(
                                        ClauseForPairs(
                                            3,
                                            [
                                                first_pair_predicates,
                                                second_pair_predicates,
                                                [],
                                            ],
                                            [
                                                first_pair_conjunctors,
                                                second_pair_conjunctors,
                                                [],
                                            ],
                                            [[True, True], [False, False], []],
                                        )
                                    )

                else:
                    second_pair_predicates = []
                    second_pair_conjunctors = []
                    third_pair_predicates = []
                    third_pair_conjunctors = []

                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[False, False], [], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[True, False], [], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[False, True], [], []],
                        )
                    )
                    clauses.append(
                        ClauseForPairs(
                            3,
                            [
                                first_pair_predicates,
                                second_pair_predicates,
                                third_pair_predicates,
                            ],
                            [
                                first_pair_conjunctors,
                                second_pair_conjunctors,
                                third_pair_conjunctors,
                            ],
                            [[True, True], [], []],
                        )
                    )

    print(len(clauses))
    return clauses


def get_objects_from_seq(seq):
    num_objects = seq.count("bbox")
    objects = []
    for i in range(num_objects):
        s = i * 15
        objects.append(
            Object(
                color=seq[s + 8],
                shape=seq[s + 11],
                size=seq[s + 14],
                bbox=seq[s + 2 : s + 6],
            )
        )

    return objects


def generate_clauses():
    predicates = [same_shape, same_color, same_size, one_is_red_triangle]
    conjunctors = ["and", "or"]

    # no pairs
    no_pairs_1 = generate_tasks_1(predicates)
    no_pairs_2 = generate_tasks_2(predicates, conjunctors)

    # two pairs
    two_pairs_1_pair_1 = generate_tasks_for_two_pairs_1(
        predicates, conjunctors, consider_second_pair=False
    )
    two_pairs_2_pair_1 = generate_tasks_for_two_pairs_1(
        predicates, conjunctors, consider_second_pair=True
    )

    two_pairs_1_pair_2 = generate_tasks_for_two_pairs_2(
        predicates, conjunctors, consider_second_pair=False
    )
    two_pairs_2_pair_2 = generate_tasks_for_two_pairs_2(
        predicates, conjunctors, consider_second_pair=True
    )

    # three pairs
    three_pairs_1_pair_1 = generate_tasks_for_three_pairs_1(
        predicates, conjunctors, consider_second_pair=False, consider_third_pair=False
    )
    three_pairs_2_pair_1 = generate_tasks_for_three_pairs_1(
        predicates, conjunctors, consider_second_pair=True, consider_third_pair=False
    )
    three_pairs_3_pair_1 = generate_tasks_for_three_pairs_1(
        predicates, conjunctors, consider_second_pair=True, consider_third_pair=True
    )

    three_pairs_1_pair_2 = generate_tasks_for_three_pairs_2(
        predicates, conjunctors, consider_second_pair=False, consider_third_pair=False
    )
    three_pairs_2_pair_2 = generate_tasks_for_three_pairs_2(
        predicates, conjunctors, consider_second_pair=True, consider_third_pair=False
    )
    three_pairs_3_pair_2 = generate_tasks_for_three_pairs_2(
        predicates, conjunctors, consider_second_pair=True, consider_third_pair=True
    )

    # sample subset from clauses
    no_pairs_1_samples = no_pairs_1
    no_pairs_2_samples = random.sample(no_pairs_2, k=10)

    two_pairs_1_pair_1_samples = two_pairs_1_pair_1 + [
        ClauseForPairs(2, [[closeby], []], [[], []], [[False], []])
    ]  # TODO: check if is 10
    two_pairs_2_pair_1_samples = random.sample(two_pairs_2_pair_1, k=25)

    two_pairs_1_pair_2_samples = random.sample(two_pairs_1_pair_2, k=25)
    two_pairs_2_pair_2_samples = random.sample(two_pairs_2_pair_2, k=25) + [
        ClauseForPairs(
            2,
            [[same_shape, same_color], [same_shape, same_color]],
            [["and"], ["and"]],
            [[False, False], [False, True]],
        )
    ]  # twoPairs

    three_pairs_1_pair_1_samples = three_pairs_1_pair_1
    three_pairs_2_pair_1_samples = random.sample(three_pairs_2_pair_1, k=25)
    three_pairs_3_pair_1_samples = random.sample(three_pairs_3_pair_1, k=25)

    three_pairs_1_pair_2_samples = random.sample(three_pairs_1_pair_2, k=25) + [
        ClauseForPairs(
            3,
            [[closeby, one_is_red_triangle], [], []],
            [["and"], [], []],
            [[False, False], [], []],
        )
    ]
    three_pairs_2_pair_2_samples = random.sample(three_pairs_2_pair_2, k=25)
    three_pairs_3_pair_2_samples = random.sample(three_pairs_3_pair_2, k=25) + [
        ClauseForPairs(
            3,
            [
                [same_shape, same_color],
                [same_shape, same_color],
                [same_shape, same_color],
            ],
            [["and"], ["and"], ["and"]],
            [[False, False], [False, True], [False, True]],
        )
    ]  # threePairs

    no_pair_clauses = no_pairs_1_samples + no_pairs_2_samples
    pair_clauses = (
        two_pairs_1_pair_1_samples
        + two_pairs_2_pair_1_samples
        + two_pairs_1_pair_2_samples
        + two_pairs_2_pair_2_samples
        + three_pairs_1_pair_1_samples
        + three_pairs_2_pair_1_samples
        + three_pairs_3_pair_1_samples
        + three_pairs_1_pair_2_samples
        + three_pairs_2_pair_2_samples
        + three_pairs_3_pair_2_samples
    )

    for c in no_pair_clauses + pair_clauses:
        print(c)

    print(f"Generated {len(no_pair_clauses + pair_clauses)} clauses.")

    pair_clauses = {
        "4_pair_1_1": two_pairs_1_pair_1,
        "4_pair_1_2": two_pairs_1_pair_2,
        "4_pair_2_1": two_pairs_2_pair_1,
        "4_pair_2_2": two_pairs_2_pair_2,
        "6_pair_1_1": three_pairs_1_pair_1,
        "6_pair_1_2": three_pairs_1_pair_2,
        "6_pair_2_1": three_pairs_2_pair_1,
        "6_pair_2_2": three_pairs_2_pair_2,
        "6_pair_3_1": three_pairs_3_pair_1,
        "6_pair_3_2": three_pairs_3_pair_2,
    }
    return no_pairs_1, no_pairs_2, pair_clauses


if __name__ == "__main__":

    generate_clauses()
