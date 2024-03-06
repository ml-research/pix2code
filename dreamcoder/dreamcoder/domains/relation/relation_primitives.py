import datetime
import os
import random

# import binutil
from functools import reduce
from dreamcoder.domains.list.listPrimitives import (
    _car,
    _cdr,
    _cons,
    _isEmpty,
    _range,
    _unfold,
    bootstrapTarget,
)
from dreamcoder.domains.relation.make_relation_tasks import make_relation_tasks
from dreamcoder.domains.text.main import (
    ConstantInstantiateVisitor,
    LearnedFeatureExtractor,
)
from dreamcoder.domains.text.makeTextTasks import delimiters, guessConstantStrings
from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import *
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tstr
from dreamcoder.utilities import numberOfCPUs


# Primitives
def _slice(x):
    return lambda y: lambda s: s[x:y]


def _index(n):
    return lambda x: x[n]


def _find(pattern):
    return lambda s: s.index(pattern)


def _split(delimiter):
    return lambda s: s.split(delimiter)


def _join(delimiter):
    return lambda ss: delimiter.join(ss)


def _strip(x):
    return x.strip()


def _eq(x):
    return lambda y: x == y


def _if(c):
    return lambda t: lambda f: t if c else f


def _and(x):
    return lambda y: x and y


def _or(x):
    return lambda y: x or y


def _gt(x):
    return lambda y: x > y


def _not(x):
    return not x


def _addition(x):
    return lambda y: x + y


def _subtraction(x):
    return lambda y: x - y


def _multiplication(x):
    return lambda y: x * y


def _negate(x):
    return -x


def _map(f):
    return lambda l: list(map(f, l))


def _fold(l):
    return lambda x0: lambda f: reduce(lambda a, x: f(x)(a), l[::-1], x0)


def _mod(x):
    return lambda y: x % y


def _max(x):
    return lambda y: max(x, y)


def _min(x):
    return lambda y: min(x, y)


def _count(x):
    return lambda i: x.count(i)


def _forall(pred):
    return lambda l: all(pred(i) for i in l)


def _exists(pred):
    return lambda l: any(pred(i) for i in l)


def _get_attribute(l):
    return lambda i: l[i]


def get_primitives():
    primitives = [
        # Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
        # Primitive("char-eq?", arrow(tcharacter, tcharacter, tboolean), _eq),
        # Primitive("str-eq?", arrow(tstr, tstr, tboolean), _eq),
        # Primitive("STRING", tstr, None),
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        # Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted), # not in first
        # Primitive("*", arrow(tint, tint, tint), _multiplication), # not in first
        # Primitive("negate", arrow(tint, tint), _negate), # not in first
        # Primitive("mod", arrow(tint, tint, tint), _mod), # not in first
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),  # remove in less prim
        Primitive("find", arrow(t0, tlist(t0), tint), _find),
        Primitive("max", arrow(tint, tint, tint), _max),  # remove in only used
        Primitive("min", arrow(tint, tint, tint), _min),  # remove in less prim
    ]

    # base primitives
    primitives = (
        primitives
        + [p for p in bootstrapTarget()]
        + [Primitive(str(j), tint, j) for j in [2000, 2001, 2002, 2003]]
    )

    return primitives


def get_baseline_primitives():
    primitives = [
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("find", arrow(t0, tlist(t0), tint), _find),
        Primitive("max", arrow(tint, tint, tint), _max),
        Primitive("min", arrow(tint, tint, tint), _min),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive(
            "unfold",
            arrow(t0, arrow(t0, tbool), arrow(t0, t1), arrow(t0, t0), tlist(t1)),
            _unfold,
        ),
        Primitive("range", arrow(tint, tlist(tint)), _range),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
    ]

    # base primitives
    primitives = (
        primitives
        + [Primitive(str(j), tint, j) for j in range(10)]
        + [Primitive(str(j), tint, j) for j in [2000, 2001, 2002, 2003]]
    )

    return primitives


def get_kandinsky_primitives():
    primitives = [
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("find", arrow(t0, tlist(t0), tint), _find),
        Primitive("max", arrow(tint, tint, tint), _max),
        Primitive("min", arrow(tint, tint, tint), _min),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        # Primitive("range", arrow(tint, tlist(tint)), _range),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
        Primitive("forall", arrow(arrow(t0, tbool), tlist(t0), tbool), _forall),
        Primitive("exists", arrow(arrow(t0, tbool), tlist(t0), tbool), _exists),
        Primitive("count", arrow(tlist(t0), t0, tint), _count),
    ]

    # base primitives
    primitives = primitives + [Primitive(str(j), tint, j) for j in range(10)]
    # primitives = primitives + [Primitive(str(j), tint, j) for j in [0,1,3,4,5,6,7,8,9,10,11,12,13,14]]

    return primitives


def get_clevr_primitives():
    primitives = [
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("find", arrow(t0, tlist(t0), tint), _find),
        Primitive("max", arrow(tint, tint, tint), _max),
        Primitive("min", arrow(tint, tint, tint), _min),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        # Primitive("range", arrow(tint, tlist(tint)), _range),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
        Primitive("forall", arrow(arrow(t0, tbool), tlist(t0), tbool), _forall),
        Primitive("exists", arrow(arrow(t0, tbool), tlist(t0), tbool), _exists),
        Primitive("count", arrow(tlist(t0), t0, tint), _count),
    ]

    # base primitives
    primitives = primitives + [Primitive(str(j), tint, j) for j in range(10)]
    # primitives = primitives + [Primitive(str(j), tint, j) for j in [0,1,3,4,5,6,7,8,9,10,11,12,13,14]] #

    return primitives


def get_clevr_primitives_unconfounded(confounded=True):
    primitives = [
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("find", arrow(t0, tlist(t0), tint), _find),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("empty", tlist(t0), []),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
        Primitive("forall", arrow(arrow(t0, tbool), tlist(t0), tbool), _forall),
        Primitive("exists", arrow(arrow(t0, tbool), tlist(t0), tbool), _exists),
    ]

    if confounded:
        # base primitives
        primitives = primitives + [
            Primitive(str(j), tint, j)
            for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]
    else:
        primitives = primitives + [
            Primitive(str(j), tint, j)
            for j in [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]
        ]
    return primitives
