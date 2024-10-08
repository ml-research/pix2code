"""
EC codebase Python library (AKA the "frontend")

Module mapping details:

TODO: remove module mapping code when backwards-compatibility is no longer required.

The below module mapping is required for backwards-compatibility with old pickle files
generated from before the EC codebase refactor. New files added to the codebase do not
need to be added to the mapping, but if the existing modules are moved, then this the
mapping needs to be updated to reflect the move or rename.

The mapping uses the following pattern:

    sys.modules[<old module path>] = <new module reference>

This is because the previous structure of the codebase was completely flat, and when refactoring
to a hierarchical files, loading previous pickle files no longer works properly. It is important
to retain the ability to read old pickle files generated from official experiments. As a workaround,
the old module paths are included below. A preferable alternative would be to export program state
into JSON files instead of pickle files to avoid issues where the underlying classes change, so that
could be a future improvement to this project. Until then, we use the module mapping workaround.

For more info, see this StackOverflow answer: https://stackoverflow.com/a/2121918/2573242
"""
import sys

from dreamcoder import (differentiation, dreamcoder, enumeration,
                        fragmentGrammar, fragmentUtilities, frontier, grammar,
                        likelihoodModel, primitiveGraph, program)
try:
    from dreamcoder import recognition
except:
    print(
        "Failure loading recognition - only acceptable if using pypy ", file=sys.stderr
    )
from dreamcoder import task, taskBatcher, type, utilities, vs
from dreamcoder.domains.arithmetic import arithmeticPrimitives
from dreamcoder.domains.list import listPrimitives
from dreamcoder.domains.list import main as list_main
from dreamcoder.domains.list import makeListTasks
# from dreamcoder.domains.regex import main as regex_main
from dreamcoder.domains.logo import logoPrimitives
from dreamcoder.domains.logo import main as logo_main
from dreamcoder.domains.logo import makeLogoTasks
from dreamcoder.domains.misc import (RobustFillPrimitives, algolispPrimitives,
                                     deepcoderPrimitives, napsPrimitives)
from dreamcoder.domains.regex import (groundtruthRegexes, makeRegexTasks,
                                      regexPrimitives)
from dreamcoder.domains.text import main as text_main
from dreamcoder.domains.text import makeTextTasks, textPrimitives
from dreamcoder.domains.tower import main as tower_main
from dreamcoder.domains.tower import (makeTowerTasks, tower_common,
                                      towerPrimitives)
from dreamcoder.domains.relation import main as relation_main
from dreamcoder.domains.relation import (make_relation_tasks, relation_primitives, generate_tasks)


sys.modules["differentiation"] = differentiation
sys.modules["ec"] = dreamcoder
sys.modules["enumeration"] = enumeration
sys.modules["fragmentGrammar"] = fragmentGrammar
sys.modules["fragmentUtilities"] = fragmentUtilities
sys.modules["frontier"] = frontier
sys.modules["grammar"] = grammar
sys.modules["likelihoodModel"] = likelihoodModel
sys.modules["program"] = program
try:
    sys.modules["recognition"] = recognition
except:
    pass
sys.modules["task"] = task
sys.modules["taskBatcher"] = taskBatcher
sys.modules["type"] = type
sys.modules["utilities"] = utilities
sys.modules["vs"] = vs
sys.modules["algolispPrimitives"] = algolispPrimitives
sys.modules["RobustFillPrimitives"] = RobustFillPrimitives
sys.modules["napsPrimitives"] = napsPrimitives
sys.modules["makeTowerTasks"] = makeTowerTasks
sys.modules["towerPrimitives"] = towerPrimitives
sys.modules["tower_common"] = tower_common
# sys.modules['tower'] = tower_main
sys.modules["groundtruthRegexes"] = groundtruthRegexes
sys.modules["regexPrimitives"] = regexPrimitives
sys.modules["makeRegexTasks"] = makeRegexTasks
# sys.modules['regexes'] = regex_main
sys.modules["deepcoderPrimitives"] = deepcoderPrimitives
sys.modules["logoPrimitives"] = logoPrimitives
sys.modules["makeLogoTasks"] = makeLogoTasks
# sys.modules['logo'] = logo_main
sys.modules["listPrimitives"] = listPrimitives
sys.modules["makeListTasks"] = makeListTasks
# sys.modules['list'] = list_main
sys.modules["arithmeticPrimitives"] = arithmeticPrimitives
sys.modules["textPrimitives"] = textPrimitives
sys.modules["makeTextTasks"] = makeTextTasks
# sys.modules['text'] = text_main
sys.modules["primitiveGraph"] = primitiveGraph
# make relation tasks
sys.modules["make_relation_tasks"] = make_relation_tasks
sys.modules["relation_primitives"] = relation_primitives
