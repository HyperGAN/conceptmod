"""Replace is a macro. Default λ=0.1 expands before parsing.

On the 2-D fixture this expansion is a recipe, not an independent
geometric win: ``blue++`` and ``red=blue`` oppose each other on a
linear antipodal map. See docs/dsl.md.
"""

PHRASE = "red~blue"

EXPECTED = [
    {"op": "exaggerate", "a": "blue", "b": "", "alpha": 0.2, "options": {}},
    {"op": "write", "a": "red", "b": "blue", "alpha": 0.4, "options": {}},
    {"op": "orthogonal", "a": "blue", "b": "red", "alpha": -0.1, "options": {}},
]
