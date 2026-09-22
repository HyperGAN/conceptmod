"""Keep+erase is composition, not a new token.

``red--`` neutralizes red. ``stripe#stripe`` pins the pattern axis to
the frozen model.
"""

PHRASE = "red--|stripe#stripe"

EXPECTED = [
    {"op": "erase", "a": "red", "b": "", "alpha": 1.0, "options": {}},
    {"op": "freeze", "a": "stripe", "b": "stripe", "alpha": 1.0, "options": {}},
]
