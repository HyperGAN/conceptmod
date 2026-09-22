"""Remap one concept onto another. A single write, not a global boost."""

PHRASE = "red=blue"

EXPECTED = [
    {"op": "write", "a": "red", "b": "blue", "alpha": 1.0, "options": {}},
]
