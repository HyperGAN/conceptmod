"""Authored phrases under examples/dsl match the live parser and the docs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from conceptmod.dsl import parse_phrase
from conceptmod.game import game

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "examples" / "dsl" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsl_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_dict(rule) -> dict:
    return {
        "op": rule.op,
        "a": rule.a,
        "b": rule.b,
        "alpha": rule.alpha,
        "options": dict(rule.options),
    }


PHRASES = ("bipolar", "keep_erase", "remap", "replace_macro")


def test_authored_phrases_match_the_parser_and_the_docs():
    for name in PHRASES:
        module = _load(name)
        rules = parse_phrase(module.PHRASE)
        assert [_as_dict(rule) for rule in rules] == module.EXPECTED
        doc = (ROOT / "docs" / "dsl" / "examples" / f"{name}.md").read_text()
        assert module.PHRASE in doc


def test_locked_game_example_scores_pass():
    module = _load("locked_game")
    doc = (ROOT / "docs" / "dsl" / "examples" / "locked_game.md").read_text()
    assert "red=blue" in doc
    assert "Music" in doc
    match = game(module.SOURCE)
    assert [rule.op for rule in match.rules] == module.EXPECTED_OPS
    row = match.score()
    assert row["verdict"] == "PASS"
    assert row["mismatches"] == []
    assert "Music" in row["note"]
