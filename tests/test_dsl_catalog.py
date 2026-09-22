"""Catalog pages describe the live parser, including the dead ops."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "docs" / "dsl" / "operators"

PAGES = {
    "exaggerate.md": ("c++", "guidance", "3", "right"),
    "erase.md": ("c--", "guidance", "0", "right"),
    "write.md": ("a=b", "right"),
    "freeze.md": ("a#b", "right"),
    "orthogonal.md": ("a%b", "right"),
    "pixel.md": ("a^b", "twoaxis", "dead"),
    "replace.md": ("a~b", "macro", "recipe"),
    "reward.md": ("not implemented", ";"),
}


def test_operator_pages_exist_and_name_the_verdict():
    index = (OPS / "index.md").read_text().lower()
    for name, needles in PAGES.items():
        text = (OPS / name).read_text().lower()
        assert name.removesuffix(".md") in index
        for needle in needles:
            assert needle in text, f"{name} missing {needle!r}"
    assert "ignored" in index or "deprecated" in index


def test_scoreboard_is_still_the_velocity_truth_table():
    scoreboard = (ROOT / "docs" / "dsl.md").read_text()
    assert "Live operators" in scoreboard
    assert "a~b" in scoreboard
    assert "docs/dsl/" in scoreboard or "dsl/README" in scoreboard


def test_adding_an_operator_does_not_fork_locked_shared():
    text = (ROOT / "docs" / "dsl" / "ADDING_OPERATORS.md").read_text()
    assert "conceptmod/dsl.py" in text
    assert "conceptmod/ops.py" in text
    assert "locked_adv_defaults" in text
    assert "docs/dsl.md" in text


def test_spec_splits_dsl_from_formulation_toys():
    text = (ROOT / "docs" / "dsl" / "SPEC.md").read_text()
    assert "locked_adv_defaults" in text
    assert "rule_loss" in text
    assert "not implemented" in text.lower() or "Not implemented" in text
    assert "Music" in text


def test_readme_teaches_parse_then_game():
    readme = (ROOT / "README.md").read_text()
    assert "loss_game" in readme
    assert "parse_phrase" in readme
    assert "docs/dsl/README.md" in readme
