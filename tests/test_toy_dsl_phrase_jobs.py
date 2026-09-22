"""Phrase-DSL job honesty: documented phrases PASS, wrong synonyms FAIL.

Geometric ``right`` comes from the 2-D suite. Formulation PASS is that
the phrase is the documented job. CPU only. Not a Music, Anima, or
Supra GPU transfer.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conceptmod import dsl, ops
from conceptmod.toys import locked_adv_defaults
from conceptmod.toys.cover_leftover import SAME_DIR_MAX, U_KEPT_MIN
from conceptmod.toys.dsl_phrase_jobs import (
    BIPOLAR_SCALE,
    DOCUMENTED_NAMES,
    REQUIRED_NEGATIVES,
    HonestyError,
    canon,
    claim_phrase_pass,
    invented_glyph,
    run_board,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def board():
    torch.set_num_threads(1)
    payload = run_board()
    payload["by_name"] = {row.name: row for row in payload["rows"]}
    return payload


def test_documented_phrases_match_docs_dsl():
    text = (ROOT / "docs" / "dsl.md").read_text()
    phrases = canon()
    for name in ("neutralize", "bipolar", "remap", "mix", "isolate"):
        assert phrases[name] in text, name
    # docs/dsl.md writes keep+erase as ``c--|k#k``. The 2-D fixture spells
    # that ``red--|stripe#stripe`` (same composition the erase suite trains).
    assert "k#k" in text
    assert "Keep+erase" in text
    assert "red%red stripe" in text
    assert "c!!" in text
    assert phrases["neutralize"] == "red--"
    assert phrases["bipolar"] == "red++"
    assert phrases["remap"] == "red=blue"
    assert phrases["keep_erase"] == "red--|stripe#stripe"
    assert phrases["mix"] == "red=red stripe"
    assert phrases["isolate"] == "red stripe=stripe"


def test_plus_bang_and_slash_are_not_parser_ops():
    assert invented_glyph("red++") is None
    assert invented_glyph("red+stripe") == "+"
    assert invented_glyph("red!!") == "!!"
    assert invented_glyph("red/") == "/"
    for phrase in ("red+stripe", "red!!", "red/"):
        with pytest.raises(dsl.DSLError):
            dsl.parse_phrase(phrase)
    assert not hasattr(dsl, "PLUS")


def test_package_exports_the_board():
    import conceptmod.toys as toys

    assert toys.run_dsl_phrase_board is run_board
    assert toys.dsl_phrase_jobs.TOY_ID == "dsl_phrase_jobs"


def test_docstring_keeps_cpu_toy_culture():
    doc = (ROOT / "conceptmod" / "toys" / "dsl_phrase_jobs.py").read_text()
    assert "A PASS is a CPU toy" in doc
    assert "not a Music, Anima, or Supra GPU transfer" in doc
    assert "Geometric ``right``" in doc
    text = (ROOT / "docs" / "formulation-toys.md").read_text()
    assert "conceptmod/toys/dsl_phrase_jobs.py" in text


def test_neutralize_pass_is_bare_erase_to_origin(board):
    row = board["by_name"]["neutralize"]
    assert row.passed, row.reasons
    assert row.geometric == "right"
    assert row.reasons == ()
    assert row.ops == ("erase",)
    assert row.phrase == "red--"
    assert abs(row.color_on_red) < 0.2
    assert row.write_cosine < 0.3
    assert row.stripe_hold > U_KEPT_MIN
    assert abs(row.pattern_on_red) < SAME_DIR_MAX
    assert "Neutralize" in dsl.describe_phrase(row.phrase)
    assert dsl.parse_phrase(row.phrase)[0].options.get("guidance") is None
    assert ops.OpDefaults.erase_guidance == 0.0


def test_bipolar_exaggerate_holds_stripe(board):
    row = board["by_name"]["bipolar"]
    assert row.passed, row.reasons
    assert row.geometric == "right"
    assert row.phrase == "red++"
    assert row.ops == ("exaggerate",)
    assert row.color_on_red > BIPOLAR_SCALE
    assert row.blue_x < -BIPOLAR_SCALE
    assert abs(row.blue_y) < SAME_DIR_MAX
    assert row.stripe_hold > U_KEPT_MIN


def test_remap_write_passes(board):
    row = board["by_name"]["remap"]
    assert row.passed, row.reasons
    assert row.geometric == "right"
    assert row.phrase == "red=blue"
    assert row.ops == ("write",)
    assert row.write_cosine > 0.7
    assert row.stripe_hold > U_KEPT_MIN
    assert abs(row.pattern_on_red) < SAME_DIR_MAX


def test_keep_erase_compose_passes(board):
    row = board["by_name"]["keep_erase"]
    assert row.passed, row.reasons
    assert row.geometric == "right"
    assert row.phrase == "red--|stripe#stripe"
    assert row.ops == ("erase", "freeze")
    assert row.color_on_red < 0.2
    assert row.stripe_hold > U_KEPT_MIN
    rules = dsl.parse_phrase(row.phrase)
    assert (rules[1].a, rules[1].b) == ("stripe", "stripe")


def test_mix_is_the_write_recipe_not_a_plus_op(board):
    row = board["by_name"]["mix"]
    assert row.passed, row.reasons
    assert row.geometric == "right"
    assert row.phrase == "red=red stripe"
    assert "+" not in row.phrase
    assert row.ops == ("write",)
    assert row.color_on_red > 0.7
    assert row.pattern_on_red > 0.7
    assert row.stripe_hold > U_KEPT_MIN


def test_isolate_is_the_write_recipe(board):
    row = board["by_name"]["isolate"]
    assert row.passed, row.reasons
    assert row.geometric == "right"
    assert row.phrase == "red stripe=stripe"
    assert row.ops == ("write",)
    assert abs(row.mix_x) < SAME_DIR_MAX
    assert row.mix_y > 0.7


def test_invented_synonyms_fail_honesty(board):
    for name, glyph in (
        ("plus_synonym", "+"),
        ("bang_synonym", "!!"),
        ("slash_synonym", "/"),
    ):
        row = board["by_name"][name]
        assert row.passed is False
        assert row.refused is False
        assert row.reasons == ("honesty",)
        assert row.geometric is None
        assert glyph in row.phrase.replace("++", "")
        assert glyph in row.detail


def test_percent_is_not_isolate(board):
    """``%`` can trip the isolate landing check by inflating the perpendicular."""
    row = board["by_name"]["percent_isolate"]
    isolate = board["by_name"]["isolate"]
    assert row.passed is False
    assert row.reasons == ("percent_isolate",)
    assert row.ops == ("orthogonal",)
    assert row.phrase == "red%red stripe"
    assert row.geometric == "right"
    assert row.phrase != canon()["isolate"]
    assert abs(row.mix_x) < SAME_DIR_MAX
    assert row.mix_y > 4.0
    assert isolate.mix_y == pytest.approx(1.0, abs=0.05)


def test_bare_erase_claimed_as_keep_erase_fails_retain(board):
    row = board["by_name"]["keep_erase_bare"]
    neutralize = board["by_name"]["neutralize"]
    assert row.passed is False
    assert row.reasons == ("retain",)
    assert row.phrase == "red--"
    assert row.ops == ("erase",)
    assert "freeze" not in row.ops
    # Bare erase is geometrically neutralize. That must not become keep+erase.
    assert row.geometric == "right"
    assert row.stripe_hold == pytest.approx(neutralize.stripe_hold)
    assert row.stripe_hold > U_KEPT_MIN


def test_pixel_and_reward_are_refused(board):
    pixel = board["by_name"]["pixel_caret"]
    reward = board["by_name"]["reward_semi"]
    assert pixel.passed is False and pixel.refused
    assert reward.passed is False and reward.refused
    assert pixel.reasons == ("dead_op",)
    assert reward.reasons == ("dead_op",)
    assert pixel.geometric is None and reward.geometric is None
    assert "render" in pixel.detail
    assert "not implemented" in reward.detail
    assert pixel.ops == ("pixel",)
    assert reward.ops == ("reward",)


def test_board_pass_requires_the_fail_arms(board):
    verdict = board["verdict"]
    assert verdict["verdict"] == "PASS"
    assert verdict["cpu_toy"] is True
    assert verdict["music_gpu_transfer"] is False
    assert verdict["anima_gpu_transfer"] is False
    assert verdict["supra_gpu_transfer"] is False
    assert verdict["documented"] == list(DOCUMENTED_NAMES)
    assert verdict["negatives"] == list(REQUIRED_NEGATIVES)
    assert verdict["cover_posture"] == "demo_1.5"
    for row in board["rows"]:
        assert row.geometric != "PASS"
        assert row.toy == "dsl_phrase_jobs"
    stamp = locked_adv_defaults()
    assert stamp["gan_mode"] == "rp"
    assert stamp["loss_type"] == "logistic"
    assert stamp["reg_arm"] == "b_cap"
    assert stamp["reg_coeff"] == 1.0
    assert stamp["reg_kappa"] == 1.0
    assert stamp["reg_norm"] == "l2"
    assert stamp["fm_weight"] == 0.0
    assert stamp["cover_weight"] == 1.5
    assert stamp["n_particles"] == 12
    for row in board["rows"]:
        assert row.adv == stamp
        assert row.adv_ok
    assert torch.zeros(1).device.type == "cpu"


def test_empty_negatives_raise(board):
    documented = [row for row in board["rows"] if row.role == "documented"]
    with pytest.raises(HonestyError, match="declared negative"):
        claim_phrase_pass(documented=documented, negatives=[])


def test_a_negative_that_passes_is_refused(board):
    documented = [row for row in board["rows"] if row.role == "documented"]
    negatives = []
    for row in board["rows"]:
        if row.role != "negative":
            continue
        if row.name == "plus_synonym":
            negatives.append(replace(row, passed=True, reasons=()))
        else:
            negatives.append(row)
    with pytest.raises(HonestyError, match="plus_synonym"):
        claim_phrase_pass(documented=documented, negatives=negatives)
