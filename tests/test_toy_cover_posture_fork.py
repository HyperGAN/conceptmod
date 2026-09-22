"""Cover posture fork: demo 1.5 and Music 1.0 are named passes, not aliases.

Swapping the pin under the other name without a drift report FAILs.
Stranger pairing, FM-on, and a thinned kappa cap FAIL on the locked shape.
This is a CPU toy, not a Music or Anima GPU transfer.
"""

from dataclasses import replace

import pytest

from conceptmod.toys.cover_leftover import LOCKED_COVER
from conceptmod.toys.cover_posture_fork import (
    DEMO_CLAIM,
    DEMO_COVER,
    MUSIC_CLAIM,
    MUSIC_COVER,
    MUSIC_REPORT,
    evaluate_claim,
    pins_match,
    run_board,
    score_claimed_posture,
)
from conceptmod.toys.locked_shared_floor import LOCKED, run_floor, score_floor
from conceptmod.toys.particle_posture import MUSIC_COVER as POSTURE_MUSIC_COVER


@pytest.fixture(scope="module")
def board():
    return {row["arm"]: row for row in run_board(log=False)}


def test_pins_are_the_reviewed_covers_and_not_substrings():
    """``cover_weight=1`` is a prefix of ``cover_weight=1.5`` and must not match."""
    assert DEMO_COVER == LOCKED_COVER == 1.5
    assert MUSIC_COVER == POSTURE_MUSIC_COVER == 1.0
    assert "cover_weight=1" in "cover_weight=1.5"
    assert pins_match(DEMO_CLAIM, 1.5, "demo")
    assert pins_match(MUSIC_CLAIM, 1.0, "music")
    assert not pins_match(DEMO_CLAIM, 1.0, "demo")
    assert not pins_match(MUSIC_CLAIM, 1.5, "music")
    assert not pins_match(MUSIC_CLAIM, 1.5, "demo")
    assert not pins_match(DEMO_CLAIM, 1.0, "music")


def test_locked_shared_shape_passes_on_the_posture_it_claims(board):
    demo = board[DEMO_CLAIM]
    music = board[MUSIC_CLAIM]
    assert demo["verdict"] == "PASS", demo["mismatches"]
    assert music["verdict"] == "PASS", music["mismatches"]
    for key in ("gan_mode", "loss_type", "fm_weight", "n_particles", "reg_class", "pairing"):
        assert demo[key] == music[key]
    assert demo["gan_mode"] == "rp" and demo["loss_type"] == "logistic"
    assert demo["fm_weight"] == 0.0
    assert demo["n_particles"] == 12
    assert demo["reg_class"] == "GradientPenalty"
    assert demo["kappa_probe_abs_err"] < 1e-5
    assert music["kappa_probe_abs_err"] < 1e-5
    assert demo["cover_weight"] == 1.5 and demo["cover_posture"] == "demo"
    assert demo["reported_drift"] == {}
    assert music["cover_weight"] == 1.0 and music["cover_posture"] == "music"
    assert music["reported_drift"] == MUSIC_REPORT
    assert demo["g_abs_err"] < 1e-5 and music["g_abs_err"] < 1e-5
    assert demo["alias_gap"] > 1e-3 and music["alias_gap"] > 1e-3
    assert demo["music_gpu_transfer"] is False and music["music_gpu_transfer"] is False
    assert demo["anima_gpu_transfer"] is False and music["anima_gpu_transfer"] is False
    assert demo["cpu_toy"] is True and music["cpu_toy"] is True


def test_demo_claim_matches_the_locked_floor_stamp():
    floor = score_floor(run_floor(LOCKED))
    assert floor["verdict"] == "PASS", floor["mismatches"]
    assert floor["cover_weight"] == DEMO_COVER
    assert floor["cover_posture"] == "demo"


def test_music_posture_still_fails_the_demo_floor_stamp():
    """The floor claims demo. Music 1.0 is a PASS only on the fork toy."""
    cfg = replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music")
    floor = score_floor(run_floor(cfg))
    assert floor["verdict"] == "FAIL"
    assert any(item.startswith("cover_weight") for item in floor["mismatches"])


def test_mislabeled_swap_fails(board):
    row = board["mislabeled_swap"]
    assert row["verdict"] == "FAIL"
    assert row["claim"] == DEMO_CLAIM
    assert row["cover_weight"] == 1.0
    assert row["cover_posture"] == "demo"
    assert row["reported_drift"] == {}
    assert row["g_abs_err"] > 1e-3
    text = " ".join(row["mismatches"])
    assert "mislabeled swap" in text
    assert "unreported cover drift" in text


def test_music_without_a_drift_report_is_a_silent_alias():
    row = evaluate_claim(
        replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music"),
        claim=MUSIC_CLAIM,
        reported_drift=None,
        arm="music_unreported",
    )
    assert row["verdict"] == "FAIL"
    text = " ".join(row["mismatches"])
    assert "unreported cover drift" in text
    assert "silent alias" in text
    assert "mislabeled swap" not in text


def test_stranger_fm_and_thinned_kappa_fail(board):
    stranger = board["stranger"]
    assert stranger["verdict"] == "FAIL"
    assert any(item.startswith("pairing") for item in stranger["mismatches"])
    assert stranger["adv_abs_err"] > 1e-5

    fm = board["fm_on"]
    assert fm["verdict"] == "FAIL"
    assert any(item.startswith("fm_weight") for item in fm["mismatches"])
    assert fm["g_abs_err"] > 1e-5

    thin = board["thinned_kappa"]
    assert thin["verdict"] == "FAIL"
    assert thin["reg_class"] == "ThinnedKappaCap"
    assert thin["kappa_probe_abs_err"] > 1e-3
    assert thin["cap_abs_err"] < 1e-5
    assert any("kappa" in item or "GradientPenalty" in item for item in thin["mismatches"])


def test_music_label_does_not_waive_stranger_fm_or_thinned_kappa():
    music_cfg = replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music")
    stranger = evaluate_claim(
        music_cfg,
        claim=MUSIC_CLAIM,
        reported_drift=dict(MUSIC_REPORT),
        pairing="stranger",
        arm="music_stranger",
    )
    assert stranger["verdict"] == "FAIL"
    assert any(item.startswith("pairing") for item in stranger["mismatches"])

    fm = evaluate_claim(
        replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music", fm_weight=0.1),
        claim=MUSIC_CLAIM,
        reported_drift=dict(MUSIC_REPORT),
        arm="music_fm",
    )
    assert fm["verdict"] == "FAIL"
    assert any(item.startswith("fm_weight") for item in fm["mismatches"])

    thin = evaluate_claim(
        music_cfg,
        claim=MUSIC_CLAIM,
        reported_drift=dict(MUSIC_REPORT),
        regularizer="thinned",
        arm="music_thinned",
    )
    assert thin["verdict"] == "FAIL"
    assert thin["reg_class"] == "ThinnedKappaCap"


def test_reporting_the_swap_does_not_keep_the_demo_claim():
    """A report can name Music. It cannot leave the claim as demo 1.5."""
    row = evaluate_claim(
        replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music"),
        claim=DEMO_CLAIM,
        reported_drift=dict(MUSIC_REPORT),
        arm="reported_but_still_demo",
    )
    assert row["verdict"] == "FAIL"
    text = " ".join(row["mismatches"])
    assert "mislabeled swap" in text


def test_unknown_claim_is_refused():
    with pytest.raises(ValueError, match="unknown posture"):
        pins_match("cover_2", 2.0, "other")
    obs = run_floor(LOCKED)
    with pytest.raises(ValueError, match="unknown posture"):
        score_claimed_posture(obs, claim="cover_2")
