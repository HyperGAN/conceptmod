"""2D sheet recipe lifted onto a tilted Field3D plane.

The guarded odd residual passes on the coordinate plane and, after the
tangent pushforward, on the tilt. Pasting those 2D numbers into x, y
without the lift correction fails. Stranger pairing, FM-on, and a
κ-hardcoded b_cap fail while the lifted geometry still holds.

CPU only. Not a Music or Anima transfer.
"""

import math

import pytest
import torch

from particlegan import GradientPenalty

from conceptmod.toys.field_lift import (
    CONTENT,
    LIFT_ANGLE,
    SLIDER,
    LiftArm,
    evaluate,
    guarded_odd,
    lift_correction,
    lifted_frame,
    naive_copy,
    plane_frame,
    pushforward,
    run_board,
)
from conceptmod.toys.locked_shared_floor import ThinnedKappaCap


@pytest.fixture(scope="module")
def board():
    return {row["arm"]: row for row in run_board()}


def test_naive_paste_is_the_plane_recipe():
    """On θ=0 the 2D coordinates already lie in the sheet. The correction is 0."""
    plane = plane_frame()
    torch.testing.assert_close(naive_copy(), pushforward(plane))
    assert float(lift_correction(plane).norm()) == pytest.approx(0.0, abs=1e-12)
    tilted = lifted_frame()
    correction = lift_correction(tilted)
    assert float(correction.norm()) == pytest.approx(math.sqrt(2.0 - math.sqrt(2.0)), rel=1e-12)
    assert float(correction[2]) == pytest.approx(SLIDER * math.sin(LIFT_ANGLE), rel=1e-12)
    assert tilted.angle == pytest.approx(LIFT_ANGLE)


def test_guard_strips_unused_e_and_matches_the_pushforward():
    for frame in (plane_frame(), lifted_frame()):
        teacher = guarded_odd(frame)
        torch.testing.assert_close(teacher, pushforward(frame))
        assert abs(float(teacher @ frame.e)) < 1e-8
        assert float(teacher @ frame.u) == pytest.approx(SLIDER)
        assert float(teacher @ frame.content) == pytest.approx(CONTENT)


def test_locked_plane_and_locked_lift_pass(board):
    plane = board["locked_2d"]
    lifted = board["locked_lift"]
    assert plane["pass"] and plane["reasons"] == ()
    assert lifted["pass"] and lifted["reasons"] == ()
    assert plane["frame"] == "plane" and lifted["frame"] == "tilted"
    for key in ("u_kept", "content_kept", "leak_ratio", "same_dir", "pole_rel_err"):
        assert lifted[key] == pytest.approx(plane[key], abs=1e-12)
    assert lifted["u_kept"] == pytest.approx(1.0, abs=1e-12)
    assert lifted["content_kept"] == pytest.approx(1.0, abs=1e-12)
    assert lifted["leak_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["same_dir"] == pytest.approx(0.0, abs=1e-8)
    assert lifted["pole_rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["fm_weight"] == 0.0 and lifted["cover_weight"] == 1.5
    assert lifted["n_particles"] == 12 and lifted["pairing"] == "pair"
    assert lifted["gan_mode"] == "rp" and lifted["reg_class"] == "GradientPenalty"
    # Matched clouds sit on the teacher, so Rp logistic is softplus(0).
    assert plane["d_loss"] == pytest.approx(math.log(2.0), abs=1e-12)
    assert lifted["d_loss"] == pytest.approx(math.log(2.0), abs=1e-12)
    assert lifted["adv_abs_err"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["g_abs_err"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["cap_abs_err"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["kappa_probe_abs_err"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["fm_term"] == pytest.approx(0.0, abs=1e-12)
    assert lifted["critic_frozen"]


def test_naive_copy_fails_the_lift(board):
    row = board["naive_copy"]
    assert not row["pass"]
    assert row["reasons"] == ("naive_lift", "undershoot")
    assert row["u_kept"] == pytest.approx(math.cos(LIFT_ANGLE), rel=1e-12)
    assert row["u_kept"] < 0.85
    assert row["content_kept"] == pytest.approx(1.0, abs=1e-12)
    assert row["leak_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert row["pole_rel_err"] == pytest.approx(0.504394, rel=1e-5)
    assert row["pole_rel_err"] > 0.20
    assert row["g_abs_err"] == pytest.approx(0.4393398, rel=1e-5)
    assert row["same_dir"] == pytest.approx(0.0, abs=1e-8)
    assert row["fm_weight"] == 0.0 and row["pairing"] == "pair"
    assert row["kappa_probe_abs_err"] == pytest.approx(0.0, abs=1e-12)


def test_stranger_fm_and_thinned_kappa_fail_with_the_lift_held(board):
    stranger = board["stranger_pairing"]
    fm = board["fm_on"]
    thin = board["thinned_kappa"]
    for row in (stranger, fm, thin):
        assert not row["pass"]
        assert row["u_kept"] == pytest.approx(1.0, abs=1e-12)
        assert row["leak_ratio"] == pytest.approx(0.0, abs=1e-12)
        assert row["pole_rel_err"] == pytest.approx(0.0, abs=1e-12)
        assert "naive_lift" not in row["reasons"]
    assert stranger["reasons"] == ("stranger_pairing",)
    assert stranger["adv_abs_err"] == pytest.approx(4.922802e-4, rel=1e-5)
    assert stranger["adv_abs_err"] > 1e-4
    assert fm["reasons"] == ("fm_on",)
    assert fm["fm_weight"] == pytest.approx(0.1)
    assert fm["fm_term"] == pytest.approx(6.360330e-3, rel=1e-5)
    assert fm["g_abs_err"] == pytest.approx(fm["fm_term"], rel=1e-6)
    assert thin["reasons"] == ("thinned_kappa",)
    assert thin["reg_class"] == "ThinnedKappaCap"
    assert thin["kappa_probe_abs_err"] == pytest.approx(0.040, abs=1e-6)
    assert thin["cap_abs_err"] == pytest.approx(0.0, abs=1e-12)


def test_thinned_center_ignores_kappa():
    faithful = GradientPenalty(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    thinned = ThinnedKappaCap(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    assert thinned.kappa == 0.2
    assert thinned.center(0) == 1.0
    assert faithful.center(0) == 0.2


def test_board_log_names_the_pass_and_the_fails(capsys):
    rows = run_board()
    assert [row["arm"] for row in rows if row["pass"]] == ["locked_2d", "locked_lift"]
    out = capsys.readouterr().out
    assert "field_lift arm=locked_lift verdict=PASS" in out
    assert "reasons=naive_lift,undershoot" in out
    assert "reasons=stranger_pairing" in out
    assert "reasons=fm_on" in out
    assert "reasons=thinned_kappa" in out


def test_unknown_placement_is_refused():
    with pytest.raises(ValueError, match="placement"):
        LiftArm("nope", "pad")
