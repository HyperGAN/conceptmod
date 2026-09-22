"""Cover / leftover / faithful-teacher gate.

Locked demo cover 1.5 + faithful_guard_e must PASS. cover=0 must undershoot.
Raw faithful (teacher drift) must copy leftover ê and FAIL. One seed.
"""

import pytest
import torch

from conceptmod.toys.cover_leftover import (
    GATE_STEPS,
    LOCKED_COVER,
    LOCKED_TEACHER,
    CoverRecipe,
    LeftoverField,
    faithful_guard_e,
    reject_unlocked,
    run_board,
)


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    path = tmp_path_factory.mktemp("cover_leftover") / "train.log"
    with path.open("w") as handle:
        rows = run_board(steps=GATE_STEPS, seed=0, log=handle)
    return {row["arm"]: row for row in rows}


def test_guard_subtracts_unused_leftover_e():
    field = LeftoverField()
    pos, neg, neu = field.poles()
    plus, minus = faithful_guard_e(pos, neg, neu, field.basis(2), field.basis(0))
    assert abs(float((plus - neu) @ field.basis(2))) < 1e-5
    assert float((plus - neu) @ field.basis(0)) == pytest.approx(field.slider)
    assert float((plus - neu) @ field.basis(1)) == pytest.approx(field.content)
    assert torch.allclose(minus, neu - (plus - neu))


def test_guard_refuses_when_e_restates_the_axis():
    """Divergent analogue: ê along content. Blend guard keeps the raw poles."""
    field = LeftoverField()
    u, content = field.basis(0), field.basis(1)
    neu = torch.zeros(field.dim)
    amplitude = 0.3 * u + 2.0 * content
    pos, neg = neu + amplitude, neu - amplitude
    plus, minus = faithful_guard_e(pos, neg, neu, content, u)
    assert torch.allclose(plus, pos)
    assert torch.allclose(minus, neg)


def test_formulation_drift_is_refused():
    with pytest.raises(ValueError, match="stranger pairing"):
        reject_unlocked({"gan_mode": "vanilla"})
    with pytest.raises(ValueError, match="FM-on"):
        reject_unlocked({"fm_weight": 1.0})
    with pytest.raises(ValueError, match="thinned b_cap"):
        reject_unlocked({"reg_kappa": 0.0})
    with pytest.raises(ValueError, match="thinned b_cap"):
        reject_unlocked({"reg_coeff": 0.1, "reg_norm": "l1"})
    with pytest.raises(ValueError, match="particle-count"):
        reject_unlocked({"n_particles": 128})
    CoverRecipe(arm="cover_zero")
    CoverRecipe(arm="teacher_drift")


def test_locked_demo_cover_passes(board):
    row = board["locked"]
    assert row["teacher"] == LOCKED_TEACHER
    assert row["cover_weight"] == LOCKED_COVER
    assert row["b_cap"] == 1.0 and row["kappa"] == 1.0 and row["norm"] == "l2"
    assert row["fm_weight"] == 0.0 and row["gan_mode"] == "rp"
    assert row["n_particles"] == 12
    assert row["pass"]
    assert row["covered"]
    assert row["u_kept"] >= 0.90
    assert row["content_kept"] >= 0.90
    assert row["leak_ratio"] <= 0.05
    assert row["same_dir"] <= 0.05


def test_cover_zero_undershoots(board):
    row = board["cover_zero"]
    assert row["teacher"] == LOCKED_TEACHER
    assert row["cover_weight"] == 0.0
    assert not row["pass"]
    assert "undershoot" in row["fail_reasons"]
    assert "teacher_leak" not in row["fail_reasons"]
    assert row["u_kept"] <= 0.70
    assert row["leak_ratio"] <= 0.20
    assert max(row["pole_rel_err_plus"], row["pole_rel_err_minus"]) > 0.25


def test_teacher_drift_copies_leftover(board):
    row = board["teacher_drift"]
    assert row["teacher"] == "faithful"
    assert row["cover_weight"] == LOCKED_COVER
    assert not row["pass"]
    assert "teacher_leak" in row["fail_reasons"]
    assert "undershoot" not in row["fail_reasons"]
    assert row["u_kept"] >= 0.90
    assert row["leak_ratio"] >= 0.35
