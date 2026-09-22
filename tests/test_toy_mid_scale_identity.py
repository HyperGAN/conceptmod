"""Mid-scale identity hold: locked passes; smile drifts fail.

CPU only. One seed. Not a Music or Anima GPU transfer.
Eval grid is -1, 0, 0.5, 1. The Anima smile sample grid drops -1.
"""

from __future__ import annotations

import pytest
import torch

from conceptmod.toys import mid_scale_identity as toy
from conceptmod.toys.cover_leftover import LeftoverField


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _plant(teacher, *, odd=None, origin=None, mid=None, even=None):
    student = toy.MidScaleResidual(teacher.concept.numel())
    with torch.no_grad():
        student.origin.copy_(teacher.identity if origin is None else origin)
        student.odd.copy_(teacher.concept if odd is None else odd)
        if even is not None:
            student.even.copy_(even)
        if mid is not None:
            student.mid.copy_(mid)
    return student


def test_eval_grid_includes_minus_one_zero_half_and_plus():
    assert toy.EVAL_SCALES == (-1.0, 0.0, 0.5, 1.0)
    assert -1.0 in toy.EVAL_SCALES
    assert toy.ANIMA_SMILE_SCALES == (0.0, 0.25, 0.5, 1.0)
    assert -1.0 not in toy.ANIMA_SMILE_SCALES


def test_mid_bump_is_zero_on_poles_and_neutral_and_one_at_half():
    assert toy.mid_bump(-1.0) == 0.0
    assert toy.mid_bump(0.0) == 0.0
    assert toy.mid_bump(1.0) == 0.0
    assert toy.mid_bump(-0.5) == 0.0
    assert toy.mid_bump(0.5) == pytest.approx(1.0)


def test_teacher_strips_leftover_and_keeps_identity_and_concept():
    field = LeftoverField()
    teacher = toy.smile_teacher(field)
    odd = teacher.plus - teacher.identity
    assert abs(float(odd @ field.basis(2))) < 1e-5
    assert float(odd @ field.basis(0)) == pytest.approx(field.slider)
    assert float(teacher.identity @ field.basis(1)) == pytest.approx(field.content)
    assert torch.allclose(teacher.minus, teacher.identity - odd)
    assert abs(float(teacher.stranger @ teacher.retain_unit)) < 1e-6


def test_score_reads_every_required_scale_and_not_a_hidden_pole():
    teacher = toy.smile_teacher()
    student = _plant(teacher)
    seen = []
    original = student.state

    def wrapped(scale):
        seen.append(float(scale))
        return original(scale)

    student.state = wrapped
    row = toy.score_hold(student, teacher=teacher)
    assert seen == [-1.0, 0.0, 0.5, 1.0]
    assert row["scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert row["pass"]
    assert row["fail_reasons"] == ""
    assert row["concept_cos_plus"] == pytest.approx(1.0)
    assert row["concept_cos_minus"] == pytest.approx(1.0)
    assert row["identity_at_0"] == pytest.approx(1.0)
    assert row["identity_at_mid"] == pytest.approx(1.0)
    assert row["device"] == "cpu"

    seen.clear()
    dropped = toy.score_hold(student, scales=toy.ANIMA_SMILE_SCALES, teacher=teacher)
    assert seen == [0.0, 0.25, 0.5, 1.0]
    assert -1.0 not in seen
    assert dropped["pass"] is False
    assert dropped["fail_reasons"] == "missing_minus"
    assert dropped["concept_cos_minus"] is None
    assert dropped["identity_at_0"] == pytest.approx(1.0)
    assert dropped["identity_at_mid"] == pytest.approx(1.0)
    assert dropped["concept_cos_plus"] == pytest.approx(1.0)


def test_hand_set_mid_collapse_fails_only_identity_at_half():
    teacher = toy.smile_teacher()
    student = _plant(teacher, mid=teacher.stranger - teacher.identity)
    row = toy.score_hold(student, teacher=teacher)
    assert row["pass"] is False
    assert row["fail_reasons"] == "identity_mid"
    assert row["identity_at_0"] == pytest.approx(1.0)
    assert row["identity_at_mid"] == pytest.approx(0.0, abs=1e-6)
    assert row["concept_cos_plus"] == pytest.approx(1.0)
    assert row["concept_cos_minus"] == pytest.approx(1.0)
    assert row["same_dir"] == pytest.approx(0.0, abs=1e-5)


def test_hand_set_polarity_flip_fails_concept_direction():
    teacher = toy.smile_teacher()
    student = _plant(teacher, odd=-teacher.concept)
    row = toy.score_hold(student, teacher=teacher)
    assert row["pass"] is False
    assert row["fail_reasons"] == "polarity"
    assert row["concept_cos_plus"] == pytest.approx(-1.0)
    assert row["concept_cos_minus"] == pytest.approx(-1.0)
    assert row["identity_at_0"] == pytest.approx(1.0)
    assert row["identity_at_mid"] == pytest.approx(1.0)


def test_stranger_pairing_fails_a_healthy_residual():
    teacher = toy.smile_teacher()
    row = toy.score_hold(_plant(teacher), pairing="stranger", teacher=teacher)
    assert row["pass"] is False
    assert row["fail_reasons"] == "stranger_pairing"
    assert row["identity_at_0"] == pytest.approx(1.0)
    assert row["concept_cos_plus"] == pytest.approx(1.0)


def test_locked_shape_reports_the_particle_omission_and_refuses_drift():
    report = toy.locked_shape_report()
    assert "no particle cloud" in report["intentional_drift"]["particles"]
    assert report["matches"]["cover_weight"] == 1.5
    assert report["matches"]["fm_weight"] == 0.0
    assert report["matches"]["eval_grid"] == [-1.0, 0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="stranger pairing"):
        toy.run_arm("locked", gan_mode="vanilla")
    with pytest.raises(ValueError, match="stranger pairing"):
        toy.run_arm("locked", pairing="stranger")
    with pytest.raises(ValueError, match="FM-on"):
        toy.run_arm("locked", fm_weight=1.0)
    with pytest.raises(ValueError, match="thinned b_cap"):
        toy.run_arm("locked", reg_kappa=0.2)
    with pytest.raises(ValueError, match="unknown arm"):
        toy.run_arm("hub128")


def test_locked_training_calls_real_b_cap(monkeypatch):
    seen = []
    original = toy.GradientPenalty.penalty

    def spy(self, *args, **kwargs):
        seen.append((self.arm, self.coeff, self.kappa, self.norm, self.lazy_k, self.target_anneal))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(toy.GradientPenalty, "penalty", spy)
    row = toy.run_arm("locked", steps=2, seed=0)
    assert seen, "b_cap penalty never ran"
    assert set(seen) == {("b_cap", 1.0, 1.0, "l2", 1, "none")}
    assert row["reg_is_gradient_penalty"]
    assert row["reg_calls"] == 2 * len(toy.EVAL_SCALES)
    assert row["train_scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert row["gan_mode"] == "rp"
    assert row["fm_weight"] == 0.0
    assert row["cover_weight"] == 1.5
    assert row["device"] == "cpu"
    assert row["scales"] == [-1.0, 0.0, 0.5, 1.0]


@pytest.fixture(scope="module")
def board():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    rows = toy.run_board(steps=toy.GATE_STEPS, seed=0)
    torch.set_num_threads(previous)
    return {row["arm"]: row for row in rows}


def test_locked_passes_concept_poles_and_identity_at_zero_and_mid(board):
    row = board["locked"]
    assert row["pass"]
    assert row["fail_reasons"] == ""
    assert row["scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert row["train_scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert [item["scale"] for item in row["per_scale"]] == [-1.0, 0.0, 0.5, 1.0]
    assert row["concept_cos_plus"] >= toy.CONCEPT_COS_MIN
    assert row["concept_cos_minus"] >= toy.CONCEPT_COS_MIN
    assert row["concept_mag_plus"] == pytest.approx(1.0, abs=0.05)
    assert row["concept_mag_minus"] == pytest.approx(1.0, abs=0.05)
    assert row["identity_at_0"] >= toy.IDENTITY_KEPT_MIN
    assert row["identity_at_mid"] >= toy.IDENTITY_KEPT_MIN
    assert row["same_dir"] <= 0.05
    assert row["cover_weight"] == 1.5
    assert row["reg_arm"] == "b_cap" and row["reg_kappa"] == 1.0
    assert row["fm_weight"] == 0.0 and row["gan_mode"] == "rp"
    assert row["device"] == "cpu"


def test_mid_collapse_keeps_poles_and_drops_identity_at_half(board):
    row = board["mid_collapse"]
    assert row["pass"] is False
    assert row["fail_reasons"] == "identity_mid"
    assert row["scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert row["concept_cos_plus"] >= toy.CONCEPT_COS_MIN
    assert row["concept_cos_minus"] >= toy.CONCEPT_COS_MIN
    assert row["identity_at_0"] >= toy.IDENTITY_KEPT_MIN
    assert row["identity_at_mid"] < 0.2
    assert row["same_dir"] <= 0.05


def test_missing_minus_fails_the_anima_smile_grid(board):
    row = board["missing_minus"]
    assert row["pass"] is False
    assert row["fail_reasons"] == "missing_minus"
    assert row["train_scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert row["scales"] == [0.0, 0.25, 0.5, 1.0]
    assert -1.0 not in row["scales"]
    assert row["concept_cos_minus"] is None
    assert row["concept_cos_plus"] >= toy.CONCEPT_COS_MIN
    assert row["identity_at_0"] >= toy.IDENTITY_KEPT_MIN
    assert row["identity_at_mid"] >= toy.IDENTITY_KEPT_MIN


def test_polarity_flip_fails_and_stranger_pairing_loses_identity(board):
    flipped = board["polarity_flipped"]
    assert flipped["pass"] is False
    assert flipped["fail_reasons"] == "polarity"
    assert flipped["concept_cos_plus"] < 0.0
    assert flipped["concept_cos_minus"] < 0.0
    assert flipped["identity_at_0"] >= toy.IDENTITY_KEPT_MIN
    assert flipped["identity_at_mid"] >= toy.IDENTITY_KEPT_MIN
    assert flipped["scales"] == [-1.0, 0.0, 0.5, 1.0]

    stranger = board["stranger"]
    assert stranger["pass"] is False
    assert "stranger_pairing" in stranger["fail_reasons"].split(",")
    assert stranger["identity_at_0"] < toy.IDENTITY_KEPT_MIN
    assert stranger["identity_at_mid"] < toy.IDENTITY_KEPT_MIN
    assert stranger["concept_cos_plus"] >= toy.CONCEPT_COS_MIN
    assert stranger["concept_cos_minus"] >= toy.CONCEPT_COS_MIN
    assert stranger["pairing"] == "stranger"
    assert stranger["scales"] == [-1.0, 0.0, 0.5, 1.0]
