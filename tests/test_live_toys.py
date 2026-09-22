"""Live analysis / erase paths call conceptmod.toys, not a private copy.

The toy modules stay the formulation source of truth. These tests are
not ``test_toy_*``: they enter through ``analysis_2d``, ``analysis_dsl``,
and ``ops_erase``.
"""

from __future__ import annotations

import pytest

from conceptmod.analysis_2d import claim_formulation, run_method
from conceptmod.analysis_dsl import run_job
from conceptmod.ops_erase import erase_keep_geometry
from conceptmod.toys import (
    LOCKED,
    LOCKED_SHARED,
    LOCKED_TEACHER,
    HonestyError,
    locked_adv_defaults,
    run_honesty_board,
    train_locked,
)


def test_public_api_is_the_locked_shared_stamp():
    adv = locked_adv_defaults()
    assert adv["loss_type"] == "logistic"
    assert adv["gan_mode"] == "rp"
    assert adv["reg_arm"] == "b_cap"
    assert adv["reg_coeff"] == 1.0
    assert adv["reg_kappa"] == 1.0
    assert adv["reg_norm"] == "l2"
    assert adv["fm_weight"] == 0.0
    assert adv["cover_weight"] == 1.5
    assert adv["n_particles"] == 12
    assert adv["particle_l2"] == 0.02
    assert adv["cover_posture"] == "demo"
    for key, value in adv.items():
        assert getattr(LOCKED, key) == value
        if key in LOCKED_SHARED:
            assert LOCKED_SHARED[key] == value
    assert callable(train_locked)
    assert callable(run_honesty_board)


def test_analysis_calls_locked_adv_and_leftover_geometry(monkeypatch):
    """A real 2-D erase step must execute the toy helpers."""
    import conceptmod.analysis_2d as analysis
    import conceptmod.ops_erase as erase

    called = {"adv": 0, "bipolar": 0, "guard": 0, "hold": 0}
    real_adv = analysis.locked_adv_defaults
    real_bipolar = erase.leftover_bipolar
    real_guard = erase.faithful_guard_e
    real_hold = erase.hold_dir

    def adv():
        called["adv"] += 1
        return real_adv()

    def bipolar(plus, minus):
        called["bipolar"] += 1
        return real_bipolar(plus, minus)

    def guard(*args, **kwargs):
        called["guard"] += 1
        return real_guard(*args, **kwargs)

    def hold(*args, **kwargs):
        called["hold"] += 1
        return real_hold(*args, **kwargs)

    monkeypatch.setattr(analysis, "locked_adv_defaults", adv)
    monkeypatch.setattr(erase, "leftover_bipolar", bipolar)
    monkeypatch.setattr(erase, "faithful_guard_e", guard)
    monkeypatch.setattr(erase, "hold_dir", hold)

    result = run_method("erase_esd", "red--", steps=1)
    assert called == {"adv": 1, "bipolar": 1, "guard": 1, "hold": 1}
    assert result.adv == locked_adv_defaults()
    assert result.adv["gan_mode"] == "rp"
    assert result.adv["fm_weight"] == 0.0
    assert result.adv["cover_weight"] == LOCKED.cover_weight
    assert result.geometry["teacher"] == LOCKED_TEACHER
    assert result.geometry["teacher_leak"] == pytest.approx(0.0, abs=1e-5)
    assert result.geometry["hold_cos"] == pytest.approx(1.0, abs=1e-5)
    assert result.geometry["same_dir"] == pytest.approx(0.0, abs=1e-5)
    assert result.geometry["same_dir_ok"] is True
    assert result.verdict != "PASS"


def test_dsl_job_carries_the_same_adv_stamp():
    result = run_job("orthogonal_noop", "red%stripe", steps=1)
    assert result.adv == locked_adv_defaults()
    assert result.adv["gan_mode"] == "rp"
    assert result.geometry["teacher"] == "faithful_guard_e"
    assert "same_dir" in result.geometry


def test_formulation_pass_without_negatives_is_refused():
    with pytest.raises(HonestyError, match="negative"):
        claim_formulation(winner=None, negatives=[])
    with pytest.raises(HonestyError, match="negative"):
        claim_formulation(winner=None, negatives=())


def test_erase_axes_use_faithful_hold():
    import torch

    erase = torch.tensor([1.0, 0.0, 0.0])
    keep = torch.tensor([0.0, 1.0, 0.0])
    report = erase_keep_geometry(erase, keep)
    assert report["teacher"] == LOCKED_TEACHER
    assert report["teacher_leak"] == pytest.approx(0.0, abs=1e-5)
    assert report["teacher_leak_ok"] is True
    assert report["hold_cos"] == pytest.approx(1.0, abs=1e-5)
    assert "same_dir" not in report

    residual = torch.tensor([0.4, 0.0, 0.0])
    poles = erase_keep_geometry(erase, keep, residual, -residual)
    assert poles["same_dir"] == pytest.approx(0.0, abs=1e-5)
    assert poles["same_dir_ok"] is True
    assert poles["leak_frac"] == pytest.approx(-1.0, abs=1e-5)
