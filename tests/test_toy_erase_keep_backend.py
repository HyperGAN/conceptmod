"""Backend-agnostic erase/keep: locked geometry passes on cpu/dummy.

The score enters ``conceptmod.ops_erase.erase_keep_geometry`` (hold_dir,
faithful_guard_e, leftover_bipolar). Drift arms fail: teacher_leak,
cover_zero undershoot, wrong leftover poles. No CUDA and no Hub.
"""

from __future__ import annotations

import pytest
import torch

from conceptmod.backends import load_backend
from conceptmod.backends.cpu import CpuBackend
from conceptmod.toys.erase_keep_backend import (
    ARMS,
    COVER_ZERO_SCALE,
    FAMILY,
    SUPRA_LATENT,
    SupraShapedStub,
    require_image_latent,
    run_board,
    score_backend,
)


@pytest.fixture(scope="module")
def board():
    rows = run_board(seed=0)
    return {(row["backend"], row["arm"]): row for row in rows}


def test_family_arms_are_named():
    assert FAMILY == "erase_keep_backend"
    assert ARMS == ("locked", "teacher_leak", "cover_zero", "wrong_poles")
    assert COVER_ZERO_SCALE == 0.5


def test_singleton_time_axis_is_refused():
    with pytest.raises(ValueError, match="C, H, W"):
        require_image_latent((16, 1, 8, 8))


def test_locked_cpu_invokes_live_erase_geometry(monkeypatch):
    """CpuBackend predict_v feeds the live ops_erase helpers."""
    import conceptmod.ops_erase as erase

    called = {"geo": 0, "hold": 0, "guard": 0, "bipolar": 0}
    real_geo = erase.erase_keep_geometry
    real_hold = erase.hold_dir
    real_guard = erase.faithful_guard_e
    real_bipolar = erase.leftover_bipolar

    def geo(*args, **kwargs):
        called["geo"] += 1
        return real_geo(*args, **kwargs)

    def hold(*args, **kwargs):
        called["hold"] += 1
        return real_hold(*args, **kwargs)

    def guard(*args, **kwargs):
        called["guard"] += 1
        return real_guard(*args, **kwargs)

    def bipolar(*args, **kwargs):
        called["bipolar"] += 1
        return real_bipolar(*args, **kwargs)

    monkeypatch.setattr(erase, "erase_keep_geometry", geo)
    monkeypatch.setattr(erase, "hold_dir", hold)
    monkeypatch.setattr(erase, "faithful_guard_e", guard)
    monkeypatch.setattr(erase, "leftover_bipolar", bipolar)

    backend = load_backend("cpu", device="cpu", lora_rank=4, seed=0)
    assert isinstance(backend, CpuBackend)
    seen = {"n": 0}
    real_v = backend.predict_v

    def wrapped(prompt, z, timestep, frozen=False):
        seen["n"] += 1
        return real_v(prompt, z, timestep, frozen)

    backend.predict_v = wrapped
    row = score_backend(backend, "locked", name="cpu", seed=0)
    assert seen["n"] == 8
    assert called == {"geo": 1, "hold": 1, "guard": 2, "bipolar": 1}
    assert row["pass"] is True
    assert row["fail_reasons"] == ""
    assert row["backend"] == "cpu"
    assert row["backend_cls"] == "CpuBackend"
    assert row["latent_shape"] == (4, 8, 8)
    assert row["teacher"] == "faithful_guard_e"
    assert row["cover_weight"] == 1.5
    assert row["teacher_leak"] == pytest.approx(0.0, abs=1e-5)
    assert row["teacher_leak_ok"] is True
    assert row["hold_cos"] == pytest.approx(1.0, abs=1e-5)
    assert row["same_dir"] == pytest.approx(0.0, abs=1e-5)
    assert row["same_dir_ok"] is True
    assert row["leak_frac"] == pytest.approx(-1.0, abs=1e-5)
    assert row["u_kept"] == pytest.approx(1.0, abs=1e-5)
    assert row["pole_rel_err"] == pytest.approx(0.0, abs=1e-5)
    assert row["axis_dot"] == pytest.approx(0.0, abs=1e-5)
    assert row["covered"] is True


def test_dummy_alias_matches_locked_cpu(board):
    cpu = board[("cpu", "locked")]
    dummy = board[("dummy", "locked")]
    assert dummy["pass"] is True
    assert dummy["backend_cls"] == "CpuBackend"
    assert dummy["latent_shape"] == cpu["latent_shape"] == (4, 8, 8)
    for key in ("teacher_leak", "hold_cos", "same_dir", "leak_frac", "u_kept", "pole_rel_err"):
        assert dummy[key] == pytest.approx(cpu[key], abs=1e-5)
    assert dummy["teacher"] == cpu["teacher"] == "faithful_guard_e"


def test_teacher_leak_fails_on_cpu(board):
    row = board[("cpu", "teacher_leak")]
    assert row["pass"] is False
    assert row["fail_reasons"] == "teacher_leak"
    assert row["teacher_leak"] == pytest.approx(1.0, abs=1e-5)
    assert row["teacher_leak_ok"] is False
    assert row["hold_cos"] is None
    assert row["axis_dot"] == pytest.approx(1.0, abs=1e-5)
    assert row["same_dir_ok"] is True
    assert row["u_kept"] == pytest.approx(1.0, abs=1e-5)
    assert "undershoot" not in row["fail_reasons"]
    assert "wrong_poles" not in row["fail_reasons"]


def test_cover_zero_undershoots_on_cpu(board):
    row = board[("cpu", "cover_zero")]
    assert row["pass"] is False
    assert row["fail_reasons"] == "undershoot"
    assert row["cover_weight"] == 0.0
    assert row["teacher"] == "faithful_guard_e"
    assert row["teacher_leak_ok"] is True
    assert row["same_dir_ok"] is True
    assert row["u_kept"] == pytest.approx(COVER_ZERO_SCALE, abs=1e-5)
    assert row["u_kept"] < 0.85
    assert row["pole_rel_err"] == pytest.approx(1.0 - COVER_ZERO_SCALE, abs=1e-5)
    assert row["pole_rel_err"] > 0.20
    assert row["hold_cos"] == pytest.approx(1.0, abs=1e-5)


def test_wrong_leftover_poles_fail_on_cpu(board):
    row = board[("cpu", "wrong_poles")]
    assert row["pass"] is False
    reasons = row["fail_reasons"].split(",")
    assert "wrong_poles" in reasons
    assert "teacher_leak" not in reasons
    assert row["same_dir"] == pytest.approx(1.0, abs=1e-5)
    assert row["same_dir_ok"] is False
    assert row["leak_frac"] == pytest.approx(1.0, abs=1e-5)
    assert row["teacher_leak_ok"] is True
    assert row["pole_rel_err"] == pytest.approx(2.0, abs=1e-5)


def test_supra_stub_repeats_the_split_without_hub(board):
    locked = board[("supra_stub", "locked")]
    assert locked["pass"] is True
    assert locked["backend_cls"] == "SupraShapedStub"
    assert locked["latent_shape"] == SUPRA_LATENT
    assert locked["teacher_leak"] == pytest.approx(0.0, abs=1e-5)
    assert locked["hold_cos"] == pytest.approx(1.0, abs=1e-5)
    assert locked["same_dir"] == pytest.approx(0.0, abs=1e-5)
    assert board[("supra_stub", "teacher_leak")]["fail_reasons"] == "teacher_leak"
    assert board[("supra_stub", "cover_zero")]["fail_reasons"] == "undershoot"
    assert "wrong_poles" in board[("supra_stub", "wrong_poles")]["fail_reasons"]
    for arm in ARMS:
        cpu = board[("cpu", arm)]
        supra = board[("supra_stub", arm)]
        assert supra["pass"] is cpu["pass"]
        assert supra["fail_reasons"] == cpu["fail_reasons"]
        assert supra["teacher_leak"] == pytest.approx(cpu["teacher_leak"], abs=1e-5)

    stub = SupraShapedStub()
    z = torch.zeros(1, *SUPRA_LATENT)
    with pytest.raises(ValueError, match="flow time"):
        stub.predict_v("erase", z, torch.tensor([500.0]), False)
    v = stub.predict_v("", z, torch.tensor([0.5]), True)
    assert v.ndim == 4
    assert tuple(v.shape) == (1, *SUPRA_LATENT)


def test_dummy_drift_matches_cpu(board):
    for arm in ("teacher_leak", "cover_zero", "wrong_poles"):
        assert board[("dummy", arm)]["pass"] is False
        assert board[("dummy", arm)]["fail_reasons"] == board[("cpu", arm)]["fail_reasons"]
        assert board[("dummy", arm)]["backend_cls"] == "CpuBackend"
