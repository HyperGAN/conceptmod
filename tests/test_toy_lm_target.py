"""Image UNI lm_target: trajectory PASSes; direct and cfg_delta FAIL.

Same field, same gates. CPU only. Not an Anima or Music GPU transfer.
"""

import ast
import inspect

import pytest
import torch

from conceptmod.toys import locked_adv_defaults
from conceptmod.toys import lm_target as lm


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def board():
    rows = lm.run_board(steps=lm.STEPS, seed=0)
    return {row["arm"]: row for row in rows}


def test_high_sigma_gap_is_shut_and_late_sigma_is_not():
    """Image UNI fact this fixture sharpens: 1-step plus≈neu, late σ is not."""
    diag = lm.field_diagnostics()
    assert diag["one_step_mse"] == pytest.approx(0.0, abs=1e-8)
    assert diag["one_step_cos"] == pytest.approx(1.0, abs=1e-6)
    assert diag["late_expr_gap"] == pytest.approx(lm.EXPR)
    assert diag["traj_expr_gap"] == pytest.approx(-1.0, abs=1e-5)
    plus = lm.frozen_velocity("plus", lm.ONE_STEP_SIGMA)
    neu = lm.frozen_velocity("neu", lm.ONE_STEP_SIGMA)
    assert torch.allclose(plus, neu)


def test_one_step_arms_have_no_expression_gradient():
    """At infer noise the locked 1-step targets do not move the residual."""
    for arm in ("direct", "cfg_delta"):
        residual = lm.NeuResidual()
        loss = lm.one_step_loss(arm, residual, lm.ONE_STEP_SIGMA)
        loss.backward()
        assert residual.a.grad is not None
        assert torch.allclose(residual.a.grad, torch.zeros_like(residual.a))
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-8)


def test_one_step_arms_would_learn_expression_at_late_sigma():
    """The student can represent the concept. σ=1 is what hides it."""
    for arm in ("direct", "cfg_delta"):
        residual = lm.NeuResidual()
        loss = lm.one_step_loss(arm, residual, sigma=0.25)
        loss.backward()
        assert residual.a.grad is not None
        assert float(residual.a.grad[1].abs()) > 0.0
        assert float(loss.detach()) > 0.0


def test_trajectory_gradient_is_on_expression_only():
    residual = lm.NeuResidual()
    loss = lm.trajectory_loss(residual)
    loss.backward()
    assert float(residual.a.grad[0].abs()) == pytest.approx(0.0, abs=1e-6)
    assert float(residual.a.grad[1].abs()) > 0.0
    assert float(loss.detach()) > 0.0


def test_locked_trajectory_passes(board):
    row = board["trajectory"]
    assert row["locked"] and row["lm_target"] == lm.LOCKED_TARGET == "trajectory"
    assert row["passed"]
    assert row["fail_reasons"] == []
    assert row["expr_gain"] == pytest.approx(1.0, abs=1e-4)
    assert row["expr_gain"] >= lm.EXPR_GAIN_MIN
    assert row["struct_hold"] == pytest.approx(1.0, abs=1e-4)
    assert row["struct_hold"] >= lm.STRUCT_HOLD_MIN
    assert row["identity_mse"] <= lm.IDENTITY_MSE_MAX
    assert row["param_l2"] == pytest.approx(lm.EXPR, abs=1e-3)
    assert row["one_step_mse"] == pytest.approx(0.0, abs=1e-8)
    assert row["adv_applies"] is False
    assert row["adv_posture"] == "locked_shared"
    assert row["fm_weight"] == 0.0
    assert row["gan_mode"] == "rp"
    assert row["loss_type"] == "logistic"
    assert row["reg_arm"] == "b_cap"
    assert row["reg_coeff"] == 1.0
    assert row["reg_kappa"] == 1.0
    assert row["reg_norm"] == "l2"


def test_direct_and_cfg_delta_fail_same_gates(board):
    locked = board["trajectory"]
    for arm in ("direct", "cfg_delta"):
        row = board[arm]
        assert row["passed"] is False
        assert row["fail_reasons"] == ["expr_gain"]
        assert row["expr_gain"] == pytest.approx(0.0, abs=1e-6)
        assert row["expr_gain"] < lm.EXPR_GAIN_MIN
        assert row["struct_hold"] >= lm.STRUCT_HOLD_MIN
        assert row["identity_mse"] <= lm.IDENTITY_MSE_MAX
        assert row["param_l2"] == pytest.approx(0.0, abs=1e-8)
        assert row["one_step_mse"] == locked["one_step_mse"]
        assert row["one_step_cos"] == locked["one_step_cos"]
        assert row["traj_expr_gap"] == locked["traj_expr_gap"]
        assert row["late_expr_gap"] == locked["late_expr_gap"]
        assert row["adv_applies"] is False


def test_adv_posture_matches_locked_shared_and_drift_is_refused(board):
    stamp = locked_adv_defaults()
    for row in board.values():
        for key in ("loss_type", "gan_mode", "reg_arm", "reg_coeff", "reg_kappa", "reg_norm", "fm_weight"):
            assert row[key] == stamp[key]
    with pytest.raises(ValueError, match="stranger pairing"):
        lm.run_arm("trajectory", gan_mode="vanilla")
    with pytest.raises(ValueError, match="FM-on"):
        lm.run_arm("direct", fm_weight=1.0)
    with pytest.raises(ValueError, match="thinned b_cap"):
        lm.run_arm("cfg_delta", reg_kappa=0.2)
    with pytest.raises(ValueError, match="thinned b_cap"):
        lm.run_arm("trajectory", reg_norm="l1")
    with pytest.raises(ValueError, match="unknown arm"):
        lm.run_arm("v9")
    with pytest.raises(ValueError, match="unknown formulation knob"):
        lm.reject_unlocked({"cover_weight": 0.0})


def test_module_does_not_import_anima_or_particlegan_toys():
    """Culture is cited in the docstring. The toy does not import those modules."""
    tree = ast.parse(inspect.getsource(lm))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    third_party = [name for name in imported if name != "__future__"]
    assert third_party == [
        "torch",
        "torch.nn.functional",
        "torch",
        "conceptmod.toys.locked_shared_floor",
    ]
    blob = " ".join(third_party)
    assert "particlegan" not in blob
    assert "anima" not in blob
