"""Residual student lands on shared both-land pairs and crashes on strangers.

The arcs and locked_shared pins come from shared_trajectory. This family adds
the residual head, the both-land mask, and the own-pad landing gate.
"""

import inspect

import pytest
import torch

import conceptmod.toys.residual_student as residual_student
import conceptmod.toys.shared_trajectory as shared_trajectory
from conceptmod.toys.residual_student import (
    LAND_TOL,
    PASS_IDENTITY_MSE,
    SLOW_IMPACT_MAX,
    SUCCESS_MIN,
    ResidualHead,
    both_land_mask,
    endpoints,
    impact,
    landing_stats,
    train,
    train_drift,
    train_locked,
)
from conceptmod.toys.shared_trajectory import LOCKED, pairing_index, trajectories


def test_composes_with_shared_trajectory_rather_than_copying_it():
    source = inspect.getsource(residual_student)
    assert "shared_trajectory" in source
    assert "def trajectories" not in source
    assert "particlegan.shared_trajectory" not in source
    assert "lib.shared_trajectory" not in source
    assert residual_student.LOCKED is shared_trajectory.LOCKED
    assert residual_student.PASS_IDENTITY_MSE is shared_trajectory.PASS_IDENTITY_MSE


def test_head_is_a_residual_on_the_slow_arc():
    slow, _fast = trajectories()
    head = ResidualHead(slow.shape[1], LOCKED["z_dim"], LOCKED["critic_hidden"])
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    z = torch.zeros(slow.shape[0], LOCKED["z_dim"])
    pred = head(slow, z)
    assert torch.equal(pred, slow)


def test_both_land_is_same_seed_only():
    slow, fast = trajectories()
    shared = pairing_index("shared", slow)
    stranger = pairing_index("stranger", slow)
    nearest = pairing_index("nearest_stranger", slow)
    assert int(both_land_mask(slow, fast, shared).sum()) == slow.shape[0]
    assert int(both_land_mask(slow, fast, stranger).sum()) == 0
    assert int(both_land_mask(slow, fast, nearest).sum()) == 0
    # A fast arc is a hard impact, so it is not a slow landing.
    assert float(impact(slow).max()) <= SLOW_IMPACT_MAX
    assert float(impact(fast).min()) > SLOW_IMPACT_MAX
    assert int(both_land_mask(fast, fast, shared).sum()) == 0


def test_stranger_touchdown_fails_the_landing_gate_without_training():
    """Copying the paired fast arc lands on the wrong pad. That is the crash."""
    slow, fast = trajectories()
    for pairing in ("stranger", "nearest_stranger"):
        index = pairing_index(pairing, slow)
        copied = fast[index]
        stats = landing_stats(copied, fast)
        assert stats["success_rate"] == 0.0
        assert stats["wrong_pad_rate"] == 1.0
        assert shared_trajectory.identity_mse(copied, fast) > PASS_IDENTITY_MSE
    own = landing_stats(fast, fast)
    assert own["success_rate"] == 1.0
    assert own["wrong_pad_rate"] == 0.0
    pads = endpoints(fast)
    gap = torch.cdist(pads, pads)
    gap.fill_diagonal_(float("inf"))
    assert LAND_TOL < float(gap.min())


def test_locked_recipe_pins_match_shared_arm():
    assert LOCKED["loss_type"] == "logistic" and LOCKED["gan_mode"] == "rp"
    assert LOCKED["reg_arm"] == "b_cap"
    assert LOCKED["reg_coeff"] == 1.0 and LOCKED["reg_kappa"] == 1.0
    assert LOCKED["reg_norm"] == "l2"
    assert LOCKED["fm_weight"] == 0.0
    assert LOCKED["cover_weight"] == 1.5 and LOCKED["cover_posture"] == "demo"
    assert LOCKED["n_particles"] == 12 and LOCKED["particle_l2"] == 0.02
    assert LOCKED["vicreg_weight"] == 0.05
    assert LOCKED["pairing"] == "shared"
    assert SUCCESS_MIN == 1.0


def test_locked_path_refuses_stranger_pairing():
    with pytest.raises(ValueError, match="refuses pairing"):
        train(pairing="stranger")
    with pytest.raises(ValueError, match="refuses pairing"):
        train(pairing="nearest_stranger", locked=True)
    with pytest.raises(ValueError, match="drift arm"):
        train(pairing="shared", locked=False)


def test_locked_shared_residual_passes():
    result = train_locked()
    assert result["locked"] is True and result["pairing"] == "shared"
    assert result["delta"] == {}
    assert result["both_land_rows"] == LOCKED["n_particles"]
    assert result["fm_weight"] == 0.0
    assert result["reg_arm"] == "b_cap" and result["reg_norm"] == "l2"
    assert result["reg_coeff"] == 1.0 and result["reg_kappa"] == 1.0
    assert result["cover_weight"] == 1.5 and result["n_particles"] == 12
    assert result["pass"] is True
    assert result["identity_mse"] <= PASS_IDENTITY_MSE
    assert result["success_rate"] >= SUCCESS_MIN
    assert result["wrong_pad_rate"] == 0.0


@pytest.mark.parametrize("pairing", ["stranger", "nearest_stranger"])
def test_stranger_pairing_fails_landing_gate(pairing):
    result = train_drift(pairing)
    assert result["locked"] is False
    assert result["delta"] == {"pairing": pairing}
    assert result["both_land_rows"] == 0
    assert result["fm_weight"] == 0.0
    assert result["reg_coeff"] == 1.0 and result["reg_kappa"] == 1.0
    assert result["pass"] is False
    assert result["identity_mse"] > PASS_IDENTITY_MSE
    # One nearest-stranger seed can graze its pad. The gate still wants every
    # seed down, and most of the cloud is on the wrong pad.
    assert result["success_rate"] <= 0.25
    assert result["wrong_pad_rate"] >= 0.5
