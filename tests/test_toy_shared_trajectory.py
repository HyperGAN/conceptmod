"""Shared-trajectory pairs pass; stranger pairs fail or are refused.

Re-homed from ParticleGAN #26. That pull was merged by mistake and is being
reverted. This module is the copy that remains.
"""

import inspect

import pytest
import torch

import conceptmod.toys.shared_trajectory as shared_trajectory
from conceptmod.toys.shared_trajectory import (
    LOCKED,
    PASS_IDENTITY_MSE,
    pairing_index,
    train,
    train_drift,
    train_locked,
    trajectories,
)


def test_family_does_not_import_particlegan_trajectory():
    """A ParticleGAN revert of #26 must not delete this gate."""
    source = inspect.getsource(shared_trajectory)
    assert "particlegan.shared_trajectory" not in source
    assert "lib.shared_trajectory" not in source
    assert shared_trajectory.train_locked is not None


def test_shared_identity_is_the_only_fixed_point_pairing():
    slow, fast = trajectories()
    shared = pairing_index("shared", slow)
    stranger = pairing_index("stranger", slow)
    nearest = pairing_index("nearest_stranger", slow)
    identity = torch.arange(slow.shape[0])
    assert torch.equal(shared, identity)
    assert int((stranger == identity).sum()) == 0
    assert int((nearest == identity).sum()) == 0
    assert not torch.equal(stranger, nearest)
    nearest_floor = float((fast - fast[nearest]).pow(2).mean())
    assert nearest_floor > PASS_IDENTITY_MSE


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


def test_locked_shared_passes():
    result = train_locked()
    assert result["locked"] is True and result["pairing"] == "shared"
    assert result["delta"] == {}
    assert result["fm_weight"] == 0.0
    assert result["reg_arm"] == "b_cap" and result["reg_norm"] == "l2"
    assert result["reg_coeff"] == 1.0 and result["reg_kappa"] == 1.0
    assert result["cover_weight"] == 1.5 and result["n_particles"] == 12
    assert result["pass"] is True
    assert result["identity_mse"] <= PASS_IDENTITY_MSE


def test_locked_path_refuses_stranger_pairing():
    with pytest.raises(ValueError, match="refuses pairing"):
        train(pairing="stranger")
    with pytest.raises(ValueError, match="refuses pairing"):
        train(pairing="nearest_stranger", locked=True)
    with pytest.raises(ValueError, match="drift arm"):
        train(pairing="shared", locked=False)


@pytest.mark.parametrize("pairing", ["stranger", "nearest_stranger"])
def test_stranger_pairing_fails_identity_gate(pairing):
    result = train_drift(pairing)
    assert result["locked"] is False
    assert result["delta"] == {"pairing": pairing}
    assert result["fm_weight"] == 0.0
    assert result["reg_coeff"] == 1.0 and result["reg_kappa"] == 1.0
    assert result["pass"] is False
    assert result["identity_mse"] > PASS_IDENTITY_MSE
