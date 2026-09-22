"""CPU gate: AE reconstruction plus locked_shared adversarial shape.

Winning arm PASSes. AE-only FAILs because RpGAN and b_cap never ran.
Stranger pairing, FM-on under b_cap, and a kappa-hardcoded thin cap are refused.
"""
import pytest
import torch
from torch import nn

from conceptmod.toys.ae_gan_hold import (
    DEMO_COVER,
    HOLD_MAX,
    N_PARTICLES,
    PARTICLE_L2,
    RECON_MAX,
    HoldConfig,
    ThinnedBCap,
    ae_only_config,
    fm_on_config,
    gate_reasons,
    locked_config,
    make_recipe,
    regularizer_mismatches,
    shape_mismatches,
    stranger_config,
    train,
)
from particlegan.grad_regularizers import GradRegularizer


_BOARD = {}


def board():
    if "locked" not in _BOARD:
        _BOARD["locked"] = train(locked_config())
        _BOARD["ae_only"] = train(ae_only_config())
    return _BOARD


def test_locked_shape_is_rpgan_bcap_fm_off_demo_cover():
    cfg = locked_config()
    assert shape_mismatches(cfg) == []
    assert cfg.cover_weight == DEMO_COVER
    assert cfg.fm_weight == 0.0
    assert cfg.n_particles == N_PARTICLES == 12
    assert cfg.particle_l2 == PARTICLE_L2 == 0.02
    assert cfg.n_particles != 128
    recipe = make_recipe(cfg)
    loss = recipe.make_loss()
    assert (loss.loss_type, loss.mode) == ("logistic", "rp")
    reg = recipe.make_gradient_penalty(norm=cfg.reg_norm, target_anneal=cfg.target_anneal)
    assert regularizer_mismatches(reg) == []
    assert reg.arm == "b_cap" and reg.coeff == 1.0 and reg.kappa == 1.0 and reg.norm == "l2"


def test_winning_combo_passes_recon_and_adversarial_hold():
    row = board()["locked"]
    assert row["verdict"] == "PASS"
    assert row["reasons"] == []
    assert row["recon_mse"] <= RECON_MAX
    assert row["recon_mse"] < row["init_recon_mse"] / 10
    assert row["hold"] <= HOLD_MAX
    assert row["bcap_applied"] == row["steps"]
    assert row["adv_steps"] == row["steps"]


def test_ae_only_fails_even_though_reconstruction_is_tight():
    row = board()["ae_only"]
    assert row["verdict"] == "FAIL"
    assert row["recon_mse"] <= RECON_MAX
    assert row["bcap_applied"] == 0
    assert row["adv_steps"] == 0
    assert any("b_cap applied 0" in reason for reason in row["reasons"])
    assert any("RpGAN generator steps 0" in reason for reason in row["reasons"])
    assert shape_mismatches(row["cfg"])


@pytest.mark.parametrize("cfg", [stranger_config(), fm_on_config(), HoldConfig(name="thin_kappa", reg_kappa=0.1)])
def test_adv_ablations_are_refused(cfg):
    assert shape_mismatches(cfg)
    with pytest.raises(ValueError, match="refusing drifted"):
        train(cfg)


def test_thinned_bcap_stub_is_not_the_champion():
    assert regularizer_mismatches(ThinnedBCap())
    thin = GradRegularizer(arm="b_cap", coeff=1.0, kappa=0.1, norm="l2", lazy_k=1)
    assert regularizer_mismatches(thin)
    assert any("kappa" in reason for reason in regularizer_mismatches(thin))


def test_untrained_recon_does_not_clear_the_fidelity_bar():
    torch.manual_seed(0)
    cfg = locked_config()
    recipe = make_recipe(cfg)
    prior = recipe.make_prior()
    encoder, decoder = nn.Sequential(nn.Linear(2, 4)), nn.Sequential(nn.Linear(2, 2))
    # A fresh decoder is far from the blobs; the bar is not vacuous.
    from conceptmod.toys.ae_gan_hold import evaluate
    opened = evaluate(encoder, decoder, prior, recipe)
    assert opened["recon_mse"] > RECON_MAX
    reasons = gate_reasons({
        "cfg": cfg,
        "recon_mse": opened["recon_mse"],
        "hold": opened["hold"],
        "bcap_applied": 0,
        "adv_steps": 0,
        "steps": cfg.steps,
    })
    assert reasons
