"""Diversity gate: locked_shared RpGAN+b_cap holds a ring; cap-off collapses.

CPU only. Run and tail::

    python -m conceptmod.toys.mode_hold
    python -m pytest -s tests/test_toy_mode_hold.py
"""

import torch

from conceptmod.toys.mode_hold import (
    COLLAPSE_MODES,
    PASS_HQ,
    PASS_MODES,
    b_cap_off_recipe,
    diversity,
    formulation_mismatches,
    locked_recipe,
    ring_means,
    train_mode_hold,
    verdict,
)
from particlegan import GANLoss, GradientPenalty


def test_locked_card_is_rpgan_b_cap_shape():
    recipe = locked_recipe()
    assert formulation_mismatches(recipe) == []
    assert recipe.gan_mode == "rp"
    assert recipe.loss_type == "logistic"
    assert (recipe.reg_arm, recipe.reg_coeff, recipe.reg_kappa, recipe.reg_norm) == (
        "b_cap",
        1.0,
        1.0,
        "l2",
    )
    assert recipe.fm_weight == 0.0
    assert recipe.cover_weight == 1.5  # demo pin, not Music 1.0
    assert recipe.n_particles == 12
    assert recipe.particle_l2 == 0.02
    loss = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    regularizer = GradientPenalty(
        arm=recipe.reg_arm,
        coeff=recipe.reg_coeff,
        kappa=recipe.reg_kappa,
        norm=recipe.reg_norm,
    )
    assert loss.mode == "rp" and regularizer.arm == "b_cap"
    assert regularizer.kappa == 1.0


def test_thinned_kappa_stub_disagrees_with_grad_regularizer():
    """kappa=1 is slack on ||g||=0.5. An explicit kappa=0.2 still binds.

    Both calls go through the published GradientPenalty. There is no local phi.
    """
    slope = torch.tensor([0.3, 0.4])  # ||g|| = 0.5
    norm = slope.norm()
    assert float(norm) == torch.tensor(0.5)

    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(slope.clone())

        def forward(self, x):
            return x @ self.w

    critic = _Linear()
    real = torch.zeros(4, 2)
    real[:, 0] = 1.0
    # ||grad|| = 0.5 < 1, so a hardcoded cap is silent and an explicit kappa=0.2 is not.
    host = GradientPenalty(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    penalty = float(host(critic, real, real).detach())
    assert penalty > 0.0
    locked = GradientPenalty(arm="b_cap", coeff=1.0, kappa=1.0, norm="l2")
    assert float(locked(critic, real, real).detach()) == 0.0


def test_stranger_pairing_is_refused():
    stranger = locked_recipe().replace(gan_mode="vanilla")
    mismatches = formulation_mismatches(stranger)
    assert any(item.startswith("gan_mode:") for item in mismatches)
    try:
        train_mode_hold(stranger, log=False)
    except ValueError as exc:
        assert "unreported drift" in str(exc)
    else:
        raise AssertionError("stranger pairing trained without a drift report")


def test_fm_on_under_b_cap_is_refused():
    fm_on = locked_recipe().replace(fm_weight=0.1)
    try:
        train_mode_hold(fm_on, log=False)
    except ValueError as exc:
        assert "fm_weight" in str(exc)
    else:
        raise AssertionError("FM-on under b_cap trained without a drift report")


def test_unreported_b_cap_off_is_refused():
    try:
        train_mode_hold(b_cap_off_recipe(), log=False)
    except ValueError as exc:
        assert "reg_arm" in str(exc)
    else:
        raise AssertionError("b_cap off trained without a drift report")


def test_locked_shared_holds_ring_modes():
    row = train_mode_hold(log=True)
    assert row["verdict"] == "PASS"
    assert row["modes"] >= PASS_MODES
    assert row["hq"] >= PASS_HQ
    assert row["reg_arm"] == "b_cap"
    assert row["fm_weight"] == 0.0


def test_b_cap_off_collapses():
    row = train_mode_hold(
        b_cap_off_recipe(),
        drift={"reg_arm": "f_none replaces b_cap", "reg_coeff": "coeff 0, cap removed"},
        log=True,
    )
    assert row["verdict"] == "FAIL"
    assert row["modes"] <= COLLAPSE_MODES
    assert row["modes"] < PASS_MODES


def test_dead_cloud_fails_the_same_gate():
    """The veto is not vacuous: a point mass at the origin covers nothing."""
    row = diversity(torch.zeros(256, 2), ring_means())
    assert verdict(row) == "FAIL"
    assert row["modes"] == 0
