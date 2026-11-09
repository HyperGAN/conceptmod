"""CPU regression floor for the locked_shared RpGAN + faithful b_cap stamp.

PASS is the demo shape (logistic Rp pair, GradientPenalty b_cap coeff=1
κ=1 l2, FM off, 12 particles, particle_l2=0.02, cover 1.5). Drifts FAIL.
This is not a Music or Anima transfer claim.
"""

from conceptmod.particle_api import make_gradient_penalty

from dataclasses import replace

from particlegan.grad_regularizers import GradientPenalty
from particlegan import GANLoss, get_recipe

from conceptmod.research.toys.locked_shared_floor import (
    COVER_WEIGHT,
    LOCKED,
    N_PARTICLES,
    ThinnedKappaCap,
    run_floor,
    score_floor,
)


def test_locked_shared_rpgan_bcap_passes():
    score = score_floor(run_floor(LOCKED))
    assert score["verdict"] == "PASS", score["mismatches"]
    assert score["adv_abs_err"] < 1e-5
    assert score["g_abs_err"] < 1e-5
    assert score["cap_abs_err"] < 1e-5
    assert score["kappa_probe_abs_err"] < 1e-5
    assert score["reg_class"] == "GradientPenalty"
    assert score["fm_weight"] == 0.0
    assert score["cover_weight"] == COVER_WEIGHT
    assert score["cover_posture"] == "demo"
    assert score["n_particles"] == N_PARTICLES
    assert score["gan_mode"] == "rp"
    assert score["loss_type"] == "logistic"


def test_thinned_kappa_bcap_fails():
    score = score_floor(run_floor(LOCKED, regularizer="thinned"))
    assert score["verdict"] == "FAIL"
    assert score["reg_class"] == "ThinnedKappaCap"
    assert score["kappa_probe_abs_err"] > 1e-3
    assert any("kappa" in item or "GradientPenalty" in item for item in score["mismatches"])
    # At locked κ=1 the thinned center matches, so the step cap can still agree.
    assert score["cap_abs_err"] < 1e-5


def test_fm_on_under_bcap_fails():
    score = score_floor(run_floor(replace(LOCKED, fm_weight=0.1)))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("fm_weight") for item in score["mismatches"])
    assert score["g_abs_err"] > 1e-5


def test_non_rp_logistic_fails():
    score = score_floor(run_floor(replace(LOCKED, gan_mode="vanilla")))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("gan_mode") for item in score["mismatches"])
    assert score["adv_abs_err"] > 1e-4


def test_stranger_pairing_fails():
    score = score_floor(run_floor(LOCKED, pairing="stranger"))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("pairing") for item in score["mismatches"])
    assert score["adv_abs_err"] > 1e-5


def test_music_cover_weight_is_drift_from_demo_stamp():
    cfg = replace(LOCKED, cover_weight=1.0, cover_posture="music")
    score = score_floor(run_floor(cfg))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("cover_weight") for item in score["mismatches"])


def test_hub_128_particle_cloud_is_not_locked():
    score = score_floor(run_floor(replace(LOCKED, n_particles=128)))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("n_particles") for item in score["mismatches"])


def test_thinned_class_ignores_kappa_argument():
    faithful = make_gradient_penalty(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    thinned = make_gradient_penalty(cls=ThinnedKappaCap, arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    assert thinned.kappa == 0.2
    assert thinned.center(0) == 1.0
    assert faithful.kappa == 0.2


def test_public_recipe_defaults_are_independent_of_the_locked_toy():
    """The toy's explicit RpGAN/b_cap lock does not override the current library."""
    recipe = get_recipe("gan")
    assert recipe.num_particles == 20_000
    assert recipe.critic_formulation == "bispf"
    assert recipe.make_loss().loss == "non_saturating"
    assert recipe.prior_regularizer == "sbnp"
    assert recipe.prior_reg == 0.01
