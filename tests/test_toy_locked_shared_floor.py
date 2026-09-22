"""CPU regression floor for the locked_shared RpGAN + faithful b_cap stamp.

PASS is the demo shape (logistic Rp pair, GradientPenalty b_cap coeff=1
κ=1 l2, FM off, 12 particles, particle_l2=0.02, cover 1.5). Drifts FAIL.
This is not a Music or Anima transfer claim.
"""

from dataclasses import replace

from particlegan import GANLoss, GradientPenalty, get_recipe

from conceptmod.toys.locked_shared_floor import (
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
    faithful = GradientPenalty(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    thinned = ThinnedKappaCap(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    assert thinned.kappa == 0.2
    assert thinned.center(0) == 1.0
    assert faithful.center(0) == 0.2


def test_public_recipe_defaults_stay_on_the_study_prior():
    """The floor's 12-particle cloud must not retune Recipe('gan')."""
    recipe = get_recipe("gan")
    loss = recipe.make_loss()
    reg = recipe.make_gradient_penalty()
    assert isinstance(loss, GANLoss)
    assert loss.loss_type == "logistic" and loss.mode == "rp"
    assert isinstance(reg, GradientPenalty)
    assert (reg.arm, reg.coeff, reg.kappa, reg.norm) == ("b_cap", 1.0, 1.0, "l2")
    assert recipe.num_particles == 20_000
    assert recipe.prior_reg == 1.0
