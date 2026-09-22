"""Radius-hold gate for the closed-loop circle orbit.

Heading and linear speed pass for the ablated residual. The radius gate does not.
Locked_shared (RpGAN logistic, real b_cap, FM off, demo cover 1.5) passes.
"""

import math

import pytest
import torch

from particlegan.grad_regularizers import GradRegularizer
from conceptmod.toys.orbit_hold import (
    DIRECTION_COS_MIN,
    RADIUS_REL_MAX,
    SPEED_REL_MAX,
    ClosedLoopRadialHead,
    HardcodedKappaBCap,
    ZeroResidualHead,
    _step,
    evaluate,
    locked_recipe,
    run_family,
)


def test_locked_shared_passes_radius_hold():
    report = evaluate(locked_recipe())
    assert report.passed
    assert report.reasons == ()
    assert report.radius_pass and report.direction_pass and report.speed_pass
    assert report.radius_rel < RADIUS_REL_MAX
    assert report.direction_cos >= DIRECTION_COS_MIN
    assert report.speed_rel <= SPEED_REL_MAX
    assert report.critic_frozen
    # Matched pairs land on the same points, so Rp logistic is softplus(0).
    assert report.d_loss == pytest.approx(math.log(2.0), abs=1e-12)
    # Frozen linear critic ||w|| = 3, phi = (3-1)^2, pen = (1/2)*(4+4).
    assert report.bcap == pytest.approx(4.0, abs=1e-12)
    assert report.fm_term == 0.0
    assert report.cover == pytest.approx(0.0, abs=1e-12)
    assert report.particle_l2_term == pytest.approx(0.0, abs=1e-12)


def test_ablated_residual_drifts_radius_with_sound_heading_and_speed():
    report = evaluate(locked_recipe(name="ablate_residual", residual=False))
    assert report.direction_pass and report.speed_pass
    assert report.direction_cos == pytest.approx(1.0, abs=1e-12)
    assert report.speed_rel == pytest.approx(0.0, abs=1e-12)
    assert not report.radius_pass
    assert not report.passed
    assert report.reasons == ("radius_drift",)
    # r_k^2 = R^2 + k*(dt*omega*R)^2, mean |r/R - 1| over k=1..32.
    growth = []
    a2 = 0.25 ** 2
    for k in range(1, 33):
        growth.append(math.sqrt(1.0 + k * a2) - 1.0)
    assert report.radius_rel == pytest.approx(sum(growth) / len(growth), rel=1e-9)
    assert report.radius_rel > 0.2


def test_one_euler_step_already_misses_the_radius_gate():
    recipe = locked_recipe()
    points = torch.tensor([[recipe.radius, 0.0]], dtype=torch.float64)
    nxt, velocity, tangent = _step(points, recipe, ZeroResidualHead())
    radius = torch.linalg.vector_norm(nxt, dim=-1).item()
    assert radius == pytest.approx(math.sqrt(1.0 + (recipe.dt * recipe.omega) ** 2), rel=1e-12)
    assert radius - 1.0 > RADIUS_REL_MAX
    cos = (velocity * tangent).sum(-1).item() / torch.linalg.vector_norm(velocity).item()
    assert cos == pytest.approx(1.0, abs=1e-12)


def test_residual_head_lands_on_the_circle():
    recipe = locked_recipe()
    points = torch.tensor([[0.0, recipe.radius], [recipe.radius, 0.0]], dtype=torch.float64)
    nxt, _, _ = _step(points, recipe, ClosedLoopRadialHead())
    radius = torch.linalg.vector_norm(nxt, dim=-1)
    torch.testing.assert_close(radius, torch.ones_like(radius))


def test_stranger_fm_hinge_and_thin_bcap_fail():
    stranger = evaluate(locked_recipe(name="stranger_vanilla", gan_mode="vanilla"))
    shuffle = evaluate(locked_recipe(name="stranger_shuffle", pairing="stranger"))
    fm = evaluate(locked_recipe(name="fm_on", fm_weight=1.0))
    hinge = evaluate(locked_recipe(name="hinge_drift", loss_type="hinge"))
    thin = evaluate(locked_recipe(name="thin_bcap"), HardcodedKappaBCap())
    for report, code in (
        (stranger, "stranger_pairing"),
        (shuffle, "stranger_pairing"),
        (fm, "fm_on"),
        (hinge, "loss_not_logistic"),
        (thin, "thinned_bcap"),
    ):
        assert not report.passed
        assert code in report.reasons
    # These refusals still hold radius. The fail is the adv drift, not the orbit.
    for report in (stranger, shuffle, fm, hinge):
        assert report.radius_pass and report.direction_pass and report.speed_pass
    assert thin.radius_pass
    # Hardcoded center 0 on ||grad|| = 3: phi = 9, pen = (1/2)*(9+9).
    assert thin.bcap == pytest.approx(9.0, abs=1e-9)
    reference = GradRegularizer("b_cap", coeff=1.0, kappa=1.0, norm="l2")
    assert type(reference).mro()[0] is GradRegularizer


def test_music_cover_and_big_cloud_are_not_locked():
    music = evaluate(locked_recipe(name="music_cover", cover_weight=1.0))
    hub = evaluate(locked_recipe(name="hub_cloud", n_particles=128))
    assert not music.passed and "cover_weight" in music.reasons
    assert not hub.passed and "n_particles" in hub.reasons


def test_family_log_one_pass(capsys):
    reports = run_family()
    out = capsys.readouterr().out
    assert sum(r.passed for r in reports) == 1
    assert reports[0].name == "locked_shared" and reports[0].passed
    assert reports[1].name == "ablate_residual" and not reports[1].passed
    assert "ORBIT arm=locked_shared verdict=PASS" in out
    assert "ORBIT arm=ablate_residual verdict=FAIL" in out
    assert "reasons=radius_drift" in out
    assert "reasons=thinned_bcap" in out
