"""CPU gate: tiny particle posture passes, Hub-128 and a dropped L2 anchor fail."""
from dataclasses import replace

import pytest

from particlegan import GradientPenalty

from conceptmod.toys.particle_posture import (
    CLOUD_MS_MAX,
    DEMO_L2,
    bcap_kappa_is_explicit,
    evaluate,
    locked_tiny,
    music_parts0,
    render_leaderboard,
    run_family,
)


@pytest.fixture(scope="module")
def family():
    torch_rows = run_family()
    return {row["name"]: row for row in torch_rows}


def _metrics(row):
    return {
        "cloud_ms": row["cloud_ms"],
        "residual_l2": row["residual_l2"],
        "anchor_drop": row["anchor_drop"],
        "finite": row["finite"],
        "has_cloud": row["has_cloud"],
        "bcap_faithful": row["bcap_faithful"],
    }


def test_locked_tiny_passes(family):
    row = family["locked_tiny"]
    assert row["passed"], row["reasons"]
    assert row["n_particles"] == 12
    assert row["particle_l2"] == pytest.approx(DEMO_L2)
    assert row["cover_weight"] == pytest.approx(1.5)
    assert row["cloud_ms"] <= CLOUD_MS_MAX
    assert row["residual_l2"] <= 0.05
    assert row["anchor_drop"] > 0.0
    assert row["routing"] == "none"
    assert row["critic_hidden"] == family["hub128_routed"]["critic_hidden"]


def test_music_parts0_passes(family):
    row = family["music_parts0"]
    assert row["passed"], row["reasons"]
    assert row["n_particles"] == 0
    assert row["has_cloud"] is False
    assert row["cover_weight"] == pytest.approx(1.0)
    assert row["vicreg_weight"] == pytest.approx(0.0)
    assert row["particle_l2"] == pytest.approx(0.0)
    assert row["residual_l2"] <= 0.05


def test_hub128_routed_cloud_fails(family):
    row = family["hub128_routed"]
    assert row["passed"] is False
    assert row["n_particles"] == 128
    assert row["cloud_ms"] > 0.2
    text = " ".join(row["reasons"])
    assert "oversized" in text
    assert "noisy" in text
    assert "routing" in text


def test_zero_particle_l2_fails(family):
    row = family["zero_particle_l2"]
    assert row["passed"] is False
    assert row["anchor_drop"] == 0.0
    assert row["n_particles"] == 12
    assert any("particle_l2" in reason for reason in row["reasons"])
    assert any("anchor" in reason for reason in row["reasons"])


def test_adv_drift_is_refused_on_a_clean_cloud(family):
    metrics = _metrics(family["locked_tiny"])
    drifts = [
        replace(locked_tiny(), name="fm_on", fm_weight=0.1),
        replace(locked_tiny(), name="vanilla", gan_mode="vanilla"),
        replace(locked_tiny(), name="kappa", reg_kappa=2.0),
        replace(locked_tiny(), name="coeff", reg_coeff=0.0),
        replace(locked_tiny(), name="norm", grad_norm="l1"),
        replace(locked_tiny(), name="route", routing="all_examples"),
        replace(locked_tiny(), name="big", n_particles=64),
        replace(locked_tiny(), name="noisy", init_std=1.0),
        replace(locked_tiny(), name="cover_swap", cover_weight=1.0),
    ]
    for cfg in drifts:
        passed, reasons = evaluate(cfg, metrics)
        assert passed is False, cfg.name
        assert reasons, cfg.name


def test_covers_stay_on_their_rows(family):
    empty = _metrics(family["music_parts0"])
    passed, reasons = evaluate(replace(music_parts0(), cover_weight=1.5), empty)
    assert passed is False
    assert any("cover" in reason for reason in reasons)


def test_hardcoded_kappa_is_not_the_particlegan_penalty():
    def honest(kappa):
        return GradientPenalty(arm="b_cap", coeff=1.0, kappa=float(kappa), norm="l2")

    def thinned(kappa):
        return GradientPenalty(arm="b_cap", coeff=1.0, kappa=1.0, norm="l2")

    assert bcap_kappa_is_explicit(honest)
    assert bcap_kappa_is_explicit(thinned) is False


def test_leaderboard_names_the_pass_and_the_fail(family):
    text = render_leaderboard([family[name] for name in (
        "locked_tiny", "music_parts0", "hub128_routed", "zero_particle_l2",
    )])
    assert "**PASS**" in text
    assert "**FAIL**" in text
    assert "hub128_routed" in text
    assert "not a Music" in text or "not a claim" in text
