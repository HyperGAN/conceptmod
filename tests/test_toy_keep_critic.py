"""Keep-critic locked_shared: frozen host critic, no MLP swap.

PASS is the demo shape (RpGAN logistic, GradRegularizer b_cap coeff=1
κ=1 l2, FM off, cover 1.5, n=12) with the host critic unmoved.
Music cover 1.0 is a different posture. A forced mlp swap is refused.
This is not a Music or Anima transfer claim.
"""

from dataclasses import replace

import pytest

from particlegan.grad_regularizers import GradRegularizer

from conceptmod.toys.keep_critic import (
    DEMO_COVER,
    HOST_ARCH,
    LOCKED,
    MUSIC_COVER,
    HostCritic,
    KeepCriticError,
    ThinnedKappaCap,
    leaderboard,
    run_keep,
    score_keep,
)
from conceptmod.toys.mlp import SimpleMLPDiscriminator


def test_locked_shared_passes_with_frozen_host_critic():
    score = score_keep(run_keep(LOCKED))
    assert score["verdict"] == "PASS", score["mismatches"]
    assert score["adv_abs_err"] < 1e-5
    assert score["g_abs_err"] < 1e-5
    assert score["cap_abs_err"] < 1e-5
    assert score["kappa_probe_abs_err"] < 1e-5
    assert score["reg_class"] == "GradRegularizer"
    assert score["fm_weight"] == 0.0
    assert score["cover_weight"] == DEMO_COVER
    assert score["cover_posture"] == "demo"
    assert score["demo_cover"] == 1.5
    assert score["music_cover"] == 1.0
    assert score["gan_mode"] == "rp"
    assert score["loss_type"] == "logistic"
    assert score["n_particles"] == 12
    assert score["critic_arch"] == HOST_ARCH
    assert score["critic_frozen"] is True
    assert score["critic_weight_delta"] == 0.0
    assert score["refused"] is False
    assert score["music_gpu_transfer"] is False
    assert score["anima_gpu_transfer"] is False
    assert score["cpu_toy"] is True


def test_host_critic_is_not_the_ring_mlp():
    critic = HostCritic()
    assert not isinstance(critic, SimpleMLPDiscriminator)
    for param in critic.parameters():
        assert param.requires_grad is False


def test_stranger_pairing_fails():
    score = score_keep(run_keep(LOCKED, pairing="stranger"))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("pairing") for item in score["mismatches"])
    assert score["adv_abs_err"] > 1e-5
    assert score["critic_frozen"] is True


def test_fm_on_fails():
    score = score_keep(run_keep(replace(LOCKED, fm_weight=0.1)))
    assert score["verdict"] == "FAIL"
    assert any(item.startswith("fm_weight") for item in score["mismatches"])
    assert score["g_abs_err"] > 1e-5


def test_thinned_kappa_fails():
    score = score_keep(run_keep(LOCKED, regularizer="thinned"))
    assert score["verdict"] == "FAIL"
    assert score["reg_class"] == "ThinnedKappaCap"
    assert score["kappa_probe_abs_err"] > 1e-3
    assert any("kappa" in item or "GradRegularizer" in item for item in score["mismatches"])
    # At locked κ=1 the hardcoded center still matches the step penalty.
    assert score["cap_abs_err"] < 1e-5
    assert score["critic_frozen"] is True


def test_music_cover_is_a_different_posture():
    cfg = replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music")
    score = score_keep(run_keep(cfg))
    assert score["verdict"] == "FAIL"
    assert score["cover_posture"] == "music"
    assert score["cover_weight"] == 1.0
    assert score["demo_cover"] == 1.5
    assert score["music_cover"] == 1.0
    assert any(item.startswith("cover_weight") for item in score["mismatches"])
    assert any(item.startswith("cover_posture") for item in score["mismatches"])


def test_stepping_the_host_critic_fails():
    score = score_keep(run_keep(replace(LOCKED, critic_frozen=False)))
    assert score["verdict"] == "FAIL"
    assert score["critic_frozen"] is False
    assert score["critic_weight_delta"] > 0.0
    assert score["critic_arch"] == HOST_ARCH
    assert any(item.startswith("critic_frozen") or item.startswith("critic_moved") for item in score["mismatches"])


def test_forced_mlp_swap_is_refused():
    with pytest.raises(KeepCriticError, match="mlp"):
        run_keep(replace(LOCKED, critic_arch="mlp"))


def test_thinned_class_ignores_kappa_argument():
    faithful = GradRegularizer(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    thinned = ThinnedKappaCap(arm="b_cap", coeff=1.0, kappa=0.2, norm="l2")
    assert thinned.kappa == 0.2
    assert thinned.center(0) == 1.0
    assert faithful.center(0) == 0.2


def test_board_locked_first_and_drifts_fail():
    rows = leaderboard()
    assert rows[0]["arm"] == "locked_shared"
    assert rows[0]["verdict"] == "PASS"
    by_name = {row["arm"]: row for row in rows}
    assert set(by_name) >= {
        "locked_shared",
        "stranger_pairing",
        "fm_on",
        "thinned_kappa",
        "music_cover_1",
        "critic_step",
        "forced_mlp_swap",
    }
    for name in (
        "stranger_pairing",
        "fm_on",
        "thinned_kappa",
        "music_cover_1",
        "critic_step",
        "forced_mlp_swap",
    ):
        assert by_name[name]["verdict"] == "FAIL", name
    assert by_name["forced_mlp_swap"]["refused"] is True
    assert by_name["forced_mlp_swap"]["critic_arch"] == "mlp"
    assert all(row["music_gpu_transfer"] is False for row in rows)
    assert all(row["anima_gpu_transfer"] is False for row in rows)
    assert all(row["cpu_toy"] is True for row in rows)
    assert all(row["demo_cover"] == 1.5 and row["music_cover"] == 1.0 for row in rows)
