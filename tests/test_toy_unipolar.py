"""Unipolar residual toy: locked RpGAN+b_cap passes; MSE-only and polarity flip fail.

CPU only. One seed (the leaderboard seed). Not a Music/Anima transfer claim.
"""
import pytest
import torch

from conceptmod.toys import unipolar as uni


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_locked_rpgan_passes_unipolar_gates(monkeypatch):
    def no_mse(*_args, **_kwargs):
        raise AssertionError("locked GAN must not call MSE")

    monkeypatch.setattr(torch.nn.functional, "mse_loss", no_mse)
    seen = []
    original = uni.GradientPenalty.penalty

    def spy(self, *args, **kwargs):
        seen.append((self.arm, self.coeff, self.kappa, self.norm, self.lazy_k, self.target_anneal))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(uni.GradientPenalty, "penalty", spy)
    row = uni.run_arm("locked_rpgan", steps=400, seed=0)
    assert row["hit"]
    assert row["cover"] >= uni.PLUS_COVER_MIN
    assert row["off_caption"] <= uni.PLUS_OFF_MAX
    assert row["neu_hold"] >= uni.PLUS_NEU_HOLD_MIN
    assert row["cos_plus"] > 0.0
    assert row["fm_weight"] == 0.0
    assert row["cover_weight"] == 0.0
    assert row["gan_mode"] == "rp"
    assert row["loss_type"] == "logistic"
    assert row["reg_is_gradient_penalty"]
    assert seen, "b_cap penalty never ran"
    assert set(seen) == {("b_cap", 1.0, 1.0, "l2", 1, "none")}
    # Scale -1 is logged and is not part of the hit.
    assert row["canary"]["scored"] is False
    assert "cover_weight" in row["shape"]["intentional_drift"]


def test_mse_only_fails_neutral_hold():
    """Plus-only MSE matches +1 and still fails neu_hold (origin takes a third)."""
    row = uni.run_arm("mse_only", steps=400, seed=0)
    assert row["hit"] is False
    assert row["cover"] >= uni.PLUS_COVER_MIN
    assert row["off_caption"] <= uni.PLUS_OFF_MAX
    assert row["neu_hold"] < uni.PLUS_NEU_HOLD_MIN
    assert row["neu_hold"] == pytest.approx(2.0 / 3.0, abs=0.02)
    assert row["origin_norm"] == pytest.approx(row["odd_norm"], abs=1e-4)
    assert row["origin_norm"] == pytest.approx(row["even_norm"], abs=1e-4)
    assert row["reg_is_gradient_penalty"] is False


def test_polarity_flipped_fails_plus_cover():
    """Same locked GAN shape, +1 real replaced by the minus pole."""
    row = uni.run_arm("polarity_flipped", steps=400, seed=0)
    assert row["hit"] is False
    assert row["cover"] < uni.PLUS_COVER_MIN
    assert row["cos_plus"] < 0.0
    assert row["gan_mode"] == "rp"
    assert row["loss_type"] == "logistic"
    assert row["reg_arm"] == "b_cap"
    assert row["reg_coeff"] == 1.0
    assert row["reg_kappa"] == 1.0
    assert row["reg_norm"] == "l2"
    assert row["fm_weight"] == 0.0
    assert row["cover_weight"] == 0.0
    assert row["neu_hold"] >= uni.PLUS_NEU_HOLD_MIN


def test_locked_shape_refuses_fm_stranger_pairing_and_thinned_cap():
    with pytest.raises(ValueError, match="drift"):
        uni.run_arm("locked_rpgan", fm_weight=1.0)
    with pytest.raises(ValueError, match="drift"):
        uni.run_arm("locked_rpgan", gan_mode="vanilla")
    with pytest.raises(ValueError, match="drift"):
        uni.run_arm("locked_rpgan", reg_kappa=0.5)
    with pytest.raises(ValueError, match="drift"):
        uni.run_arm("locked_rpgan", reg_norm="l1")
    with pytest.raises(ValueError, match="drift"):
        uni.run_arm("locked_rpgan", cover_weight=uni.DEMO_LOCKED_COVER)
    with pytest.raises(ValueError, match="refuses"):
        uni.locked_recipe(fm_weight=1.0)
