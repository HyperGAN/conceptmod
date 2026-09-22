"""Leaderboard honesty: PASS needs locked_shared and a failing bad arm.

CPU toy. No Music or Anima GPU claim. Tail the same lines with::

    python -m conceptmod.toys.leaderboard_honesty
"""

from __future__ import annotations

import pytest
import torch

from particlegan.grad_regularizers import GradRegularizer
from conceptmod.toys.leaderboard_honesty import (
    GRAD_MED_MAX,
    LOCKED_SHARED,
    TRAVEL_MIN,
    CellResult,
    HonestyError,
    claim_pass,
    demo_arms,
    make_arm,
    make_regularizer,
    run_cell,
    run_honesty_board,
)

_BOARD = {}


def board():
    if "board" not in _BOARD:
        _BOARD["board"] = run_honesty_board(demo_arms())
    return _BOARD["board"]


def cells():
    return {cell.name: cell for cell in board()["cells"]}


def _cell(**overrides) -> CellResult:
    base = dict(
        name="locked_shared",
        role="locked_shared",
        toy="two_pole_cloud",
        won=True,
        mean_abs=0.5,
        grad_med=0.4,
        nearest=0.5,
        cover_score=0.75,
        order=1,
        steps=80,
        seed=0,
        drift=(),
    )
    base.update(overrides)
    return CellResult(**base)


# -- harness: PASS is refused without a real negative ----------------------


def test_claim_pass_without_a_negative_errors():
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_pass(winner=_cell(), negatives=[])
    with pytest.raises(HonestyError, match="without a declared negative"):
        run_honesty_board((demo_arms()[0],))


def test_first_cell_must_be_locked_shared():
    stranger = demo_arms()[1]
    with pytest.raises(HonestyError, match="FIRST"):
        run_honesty_board((stranger, demo_arms()[0]))


def test_claim_pass_rejects_a_bad_arm_that_won():
    winner = _cell()
    fake_negative = _cell(name="not_bad", role="negative", won=True, order=2)
    with pytest.raises(HonestyError, match="did not fail"):
        claim_pass(winner=winner, negatives=[fake_negative])


def test_claim_pass_rejects_a_locked_loss():
    winner = _cell(won=False)
    negative = _cell(name="stranger_pairing", role="negative", won=False, order=2)
    with pytest.raises(HonestyError, match="did not win"):
        claim_pass(winner=winner, negatives=[negative])


def test_claim_pass_rejects_a_negative_from_another_toy():
    negative = _cell(name="other", role="negative", won=False, toy="other_toy", order=2)
    with pytest.raises(HonestyError, match="same toy"):
        claim_pass(winner=_cell(), negatives=[negative])


def test_unreported_drift_is_refused():
    with pytest.raises(HonestyError, match="without reported drift"):
        make_arm("fm_on", "negative", fm_weight=1.0)
    with pytest.raises(HonestyError, match="without reported drift"):
        make_arm("vanilla", "negative", gan_mode="vanilla")
    with pytest.raises(HonestyError, match="does not match"):
        make_arm(
            "stranger_pairing",
            "negative",
            reported_drift={"fm_weight": 1.0},
            pairing="stranger",
        )


def test_locked_shared_cannot_silently_drift():
    with pytest.raises(HonestyError, match="cannot drift"):
        make_arm("locked_shared", "locked_shared", fm_weight=1.0)


def test_critic_swap_is_refused():
    with pytest.raises(HonestyError, match="critic"):
        make_arm(
            "mlp_swap",
            "negative",
            reported_drift={"critic_arch": "mlp_swap"},
            critic_arch="mlp_swap",
        )


def test_hub_particle_count_is_refused():
    arm = make_arm(
        "hub_gmix",
        "negative",
        reported_drift={"n_particles": 128},
        n_particles=128,
    )
    with pytest.raises(HonestyError, match="n=12"):
        run_cell(arm, steps=1, seed=0, order=2)


def test_reported_non_rp_recipe_is_not_scored_here():
    arm = make_arm(
        "vanilla",
        "negative",
        reported_drift={"gan_mode": "vanilla"},
        gan_mode="vanilla",
    )
    with pytest.raises(HonestyError, match="RpGAN"):
        run_cell(arm, steps=1, seed=0, order=2)


def test_locked_regularizer_is_the_real_b_cap():
    reg = make_regularizer(LOCKED_SHARED)
    assert type(reg) is GradRegularizer
    assert reg.arm == "b_cap"
    assert reg.coeff == pytest.approx(1.0)
    assert reg.kappa == pytest.approx(1.0)
    assert reg.norm == "l2"
    norms = torch.tensor([0.5, 1.0, 2.0])
    assert torch.allclose(reg._phi(norms, center=1.0), torch.relu(norms - 1).pow(2))
    stub = make_regularizer({**LOCKED_SHARED, "reg_impl": "thinned_hardcoded_kappa"})
    assert type(stub) is not GradRegularizer
    assert stub.requested_kappa == pytest.approx(1.0)
    assert stub.kappa == pytest.approx(100.0)


# -- demo toy: locked_shared wins, both declared bad arms fail --------------


def test_locked_shared_wins_and_bad_arms_fail():
    result = board()
    verdict = result["verdict"]
    assert verdict["verdict"] == "PASS"
    assert verdict["winner"] == "locked_shared"
    assert verdict["negatives"] == ["stranger_pairing", "thinned_b_cap"]
    assert verdict["music_gpu_transfer"] is False
    assert verdict["anima_gpu_transfer"] is False
    assert verdict["cpu_toy"] is True
    assert verdict["cover_posture"] == "demo_1.5"

    locked = cells()["locked_shared"]
    stranger = cells()["stranger_pairing"]
    thinned = cells()["thinned_b_cap"]
    assert [cell.order for cell in result["cells"]] == [1, 2, 3]
    assert locked.won is True
    assert locked.mean_abs >= TRAVEL_MIN
    assert locked.grad_med <= GRAD_MED_MAX
    assert locked.mean_abs == pytest.approx(0.5144, abs=0.02)
    assert locked.grad_med == pytest.approx(0.4197, abs=0.02)
    assert locked.drift == ()

    assert stranger.won is False
    assert stranger.mean_abs == pytest.approx(0.0, abs=1e-6)
    assert stranger.grad_med <= GRAD_MED_MAX

    assert thinned.won is False
    assert thinned.mean_abs >= TRAVEL_MIN
    assert thinned.grad_med > GRAD_MED_MAX
    # Higher logged cover than locked_shared, and still a fail.
    assert thinned.cover_score > locked.cover_score


def test_pass_from_the_measured_cells_still_needs_the_negatives():
    measured = board()["cells"]
    verdict = claim_pass(winner=measured[0], negatives=measured[1:])
    assert verdict["verdict"] == "PASS"
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_pass(winner=measured[0], negatives=[])
