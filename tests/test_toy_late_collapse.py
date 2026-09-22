"""Late-collapse selection: keep the earlier good checkpoint.

The synthetic curve is good at step 3 and collapsed at the last step.
Train loss keeps falling, so last-step and best-train-loss export the collapse.
The val gate does not.

CPU toy. No Music or Anima GPU claim. Tail the same lines with::

    python -m conceptmod.toys.late_collapse
"""

from __future__ import annotations

import pytest

from conceptmod.toys.leaderboard_honesty import HonestyError as BoardHonestyError
from conceptmod.toys.late_collapse import (
    GOOD_STEP,
    LATE_STEP,
    TOY_ID,
    VAL_GATE_MAX,
    Checkpoint,
    Export,
    HonestyError,
    assert_locked_posture,
    claim_selection_pass,
    curve_holds_lesson,
    demo_arms,
    make_arm,
    run_arm,
    run_selection_board,
    select_best_train,
    select_last_step,
    select_val_gate,
    synthetic_curve,
)

_BOARD = {}


def board():
    if "board" not in _BOARD:
        _BOARD["board"] = run_selection_board(demo_arms())
    return _BOARD["board"]


def exports():
    return {row.name: row for row in board()["exports"]}


def _export(**overrides) -> Export:
    base = dict(
        name="locked_val_gate",
        role="locked_shared",
        toy=TOY_ID,
        rule="val_gate",
        step=GOOD_STEP,
        train_loss=0.32,
        val_error=0.06,
        collapsed=False,
        won=True,
        order=1,
    )
    base.update(overrides)
    return Export(**base)


def _negative(**overrides) -> Export:
    base = dict(
        name="last_step",
        role="negative",
        toy=TOY_ID,
        rule="last_step",
        step=LATE_STEP,
        train_loss=0.01,
        val_error=1.40,
        collapsed=True,
        won=False,
        order=2,
    )
    base.update(overrides)
    return Export(**base)


# -- curve: early good, late collapsed, train loss blind --------------------


def test_honesty_error_is_the_leaderboard_type():
    assert HonestyError is BoardHonestyError


def test_curve_is_early_good_and_late_collapsed():
    curve = synthetic_curve()
    curve_holds_lesson(curve)
    assert [row.step for row in curve] == list(range(8))
    trains = [row.train_loss for row in curve]
    assert trains == sorted(trains, reverse=True)
    assert all(earlier > later for earlier, later in zip(trains, trains[1:]))

    good = curve[GOOD_STEP]
    late = curve[LATE_STEP]
    assert good.collapsed is False
    assert good.val_error <= VAL_GATE_MAX
    assert good.val_error == min(row.val_error for row in curve)
    assert late.collapsed is True
    assert late.val_error > VAL_GATE_MAX
    assert late.train_loss == min(trains)
    # Training continued one in-gate step past the export, then collapsed.
    assert curve[GOOD_STEP + 1].collapsed is False
    assert curve[GOOD_STEP + 1].val_error > good.val_error
    assert curve[GOOD_STEP + 2].collapsed is True


def test_locked_posture_is_the_shared_stamp():
    assert_locked_posture()


# -- selectors --------------------------------------------------------------


def test_val_gate_exports_the_earlier_good_checkpoint():
    picked = select_val_gate(synthetic_curve())
    assert picked.step == GOOD_STEP
    assert picked.train_loss == 0.32
    assert picked.val_error == 0.06
    assert picked.collapsed is False


def test_val_gate_refuses_a_collapsed_point_with_a_forged_best_val():
    curve = list(synthetic_curve())
    late = curve[-1]
    curve[-1] = Checkpoint(
        step=late.step,
        train_loss=late.train_loss,
        val_error=0.01,
        collapsed=True,
    )
    picked = select_val_gate(curve)
    assert picked.step == GOOD_STEP
    assert select_last_step(curve).step == LATE_STEP
    assert select_last_step(curve).val_error == 0.01
    # Best train loss still ignores the forged val and the collapsed flag.
    assert select_best_train(curve).step == LATE_STEP
    assert select_best_train(curve).collapsed is True


def test_gate_blind_rules_read_only_their_column():
    # Lowest train loss sits on the worst val. Lowest val sits on the worst train.
    curve = (
        Checkpoint(step=0, train_loss=0.10, val_error=0.90, collapsed=True),
        Checkpoint(step=1, train_loss=0.50, val_error=0.04, collapsed=False),
    )
    assert select_best_train(curve).step == 0
    assert select_last_step(curve).step == 1
    assert select_val_gate(curve).step == 1


def test_empty_curve_is_refused():
    with pytest.raises(HonestyError, match="empty curve"):
        select_val_gate(())
    with pytest.raises(HonestyError, match="empty curve"):
        select_last_step(())
    with pytest.raises(HonestyError, match="empty curve"):
        select_best_train(())


def test_bad_arm_cannot_silently_use_the_val_gate():
    with pytest.raises(HonestyError, match="cannot use the locked val gate"):
        make_arm("sneaky", "negative", "val_gate")
    with pytest.raises(HonestyError, match="val gate only"):
        make_arm("locked_val_gate", "locked_shared", "last_step")
    with pytest.raises(HonestyError, match="unknown selection rule"):
        make_arm("other", "negative", "best_val_without_refusal")


# -- board: locked passes, both gate-blind arms fail ------------------------


def test_locked_rule_passes_and_blind_arms_fail():
    result = board()
    verdict = result["verdict"]
    assert verdict["verdict"] == "PASS"
    assert verdict["winner"] == "locked_val_gate"
    assert verdict["exported_step"] == GOOD_STEP
    assert verdict["exported_train_loss"] == 0.32
    assert verdict["exported_val_error"] == 0.06
    assert verdict["negatives"] == ["last_step", "best_train_loss"]
    assert verdict["toy"] == TOY_ID
    assert verdict["music_gpu_transfer"] is False
    assert verdict["anima_gpu_transfer"] is False
    assert verdict["cpu_toy"] is True
    assert verdict["cover_posture"] == "demo_1.5"

    locked = exports()["locked_val_gate"]
    last = exports()["last_step"]
    best_train = exports()["best_train_loss"]
    assert [row.order for row in result["exports"]] == [1, 2, 3]
    assert locked.won is True
    assert locked.rule == "val_gate"
    assert locked.step == GOOD_STEP
    assert locked.collapsed is False

    assert last.won is False
    assert last.step == LATE_STEP
    assert last.collapsed is True
    assert last.train_loss == 0.01
    assert last.val_error == 1.40

    assert best_train.won is False
    assert best_train.step == LATE_STEP
    assert best_train.collapsed is True
    assert best_train.train_loss == last.train_loss
    assert best_train.val_error == last.val_error
    assert best_train.step != locked.step


def test_pass_from_the_measured_exports_still_needs_the_negatives():
    measured = board()["exports"]
    verdict = claim_selection_pass(winner=measured[0], negatives=measured[1:])
    assert verdict["verdict"] == "PASS"
    assert verdict["exported_step"] == GOOD_STEP
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_selection_pass(winner=measured[0], negatives=[])


# -- honesty: empty negatives and a dishonest board -------------------------


def test_claim_without_a_negative_errors():
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_selection_pass(winner=_export(), negatives=[])
    with pytest.raises(HonestyError, match="without a declared negative"):
        run_selection_board((demo_arms()[0],))


def test_first_cell_must_be_the_locked_val_gate():
    last, best = demo_arms()[1], demo_arms()[2]
    with pytest.raises(HonestyError, match="FIRST"):
        run_selection_board((last, demo_arms()[0], best))


def test_claim_rejects_a_bad_arm_that_exported_the_good_checkpoint():
    with pytest.raises(HonestyError, match="did not fail"):
        claim_selection_pass(
            winner=_export(),
            negatives=[_negative(step=GOOD_STEP, val_error=0.06, collapsed=False, won=True)],
        )


def test_claim_rejects_the_collapsed_late_pick_as_the_winner():
    late = _export(
        step=LATE_STEP,
        train_loss=0.01,
        val_error=1.40,
        collapsed=True,
        won=False,
    )
    with pytest.raises(HonestyError, match="collapsed late pick"):
        claim_selection_pass(winner=late, negatives=[_negative(name="best_train_loss", order=2)])


def test_dishonest_won_flag_errors():
    with pytest.raises(HonestyError, match="dishonest board"):
        claim_selection_pass(winner=_export(won=False), negatives=[_negative()])
    with pytest.raises(HonestyError, match="dishonest board"):
        claim_selection_pass(
            winner=_export(),
            negatives=[_negative(won=True)],
        )


def test_claim_rejects_a_negative_that_missed_the_collapse():
    in_gate_but_late = _negative(step=4, train_loss=0.20, val_error=0.11, collapsed=False)
    with pytest.raises(HonestyError, match="collapsed late pick"):
        claim_selection_pass(winner=_export(), negatives=[in_gate_but_late])


def test_two_pole_claim_pass_stays_closed():
    """This toy does not extend claim_pass. The two-pole gate still refuses it."""
    from conceptmod.toys.leaderboard_honesty import CellResult, claim_pass

    winner = CellResult(
        name="locked_shared",
        role="locked_shared",
        toy=TOY_ID,
        won=True,
        mean_abs=0.5,
        grad_med=0.4,
        nearest=0.5,
        cover_score=0.75,
        order=1,
        steps=8,
        seed=0,
    )
    negative = CellResult(
        name="last_step",
        role="negative",
        toy=TOY_ID,
        won=False,
        mean_abs=0.0,
        grad_med=0.0,
        nearest=1.0,
        cover_score=0.0,
        order=2,
        steps=8,
        seed=0,
    )
    with pytest.raises(HonestyError, match="two_pole_cloud"):
        claim_pass(winner=winner, negatives=[negative])
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_pass(winner=winner, negatives=[])


def test_claim_rejects_a_negative_from_another_toy():
    with pytest.raises(HonestyError, match="same toy"):
        claim_selection_pass(
            winner=_export(),
            negatives=[_negative(toy="two_pole_cloud")],
        )


def test_dishonest_curve_cannot_claim_pass():
    # Train loss still falls, but the last step is the best val, so there is
    # no late collapse for the gate to refuse.
    flat = []
    for step, train, _val in (
        (0, 1.20, 0.90),
        (1, 0.80, 0.40),
        (2, 0.50, 0.18),
        (3, 0.32, 0.06),
        (4, 0.20, 0.05),
        (5, 0.11, 0.04),
        (6, 0.05, 0.03),
        (7, 0.01, 0.02),
    ):
        flat.append(
            Checkpoint(
                step=step,
                train_loss=train,
                val_error=_val,
                collapsed=_val > VAL_GATE_MAX,
            )
        )
    with pytest.raises(HonestyError, match="dishonest curve"):
        run_selection_board(demo_arms(), curve=tuple(flat))


def test_run_arm_marks_only_the_good_export_won():
    locked = run_arm(demo_arms()[0], order=1)
    last = run_arm(demo_arms()[1], order=2)
    assert locked.won is True and locked.step == GOOD_STEP
    assert last.won is False and last.collapsed is True
