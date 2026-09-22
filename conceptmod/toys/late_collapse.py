"""Late-collapse selection honesty (Lunar #23).

Training continues past a good checkpoint. On the synthetic curve the early
window is good and the late checkpoints are collapsed. Train loss keeps
falling, so last-step and best-train-loss (no val gate) both export the
collapsed late pick. The locked rule refuses that pick and exports the
earlier good checkpoint.

The curve is the whole toy. It does not run Adam, and it does not retune
the locked_shared adv stamp (RpGAN logistic, ``b_cap`` coeff=1 kappa=1,
FM off, demo cover 1.5, n=12, ``particle_l2=0.02``).

``claim_selection_pass`` follows :func:`conceptmod.toys.leaderboard_honesty.claim_pass`:
an empty negative list raises :class:`HonestyError`, and a won flag that
disagrees with the export is a dishonest board. ``claim_pass`` itself stays
the two-pole gate; this module does not loosen it.

CPU only. A PASS here does not transfer to Music or Anima.

Tail the log with::

    python -m conceptmod.toys.late_collapse
"""

from __future__ import annotations

from dataclasses import dataclass

from conceptmod.toys.leaderboard_honesty import LOCKED_SHARED, HonestyError

TOY_ID = "late_collapse_selection"
VAL_GATE_MAX = 0.25
GOOD_STEP = 3
LATE_STEP = 7

# (step, train_loss, val_error). Train loss is strictly decreasing, so the
# best train loss is the last step. Val error bottoms at GOOD_STEP, stays
# inside the gate for one more step, then collapses.
_RAW_CURVE = (
    (0, 1.20, 0.90),
    (1, 0.80, 0.40),
    (2, 0.50, 0.18),
    (3, 0.32, 0.06),
    (4, 0.20, 0.11),
    (5, 0.11, 0.62),
    (6, 0.05, 0.95),
    (7, 0.01, 1.40),
)

_LOCKED_POSTURE = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "fm_weight": 0.0,
    "cover_weight": 1.5,
    "n_particles": 12,
    "particle_l2": 0.02,
}

_RULES = ("val_gate", "last_step", "best_train_loss")


@dataclass(frozen=True)
class Checkpoint:
    step: int
    train_loss: float
    val_error: float
    collapsed: bool


@dataclass(frozen=True)
class Arm:
    name: str
    role: str
    rule: str


@dataclass(frozen=True)
class Export:
    name: str
    role: str
    toy: str
    rule: str
    step: int
    train_loss: float
    val_error: float
    collapsed: bool
    won: bool
    order: int


def synthetic_curve() -> tuple[Checkpoint, ...]:
    """Early good, late collapsed. Train loss never sees the collapse."""
    rows = []
    for step, train_loss, val_error in _RAW_CURVE:
        rows.append(
            Checkpoint(
                step=step,
                train_loss=train_loss,
                val_error=val_error,
                collapsed=val_error > VAL_GATE_MAX,
            )
        )
    return tuple(rows)


def curve_holds_lesson(curve: tuple[Checkpoint, ...] | list[Checkpoint]) -> None:
    """Refuse a curve that no longer separates the good export from the late collapse."""
    rows = tuple(curve)
    if len(rows) < 2:
        raise HonestyError("dishonest curve: late collapse needs more than one checkpoint")
    steps = [row.step for row in rows]
    if steps != list(range(len(rows))):
        raise HonestyError("dishonest curve: steps must be 0..n-1 in order")
    trains = [row.train_loss for row in rows]
    if any(later >= earlier for earlier, later in zip(trains, trains[1:])):
        raise HonestyError("dishonest curve: train loss must keep falling into the collapse")
    last = rows[-1]
    if last.step != LATE_STEP or not last.collapsed or last.val_error <= VAL_GATE_MAX:
        raise HonestyError("dishonest curve: late collapse is not the last pick")
    if min(rows, key=lambda row: (row.train_loss, row.step)).step != last.step:
        raise HonestyError("dishonest curve: best train loss is not the collapsed late pick")
    if GOOD_STEP >= len(rows):
        raise HonestyError("dishonest curve: good checkpoint is missing")
    good = rows[GOOD_STEP]
    if good.collapsed or good.val_error > VAL_GATE_MAX:
        raise HonestyError("dishonest curve: the early checkpoint is not inside the val gate")
    eligible = [row for row in rows if _eligible(row)]
    if not eligible:
        raise HonestyError("dishonest curve: val gate refused every checkpoint")
    best = min(eligible, key=lambda row: (row.val_error, row.step))
    if best.step != GOOD_STEP:
        raise HonestyError("dishonest curve: val gate would not keep the earlier good checkpoint")
    later_in_gate = [row for row in eligible if row.step > GOOD_STEP]
    if not later_in_gate or any(row.val_error <= good.val_error for row in later_in_gate):
        raise HonestyError("dishonest curve: training did not continue past the good checkpoint")
    if not any(row.collapsed and row.step > GOOD_STEP for row in rows):
        raise HonestyError("dishonest curve: nothing collapsed after the good checkpoint")


def assert_locked_posture() -> None:
    """This toy reads the locked_shared stamp and does not retune it."""
    for key, value in _LOCKED_POSTURE.items():
        if key not in LOCKED_SHARED or LOCKED_SHARED[key] != value:
            found = LOCKED_SHARED.get(key, "<missing>")
            raise HonestyError(f"locked_shared stamp drifted on {key}: {found!r}")


def _eligible(row: Checkpoint) -> bool:
    return (not row.collapsed) and row.val_error <= VAL_GATE_MAX


def select_val_gate(curve: tuple[Checkpoint, ...] | list[Checkpoint]) -> Checkpoint:
    """Min val error among checkpoints the gate has not refused.

    A collapsed flag is refused even when its val error was forged downward.
    Ties keep the earlier step.
    """
    rows = tuple(curve)
    if not rows:
        raise HonestyError("empty curve")
    alive = [row for row in rows if _eligible(row)]
    if not alive:
        raise HonestyError("val gate refused every checkpoint; no good export")
    return min(alive, key=lambda row: (row.val_error, row.step))


def select_last_step(curve: tuple[Checkpoint, ...] | list[Checkpoint]) -> Checkpoint:
    """Final index. Val error and the collapsed flag are not read."""
    rows = tuple(curve)
    if not rows:
        raise HonestyError("empty curve")
    return rows[-1]


def select_best_train(curve: tuple[Checkpoint, ...] | list[Checkpoint]) -> Checkpoint:
    """Argmin train loss. Val error and the collapsed flag are not keys."""
    rows = tuple(curve)
    if not rows:
        raise HonestyError("empty curve")
    return min(rows, key=lambda row: (row.train_loss, row.step))


_SELECTORS = {
    "val_gate": select_val_gate,
    "last_step": select_last_step,
    "best_train_loss": select_best_train,
}


def make_arm(name: str, role: str, rule: str) -> Arm:
    """Locked cell is the val gate. A bad arm has to name a gate-blind rule."""
    if role not in ("locked_shared", "negative"):
        raise HonestyError("role must be locked_shared or negative")
    if rule not in _RULES:
        raise HonestyError(f"unknown selection rule {rule!r}")
    if role == "locked_shared":
        if name != "locked_val_gate" or rule != "val_gate":
            raise HonestyError("locked_shared selection is the val gate only")
    elif rule == "val_gate":
        raise HonestyError("a declared bad arm cannot use the locked val gate")
    return Arm(name=name, role=role, rule=rule)


def _is_good_export(row: Checkpoint | Export) -> bool:
    return (
        row.step == GOOD_STEP
        and not row.collapsed
        and row.val_error <= VAL_GATE_MAX
    )


def run_arm(
    arm: Arm,
    curve: tuple[Checkpoint, ...] | list[Checkpoint] | None = None,
    *,
    order: int = 1,
) -> Export:
    """Apply one selection rule and score the export against the good checkpoint."""
    rows = synthetic_curve() if curve is None else tuple(curve)
    if not rows:
        raise HonestyError("empty curve")
    picked = _SELECTORS[arm.rule](rows)
    good = _is_good_export(picked)
    return Export(
        name=arm.name,
        role=arm.role,
        toy=TOY_ID,
        rule=arm.rule,
        step=picked.step,
        train_loss=picked.train_loss,
        val_error=picked.val_error,
        collapsed=picked.collapsed,
        won=good,
        order=order,
    )


def log_export(row: Export) -> None:
    print(
        f"late_collapse order={row.order} arm={row.name} role={row.role} "
        f"rule={row.rule} won={int(row.won)} step={row.step} "
        f"train_loss={row.train_loss:.4f} val_error={row.val_error:.4f} "
        f"collapsed={int(row.collapsed)}",
        flush=True,
    )


def claim_selection_pass(
    *,
    winner: Export,
    negatives: list[Export] | tuple[Export, ...],
) -> dict:
    """PASS only when the val gate exported the good checkpoint and every bad arm collapsed.

    An empty negative list is an error. A won flag that disagrees with the
    measured export is a dishonest board.
    """
    if not negatives:
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    if winner.role != "locked_shared" or winner.name != "locked_val_gate":
        raise HonestyError("PASS requires the winning cell to be locked_val_gate")
    if winner.toy != TOY_ID:
        raise HonestyError("PASS is only defined for the late_collapse_selection toy")
    if winner.rule != "val_gate":
        raise HonestyError("locked export did not use the val gate; refusing PASS")
    if winner.won != _is_good_export(winner):
        raise HonestyError("dishonest board: won flag does not match the export")
    if winner.collapsed:
        raise HonestyError("locked export is the collapsed late pick; refusing PASS")
    if not _is_good_export(winner) or not winner.won:
        raise HonestyError(
            "locked selection did not export the good checkpoint; refusing PASS"
        )
    for neg in negatives:
        if neg.role != "negative":
            raise HonestyError("PASS negatives must be declared bad arms")
        if neg.toy != winner.toy:
            raise HonestyError("declared bad arm was not scored on the same toy")
        if neg.won != _is_good_export(neg):
            raise HonestyError("dishonest board: won flag does not match the export")
        if _is_good_export(neg) or neg.won:
            raise HonestyError(
                f"declared bad arm {neg.name!r} did not fail; refusing PASS"
            )
        if not neg.collapsed or neg.step == GOOD_STEP:
            raise HonestyError(
                f"declared bad arm {neg.name!r} did not export the collapsed late pick; "
                "refusing PASS"
            )
    return {
        "verdict": "PASS",
        "winner": winner.name,
        "exported_step": winner.step,
        "exported_train_loss": winner.train_loss,
        "exported_val_error": winner.val_error,
        "negatives": [neg.name for neg in negatives],
        "toy": TOY_ID,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
        "cover_posture": "demo_1.5",
    }


def log_verdict(verdict: dict) -> None:
    names = ",".join(verdict["negatives"])
    print(
        f"late_collapse verdict={verdict['verdict']} winner={verdict['winner']} "
        f"step={verdict['exported_step']} negatives={names} "
        f"music_gpu_transfer={int(verdict['music_gpu_transfer'])} "
        f"anima_gpu_transfer={int(verdict['anima_gpu_transfer'])}",
        flush=True,
    )


def demo_arms() -> tuple[Arm, Arm, Arm]:
    """Val gate first, then last-step and best-train-loss."""
    return (
        make_arm("locked_val_gate", "locked_shared", "val_gate"),
        make_arm("last_step", "negative", "last_step"),
        make_arm("best_train_loss", "negative", "best_train_loss"),
    )


def run_selection_board(
    arms: tuple[Arm, ...] | list[Arm],
    curve: tuple[Checkpoint, ...] | list[Checkpoint] | None = None,
) -> dict:
    """Run the locked val gate first. Refuse a curve or a board that hides the collapse."""
    assert_locked_posture()
    rows = synthetic_curve() if curve is None else tuple(curve)
    curve_holds_lesson(rows)
    if not arms or arms[0].role != "locked_shared" or arms[0].name != "locked_val_gate":
        raise HonestyError("FIRST cell must be locked_shared")
    if arms[0].rule != "val_gate":
        raise HonestyError("FIRST cell must use the val gate")
    if not any(arm.role == "negative" for arm in arms[1:]):
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    exports = []
    for order, arm in enumerate(arms, start=1):
        row = run_arm(arm, rows, order=order)
        log_export(row)
        exports.append(row)
    verdict = claim_selection_pass(winner=exports[0], negatives=exports[1:])
    log_verdict(verdict)
    return {"verdict": verdict, "exports": exports, "curve": rows}


def main() -> int:
    board = run_selection_board(demo_arms())
    return 0 if board["verdict"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
