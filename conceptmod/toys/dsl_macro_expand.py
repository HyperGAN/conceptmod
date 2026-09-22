"""Replace-macro (``~``) expansion honesty.

``a~b`` is a macro, not a fourth loss. ``docs/dsl.md`` and
``docs/dsl/operators/replace.md`` expand default ``λ = 0.1`` as

    b++:2λ | a=b:4λ | b%a:−λ

which for ``red~blue`` is exaggerate blue (``0.2``), write red toward
blue (``0.4``), and orthogonal ``blue%red`` with negative alpha
(``−0.1``). :func:`conceptmod.dsl.parse_phrase` is scored against that
written triple. The triple is not read back out of the parser.

On the 2-D fixture the macro is **recipe**. ``blue++`` and ``red=blue``
oppose each other, so a geometric ``right`` is a FAIL even when the
three rules match. An optional secondary row may record that
``loss_game("red~blue")`` stepped those rules under locked_shared.
That wiring PASS is expand-and-step honesty. It is not geometric right.

Named wrong expands FAIL: wrong op order, swapped ``a``/``b``, wrong
alphas, dropping ``%``, turning ``~`` into a single write, claiming
geometric ``right``, and inventing a live ``replace`` loss instead of
expanding.

``claim_expand_pass`` follows
:func:`conceptmod.toys.leaderboard_honesty.claim_pass`: an empty
negative list raises :class:`HonestyError`. ``claim_pass`` itself stays
the two-pole gate; this module does not loosen it.

CPU only. A PASS here is expand honesty (and, when the board asks, the
locked_shared game wiring). It is not a Music or Anima GPU transfer.

Tail the log with::

    python -m conceptmod.toys.dsl_macro_expand
"""

from __future__ import annotations

from dataclasses import dataclass

from conceptmod import dsl
from conceptmod.toys.leaderboard_honesty import HonestyError

TOY_ID = "dsl_macro_expand"

# Written contract from docs/dsl.md. Not dsl.DEFAULT_REPLACE_LAMBDA, so a
# drifted default fails the locked arm instead of moving the target.
SOURCE = "red"  # a: the prompt being replaced
TARGET = "blue"  # b: what should stand there
DOCUMENTED_LAMBDA = 0.1
PHRASE = f"{SOURCE}~{TARGET}"
GEOMETRIC_VERDICT = "recipe"
FORBIDDEN_GEOMETRIC = "right"

# rule_loss implements these. ``reward`` is parsed and raises.
# ``replace`` is not one of them.
LIVE_LOSS_OPS = frozenset({
    dsl.EXAGGERATE,
    dsl.ERASE,
    dsl.WRITE,
    dsl.FREEZE,
    dsl.ORTHOGONAL,
    dsl.PIXEL,
})

_KINDS = (
    "documented",
    "wrong_order",
    "swapped_roles",
    "wrong_alphas",
    "dropped_percent",
    "single_write",
    "geometric_right",
    "fourth_loss",
)

_NEGATIVE_KINDS = tuple(kind for kind in _KINDS if kind != "documented")


@dataclass(frozen=True)
class ExpandedRule:
    """One expanded rule. ``options`` is empty on the documented triple."""

    op: str
    a: str
    b: str
    alpha: float
    options: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Arm:
    name: str
    role: str
    kind: str


@dataclass(frozen=True)
class ExpandReport:
    name: str
    role: str
    toy: str
    kind: str
    rules: tuple[ExpandedRule, ...]
    text_rules: tuple[ExpandedRule, ...] | None
    geometric_verdict: str
    won: bool
    reasons: tuple[str, ...]
    order: int


def documented_triple(
    source: str = SOURCE,
    target: str = TARGET,
    lambda_value: float = DOCUMENTED_LAMBDA,
) -> tuple[ExpandedRule, ...]:
    """``b++:2λ | a=b:4λ | b%a:−λ`` from the scoreboard, not from the parser."""
    return (
        ExpandedRule(dsl.EXAGGERATE, target, "", 2 * lambda_value),
        ExpandedRule(dsl.WRITE, source, target, 4 * lambda_value),
        ExpandedRule(dsl.ORTHOGONAL, target, source, -lambda_value),
    )


def documented_text(
    source: str = SOURCE,
    target: str = TARGET,
    lambda_value: float = DOCUMENTED_LAMBDA,
) -> str:
    """The same triple spelled as a phrase, so ``~`` is not required to read it."""
    exaggerate, write, orthogonal = documented_triple(source, target, lambda_value)
    return (
        f"{target}++:{exaggerate.alpha:g} | "
        f"{source}={target}:{write.alpha:g} | "
        f"{target}%{source}:{orthogonal.alpha:g}"
    )


def _options(rule: dsl.Rule) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), float(value)) for key, value in rule.options.items()))


def view_rules(rules: list[dsl.Rule] | tuple[dsl.Rule, ...]) -> tuple[ExpandedRule, ...]:
    """Project parser rules onto the expand record."""
    return tuple(
        ExpandedRule(rule.op, rule.a, rule.b, float(rule.alpha), _options(rule))
        for rule in rules
    )


def failure_reasons(
    rules: tuple[ExpandedRule, ...],
    geometric_verdict: str,
    text_rules: tuple[ExpandedRule, ...] | None = None,
) -> tuple[str, ...]:
    """Named reasons a candidate is not the documented expand.

    Geometric ``right`` fails even when the three rules match. A live
    ``replace`` op fails even when the rest of the phrase is empty.
    """
    reasons: list[str] = []
    want = documented_triple()
    ops = [rule.op for rule in rules]
    want_ops = [rule.op for rule in want]
    invented = any(rule.op == "replace" or rule.op not in LIVE_LOSS_OPS for rule in rules)
    if invented:
        reasons.append("fourth_loss")
    if len(rules) == 1 and rules[0].op == dsl.WRITE and not invented:
        reasons.append("single_write")
    elif not invented:
        roles_match = (
            len(rules) == len(want)
            and ops == want_ops
            and all((got.a, got.b) == (exp.a, exp.b) for got, exp in zip(rules, want))
        )
        alphas_match = roles_match and all(
            got.alpha == exp.alpha for got, exp in zip(rules, want)
        )
        if not any(rule.op == dsl.ORTHOGONAL for rule in rules):
            reasons.append("dropped_percent")
        elif ops != want_ops:
            reasons.append("op_order")
        elif not roles_match:
            reasons.append("swapped_roles")
        elif not alphas_match:
            reasons.append("alphas")
        elif any(got.options != exp.options for got, exp in zip(rules, want)):
            reasons.append("options")
    if geometric_verdict == FORBIDDEN_GEOMETRIC:
        reasons.append("geometric_right")
    elif geometric_verdict != GEOMETRIC_VERDICT:
        reasons.append("geometric_verdict")
    if text_rules is not None and text_rules != rules:
        reasons.append("macro_not_text")
    if rules != want and not reasons:
        reasons.append("expand_mismatch")
    return tuple(dict.fromkeys(reasons))


def _wrong_order() -> tuple[ExpandedRule, ...]:
    exaggerate, write, orthogonal = documented_triple()
    return (write, exaggerate, orthogonal)


def _swapped_roles() -> tuple[ExpandedRule, ...]:
    """``blue~red`` roles under the ``red~blue`` name: a and b exchanged."""
    lam = DOCUMENTED_LAMBDA
    return (
        ExpandedRule(dsl.EXAGGERATE, SOURCE, "", 2 * lam),
        ExpandedRule(dsl.WRITE, TARGET, SOURCE, 4 * lam),
        ExpandedRule(dsl.ORTHOGONAL, SOURCE, TARGET, -lam),
    )


def _wrong_alphas() -> tuple[ExpandedRule, ...]:
    return tuple(
        ExpandedRule(rule.op, rule.a, rule.b, 1.0) for rule in documented_triple()
    )


def _dropped_percent() -> tuple[ExpandedRule, ...]:
    exaggerate, write, _orthogonal = documented_triple()
    return (exaggerate, write)


def _single_write() -> tuple[ExpandedRule, ...]:
    return (ExpandedRule(dsl.WRITE, SOURCE, TARGET, 1.0),)


def _fourth_loss() -> tuple[ExpandedRule, ...]:
    return (ExpandedRule("replace", SOURCE, TARGET, DOCUMENTED_LAMBDA),)


def _documented_views() -> tuple[tuple[ExpandedRule, ...], tuple[ExpandedRule, ...]]:
    macro = view_rules(dsl.parse_phrase(PHRASE))
    text = view_rules(dsl.parse_phrase(documented_text()))
    return macro, text


_HAND = {
    "wrong_order": _wrong_order,
    "swapped_roles": _swapped_roles,
    "wrong_alphas": _wrong_alphas,
    "dropped_percent": _dropped_percent,
    "single_write": _single_write,
    "geometric_right": documented_triple,
    "fourth_loss": _fourth_loss,
}


def make_arm(name: str, role: str, kind: str) -> Arm:
    """Locked cell reads the parser. A bad arm has to name a wrong expand."""
    if role not in ("locked_shared", "negative"):
        raise HonestyError("role must be locked_shared or negative")
    if kind not in _KINDS:
        raise HonestyError(f"unknown expand kind {kind!r}")
    if role == "locked_shared":
        if name != "locked_expand" or kind != "documented":
            raise HonestyError("locked_shared expand is the documented parser triple only")
    elif kind == "documented":
        raise HonestyError("a declared bad arm cannot use the documented expand")
    return Arm(name=name, role=role, kind=kind)


def run_arm(arm: Arm, *, order: int = 1) -> ExpandReport:
    """Score one expand. The locked arm reads ``parse_phrase``; negatives are hand-rolled."""
    if arm.kind == "documented":
        rules, text_rules = _documented_views()
        geometric = GEOMETRIC_VERDICT
    else:
        rules = _HAND[arm.kind]()
        text_rules = None
        geometric = FORBIDDEN_GEOMETRIC if arm.kind == "geometric_right" else GEOMETRIC_VERDICT
    reasons = failure_reasons(rules, geometric, text_rules)
    return ExpandReport(
        name=arm.name,
        role=arm.role,
        toy=TOY_ID,
        kind=arm.kind,
        rules=rules,
        text_rules=text_rules,
        geometric_verdict=geometric,
        won=not reasons,
        reasons=reasons,
        order=order,
    )


def log_report(row: ExpandReport) -> None:
    ops = ",".join(rule.op for rule in row.rules)
    alphas = ",".join(f"{rule.alpha:.3g}" for rule in row.rules)
    flag = "PASS" if row.won else "FAIL"
    why = (" " + " ".join(row.reasons)) if row.reasons else ""
    print(
        f"dsl_macro_expand order={row.order} arm={row.name} role={row.role} "
        f"kind={row.kind} won={int(row.won)} ops={ops} alphas={alphas} "
        f"geometric={row.geometric_verdict} {flag}{why}",
        flush=True,
    )


def _measured_reasons(row: ExpandReport) -> tuple[str, ...]:
    return failure_reasons(row.rules, row.geometric_verdict, row.text_rules)


def claim_expand_pass(
    *,
    winner: ExpandReport,
    negatives: list[ExpandReport] | tuple[ExpandReport, ...],
) -> dict:
    """PASS only when the parser matched the documented triple and every bad arm failed.

    An empty negative list is an error. A won flag or a reason list that
    disagrees with the measured expand is a dishonest board. Geometric
    ``right`` is not a PASS on this fixture.
    """
    if not negatives:
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    if winner.role != "locked_shared" or winner.name != "locked_expand":
        raise HonestyError("PASS requires the winning cell to be locked_expand")
    if winner.toy != TOY_ID:
        raise HonestyError("PASS is only defined for the dsl_macro_expand toy")
    if winner.kind != "documented":
        raise HonestyError("locked expand did not read the parser; refusing PASS")
    if winner.text_rules is None:
        raise HonestyError("locked expand is missing the spelled triple; refusing PASS")
    measured = _measured_reasons(winner)
    if winner.won != (not measured) or winner.reasons != measured:
        raise HonestyError("dishonest board: won flag does not match the expansion")
    if winner.geometric_verdict == FORBIDDEN_GEOMETRIC:
        raise HonestyError(
            "locked expand claimed geometric right; the fixture verdict is recipe"
        )
    if any(rule.op == "replace" or rule.op not in LIVE_LOSS_OPS for rule in winner.rules):
        raise HonestyError("locked expand invented a fourth loss; refusing PASS")
    if measured or not winner.won:
        raise HonestyError(
            "locked expand does not match the documented triple; refusing PASS"
        )
    ops = [rule.op for rule in winner.rules]
    if ops != [dsl.EXAGGERATE, dsl.WRITE, dsl.ORTHOGONAL]:
        raise HonestyError("locked expand ops are not exaggerate, write, orthogonal")
    alphas = [rule.alpha for rule in winner.rules]
    if alphas != [0.2, 0.4, -0.1]:
        raise HonestyError("locked expand alphas are not 0.2, 0.4, -0.1")
    if winner.rules[2].alpha >= 0:
        raise HonestyError("orthogonal blue%red must carry the negative alpha")
    for neg in negatives:
        if neg.role != "negative":
            raise HonestyError("PASS negatives must be declared bad arms")
        if neg.toy != winner.toy:
            raise HonestyError("declared bad arm was not scored on the same toy")
        neg_measured = _measured_reasons(neg)
        if neg.won != (not neg_measured) or neg.reasons != neg_measured:
            raise HonestyError("dishonest board: won flag does not match the expansion")
        if not neg_measured or neg.won:
            raise HonestyError(
                f"declared bad arm {neg.name!r} did not fail; refusing PASS"
            )
    return {
        "verdict": "PASS",
        "winner": winner.name,
        "ops": ops,
        "alphas": alphas,
        "roles": [
            {"op": rule.op, "a": rule.a, "b": rule.b} for rule in winner.rules
        ],
        "geometric_verdict": GEOMETRIC_VERDICT,
        "geometry_right": False,
        "negatives": [neg.name for neg in negatives],
        "toy": TOY_ID,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
    }


def log_verdict(verdict: dict) -> None:
    names = ",".join(verdict["negatives"])
    game = verdict.get("game_wiring")
    game_bit = f" game_wiring={game}" if game is not None else ""
    print(
        f"dsl_macro_expand verdict={verdict['verdict']} winner={verdict['winner']} "
        f"ops={','.join(verdict['ops'])} geometric={verdict['geometric_verdict']} "
        f"geometry_right={int(verdict['geometry_right'])} negatives={names}"
        f"{game_bit} music_gpu_transfer={int(verdict['music_gpu_transfer'])} "
        f"anima_gpu_transfer={int(verdict['anima_gpu_transfer'])}",
        flush=True,
    )


def demo_arms() -> tuple[Arm, ...]:
    """Documented parser first, then each named wrong expand."""
    locked = make_arm("locked_expand", "locked_shared", "documented")
    negatives = tuple(
        make_arm(kind, "negative", kind) for kind in _NEGATIVE_KINDS
    )
    return (locked, *negatives)


def game_wiring_row(phrase: str = PHRASE) -> dict:
    """Secondary row: ``loss_game`` stepped the expanded rules under locked_shared.

    The formulation verdict on that step is wiring. The geometric verdict
    stays ``recipe``. This row does not claim ``right``.
    """
    from conceptmod.game import NOT_A_TRANSFER, loss_game

    match = loss_game(phrase)
    devices = {param.device.type for param in match.backend.trainable_parameters()}
    if devices != {"cpu"}:
        raise HonestyError(f"replace-macro toy left the game on {sorted(devices)}")
    score = match.score()
    rules = view_rules(match.rules)
    want = documented_triple()
    wiring_ok = (
        score["verdict"] == "PASS"
        and score["pairing"] == "locked_shared"
        and score["claim"] == "locked"
        and not score["mismatches"]
        and rules == want
        and score["note"] == NOT_A_TRANSFER
    )
    return {
        "name": "game_wiring",
        "phrase": phrase,
        "verdict": "PASS" if wiring_ok else "FAIL",
        "pairing": score["pairing"],
        "claim": score["claim"],
        "game_verdict": score["verdict"],
        "mismatches": list(score["mismatches"]),
        "ops": [rule.op for rule in rules],
        "alphas": [rule.alpha for rule in rules],
        "geometric_verdict": GEOMETRIC_VERDICT,
        "geometry_right": False,
        "note": score["note"],
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
    }


def accept_game_row(row: dict) -> dict:
    """Refuse a game row that claims geometric ``right`` for bare ``red~blue``."""
    if row.get("geometry_right") or row.get("geometric_verdict") == FORBIDDEN_GEOMETRIC:
        raise HonestyError(
            "game wiring claimed geometric right; the fixture verdict is recipe"
        )
    if row.get("geometric_verdict") != GEOMETRIC_VERDICT:
        raise HonestyError(
            "game wiring geometric verdict must stay recipe"
        )
    if row.get("verdict") != "PASS":
        raise HonestyError(
            "game wiring did not step the documented triple under locked_shared"
        )
    if row.get("music_gpu_transfer") or row.get("anima_gpu_transfer"):
        raise HonestyError("game wiring is not a Music or Anima GPU transfer")
    return row


def run_board(
    arms: tuple[Arm, ...] | list[Arm] | None = None,
    *,
    include_game: bool = True,
) -> dict:
    """Run the documented expand first. Refuse a board with no bad arm.

    ``include_game`` adds the locked_shared wiring row. That row is not a
    geometric ``right``.
    """
    chosen = demo_arms() if arms is None else tuple(arms)
    if not chosen or chosen[0].role != "locked_shared" or chosen[0].name != "locked_expand":
        raise HonestyError("FIRST cell must be locked_expand")
    if chosen[0].kind != "documented":
        raise HonestyError("FIRST cell must read the documented parser triple")
    if not any(arm.role == "negative" for arm in chosen[1:]):
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    reports = []
    for order, arm in enumerate(chosen, start=1):
        row = run_arm(arm, order=order)
        log_report(row)
        reports.append(row)
    verdict = claim_expand_pass(winner=reports[0], negatives=reports[1:])
    game = None
    if include_game:
        game = accept_game_row(game_wiring_row())
        verdict = {**verdict, "game_wiring": game["verdict"]}
    log_verdict(verdict)
    return {"verdict": verdict, "reports": reports, "game": game}


def main() -> int:
    board = run_board()
    return 0 if board["verdict"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
