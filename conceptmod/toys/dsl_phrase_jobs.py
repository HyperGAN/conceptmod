"""Phrase-DSL job honesty on the CPU 2-D field.

Geometric ``right`` from ``docs/dsl.md`` is the job verdict. The live
runners are :func:`conceptmod.analysis_dsl.run_job` and
:func:`conceptmod.analysis_2d.run_method`. Formulation PASS is honesty
that the phrase is that documented job. A first-class ``+``, ``!!``, or
``/`` synonym, ``%`` used as isolate, a keep+erase row that dropped
``k#k``, and dead ``^`` / ``;`` claimed as right all FAIL.

The adv stamp is :func:`conceptmod.toys.locked_adv_defaults`. Keep gates
are ``U_KEPT_MIN`` and ``SAME_DIR_MAX`` from the cover / leftover toy,
read by the 2-D suite. This module does not invent a second recipe, does
not vendor ParticleGAN primitives, and does not add synonym operators.

``claim_phrase_pass`` follows :func:`conceptmod.toys.claim_pass`: an
empty negative list raises :class:`HonestyError`. ``claim_pass`` itself
stays the two-pole gate; this module does not loosen it.

A PASS is a CPU toy. It is not a Music, Anima, or Supra GPU transfer.

Tail the log with::

    python -m conceptmod.toys.dsl_phrase_jobs
"""

from __future__ import annotations

from dataclasses import dataclass

from conceptmod import dsl, ops
from conceptmod.toys.cover_leftover import SAME_DIR_MAX, U_KEPT_MIN
from conceptmod.toys.leaderboard_honesty import HonestyError

TOY_ID = "dsl_phrase_jobs"

# Bipolar landing the job test already gates (``red++`` → color ~+3, blue ~−3).
# Keep floors stay the cover toy's constants. Not an adv recipe.
BIPOLAR_SCALE = 2.0

DOCUMENTED_NAMES = (
    "neutralize",
    "bipolar",
    "remap",
    "keep_erase",
    "mix",
    "isolate",
)
REQUIRED_NEGATIVES = (
    "plus_synonym",
    "bang_synonym",
    "slash_synonym",
    "percent_isolate",
    "keep_erase_bare",
    "pixel_caret",
    "reward_semi",
)


@dataclass(frozen=True)
class Arm:
    name: str
    role: str
    job: str
    phrase: str


@dataclass(frozen=True)
class Row:
    name: str
    role: str
    job: str
    phrase: str
    passed: bool
    reasons: tuple[str, ...]
    toy: str = TOY_ID
    geometric: str | None = None
    refused: bool = False
    stripe_hold: float | None = None
    color_on_red: float | None = None
    pattern_on_red: float | None = None
    write_cosine: float | None = None
    blue_x: float | None = None
    blue_y: float | None = None
    mix_x: float | None = None
    mix_y: float | None = None
    ops: tuple[str, ...] = ()
    adv: dict | None = None
    adv_ok: bool = False
    detail: str = ""

    def line(self) -> str:
        flag = "PASS" if self.passed else ("REFUSE" if self.refused else "FAIL")
        why = (" " + " ".join(self.reasons)) if self.reasons else ""
        geo = self.geometric or "-"
        hold = f"{self.stripe_hold:.3f}" if self.stripe_hold is not None else "-"
        return (
            f"{self.name} job={self.job} phrase={self.phrase} "
            f"geometric={geo} stripe_hold={hold} {flag}{why}"
        )


def _names():
    from conceptmod.analysis_2d import COLOR, COLOR_OPP, KEEP

    return COLOR, COLOR_OPP, KEEP


def canon() -> dict[str, str]:
    """Documented phrases. Same strings as ``docs/dsl.md``."""
    color, opp, keep = _names()
    return {
        "neutralize": f"{color}--",
        "bipolar": f"{color}++",
        "remap": f"{color}={opp}",
        "keep_erase": f"{color}--|{keep}#{keep}",
        "mix": f"{color}={color} {keep}",
        "isolate": f"{color} {keep}={keep}",
    }


def documented_arms() -> tuple[Arm, ...]:
    phrases = canon()
    return tuple(
        Arm(name, "documented", name, phrases[name]) for name in DOCUMENTED_NAMES
    )


def negative_arms() -> tuple[Arm, ...]:
    """Inventions and dead ops. None of these is a new live operator."""
    color, _opp, keep = _names()
    return (
        Arm("plus_synonym", "negative", "mix", f"{color}+{keep}"),
        Arm("bang_synonym", "negative", "neutralize", f"{color}!!"),
        Arm("slash_synonym", "negative", "neutralize", f"{color}/"),
        Arm("percent_isolate", "negative", "isolate", f"{color}%{color} {keep}"),
        Arm("keep_erase_bare", "negative", "keep_erase", f"{color}--"),
        Arm("pixel_caret", "negative", "dead_pixel", f"{color}^{_opp}"),
        Arm("reward_semi", "negative", "dead_reward", f";{color}"),
    )


def invented_glyph(phrase: str) -> str | None:
    """``+`` / ``!!`` / ``/`` are not live ops. ``++`` stays exaggerate."""
    if "+" in phrase.replace("++", ""):
        return "+"
    if "!!" in phrase:
        return "!!"
    for part in phrase.split("|"):
        concept = part.split(":", 1)[0].strip()
        if concept.endswith("/"):
            return "/"
    return None


def dead_glyph(phrase: str) -> str | None:
    """Pixel ``^`` and ImageReward ``;``. Left dead on this fixture."""
    for part in phrase.split("|"):
        concept = part.split(":", 1)[0].replace("@", "").strip()
        if "^" in concept:
            return "^"
        if concept.startswith(";"):
            return ";"
    return None


def _stamp() -> dict:
    from conceptmod.toys import locked_adv_defaults

    return locked_adv_defaults()


def _budget() -> tuple[int, float, int]:
    from conceptmod.analysis_2d import DEFAULT_LR, DEFAULT_SEED, DEFAULT_STEPS

    return DEFAULT_STEPS, DEFAULT_LR, DEFAULT_SEED


def _ops_of(phrase: str) -> tuple[str, ...]:
    return tuple(rule.op for rule in dsl.parse_phrase(phrase))


def has_keep_pin(phrase: str) -> bool:
    """True when the phrase freezes the keep concept onto itself."""
    _color, _opp, keep = _names()
    for rule in dsl.parse_phrase(phrase):
        if rule.op == dsl.FREEZE and rule.a == keep and rule.b == keep:
            return True
    return False


def _blank(arm: Arm, *, reasons: tuple[str, ...], detail: str,
           refused: bool = False, ops_names: tuple[str, ...] = (),
           geometric: str | None = None, **probes) -> Row:
    stamp = _stamp()
    return Row(
        name=arm.name,
        role=arm.role,
        job=arm.job,
        phrase=arm.phrase,
        passed=False,
        reasons=reasons,
        geometric=geometric,
        refused=refused,
        ops=ops_names,
        adv=stamp,
        adv_ok=True,
        detail=detail,
        **probes,
    )


def _from_result(arm: Arm, result, *, reasons: list[str], detail: str) -> Row:
    from conceptmod.analysis_2d import COLOR, COLOR_OPP, KEEP

    stamp = _stamp()
    adv_ok = result.adv == stamp
    if not adv_ok:
        reasons.append("adv_stamp")
    blue = result.points_after.get(COLOR_OPP)
    mix = result.points_after.get(f"{COLOR} {KEEP}")
    passed = not reasons and arm.role == "documented" and result.verdict == "right"
    return Row(
        name=arm.name,
        role=arm.role,
        job=arm.job,
        phrase=arm.phrase,
        passed=passed,
        reasons=tuple(reasons),
        geometric=result.verdict,
        refused=False,
        stripe_hold=result.after.stripe_hold,
        color_on_red=result.after.color_on_red,
        pattern_on_red=result.after.pattern_on_red,
        write_cosine=result.after.write_cosine,
        blue_x=None if blue is None else blue[0],
        blue_y=None if blue is None else blue[1],
        mix_x=None if mix is None else mix[0],
        mix_y=None if mix is None else mix[1],
        ops=_ops_of(arm.phrase),
        adv=stamp if adv_ok else result.adv,
        adv_ok=adv_ok,
        detail=detail,
    )


def _train(kind: str, name: str, phrase: str, cache: dict):
    from conceptmod.analysis_2d import run_method
    from conceptmod.analysis_dsl import run_job

    steps, lr, seed = _budget()
    key = (kind, name, phrase, steps, seed)
    if key not in cache:
        kwargs = {"steps": steps, "lr": lr, "seed": seed}
        if kind == "job":
            cache[key] = run_job(name, phrase, **kwargs)
        else:
            cache[key] = run_method(name, phrase, **kwargs)
    return cache[key]


def _suite_for(job: str) -> tuple[str, str]:
    return {
        "neutralize": ("job", "neutralize"),
        "bipolar": ("method", "exaggerate"),
        "remap": ("method", "write"),
        "keep_erase": ("method", "erase_esd_freeze"),
        "mix": ("job", "mix_write"),
        "isolate": ("job", "isolate_write"),
    }[job]


def _refuse_dead(arm: Arm, glyph: str) -> Row:
    from conceptmod.analysis_2d import TwoAxisBackend
    from conceptmod.analysis_dsl import dead_ops

    dead = dead_ops()
    if glyph not in dead:
        raise HonestyError(
            f"{glyph!r} is not a dead op on this fixture; refusing to score it as live"
        )
    if hasattr(TwoAxisBackend, "render") and glyph == "^":
        raise HonestyError("TwoAxisBackend grew a render; pixel ^ stays dead here")
    rules = dsl.parse_phrase(arm.phrase)
    backend = TwoAxisBackend()
    ctx = ops.StepContext(backend, stop_index=2, seed=0, cfg=ops.OpDefaults())
    try:
        ops.rule_loss(rules[0], ctx)
    except NotImplementedError as exc:
        if glyph != ";":
            raise
        detail = str(exc)
    except AttributeError as exc:
        if glyph != "^" or "render" not in str(exc):
            raise
        detail = str(exc)
    else:
        return _blank(
            arm,
            reasons=("dead_op", "implemented"),
            detail=f"{glyph} ran on the 2-D fixture; it must stay dead",
            ops_names=_ops_of(arm.phrase),
        )
    return _blank(
        arm,
        reasons=("dead_op",),
        refused=True,
        detail=detail,
        ops_names=_ops_of(arm.phrase),
    )


def _score_documented(arm: Arm, cache: dict) -> Row:
    phrases = canon()
    if arm.phrase != phrases[arm.job]:
        return _blank(
            arm,
            reasons=("honesty",),
            detail=f"{arm.job} phrase {arm.phrase!r} is not the documented {phrases[arm.job]!r}",
        )
    kind, suite_name = _suite_for(arm.job)
    result = _train(kind, suite_name, arm.phrase, cache)
    reasons: list[str] = []
    if result.verdict != "right":
        reasons.append("geometric")
    detail = result.note
    if arm.job == "neutralize":
        reading = dsl.describe_phrase(arm.phrase)
        guidance = dsl.parse_phrase(arm.phrase)[0].options.get("guidance")
        if "Neutralize" not in reading:
            reasons.append("honesty")
        if guidance not in (None, 0, 0.0):
            reasons.append("guidance")
        if ops.OpDefaults.erase_guidance != 0.0:
            reasons.append("guidance")
        detail = reading
    elif arm.job == "bipolar":
        blue_x = result.points_after[_names()[1]][0]
        blue_y = result.points_after[_names()[1]][1]
        scaled = (
            result.after.color_on_red > BIPOLAR_SCALE
            and blue_x < -BIPOLAR_SCALE
            and abs(blue_y) < SAME_DIR_MAX
            and result.after.stripe_hold > U_KEPT_MIN
        )
        if not scaled:
            reasons.append("bipolar")
    elif arm.job == "keep_erase":
        if not has_keep_pin(arm.phrase):
            reasons.append("retain")
    elif arm.job == "mix":
        if invented_glyph(arm.phrase) is not None:
            reasons.append("honesty")
        rules = dsl.parse_phrase(arm.phrase)
        if len(rules) != 1 or rules[0].op != dsl.WRITE:
            reasons.append("honesty")
    elif arm.job == "isolate":
        if "%" in arm.phrase:
            reasons.append("percent_isolate")
        rules = dsl.parse_phrase(arm.phrase)
        if len(rules) != 1 or rules[0].op != dsl.WRITE:
            reasons.append("honesty")
    return _from_result(arm, result, reasons=reasons, detail=detail)


def _score_percent(arm: Arm, cache: dict) -> Row:
    """``%`` is the wrong isolate tool. A landing does not make it the job."""
    if "%" not in arm.phrase:
        raise HonestyError("percent_isolate arm has no %")
    isolate = canon()["isolate"]
    if arm.phrase == isolate:
        raise HonestyError("percent arm collapsed into the documented isolate write")
    rules = dsl.parse_phrase(arm.phrase)
    if len(rules) != 1 or rules[0].op != dsl.ORTHOGONAL:
        return _blank(
            arm,
            reasons=("honesty",),
            detail="% isolate claim did not parse as orthogonal",
            ops_names=tuple(rule.op for rule in rules),
        )
    result = _train("job", "isolate_write", arm.phrase, cache)
    detail = (
        "% orthogonalizes; it does not project. "
        f"Isolate is `{isolate}`. Suite isolate probe said {result.verdict}."
    )
    # Always a formulation fail. Geometric ``right`` would still be the wrong op.
    row = _from_result(arm, result, reasons=["percent_isolate"], detail=detail)
    if row.passed:
        raise HonestyError("percent isolate was marked PASS")
    return row


def _score_retain(arm: Arm, cache: dict) -> Row:
    """Keep+erase without ``k#k`` is bare neutralize, even when stripe holds."""
    if has_keep_pin(arm.phrase):
        raise HonestyError("negative keep+erase arm still has the freeze pin")
    phrases = canon()
    if arm.phrase == phrases["neutralize"]:
        result = _train("job", "neutralize", arm.phrase, cache)
    else:
        result = _train("method", "erase_esd", arm.phrase, cache)
    detail = (
        f"keep+erase is `{phrases['keep_erase']}`; "
        f"{arm.phrase!r} dropped the freeze arm"
    )
    row = _from_result(arm, result, reasons=["retain"], detail=detail)
    if row.passed:
        raise HonestyError("bare erase was marked keep+erase PASS")
    return row


def score_arm(arm: Arm, cache: dict | None = None) -> Row:
    """Score one phrase claim. Inventions and dead ops do not train a new loss."""
    if arm.role not in ("documented", "negative"):
        raise HonestyError("role must be documented or negative")
    store = cache if cache is not None else {}
    glyph = invented_glyph(arm.phrase)
    if glyph is not None:
        try:
            dsl.parse_phrase(arm.phrase)
        except dsl.DSLError as exc:
            parsed = f"parser rejected it ({exc})"
        else:
            parsed = "parser accepted a synonym; still not the documented job"
        return _blank(
            arm,
            reasons=("honesty",),
            detail=f"invented {glyph!r} synonym; {parsed}",
        )
    dead = dead_glyph(arm.phrase)
    if dead is not None:
        return _refuse_dead(arm, dead)
    if arm.role == "documented":
        if arm.job not in DOCUMENTED_NAMES:
            raise HonestyError(f"unknown documented job {arm.job!r}")
        return _score_documented(arm, store)
    if arm.job == "isolate" and "%" in arm.phrase:
        return _score_percent(arm, store)
    if arm.job == "keep_erase":
        return _score_retain(arm, store)
    raise HonestyError(f"unknown negative arm {arm.name!r}")


def claim_phrase_pass(
    *,
    documented: list[Row] | tuple[Row, ...],
    negatives: list[Row] | tuple[Row, ...],
) -> dict:
    """PASS only when every documented phrase job holds and every bad arm fails.

    An empty negative list is an error, including when every documented row
    looks perfect. Geometric ``right`` is not this verdict.
    """
    if not negatives:
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    if not documented:
        raise HonestyError("PASS requires the documented phrase jobs")
    seen = [row.name for row in documented]
    if seen != list(DOCUMENTED_NAMES):
        raise HonestyError(
            "PASS requires the documented jobs in order "
            f"{list(DOCUMENTED_NAMES)}; got {seen}"
        )
    for row in documented:
        if row.role != "documented":
            raise HonestyError("PASS documented rows must use role documented")
        if row.toy != TOY_ID:
            raise HonestyError("PASS is only defined for the dsl_phrase_jobs toy")
        if row.geometric == "PASS":
            raise HonestyError("geometric verdict must stay right, not PASS")
        if row.geometric != "right" or not row.passed or row.reasons or row.refused:
            raise HonestyError(
                f"documented job {row.name!r} did not pass; refusing PASS"
            )
        if row.phrase != canon()[row.job]:
            raise HonestyError(f"documented job {row.name!r} drifted off its phrase")
    neg_names = [row.name for row in negatives]
    missing = [name for name in REQUIRED_NEGATIVES if name not in neg_names]
    if missing:
        raise HonestyError(f"PASS is missing declared negatives {missing}")
    for neg in negatives:
        if neg.role != "negative":
            raise HonestyError("PASS negatives must be declared bad arms")
        if neg.toy != TOY_ID:
            raise HonestyError("declared bad arm was not scored on the same toy")
        if neg.passed or not neg.reasons:
            raise HonestyError(
                f"declared bad arm {neg.name!r} did not fail; refusing PASS"
            )
        if neg.geometric == "PASS":
            raise HonestyError("a negative geometric verdict was labeled PASS")
    return {
        "verdict": "PASS",
        "documented": list(DOCUMENTED_NAMES),
        "negatives": neg_names,
        "toy": TOY_ID,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "supra_gpu_transfer": False,
        "cpu_toy": True,
        "cover_posture": "demo_1.5",
    }


def log_row(row: Row) -> None:
    print(row.line(), flush=True)


def log_verdict(verdict: dict) -> None:
    names = ",".join(verdict["negatives"])
    print(
        f"dsl_phrase_jobs verdict={verdict['verdict']} "
        f"documented={','.join(verdict['documented'])} negatives={names} "
        f"music_gpu_transfer={int(verdict['music_gpu_transfer'])} "
        f"anima_gpu_transfer={int(verdict['anima_gpu_transfer'])} "
        f"supra_gpu_transfer={int(verdict['supra_gpu_transfer'])}",
        flush=True,
    )


def run_board(arms: tuple[Arm, ...] | list[Arm] | None = None) -> dict:
    """Score documented phrases, then the named failures. Claim PASS last."""
    if arms is None:
        arms = (*documented_arms(), *negative_arms())
    cache: dict = {}
    rows: list[Row] = []
    for arm in arms:
        row = score_arm(arm, cache)
        log_row(row)
        rows.append(row)
    documented = [row for row in rows if row.role == "documented"]
    negatives = [row for row in rows if row.role == "negative"]
    verdict = claim_phrase_pass(documented=documented, negatives=negatives)
    steps, lr, seed = _budget()
    verdict = {**verdict, "steps": steps, "lr": lr, "seed": seed}
    log_verdict(verdict)
    return {"verdict": verdict, "rows": rows}


def main() -> int:
    board = run_board()
    return 0 if board["verdict"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
