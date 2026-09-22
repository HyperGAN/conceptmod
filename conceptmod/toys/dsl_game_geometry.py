"""Bridge locked_shared game wiring to geometric ``right``.

``conceptmod.game.loss_game`` can score ``PASS`` on one CPU step because
the adversary matched :func:`conceptmod.toys.locked_adv_defaults`. That
row is wiring. It is not a geometric verdict from ``docs/dsl.md``.

A bridge row here PASSes only when both are true:

1. The phrase steps under ``locked_shared`` with the locked stamp
   (``loss_game`` / ``game``, claim locked, mismatches empty).
2. After the 2-D fixture train, the field is geometric ``right`` for
   that phrase. The verdict comes from :func:`conceptmod.analysis_2d.run_method`
   or :func:`conceptmod.analysis_dsl.run_job` — the same scorers
   ``tests/test_2d_analysis.py`` and ``tests/test_dsl_jobs.py`` gate.
   This module does not copy those thresholds.

A game ``PASS`` with skipped or faked geometry FAILs (``wiring_only``).
Geometric ``right`` with an unlocked claim, FM-on, or stranger pairing
FAILs (``geometry_only``). ``^`` and ``;`` still refuse inside
``loss_game``; claiming PASS on that refusal FAILs. ``a~b`` stays
``recipe`` on the job board and does not become a silent ``right``.

Budget is the 2-D fixture (``DEFAULT_STEPS`` 40, ``DEFAULT_LR`` 8e-2,
``DEFAULT_SEED``). That budget is not a new adv recipe. The stamp stays
``locked_adv_defaults()``.

A bridge PASS is a CPU toy. It is not a Music or Anima GPU transfer.
A game PASS alone is not this gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from conceptmod.toys.leaderboard_honesty import HonestyError


TOY_ID = "dsl_game_geometry"

NOT_A_TRANSFER = (
    "CPU toy. Bridge PASS needs a locked_shared game step and geometric "
    "right from the 2-D phrase scoreboard. A game PASS alone is not that "
    "verdict. Not a Music or Anima GPU transfer."
)

# Scorers that own geometric right / needs help / recipe. A string the
# caller typed is not one of these.
REAL_SOURCES = frozenset({
    "analysis_2d.run_method",
    "analysis_dsl.run_job",
})

_BRIDGE_METHODS = ("write", "erase_esd", "erase_esd_freeze", "exaggerate")


@dataclass(frozen=True)
class BridgeArm:
    """One scoreboard row. ``game_kwargs`` override :func:`loss_game`."""

    name: str
    phrase: str
    role: str
    kind: str
    scorer: str | None = None
    scorer_name: str | None = None
    game_kwargs: tuple[tuple[str, object], ...] = ()
    claimed_pass: bool = False
    claimed_geometry: str | None = None
    dead_op: str | None = None


def fixture_budget() -> dict:
    """Adam budget of the 2-D analysis fixture. Not an adv recipe.

    Steps, lr, and seed are read from :mod:`conceptmod.analysis_2d`.
    They are not keys on ``locked_adv_defaults()``.
    """
    from conceptmod.analysis_2d import DEFAULT_LR, DEFAULT_SEED, DEFAULT_STEPS

    return {"steps": DEFAULT_STEPS, "lr": DEFAULT_LR, "seed": DEFAULT_SEED}


def arms() -> tuple[BridgeArm, ...]:
    """PASS phrases from the 2-D method suite, plus the named mismatch arms.

    Phrases are the suite's own strings (``red=blue``, bare ``red--``,
    ``red--|stripe#stripe``, ``red++``). The replace macro is the job
    board's ``red~blue``, which scores ``recipe``.
    """
    from conceptmod.analysis_2d import COLOR, COLOR_OPP, KEEP, method_specs
    from conceptmod.analysis_dsl import job_specs

    methods = {spec["name"]: spec for spec in method_specs()}
    jobs = {spec["name"]: spec for spec in job_specs()}
    for name in _BRIDGE_METHODS:
        spec = methods[name]
        if spec["erase_mode"] is not None:
            raise HonestyError(
                f"{name} is not a bare phrase method (erase_mode={spec['erase_mode']!r})"
            )
    write = methods["write"]["phrase"]
    erase = methods["erase_esd"]["phrase"]
    freeze = methods["erase_esd_freeze"]["phrase"]
    exaggerate = methods["exaggerate"]["phrase"]
    recipe = jobs["replace_macro"]["phrase"]
    expected = {
        "write": f"{COLOR}={COLOR_OPP}",
        "erase_esd": f"{COLOR}--",
        "erase_esd_freeze": f"{COLOR}--|{KEEP}#{KEEP}",
        "exaggerate": f"{COLOR}++",
        "replace_macro": f"{COLOR}~{COLOR_OPP}",
    }
    got = {
        "write": write,
        "erase_esd": erase,
        "erase_esd_freeze": freeze,
        "exaggerate": exaggerate,
        "replace_macro": recipe,
    }
    if got != expected:
        raise HonestyError(
            "phrase suite drifted from the documented 2-D scoreboard: "
            f"{got!r} != {expected!r}"
        )
    if (COLOR, COLOR_OPP, KEEP) != ("red", "blue", "stripe"):
        raise HonestyError("2-D fixture axes are not red/blue × stripe")

    def bridge(name: str, phrase: str, scorer_name: str) -> BridgeArm:
        return BridgeArm(
            name=name,
            phrase=phrase,
            role="bridge",
            kind="both",
            scorer="method",
            scorer_name=scorer_name,
        )

    def negative(name: str, *, kind: str, phrase: str = write, **kwargs) -> BridgeArm:
        return BridgeArm(name=name, phrase=phrase, role="negative", kind=kind, **kwargs)

    return (
        bridge("write", write, "write"),
        bridge("erase_bare", erase, "erase_esd"),
        bridge("erase_freeze", freeze, "erase_esd_freeze"),
        bridge("exaggerate", exaggerate, "exaggerate"),
        negative(
            "wiring_only",
            kind="wiring_only",
            scorer="method",
            scorer_name="write",
            claimed_pass=True,
        ),
        negative(
            "faked_geometry",
            kind="faked_geometry",
            scorer="method",
            scorer_name="write",
            claimed_pass=True,
            claimed_geometry="right",
        ),
        negative(
            "geometry_only_stranger",
            kind="geometry_only",
            scorer="method",
            scorer_name="write",
            game_kwargs=(("pairing", "stranger"), ("claim", "drift")),
        ),
        negative(
            "geometry_only_fm",
            kind="geometry_only",
            scorer="method",
            scorer_name="write",
            game_kwargs=(("fm_weight", 0.1), ("claim", "drift")),
        ),
        negative(
            "locked_stranger_refuse",
            kind="geometry_only",
            scorer="method",
            scorer_name="write",
            game_kwargs=(("pairing", "stranger"), ("claim", "locked")),
        ),
        negative(
            "locked_fm_refuse",
            kind="geometry_only",
            scorer="method",
            scorer_name="write",
            game_kwargs=(("fm_weight", 0.1), ("claim", "locked")),
        ),
        negative(
            "dead_pixel",
            kind="dead",
            phrase="a painting of a house^a photo of a house",
            dead_op="^",
            claimed_pass=True,
        ),
        negative(
            "dead_reward",
            kind="dead",
            phrase=";a bright sunset",
            dead_op=";",
            claimed_pass=True,
        ),
        negative(
            "replace_macro",
            kind="recipe",
            phrase=recipe,
            scorer="job",
            scorer_name="replace_macro",
            claimed_pass=True,
            claimed_geometry="right",
        ),
    )


def score_geometry(arm: BridgeArm, *, steps: int, lr: float, seed: int) -> dict:
    """Train ``arm.phrase`` and return the live job/method verdict.

    ``method`` arms call :func:`conceptmod.analysis_2d.run_method`.
    ``job`` arms call :func:`conceptmod.analysis_dsl.run_job`.
    """
    if arm.scorer not in ("method", "job") or not arm.scorer_name:
        raise HonestyError(f"{arm.name} has no geometric scorer")
    from conceptmod.toys import locked_adv_defaults

    stamp = locked_adv_defaults()
    if arm.scorer == "method":
        from conceptmod.analysis_2d import run_method

        result = run_method(
            arm.scorer_name, arm.phrase, steps=steps, lr=lr, seed=seed,
        )
        source = "analysis_2d.run_method"
    else:
        from conceptmod.analysis_dsl import run_job

        result = run_job(
            arm.scorer_name, arm.phrase, steps=steps, lr=lr, seed=seed,
        )
        source = "analysis_dsl.run_job"
    if result.phrase != arm.phrase or result.name != arm.scorer_name:
        raise HonestyError(
            "scorer returned a different phrase or job "
            f"({result.name!r} {result.phrase!r})"
        )
    if result.verdict == "PASS":
        raise HonestyError(
            "geometric scorer returned PASS; right / needs help / recipe "
            "are not a formulation PASS"
        )
    return {
        "verdict": result.verdict,
        "note": result.note,
        "source": source,
        "phrase": result.phrase,
        "name": result.name,
        "stamp_ok": result.adv == stamp,
        "steps": steps,
        "lr": lr,
        "seed": seed,
    }


def score_game(
    phrase: str,
    *,
    seed: int,
    claimed_pass: bool = False,
    **kwargs,
) -> dict:
    """One ``loss_game`` score. A locked-claim refusal stays a refusal.

    ``LockedClaimError`` and dead-op ``GameError`` are caught and recorded.
    They are not turned into PASS. ``game.py`` still raises for the caller
    who invokes ``loss_game`` directly.
    """
    from conceptmod.dsl import DSLError
    from conceptmod.game import GameError, loss_game
    from conceptmod.toys import locked_adv_defaults

    claim = kwargs.get("claim", "locked")
    pairing = kwargs.get("pairing", "locked_shared")
    try:
        match = loss_game(phrase, seed=seed, **kwargs)
    except (GameError, DSLError) as exc:
        return {
            "verdict": "FAIL",
            "refused": True,
            "claim": claim,
            "pairing": pairing,
            "mismatches": _refuse_mismatches(str(exc)),
            "error": str(exc),
            "stamp_ok": False,
            "claimed_pass": claimed_pass,
            "step": 0,
        }
    row = match.score()
    stamp = locked_adv_defaults()
    return {
        "verdict": row["verdict"],
        "refused": False,
        "claim": row["claim"],
        "pairing": row["pairing"],
        "mismatches": list(row["mismatches"]),
        "error": "",
        "stamp_ok": row["adv"] == stamp and match.stamp == stamp,
        "claimed_pass": claimed_pass,
        "step": row["step"],
    }


def _refuse_mismatches(message: str) -> list[str]:
    tags: list[str] = []
    if "stranger pairing refused" in message:
        tags.append("stranger_pairing")
    if "FM-on under b_cap refused" in message:
        tags.append("fm_on")
    if "thinned b_cap refused" in message:
        tags.append("thinned_kappa")
    if "no render" in message or message.startswith("pixel"):
        tags.append("dead_pixel")
    if "ImageReward" in message or "not implemented" in message:
        tags.append("dead_reward")
    if not tags:
        tags.append("game_refused")
    return tags


def _skipped_geometry(phrase: str, budget: dict) -> dict:
    return {
        "verdict": "skipped",
        "note": "loss_game PASS was not scored on the 2-D phrase board",
        "source": "skipped",
        "phrase": phrase,
        "name": "",
        "stamp_ok": False,
        **budget,
    }


def _faked_geometry(phrase: str, budget: dict) -> dict:
    return {
        "verdict": "right",
        "note": "caller asserted right without analysis_2d / analysis_dsl",
        "source": "faked",
        "phrase": phrase,
        "name": "",
        "stamp_ok": False,
        **budget,
    }


def _dead_geometry(phrase: str, op: str, budget: dict) -> dict:
    from conceptmod.analysis_dsl import dead_ops

    notes = dead_ops()
    if op not in notes:
        raise HonestyError(f"{op!r} is not a documented dead op on this fixture")
    return {
        "verdict": "dead",
        "note": notes[op],
        "source": "analysis_dsl.dead_ops",
        "phrase": phrase,
        "name": op,
        "stamp_ok": False,
        **budget,
    }


def _geometry_for(arm: BridgeArm, cache: dict, budget: dict) -> dict:
    if arm.kind == "wiring_only":
        return _skipped_geometry(arm.phrase, budget)
    if arm.kind == "faked_geometry":
        return _faked_geometry(arm.phrase, budget)
    if arm.kind == "dead":
        if not arm.dead_op:
            raise HonestyError(f"{arm.name} is dead but names no op")
        return _dead_geometry(arm.phrase, arm.dead_op, budget)
    key = (
        arm.scorer,
        arm.scorer_name,
        arm.phrase,
        budget["steps"],
        budget["lr"],
        budget["seed"],
    )
    if key not in cache:
        cache[key] = score_geometry(
            arm, steps=budget["steps"], lr=budget["lr"], seed=budget["seed"],
        )
    return cache[key]


def _game_for(arm: BridgeArm, cache: dict, seed: int) -> dict:
    # ``claimed_pass`` is an arm flag, not a second game. Share the step.
    key = (arm.phrase, arm.game_kwargs, seed)
    if key not in cache:
        cache[key] = score_game(
            arm.phrase,
            seed=seed,
            claimed_pass=arm.claimed_pass,
            **dict(arm.game_kwargs),
        )
    return cache[key]


def _append(mismatches: list[str], tag: str) -> None:
    if tag not in mismatches:
        mismatches.append(tag)


def _row(arm: BridgeArm, game: dict, geometry: dict, budget: dict) -> dict:
    mismatches: list[str] = []
    geo_verdict = geometry["verdict"]
    geo_source = geometry["source"]
    kind = arm.kind

    if kind == "dead" or geo_verdict == "dead":
        _append(mismatches, "dead_phrase")
    elif kind == "faked_geometry" or geo_source == "faked":
        _append(mismatches, "wiring_only")
        _append(mismatches, "faked_geometry")
    elif kind == "wiring_only" or geo_source == "skipped" or geo_source not in REAL_SOURCES:
        _append(mismatches, "wiring_only")
    elif geo_verdict == "recipe":
        _append(mismatches, "recipe_not_independently_right")
    elif geo_verdict != "right":
        _append(mismatches, "geometry_not_right")

    if game["refused"]:
        if kind == "dead":
            _append(mismatches, "dead_phrase_refused")
        elif game["claim"] == "locked":
            _append(mismatches, "locked_claim_refused")
        else:
            _append(mismatches, "game_refused")
        for tag in game["mismatches"]:
            _append(mismatches, tag)
    else:
        if game["claim"] != "locked":
            _append(mismatches, "unlocked_claim")
        if game["verdict"] != "PASS":
            _append(mismatches, "game_not_pass")
        if not game["stamp_ok"]:
            _append(mismatches, "stamp_drift")
        for tag in game["mismatches"]:
            _append(mismatches, tag)

    if geo_source in REAL_SOURCES and not geometry.get("stamp_ok", False):
        _append(mismatches, "stamp_drift")

    geometry_right = (
        geo_verdict == "right"
        and geo_source in REAL_SOURCES
        and geometry.get("stamp_ok", False)
        and geometry["phrase"] == arm.phrase
    )
    game_locked_pass = (
        not game["refused"]
        and game["verdict"] == "PASS"
        and game["claim"] == "locked"
        and game["stamp_ok"]
        and not game["mismatches"]
    )
    if geometry_right and not game_locked_pass:
        _append(mismatches, "geometry_only")
    claimed_right = arm.claimed_geometry == "right" and not geometry_right
    if claimed_right:
        _append(mismatches, "claimed_geometry_right")
    if arm.claimed_pass and not (game_locked_pass and geometry_right):
        _append(mismatches, "claimed_pass")

    verdict = "PASS" if game_locked_pass and geometry_right and not mismatches else "FAIL"
    if arm.role == "negative" and verdict == "PASS":
        raise HonestyError(f"negative arm {arm.name} scored PASS")
    if arm.role == "bridge" and kind != "both":
        raise HonestyError("bridge role is only a locked game plus geometric right")
    if arm.role not in ("bridge", "negative"):
        raise HonestyError(f"unknown role {arm.role!r}")

    return {
        "arm": arm.name,
        "phrase": arm.phrase,
        "role": arm.role,
        "kind": kind,
        "verdict": verdict,
        "mismatches": mismatches,
        "game_verdict": "REFUSED" if game["refused"] else game["verdict"],
        "claim": game["claim"],
        "pairing": game["pairing"],
        "game_steps": game["step"],
        "geometry": geo_verdict,
        "geometry_source": geo_source,
        "geometry_name": geometry.get("name", ""),
        "geometry_note": geometry.get("note", ""),
        "stamp_ok": bool(game["stamp_ok"]) and (
            geo_source not in REAL_SOURCES or bool(geometry.get("stamp_ok"))
        ),
        "steps": budget["steps"],
        "lr": budget["lr"],
        "seed": budget["seed"],
        "budget": "2d_fixture",
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
        "game_pass_is_sufficient": False,
        "error": game.get("error", ""),
        "note": NOT_A_TRANSFER,
    }


def evaluate(
    arm: BridgeArm,
    *,
    budget: dict | None = None,
    geometry_cache: dict | None = None,
    game_cache: dict | None = None,
) -> dict:
    """Score one arm. Geometric trains are cached by phrase and budget."""
    budget = dict(fixture_budget() if budget is None else budget)
    for key in ("steps", "lr", "seed"):
        if key not in budget:
            raise HonestyError(f"fixture budget is missing {key}")
    geometry_cache = {} if geometry_cache is None else geometry_cache
    game_cache = {} if game_cache is None else game_cache
    geometry = _geometry_for(arm, geometry_cache, budget)
    game = _game_for(arm, game_cache, int(budget["seed"]))
    return _row(arm, game, geometry, budget)


def claim_bridge_pass(row: dict) -> dict:
    """Accept one row as a bridge PASS.

    Game PASS with skipped, faked, dead, or recipe geometry raises
    :class:`HonestyError`. Geometric ``right`` without a locked game
    raises too.
    """
    if row.get("game_pass_is_sufficient"):
        raise HonestyError("game PASS is not a geometric verdict")
    if row.get("music_gpu_transfer") or row.get("anima_gpu_transfer"):
        raise HonestyError("a bridge PASS is not a Music or Anima GPU transfer")
    if row["geometry_source"] not in REAL_SOURCES or row["geometry"] != "right":
        raise HonestyError(
            "game PASS is not geometric right "
            f"(geometry={row['geometry']!r} source={row['geometry_source']!r})"
        )
    if row["game_verdict"] != "PASS" or row["claim"] != "locked":
        raise HonestyError("geometric right without a locked game is not a bridge PASS")
    if row["role"] != "bridge":
        raise HonestyError(f"{row['arm']} is not a bridge arm")
    if row["mismatches"] or row["verdict"] != "PASS":
        raise HonestyError(
            "bridge PASS cannot carry mismatches: " + ", ".join(row["mismatches"])
        )
    if not row["stamp_ok"] or row["budget"] != "2d_fixture":
        raise HonestyError("bridge PASS left the locked stamp or the 2-D fixture budget")
    return {
        "verdict": "PASS",
        "arm": row["arm"],
        "phrase": row["phrase"],
        "toy": TOY_ID,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
        "game_pass_is_sufficient": False,
        "budget": "2d_fixture",
    }


def claim_board(rows: list[dict] | tuple[dict, ...]) -> dict:
    """Family PASS: every bridge arm passed, every mismatch arm failed.

    An empty negative list raises :class:`HonestyError`, same rule as
    :func:`conceptmod.toys.claim_pass`.
    """
    rows = list(rows)
    by_arm = {row["arm"]: row for row in rows}
    if len(by_arm) != len(rows):
        raise HonestyError("duplicate arm on the bridge board")
    negatives = [row for row in rows if row["role"] == "negative"]
    if not negatives:
        raise HonestyError(
            "bridge PASS cannot be claimed without a declared negative on the same toy"
        )
    catalog = arms()
    required_bridge = [arm.name for arm in catalog if arm.role == "bridge"]
    required_neg = [arm.name for arm in catalog if arm.role == "negative"]
    missing = [name for name in required_bridge + required_neg if name not in by_arm]
    if missing:
        raise HonestyError(f"board is missing arms {missing}")
    winners = []
    for name in required_bridge:
        row = by_arm[name]
        winners.append(claim_bridge_pass(row)["arm"])
    failed = []
    for name in required_neg:
        row = by_arm[name]
        if row["verdict"] != "FAIL" or row["role"] != "negative":
            raise HonestyError(
                f"declared mismatch arm {name!r} did not fail; refusing PASS"
            )
        failed.append(name)
    return {
        "verdict": "PASS",
        "winner": winners,
        "negatives": failed,
        "toy": TOY_ID,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
        "game_pass_is_sufficient": False,
        "budget": "2d_fixture",
    }


def run_board(
    *,
    steps: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
    log: bool = False,
) -> list[dict]:
    """Score every catalog arm. Bridge phrases train once and are reused."""
    budget = fixture_budget()
    if steps is not None:
        budget["steps"] = steps
    if lr is not None:
        budget["lr"] = lr
    if seed is not None:
        budget["seed"] = seed
    geometry_cache: dict = {}
    game_cache: dict = {}
    rows = []
    for arm in arms():
        row = evaluate(
            arm,
            budget=budget,
            geometry_cache=geometry_cache,
            game_cache=game_cache,
        )
        if log:
            print(_line(row), flush=True)
        rows.append(row)
    return rows


def _line(row: dict) -> str:
    why = ",".join(row["mismatches"]) or "-"
    return (
        "dsl_game_geometry arm=%s phrase=%s game=%s claim=%s geometry=%s "
        "source=%s verdict=%s mismatches=%s"
        % (
            row["arm"],
            row["phrase"],
            row["game_verdict"],
            row["claim"],
            row["geometry"],
            row["geometry_source"],
            row["verdict"],
            why,
        )
    )


def format_board(rows: list[dict]) -> str:
    header = "| arm | phrase | game | claim | geometry | source | gate |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        why = row["verdict"] if row["verdict"] == "PASS" else (
            row["verdict"] + " " + " ".join(row["mismatches"])
        )
        phrase = row["phrase"].replace("|", "\\|")
        lines.append(
            "| %s | `%s` | %s | %s | %s | %s | %s |"
            % (
                row["arm"],
                phrase,
                row["game_verdict"],
                row["claim"],
                row["geometry"],
                row["geometry_source"],
                why,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = run_board(log=True)
    print(format_board(rows), end="")
    try:
        family = claim_board(rows)
    except HonestyError as exc:
        print(f"dsl_game_geometry family=FAIL {exc}", flush=True)
        return 1
    print(
        "dsl_game_geometry family=%s pass=%d fail=%d budget=2d_fixture "
        "game_pass_is_sufficient=0"
        % (family["verdict"], len(family["winner"]), len(family["negatives"])),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
