"""Cross-toy formulation suite leaderboard (CPU).

Answers: which adv config(s) PASS every applicable CPU toy?

This is **not** :mod:`conceptmod.toys.leaderboard_honesty` (that module is the
single-toy two-pole honesty gate). This module scores a small set of
**candidate configs** across the formulation toy families and prints a
config × toy matrix (PASS / FAIL / N/A).

Applicability
-------------
* **stamp** — locked_shared / demo stamp toys. A suite *stamp sweep* winner
  must PASS every applicable stamp column (N/A excluded).
* **posture** — :mod:`cover_posture_fork`. Demo 1.5 and Music 1.0 can both
  PASS under their own claims; the suite does not force one cover pin to
  win both.
* **dsl** — phrase / expand / game-geometry honesty. Scored under
  ``locked_shared`` only (or N/A for cover drifts). Not a second phrase
  recipe per config.

Culture: a suite PASS is a CPU formulation result. It is **not** a Music /
Anima / Supra GPU transfer.

Run::

    python -m conceptmod.toys.suite_leaderboard
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Callable, Iterable, Mapping

from conceptmod.toys.leaderboard_honesty import HonestyError
from conceptmod.toys.locked_shared_floor import (
    COVER_WEIGHT,
    FM_WEIGHT,
    LOCKED,
    N_PARTICLES,
    PARTICLE_L2,
    LockedSharedFloor,
    run_floor,
    score_floor,
)

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_NA = "N/A"

KIND_STAMP = "stamp"
KIND_POSTURE = "posture"
KIND_DSL = "dsl"


@dataclass(frozen=True)
class SuiteCandidate:
    """One row on the suite board.

    Fields that match :data:`LOCKED` are the demo locked_shared stamp.
    Named drifts must be reported when a toy demands a drift report.
    """

    name: str
    cover_weight: float = COVER_WEIGHT
    cover_posture: str = "demo"
    fm_weight: float = FM_WEIGHT
    gan_mode: str = "rp"
    n_particles: int = N_PARTICLES
    particle_l2: float = PARTICLE_L2
    pairing: str = "pair"  # pair | stranger
    regularizer: str = "faithful"  # faithful | thinned
    reported_drift: Mapping[str, object] = field(default_factory=dict)

    def is_locked_shared(self) -> bool:
        return (
            self.cover_weight == COVER_WEIGHT
            and self.cover_posture == "demo"
            and self.fm_weight == FM_WEIGHT
            and self.gan_mode == "rp"
            and self.n_particles == N_PARTICLES
            and self.particle_l2 == PARTICLE_L2
            and self.pairing == "pair"
            and self.regularizer == "faithful"
        )

    def floor_cfg(self) -> LockedSharedFloor:
        return replace(
            LOCKED,
            cover_weight=self.cover_weight,
            cover_posture=self.cover_posture,
            fm_weight=self.fm_weight,
            gan_mode=self.gan_mode,
            n_particles=self.n_particles,
            particle_l2=self.particle_l2,
        )

    def drift_keys(self) -> frozenset[str]:
        keys = []
        if self.cover_weight != COVER_WEIGHT or self.cover_posture != "demo":
            keys.append("cover")
        if self.fm_weight != FM_WEIGHT:
            keys.append("fm")
        if self.gan_mode != "rp":
            keys.append("gan_mode")
        if self.n_particles != N_PARTICLES:
            keys.append("n_particles")
        if self.particle_l2 != PARTICLE_L2:
            keys.append("particle_l2")
        if self.pairing != "pair":
            keys.append("pairing")
        if self.regularizer != "faithful":
            keys.append("thinned")
        return frozenset(keys)


@dataclass(frozen=True)
class SuiteCell:
    config: str
    toy: str
    kind: str
    verdict: str
    reason: str = ""

    def line(self) -> str:
        why = self.reason or "-"
        return f"suite config={self.config} toy={self.toy} kind={self.kind} verdict={self.verdict} reason={why}"


def default_candidates() -> tuple[SuiteCandidate, ...]:
    """Candidate rows. Demo lock first; named drifts after."""
    music_report = {"cover_weight": 1.0, "cover_posture": "music"}
    return (
        SuiteCandidate(name="locked_shared"),
        SuiteCandidate(
            name="music_cover_1_0",
            cover_weight=1.0,
            cover_posture="music",
            reported_drift=music_report,
        ),
        SuiteCandidate(
            name="stranger_pair",
            pairing="stranger",
            reported_drift={"pairing": "stranger"},
        ),
        SuiteCandidate(
            name="fm_on",
            fm_weight=0.1,
            reported_drift={"fm_weight": 0.1},
        ),
        SuiteCandidate(
            name="thinned_kappa",
            regularizer="thinned",
            reported_drift={"regularizer": "thinned"},
        ),
        SuiteCandidate(
            name="vanilla_logistic",
            gan_mode="vanilla",
            reported_drift={"gan_mode": "vanilla"},
        ),
        SuiteCandidate(
            name="hub128",
            n_particles=128,
            reported_drift={"n_particles": 128},
        ),
    )


def _cell(
    candidate: SuiteCandidate,
    toy: str,
    kind: str,
    verdict: str,
    reason: str = "",
) -> SuiteCell:
    return SuiteCell(
        config=candidate.name,
        toy=toy,
        kind=kind,
        verdict=verdict,
        reason=reason,
    )


def _pass(candidate: SuiteCandidate, toy: str, kind: str, reason: str = "") -> SuiteCell:
    return _cell(candidate, toy, kind, VERDICT_PASS, reason or "locked arm / stamp match")


def _fail(candidate: SuiteCandidate, toy: str, kind: str, reason: str) -> SuiteCell:
    return _cell(candidate, toy, kind, VERDICT_FAIL, reason)


def _na(candidate: SuiteCandidate, toy: str, kind: str, reason: str) -> SuiteCell:
    return _cell(candidate, toy, kind, VERDICT_NA, reason)


def _verdict_from_bool(ok: bool) -> str:
    return VERDICT_PASS if ok else VERDICT_FAIL


# ---------------------------------------------------------------------------
# Per-family scorers (reuse existing entrypoints; do not reimplement trains)
# ---------------------------------------------------------------------------


def score_locked_shared_floor(candidate: SuiteCandidate) -> SuiteCell:
    toy = "locked_shared_floor"
    obs = run_floor(
        candidate.floor_cfg(),
        regularizer=candidate.regularizer,
        pairing=candidate.pairing,
        arm=candidate.name,
    )
    score = score_floor(obs)
    why = "; ".join(score["mismatches"]) if score["mismatches"] else "demo LOCKED stamp"
    return _cell(candidate, toy, KIND_STAMP, score["verdict"], why)


def score_keep_critic(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys import keep_critic as kc

    toy = "keep_critic"
    cfg = replace(
        kc.LOCKED,
        cover_weight=candidate.cover_weight,
        cover_posture=candidate.cover_posture,
        fm_weight=candidate.fm_weight,
        gan_mode=candidate.gan_mode,
        n_particles=candidate.n_particles,
        particle_l2=candidate.particle_l2,
    )
    obs = kc.run_keep(
        cfg,
        regularizer=candidate.regularizer,
        pairing=candidate.pairing,
        arm=candidate.name,
    )
    score = kc.score_keep(obs)
    why = "; ".join(score["mismatches"]) if score["mismatches"] else "frozen host + demo stamp"
    return _cell(candidate, toy, KIND_STAMP, score["verdict"], why)


@lru_cache(maxsize=1)
def _honesty_board():
    from conceptmod.toys.leaderboard_honesty import demo_arms, run_honesty_board

    return run_honesty_board(demo_arms())


def score_leaderboard_honesty(candidate: SuiteCandidate) -> SuiteCell:
    toy = "leaderboard_honesty"
    board = _honesty_board()
    cells = {cell.name: cell for cell in board["cells"]}
    arm_map = {
        "locked_shared": "locked_shared",
        "stranger_pair": "stranger_pairing",
        "thinned_kappa": "thinned_b_cap",
    }
    if candidate.name in arm_map:
        cell = cells[arm_map[candidate.name]]
        if candidate.name == "locked_shared":
            ok = board["verdict"]["verdict"] == VERDICT_PASS and cell.won
            why = "locked_shared won; declared negatives failed"
        else:
            # Config PASSes the toy only when its arm wins. Declared
            # negatives are supposed to lose → suite cell is FAIL.
            ok = bool(cell.won)
            why = f"honesty arm={cell.name} won={int(cell.won)}"
        return _cell(candidate, toy, KIND_STAMP, _verdict_from_bool(ok), why)
    if "cover" in candidate.drift_keys():
        return _na(candidate, toy, KIND_STAMP, "cover posture not an honesty cell")
    if candidate.drift_keys() & {"fm", "gan_mode", "n_particles", "particle_l2"}:
        return _na(
            candidate,
            toy,
            KIND_STAMP,
            "no honesty arm for this adv drift; floor/keep catch it",
        )
    return _fail(candidate, toy, KIND_STAMP, "not the locked honesty winner")


@lru_cache(maxsize=1)
def _shared_traj_board():
    from conceptmod.toys import shared_trajectory as st

    locked = st.train_locked(echo=False)
    stranger = st.train_drift("stranger", echo=False)
    nearest = st.train_drift("nearest_stranger", echo=False)
    return {"shared": locked, "stranger": stranger, "nearest_stranger": nearest}


def score_shared_trajectory(candidate: SuiteCandidate) -> SuiteCell:
    toy = "shared_trajectory"
    board = _shared_traj_board()
    if candidate.is_locked_shared():
        row = board["shared"]
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(row["pass"]),
            f"identity_mse={row['identity_mse']:.5f}",
        )
    if "pairing" in candidate.drift_keys() and candidate.pairing == "stranger":
        row = board["stranger"]
        return _fail(
            candidate,
            toy,
            KIND_STAMP,
            f"stranger identity_mse={row['identity_mse']:.5f}",
        )
    if candidate.drift_keys() <= {"pairing"}:
        return _fail(candidate, toy, KIND_STAMP, "pairing drift from shared")
    return _na(candidate, toy, KIND_STAMP, "pairing-only family; other knobs not forked")


@lru_cache(maxsize=1)
def _residual_board():
    from conceptmod.toys import residual_student as rs

    return {row["pairing"]: row for row in rs.run_family(echo=False)}


def score_residual_student(candidate: SuiteCandidate) -> SuiteCell:
    toy = "residual_student"
    board = _residual_board()
    if candidate.is_locked_shared():
        row = board["shared"]
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(row["pass"]),
            f"identity_mse={row['identity_mse']:.5f} success={row['success_rate']:.3f}",
        )
    if "pairing" in candidate.drift_keys() and candidate.pairing == "stranger":
        row = board["stranger"]
        return _fail(
            candidate,
            toy,
            KIND_STAMP,
            f"stranger success={row['success_rate']:.3f}",
        )
    return _na(candidate, toy, KIND_STAMP, "pairing-only residual family")


def score_orbit_hold(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.orbit_hold import (
        HardcodedKappaBCap,
        evaluate,
        locked_recipe,
    )

    toy = "orbit_hold"
    if candidate.is_locked_shared():
        report = evaluate(locked_recipe(name="locked_shared"))
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(report.passed),
            ",".join(report.reasons) or "radius hold",
        )
    if "pairing" in candidate.drift_keys():
        report = evaluate(locked_recipe(name="stranger_shuffle", pairing="stranger"))
        return _fail(candidate, toy, KIND_STAMP, ",".join(report.reasons) or "stranger")
    if "fm" in candidate.drift_keys():
        report = evaluate(locked_recipe(name="fm_on", fm_weight=1.0))
        return _fail(candidate, toy, KIND_STAMP, ",".join(report.reasons) or "fm_on")
    if "thinned" in candidate.drift_keys():
        report = evaluate(locked_recipe(name="thin_bcap"), HardcodedKappaBCap())
        return _fail(candidate, toy, KIND_STAMP, ",".join(report.reasons) or "thinned_bcap")
    if "gan_mode" in candidate.drift_keys():
        report = evaluate(locked_recipe(name="stranger_vanilla", gan_mode="vanilla"))
        return _fail(candidate, toy, KIND_STAMP, ",".join(report.reasons) or "vanilla")
    if "cover" in candidate.drift_keys() or "n_particles" in candidate.drift_keys():
        return _na(candidate, toy, KIND_STAMP, "orbit does not fork cover / hub cloud")
    return _fail(candidate, toy, KIND_STAMP, "not locked_shared orbit")


@lru_cache(maxsize=1)
def _unipolar_locked():
    from conceptmod.toys.unipolar import run_arm

    return run_arm("locked_rpgan", steps=400, seed=0)


def score_unipolar(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.unipolar import UnipolarRecipe, _refuse_unless_locked_adv

    toy = "unipolar"
    if candidate.is_locked_shared():
        row = _unipolar_locked()
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(row["hit"]),
            "locked_rpgan" if row["hit"] else "unipolar gates missed",
        )
    # Unipolar train cover is 0; Music cover is not this gate. Adv drifts refuse.
    recipe = UnipolarRecipe(
        arm="locked_rpgan",
        fm_weight=candidate.fm_weight if "fm" in candidate.drift_keys() else 0.0,
        gan_mode=candidate.gan_mode,
        cover_weight=0.0,
    )
    if "thinned" in candidate.drift_keys():
        recipe = UnipolarRecipe(arm="locked_rpgan", reg_kappa=0.2)
    if "pairing" in candidate.drift_keys() or "gan_mode" in candidate.drift_keys():
        recipe = UnipolarRecipe(arm="locked_rpgan", gan_mode="vanilla")
    if "fm" in candidate.drift_keys():
        recipe = UnipolarRecipe(arm="locked_rpgan", fm_weight=0.1)
    try:
        if candidate.drift_keys() & {"fm", "gan_mode", "pairing", "thinned"}:
            _refuse_unless_locked_adv(recipe)
    except ValueError as exc:
        return _fail(candidate, toy, KIND_STAMP, f"refused: {exc}")
    if "cover" in candidate.drift_keys() or "n_particles" in candidate.drift_keys():
        return _na(
            candidate,
            toy,
            KIND_STAMP,
            "unipolar train cover is 0; cloud omitted",
        )
    return _fail(candidate, toy, KIND_STAMP, "not locked_rpgan")


@lru_cache(maxsize=1)
def _ae_gan_locked():
    from conceptmod.toys.ae_gan_hold import locked_config, train

    return train(locked_config())


def score_ae_gan_hold(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.ae_gan_hold import (
        fm_on_config,
        shape_mismatches,
        stranger_config,
    )

    toy = "ae_gan_hold"
    if candidate.is_locked_shared():
        row = _ae_gan_locked()
        ok = row["verdict"] == VERDICT_PASS
        why = "; ".join(row["reasons"]) if row["reasons"] else "locked AE-GAN hold"
        return _cell(candidate, toy, KIND_STAMP, _verdict_from_bool(ok), why)
    if "gan_mode" in candidate.drift_keys() or "pairing" in candidate.drift_keys():
        bad = shape_mismatches(stranger_config())
        return _fail(candidate, toy, KIND_STAMP, "; ".join(bad) or "stranger_pairing")
    if "fm" in candidate.drift_keys():
        bad = shape_mismatches(fm_on_config())
        return _fail(candidate, toy, KIND_STAMP, "; ".join(bad) or "fm_on")
    if "thinned" in candidate.drift_keys():
        return _fail(candidate, toy, KIND_STAMP, "thinned b_cap refused on AE-GAN hold")
    if "cover" in candidate.drift_keys() or "n_particles" in candidate.drift_keys():
        return _na(candidate, toy, KIND_STAMP, "AE-GAN hold forks adv shape, not music cover/hub")
    return _fail(candidate, toy, KIND_STAMP, "not locked AE-GAN")


@lru_cache(maxsize=1)
def _cover_leftover_locked():
    from conceptmod.toys.cover_leftover import CoverRecipe, fit_cover_leftover

    return fit_cover_leftover(CoverRecipe(arm="locked"))


def score_cover_leftover(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.cover_leftover import reject_unlocked

    toy = "cover_leftover"
    if candidate.is_locked_shared():
        row = _cover_leftover_locked()
        why = row.get("fail_reasons") or "demo cover + faithful guard"
        return _cell(candidate, toy, KIND_STAMP, _verdict_from_bool(row["pass"]), why)
    overrides = {}
    if "fm" in candidate.drift_keys():
        overrides["fm_weight"] = candidate.fm_weight
    if "gan_mode" in candidate.drift_keys() or "pairing" in candidate.drift_keys():
        overrides["gan_mode"] = "vanilla"
    if "thinned" in candidate.drift_keys():
        overrides["reg_kappa"] = 0.2
    if "n_particles" in candidate.drift_keys():
        overrides["n_particles"] = candidate.n_particles
    if overrides:
        try:
            reject_unlocked(overrides, arm="locked")
        except ValueError as exc:
            return _fail(candidate, toy, KIND_STAMP, f"refused: {exc}")
    if "cover" in candidate.drift_keys():
        return _fail(
            candidate,
            toy,
            KIND_STAMP,
            "Music cover 1.0 is not the leftover demo pin 1.5",
        )
    return _na(candidate, toy, KIND_STAMP, "no cover/leftover arm for this candidate")


@lru_cache(maxsize=1)
def _erase_keep_locked_ok() -> bool:
    from conceptmod.toys.erase_keep_backend import ARMS, score_backend
    from conceptmod.backends import load_backend

    backend = load_backend("cpu", device="cpu", lora_rank=4, seed=0)
    row = score_backend(backend, "locked", name="cpu", seed=0)
    assert "locked" in ARMS
    return bool(row["pass"])


def score_erase_keep_backend(candidate: SuiteCandidate) -> SuiteCell:
    toy = "erase_keep_backend"
    if candidate.is_locked_shared():
        ok = _erase_keep_locked_ok()
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(ok),
            "cpu locked geometry" if ok else "locked geometry failed",
        )
    if "cover" in candidate.drift_keys() and candidate.cover_weight == 0.0:
        return _fail(candidate, toy, KIND_STAMP, "cover_zero undershoot")
    # Cover posture / adv pairing are not this backend's arms.
    return _na(
        candidate,
        toy,
        KIND_STAMP,
        "erase/keep backend arms are geometry plants, not adv-config rows",
    )


@lru_cache(maxsize=1)
def _particle_board():
    from conceptmod.toys.particle_posture import run_family

    return {row["name"]: row for row in run_family()}


def score_particle_posture(candidate: SuiteCandidate) -> SuiteCell:
    toy = "particle_posture"
    board = _particle_board()
    if candidate.is_locked_shared():
        row = board["locked_tiny"]
        why = "; ".join(row["reasons"]) if row["reasons"] else "locked_tiny n=12"
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(row["passed"]),
            why,
        )
    if "cover" in candidate.drift_keys() and candidate.n_particles == 0:
        row = board["music_parts0"]
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(row["passed"]),
            "music_parts0 empty cloud",
        )
    if "n_particles" in candidate.drift_keys() and candidate.n_particles == 128:
        row = board["hub128_routed"]
        why = "; ".join(row["reasons"]) if row["reasons"] else "hub128_routed"
        return _fail(candidate, toy, KIND_STAMP, why)
    if "cover" in candidate.drift_keys():
        # Music cover with n=12 is not music_parts0; leftover of posture fork.
        return _na(
            candidate,
            toy,
            KIND_STAMP,
            "music cover with n=12 is cover_posture_fork, not music_parts0",
        )
    return _na(candidate, toy, KIND_STAMP, "no particle-posture arm for this drift")


@lru_cache(maxsize=1)
def _mode_hold_locked():
    from conceptmod.toys.mode_hold import locked_recipe, train_mode_hold, verdict

    row = train_mode_hold(locked_recipe(), log=False)
    row = dict(row)
    row["gate"] = verdict(row)
    return row


def score_mode_hold(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.mode_hold import (
        ModeHoldRecipe,
        formulation_mismatches,
        require_drift_report,
    )

    toy = "mode_hold"
    if candidate.is_locked_shared():
        row = _mode_hold_locked()
        ok = row["gate"] == VERDICT_PASS
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(ok),
            f"modes={row.get('modes')} HQ={row.get('hq', '?')}",
        )
    recipe = ModeHoldRecipe()
    drift = None
    if "fm" in candidate.drift_keys():
        recipe = ModeHoldRecipe(fm_weight=0.1)
        drift = {"fm_weight": "fm_on under b_cap"}
    elif "gan_mode" in candidate.drift_keys() or "pairing" in candidate.drift_keys():
        recipe = ModeHoldRecipe(gan_mode="vanilla")
        drift = {"gan_mode": "stranger/vanilla pairing"}
    elif "thinned" in candidate.drift_keys():
        return _fail(candidate, toy, KIND_STAMP, "thinned kappa stub refused unless named")
    elif "cover" in candidate.drift_keys():
        recipe = ModeHoldRecipe(cover_weight=candidate.cover_weight)
        drift = {"cover_weight": "music cover posture"}
    elif "n_particles" in candidate.drift_keys():
        recipe = ModeHoldRecipe(n_particles=candidate.n_particles)
        drift = {"n_particles": "hub-scale cloud"}
    else:
        return _na(candidate, toy, KIND_STAMP, "mode-hold does not fork this candidate")
    try:
        require_drift_report(recipe, drift)
    except ValueError as exc:
        return _fail(candidate, toy, KIND_STAMP, f"refused: {exc}")
    bad = formulation_mismatches(recipe)
    if bad:
        return _fail(candidate, toy, KIND_STAMP, "; ".join(bad))
    return _fail(candidate, toy, KIND_STAMP, "drift from locked mode-hold")


@lru_cache(maxsize=1)
def _late_collapse_board():
    from conceptmod.toys.late_collapse import demo_arms, run_selection_board

    return run_selection_board(demo_arms())


def score_late_collapse(candidate: SuiteCandidate) -> SuiteCell:
    toy = "late_collapse"
    board = _late_collapse_board()
    if candidate.is_locked_shared():
        ok = board["verdict"]["verdict"] == VERDICT_PASS
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(ok),
            "val gate exports good checkpoint" if ok else "selection board failed",
        )
    # Selection rule is not an adv-config knob.
    return _na(candidate, toy, KIND_STAMP, "checkpoint selection, not an adv-config arm")


def score_field_lift(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.field_lift import LiftArm, evaluate

    toy = "field_lift"
    if candidate.is_locked_shared():
        plane = evaluate(LiftArm("locked_2d", "lift", frame="plane"))
        lift = evaluate(LiftArm("locked_lift", "lift", frame="tilted"))
        ok = plane["pass"] and lift["pass"]
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(ok),
            "locked_2d+locked_lift" if ok else f"{plane['reasons']}|{lift['reasons']}",
        )
    if "pairing" in candidate.drift_keys():
        row = evaluate(LiftArm("stranger_pairing", "lift", frame="tilted", pairing="stranger"))
        return _fail(candidate, toy, KIND_STAMP, ",".join(row["reasons"]) or "stranger")
    if "fm" in candidate.drift_keys():
        row = evaluate(LiftArm("fm_on", "lift", frame="tilted", fm_weight=0.1))
        return _fail(candidate, toy, KIND_STAMP, ",".join(row["reasons"]) or "fm_on")
    if "thinned" in candidate.drift_keys():
        row = evaluate(LiftArm("thinned_kappa", "lift", frame="tilted", regularizer="thinned"))
        return _fail(candidate, toy, KIND_STAMP, ",".join(row["reasons"]) or "thinned")
    if "gan_mode" in candidate.drift_keys():
        row = evaluate(LiftArm("stranger_pairing", "lift", frame="tilted", gan_mode="vanilla"))
        return _fail(candidate, toy, KIND_STAMP, ",".join(row["reasons"]) or "vanilla")
    return _na(candidate, toy, KIND_STAMP, "field lift does not fork cover/hub here")


@lru_cache(maxsize=1)
def _lm_target_locked():
    from conceptmod.toys.lm_target import run_arm

    return run_arm("trajectory")


def score_lm_target(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.lm_target import reject_unlocked

    toy = "lm_target"
    if candidate.is_locked_shared():
        row = _lm_target_locked()
        ok = bool(row["passed"])
        why = ",".join(row["fail_reasons"]) if row["fail_reasons"] else "trajectory"
        return _cell(candidate, toy, KIND_STAMP, _verdict_from_bool(ok), why)
    overrides = {}
    if "fm" in candidate.drift_keys():
        overrides["fm_weight"] = candidate.fm_weight
    if "gan_mode" in candidate.drift_keys() or "pairing" in candidate.drift_keys():
        overrides["gan_mode"] = "vanilla"
    if "thinned" in candidate.drift_keys():
        overrides["reg_kappa"] = 0.2
    if overrides:
        try:
            reject_unlocked(overrides)
        except ValueError as exc:
            return _fail(candidate, toy, KIND_STAMP, f"refused: {exc}")
    return _na(candidate, toy, KIND_STAMP, "lm_target forks trajectory target, not this drift")


@lru_cache(maxsize=1)
def _unused_board():
    from conceptmod.toys.unused_token_hold import run_family

    return {row["name"]: row for row in run_family()}


def score_unused_token_hold(candidate: SuiteCandidate) -> SuiteCell:
    toy = "unused_token_hold"
    board = _unused_board()
    if candidate.is_locked_shared():
        row = board["locked_shared"]
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(row["passed"]),
            ",".join(row.get("reasons") or ()) or "unused hold",
        )
    arm_map = {
        "stranger_pair": "stranger_pairing",
        "fm_on": "fm_on",
        "thinned_kappa": "thin_bcap",
        "vanilla_logistic": "stranger_vanilla",
    }
    if candidate.name in arm_map:
        row = board[arm_map[candidate.name]]
        return _fail(
            candidate,
            toy,
            KIND_STAMP,
            ",".join(row.get("reasons") or ()) or arm_map[candidate.name],
        )
    return _na(candidate, toy, KIND_STAMP, "unused-token hold has no arm for this candidate")


@lru_cache(maxsize=1)
def _path_suffix_board():
    from conceptmod.toys.path_suffix_lora import run_board

    return {row.name: row for row in run_board()}


def score_path_suffix_lora(candidate: SuiteCandidate) -> SuiteCell:
    toy = "path_suffix_lora"
    board = _path_suffix_board()
    if candidate.is_locked_shared():
        locked = board["locked_suffix"]
        regex = board["locked_regex"]
        ok = locked.passed and regex.passed
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(ok),
            "locked_suffix+locked_regex" if ok else "suffix coverage failed",
        )
    return _na(candidate, toy, KIND_STAMP, "LoRA path targets, not an adv-config knob")


@lru_cache(maxsize=1)
def _mid_scale_locked():
    from conceptmod.toys.mid_scale_identity import run_arm

    return run_arm("locked")


def score_mid_scale_identity(candidate: SuiteCandidate) -> SuiteCell:
    from conceptmod.toys.mid_scale_identity import reject_unlocked

    toy = "mid_scale_identity"
    if candidate.is_locked_shared():
        row = _mid_scale_locked()
        ok = bool(row.get("pass"))
        return _cell(
            candidate,
            toy,
            KIND_STAMP,
            _verdict_from_bool(ok),
            row.get("fail_reasons") or "locked mid-scale grid",
        )
    overrides = {}
    if "fm" in candidate.drift_keys():
        overrides["fm_weight"] = candidate.fm_weight
    if "gan_mode" in candidate.drift_keys() or "pairing" in candidate.drift_keys():
        overrides["gan_mode"] = "vanilla"
    if "thinned" in candidate.drift_keys():
        overrides["reg_kappa"] = 0.2
    if overrides:
        try:
            reject_unlocked(overrides, arm="locked")
        except ValueError as exc:
            return _fail(candidate, toy, KIND_STAMP, f"refused: {exc}")
    if "pairing" in candidate.drift_keys():
        return _fail(candidate, toy, KIND_STAMP, "stranger_pairing")
    return _na(candidate, toy, KIND_STAMP, "mid-scale forks eval grid / residual, not this drift")


def score_cover_posture_demo(candidate: SuiteCandidate) -> SuiteCell:
    """Posture column under the demo_1_5 claim."""
    from conceptmod.toys.cover_posture_fork import DEMO_CLAIM, evaluate_claim

    toy = "cover_posture_fork_demo"
    row = evaluate_claim(
        candidate.floor_cfg(),
        claim=DEMO_CLAIM,
        reported_drift=dict(candidate.reported_drift),
        pairing=candidate.pairing,
        regularizer=candidate.regularizer,
        arm=candidate.name,
        log=False,
    )
    why = "; ".join(row["mismatches"]) if row["mismatches"] else "demo_1_5 claim"
    return _cell(candidate, toy, KIND_POSTURE, row["verdict"], why)


def score_cover_posture_music(candidate: SuiteCandidate) -> SuiteCell:
    """Posture column under the music_1_0 claim."""
    from conceptmod.toys.cover_posture_fork import MUSIC_CLAIM, MUSIC_REPORT, evaluate_claim

    toy = "cover_posture_fork_music"
    reported = dict(candidate.reported_drift)
    if candidate.name == "music_cover_1_0" and not reported:
        reported = dict(MUSIC_REPORT)
    row = evaluate_claim(
        candidate.floor_cfg(),
        claim=MUSIC_CLAIM,
        reported_drift=reported,
        pairing=candidate.pairing,
        regularizer=candidate.regularizer,
        arm=candidate.name,
        log=False,
    )
    why = "; ".join(row["mismatches"]) if row["mismatches"] else "music_1_0 claim"
    return _cell(candidate, toy, KIND_POSTURE, row["verdict"], why)


@lru_cache(maxsize=1)
def _dsl_macro_board():
    from conceptmod.toys.dsl_macro_expand import run_board

    return run_board()


def score_dsl_macro_expand(candidate: SuiteCandidate) -> SuiteCell:
    toy = "dsl_macro_expand"
    if not candidate.is_locked_shared():
        return _na(
            candidate,
            toy,
            KIND_DSL,
            "phrase expand scored once under locked_shared only",
        )
    board = _dsl_macro_board()
    ok = board["verdict"]["verdict"] == VERDICT_PASS
    return _cell(
        candidate,
        toy,
        KIND_DSL,
        _verdict_from_bool(ok),
        "documented triple + negatives" if ok else "expand board failed",
    )


@lru_cache(maxsize=1)
def _dsl_phrase_board():
    from conceptmod.toys.dsl_phrase_jobs import run_board

    return run_board()


def score_dsl_phrase_jobs(candidate: SuiteCandidate) -> SuiteCell:
    toy = "dsl_phrase_jobs"
    if not candidate.is_locked_shared():
        return _na(
            candidate,
            toy,
            KIND_DSL,
            "phrase jobs scored once under locked_shared only",
        )
    board = _dsl_phrase_board()
    ok = board["verdict"]["verdict"] == VERDICT_PASS
    return _cell(
        candidate,
        toy,
        KIND_DSL,
        _verdict_from_bool(ok),
        "documented phrases + negatives" if ok else "phrase board failed",
    )


@lru_cache(maxsize=1)
def _dsl_game_board():
    from conceptmod.toys.dsl_game_geometry import run_board

    return {row["arm"]: row for row in run_board(log=False)}


def score_dsl_game_geometry(candidate: SuiteCandidate) -> SuiteCell:
    toy = "dsl_game_geometry"
    board = _dsl_game_board()
    if candidate.is_locked_shared():
        # Bridge PASSes when write/erase/exaggerate rows PASS.
        pass_arms = ("write", "erase_bare", "erase_freeze", "exaggerate")
        ok = all(board[name]["verdict"] == VERDICT_PASS for name in pass_arms)
        return _cell(
            candidate,
            toy,
            KIND_DSL,
            _verdict_from_bool(ok),
            "locked game + geometric right" if ok else "bridge PASS arms failed",
        )
    if "cover" in candidate.drift_keys() or "n_particles" in candidate.drift_keys():
        return _na(candidate, toy, KIND_DSL, "DSL game/geometry does not fork cover/hub")
    if "pairing" in candidate.drift_keys() or "gan_mode" in candidate.drift_keys():
        row = board.get("geometry_only_stranger") or board.get("locked_stranger_refuse")
        return _fail(
            candidate,
            toy,
            KIND_DSL,
            (row and ",".join(row.get("mismatches") or ())) or "stranger/unlocked claim",
        )
    if "fm" in candidate.drift_keys():
        row = board.get("geometry_only_fm") or board.get("locked_fm_refuse")
        return _fail(
            candidate,
            toy,
            KIND_DSL,
            (row and ",".join(row.get("mismatches") or ())) or "fm_on/unlocked claim",
        )
    if "thinned" in candidate.drift_keys():
        return _na(candidate, toy, KIND_DSL, "no thinned-kappa bridge arm")
    return _na(candidate, toy, KIND_DSL, "no DSL game arm for this candidate")


# Catalog: name → (kind, scorer). Order is the matrix column order.
TOY_CATALOG: tuple[tuple[str, str, Callable[[SuiteCandidate], SuiteCell]], ...] = (
    ("locked_shared_floor", KIND_STAMP, score_locked_shared_floor),
    ("leaderboard_honesty", KIND_STAMP, score_leaderboard_honesty),
    ("shared_trajectory", KIND_STAMP, score_shared_trajectory),
    ("residual_student", KIND_STAMP, score_residual_student),
    ("orbit_hold", KIND_STAMP, score_orbit_hold),
    ("unipolar", KIND_STAMP, score_unipolar),
    ("ae_gan_hold", KIND_STAMP, score_ae_gan_hold),
    ("cover_leftover", KIND_STAMP, score_cover_leftover),
    ("erase_keep_backend", KIND_STAMP, score_erase_keep_backend),
    ("particle_posture", KIND_STAMP, score_particle_posture),
    ("mode_hold", KIND_STAMP, score_mode_hold),
    ("late_collapse", KIND_STAMP, score_late_collapse),
    ("keep_critic", KIND_STAMP, score_keep_critic),
    ("lm_target", KIND_STAMP, score_lm_target),
    ("field_lift", KIND_STAMP, score_field_lift),
    ("unused_token_hold", KIND_STAMP, score_unused_token_hold),
    ("path_suffix_lora", KIND_STAMP, score_path_suffix_lora),
    ("mid_scale_identity", KIND_STAMP, score_mid_scale_identity),
    ("cover_posture_fork_demo", KIND_POSTURE, score_cover_posture_demo),
    ("cover_posture_fork_music", KIND_POSTURE, score_cover_posture_music),
    ("dsl_macro_expand", KIND_DSL, score_dsl_macro_expand),
    ("dsl_phrase_jobs", KIND_DSL, score_dsl_phrase_jobs),
    ("dsl_game_geometry", KIND_DSL, score_dsl_game_geometry),
)


def toy_names(*, kinds: Iterable[str] | None = None) -> tuple[str, ...]:
    wanted = None if kinds is None else frozenset(kinds)
    return tuple(
        name for name, kind, _ in TOY_CATALOG if wanted is None or kind in wanted
    )


def score_cell(candidate: SuiteCandidate, toy: str) -> SuiteCell:
    for name, _kind, scorer in TOY_CATALOG:
        if name == toy:
            return scorer(candidate)
    raise KeyError(f"unknown suite toy {toy!r}")


def run_suite(
    candidates: tuple[SuiteCandidate, ...] | list[SuiteCandidate] | None = None,
    *,
    toys: Iterable[str] | None = None,
    kinds: Iterable[str] | None = None,
    log: bool = False,
) -> dict:
    """Score every candidate × toy cell and summarize sweep winners."""
    rows = tuple(default_candidates() if candidates is None else candidates)
    if toys is not None:
        columns = tuple(toys)
    else:
        columns = toy_names(kinds=kinds)
    cells: list[SuiteCell] = []
    matrix: dict[str, dict[str, SuiteCell]] = {c.name: {} for c in rows}
    for candidate in rows:
        for toy in columns:
            cell = score_cell(candidate, toy)
            cells.append(cell)
            matrix[candidate.name][toy] = cell
            if log:
                print(cell.line(), flush=True)

    stamp_cols = [name for name, kind, _ in TOY_CATALOG if name in columns and kind == KIND_STAMP]
    posture_cols = [
        name for name, kind, _ in TOY_CATALOG if name in columns and kind == KIND_POSTURE
    ]
    dsl_cols = [name for name, kind, _ in TOY_CATALOG if name in columns and kind == KIND_DSL]

    stamp_winners = [
        c.name for c in rows if passes_all_applicable(c.name, matrix, stamp_cols)
    ]
    posture_demo_pass = [
        c.name
        for c in rows
        if "cover_posture_fork_demo" in matrix[c.name]
        and matrix[c.name]["cover_posture_fork_demo"].verdict == VERDICT_PASS
    ]
    posture_music_pass = [
        c.name
        for c in rows
        if "cover_posture_fork_music" in matrix[c.name]
        and matrix[c.name]["cover_posture_fork_music"].verdict == VERDICT_PASS
    ]
    dsl_pass = [
        c.name for c in rows if passes_all_applicable(c.name, matrix, dsl_cols)
    ]

    return {
        "candidates": [c.name for c in rows],
        "toys": list(columns),
        "stamp_toys": stamp_cols,
        "posture_toys": posture_cols,
        "dsl_toys": dsl_cols,
        "cells": cells,
        "matrix": matrix,
        "stamp_winners": stamp_winners,
        "posture_demo_pass": posture_demo_pass,
        "posture_music_pass": posture_music_pass,
        "dsl_pass": dsl_pass,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "supra_gpu_transfer": False,
        "cpu_toy": True,
    }


def passes_all_applicable(
    config: str,
    matrix: Mapping[str, Mapping[str, SuiteCell]],
    toys: Iterable[str],
) -> bool:
    """True when every non-N/A cell for ``config`` is PASS. No applicable → False."""
    applicable = []
    for toy in toys:
        cell = matrix[config][toy]
        if cell.verdict == VERDICT_NA:
            continue
        applicable.append(cell)
    if not applicable:
        return False
    return all(cell.verdict == VERDICT_PASS for cell in applicable)


def applicable_failures(
    config: str,
    matrix: Mapping[str, Mapping[str, SuiteCell]],
    toys: Iterable[str],
) -> list[SuiteCell]:
    return [
        matrix[config][toy]
        for toy in toys
        if matrix[config][toy].verdict == VERDICT_FAIL
    ]


def claim_stamp_sweep_pass(
    *,
    winner: str,
    negatives: list[str] | tuple[str, ...],
    suite: dict,
) -> dict:
    """PASS only when ``winner`` clears every applicable stamp toy and each negative fails.

    An empty negative list raises :class:`HonestyError`, including when the
    winner looks perfect.
    """
    if not negatives:
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the stamp sweep"
        )
    matrix = suite["matrix"]
    stamp_toys = suite["stamp_toys"]
    if winner not in matrix:
        raise HonestyError(f"unknown winner config {winner!r}")
    if not passes_all_applicable(winner, matrix, stamp_toys):
        fails = applicable_failures(winner, matrix, stamp_toys)
        why = ", ".join(f"{c.toy}:{c.reason}" for c in fails) or "no applicable stamp toys"
        raise HonestyError(f"{winner!r} did not PASS every applicable stamp toy ({why})")
    for name in negatives:
        if name not in matrix:
            raise HonestyError(f"unknown negative config {name!r}")
        if passes_all_applicable(name, matrix, stamp_toys):
            raise HonestyError(
                f"declared bad config {name!r} did not fail the stamp sweep; refusing PASS"
            )
        if not applicable_failures(name, matrix, stamp_toys):
            # All N/A would make passes_all_applicable False already, but be explicit.
            raise HonestyError(
                f"declared bad config {name!r} has no applicable stamp FAIL"
            )
    return {
        "verdict": VERDICT_PASS,
        "winner": winner,
        "negatives": list(negatives),
        "stamp_toys": list(stamp_toys),
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "supra_gpu_transfer": False,
        "cpu_toy": True,
    }


def format_matrix(suite: dict) -> str:
    """Readable config × toy matrix plus sweep summary."""
    toys = suite["toys"]
    # Compact header: truncate long toy names for terminal width.
    short = []
    for name in toys:
        if name.startswith("cover_posture_fork_"):
            short.append(name.replace("cover_posture_fork_", "posture_"))
        elif name.startswith("dsl_"):
            short.append(name)
        else:
            short.append(name[:18])
    header = "| config | " + " | ".join(short) + " |"
    sep = "|---|" + "|".join(["---:"] * len(toys)) + "|"
    lines = [
        "SUITE formulation leaderboard (CPU). PASS all stamp ≠ Music/Anima/Supra transfer.",
        header,
        sep,
    ]
    for config in suite["candidates"]:
        cells = suite["matrix"][config]
        vals = []
        for toy in toys:
            cell = cells[toy]
            if cell.verdict == VERDICT_PASS:
                vals.append("PASS")
            elif cell.verdict == VERDICT_FAIL:
                vals.append("FAIL")
            else:
                vals.append("N/A")
        lines.append("| " + config + " | " + " | ".join(vals) + " |")

    na_notes = []
    for cell in suite["cells"]:
        if cell.verdict == VERDICT_NA:
            na_notes.append(f"  - {cell.config} × {cell.toy}: {cell.reason}")

    lines.append("")
    lines.append(
        "stamp_winners="
        + (",".join(suite["stamp_winners"]) if suite["stamp_winners"] else "(none)")
    )
    lines.append(
        "posture_demo_pass="
        + (",".join(suite["posture_demo_pass"]) if suite["posture_demo_pass"] else "(none)")
    )
    lines.append(
        "posture_music_pass="
        + (",".join(suite["posture_music_pass"]) if suite["posture_music_pass"] else "(none)")
    )
    lines.append(
        "dsl_pass=" + (",".join(suite["dsl_pass"]) if suite["dsl_pass"] else "(none)")
    )
    lines.append(
        "music_gpu_transfer=0 anima_gpu_transfer=0 supra_gpu_transfer=0 cpu_toy=1"
    )
    if na_notes:
        lines.append("N/A cells:")
        lines.extend(na_notes)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kinds",
        nargs="*",
        choices=(KIND_STAMP, KIND_POSTURE, KIND_DSL),
        default=None,
        help="limit columns to these kinds (default: all)",
    )
    parser.add_argument(
        "--stamp-only",
        action="store_true",
        help="shortcut for --kinds stamp",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print the matrix only (no per-cell lines)",
    )
    args = parser.parse_args(argv)
    kinds = (KIND_STAMP,) if args.stamp_only else args.kinds
    suite = run_suite(kinds=kinds, log=not args.quiet)
    print(format_matrix(suite), end="")
    if suite["stamp_winners"]:
        # Honesty: declare the sweep only with the known failing drifts present.
        negatives = [
            name
            for name in ("stranger_pair", "fm_on", "thinned_kappa", "vanilla_logistic", "hub128", "music_cover_1_0")
            if name in suite["candidates"] and name not in suite["stamp_winners"]
        ]
        if len(suite["stamp_winners"]) == 1 and negatives:
            claim = claim_stamp_sweep_pass(
                winner=suite["stamp_winners"][0],
                negatives=tuple(negatives),
                suite=suite,
            )
            print(
                f"honesty stamp_sweep={claim['verdict']} winner={claim['winner']} "
                f"negatives={','.join(claim['negatives'])}",
                flush=True,
            )
    return 0 if suite["stamp_winners"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
