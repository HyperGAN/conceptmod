"""Cover-posture fork honesty.

Demo cover 1.5 and Music cover 1.0 are two named postures of one
locked_shared shape. They are not silent aliases. A row PASSes only on
the posture it claims. Swapping the weight under the other name, or
shipping the Music pin with no drift report, FAILs.

Reviewed against HyperGAN/particle-sliders:

* ``analysis/slider2d/locked_baseline_defaults.py`` — ``LOCKED["cover_weight"]``
  is 1.5 (demo / #94).
* ``docs/music-arm-b-gates.md`` and ``tests/test_music_arm_b_gates.py`` —
  Music Arm B is pole/cover **1.0**. Cover 1.5 on that row is drift.
  A substring check is the alias bug: ``"cover_weight=1"`` sits inside
  ``"cover_weight=1.5"``. This gate compares the numeric pin and the
  posture name.
* ``analysis/slider2d/notes/MUSIC_TRANSFER_RECIPE_LOCKED_20260909.md`` —
  do not copy demo cover 1.5 across as the Music start. Pole weight 1.0
  is the Arm B cover analogue.

The shared shape is the locked floor: RpGAN logistic, ParticleGAN
``GradientPenalty`` ``b_cap`` (coeff=1, kappa=1, norm=l2), FM off, n=12,
``particle_l2=0.02``. The cover pin is the only fork. ``score_floor``
still claims demo only, so Music 1.0 keeps failing that stamp. This
module does not rewrite the Wave-1 cover / leftover geometry toy; it
reads ``LOCKED_COVER`` from there and runs the floor's CPU step.

Budget is the floor's 8-step identity loop, not a Music train.
A PASS here is a CPU toy. It is not a Music or Anima GPU transfer.
"""

from __future__ import annotations

import json
from dataclasses import replace

from particlegan import GANLoss, GradientPenalty

from conceptmod.toys.cover_leftover import LOCKED_COVER
from conceptmod.toys.locked_shared_floor import (
    COVER_WEIGHT,
    FM_WEIGHT,
    LOCKED,
    PARTICLE_L2,
    FloorObs,
    LockedSharedFloor,
    _kappa_probe,
    cover_mse,
    run_floor,
)
from conceptmod.toys.particle_posture import MUSIC_COVER as MUSIC_POLE_COVER


if LOCKED_COVER != COVER_WEIGHT:
    raise RuntimeError(
        "demo cover pins diverged: "
        f"cover_leftover {LOCKED_COVER} vs locked floor {COVER_WEIGHT}"
    )
if MUSIC_POLE_COVER == LOCKED_COVER:
    raise RuntimeError("Music cover and demo cover collapsed into one pin")
if MUSIC_POLE_COVER != 1.0:
    raise RuntimeError(
        f"Music cover pin {MUSIC_POLE_COVER} is not the Arm B pole 1.0"
    )

# Demo lock, composed from the leftover toy and the floor stamp.
DEMO_COVER = LOCKED_COVER
MUSIC_COVER = MUSIC_POLE_COVER
DEMO_CLAIM = "demo_1_5"
MUSIC_CLAIM = "music_1_0"
DEMO_LABEL = "demo"
MUSIC_LABEL = "music"

POSTURES = {
    DEMO_CLAIM: {"cover_weight": DEMO_COVER, "cover_posture": DEMO_LABEL},
    MUSIC_CLAIM: {"cover_weight": MUSIC_COVER, "cover_posture": MUSIC_LABEL},
}

# Named delta from the demo lock. Both keys are required; a weight-only
# note still leaves the posture unlabeled.
MUSIC_REPORT = {
    "cover_weight": MUSIC_COVER,
    "cover_posture": MUSIC_LABEL,
}

# Locked shape. Cover weight and cover posture are the fork, not this list.
SHAPE_FIELDS = (
    "loss_type",
    "gan_mode",
    "reg_arm",
    "reg_coeff",
    "reg_kappa",
    "reg_norm",
    "reg_lazy",
    "target_anneal",
    "fm_weight",
    "particle_l2",
    "n_particles",
    "z_dim",
)

_ERR_ATOL = 1e-5


def pins_match(claim: str, cover_weight: float, cover_posture: str) -> bool:
    """True when the numeric pin and the posture name are that claim.

    Equality, not a prefix. ``cover_weight=1`` is a substring of
    ``cover_weight=1.5`` and must not count as Music 1.0 or as demo 1.5.
    """
    if claim not in POSTURES:
        raise ValueError(
            f"unknown posture claim {claim!r} (expected one of {sorted(POSTURES)})"
        )
    spec = POSTURES[claim]
    return cover_weight == spec["cover_weight"] and cover_posture == spec["cover_posture"]


def _generator_target(obs: FloorObs, cover_weight: float, rp: GANLoss) -> torch.Tensor:
    return (
        rp.g_loss(obs.g_fake, obs.g_real)
        + PARTICLE_L2 * obs.parts.pow(2).mean()
        + float(cover_weight) * cover_mse(obs.offsets)
    )


def score_claimed_posture(
    obs: FloorObs,
    *,
    claim: str,
    reported_drift: dict | None = None,
) -> dict:
    """PASS when locked_shared holds and the claim matches the cover that ran.

    ``reported_drift`` is required for Music 1.0 (the named delta from the
    demo lock) and must be empty for demo 1.5. A swapped pin with no report
    is an unreported drift.
    """
    if claim not in POSTURES:
        raise ValueError(
            f"unknown posture claim {claim!r} (expected one of {sorted(POSTURES)})"
        )
    spec = POSTURES[claim]
    reported = {} if reported_drift is None else dict(reported_drift)
    bad: list[str] = []

    for field in SHAPE_FIELDS:
        got, want = getattr(obs.cfg, field), getattr(LOCKED, field)
        if got != want:
            bad.append(f"{field}: {got!r} != locked_shared {want!r}")
    if obs.pairing != "pair":
        bad.append(f"pairing: {obs.pairing!r} != locked_shared 'pair'")
    if obs.reg_cls is not GradientPenalty:
        bad.append(
            f"regularizer: {obs.reg_cls.__name__} is not particlegan.GradientPenalty"
        )
    if obs.n_particles != LOCKED.n_particles:
        bad.append(
            f"n_particles: cloud {obs.n_particles} != locked_shared {LOCKED.n_particles}"
        )

    rp = GANLoss("logistic", "rp")
    d_want = rp.d_loss(obs.d_real, obs.d_fake)
    spread_want = rp.d_loss(obs.spread_real, obs.spread_fake)
    adv_err = max(
        abs(float(obs.d_adv.detach()) - float(d_want.detach())),
        abs(float(obs.spread_adv.detach()) - float(spread_want.detach())),
    )
    g_claimed = _generator_target(obs, spec["cover_weight"], rp)
    g_demo = _generator_target(obs, DEMO_COVER, rp)
    g_music = _generator_target(obs, MUSIC_COVER, rp)
    g_err = abs(float(obs.g_total.detach()) - float(g_claimed.detach()))
    alias_gap = abs(float(g_demo.detach()) - float(g_music.detach()))
    cap_want = GradientPenalty(
        arm="b_cap", coeff=1.0, kappa=1.0, norm="l2",
    ).penalty(obs.critic_snap, obs.real, obs.fake)[0]
    cap_err = abs(float(obs.penalty.detach()) - float(cap_want.detach()))
    probe_err, probe_bad = _kappa_probe(obs.reg_cls)

    if adv_err > _ERR_ATOL:
        bad.append(f"adv: abs err {adv_err:.3e} vs RpGAN logistic pair")
    if g_err > _ERR_ATOL:
        bad.append(
            f"g: abs err {g_err:.3e} vs RpGAN + particle_l2={PARTICLE_L2} "
            f"+ {claim} cover {spec['cover_weight']}"
        )
    if cap_err > _ERR_ATOL:
        bad.append(f"b_cap: abs err {cap_err:.3e} vs GradientPenalty coeff=1 κ=1 l2")
    if probe_bad:
        bad.append(probe_bad)
    if alias_gap <= _ERR_ATOL:
        bad.append(
            "cover weights aliased: demo 1.5 and Music 1.0 produced the same generator term"
        )

    got_w = obs.cfg.cover_weight
    got_p = obs.cfg.cover_posture
    if not pins_match(claim, got_w, got_p):
        bad.append(
            "mislabeled swap: claim %s expects cover_weight=%s cover_posture=%r, "
            "got cover_weight=%s cover_posture=%r"
            % (claim, spec["cover_weight"], spec["cover_posture"], got_w, got_p)
        )

    claim_drift: dict = {}
    if got_w != spec["cover_weight"]:
        claim_drift["cover_weight"] = got_w
    if got_p != spec["cover_posture"]:
        claim_drift["cover_posture"] = got_p
    expected_report = {} if claim == DEMO_CLAIM else dict(MUSIC_REPORT)
    if claim_drift and reported != claim_drift:
        if not reported:
            bad.append(
                "unreported cover drift: swapping demo 1.5 and Music 1.0 "
                "without reporting the drift"
            )
        else:
            bad.append(
                "reported drift does not match the swap "
                f"(reported {reported!r}, actual {claim_drift!r})"
            )
    elif reported != expected_report:
        if not reported and expected_report:
            bad.append(
                "unreported cover drift: Music cover 1.0 is a named posture, "
                "not a silent alias of demo 1.5"
            )
        else:
            bad.append(
                "reported drift does not match the claimed posture "
                f"(reported {reported!r}, expected {expected_report!r})"
            )

    return {
        "claim": claim,
        "verdict": "PASS" if not bad else "FAIL",
        "mismatches": bad,
        "cover_weight": got_w,
        "cover_posture": got_p,
        "reported_drift": reported,
        "adv_abs_err": adv_err,
        "g_abs_err": g_err,
        "cap_abs_err": cap_err,
        "kappa_probe_abs_err": probe_err,
        "alias_gap": alias_gap,
        "fm_weight": obs.cfg.fm_weight,
        "gan_mode": obs.cfg.gan_mode,
        "loss_type": obs.cfg.loss_type,
        "n_particles": obs.n_particles,
        "pairing": obs.pairing,
        "reg_class": (
            "GradientPenalty" if obs.reg_cls is GradientPenalty else obs.reg_cls.__name__
        ),
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
    }


def _cfg(*, cover_weight: float, cover_posture: str, fm_weight: float = FM_WEIGHT) -> LockedSharedFloor:
    return replace(
        LOCKED,
        cover_weight=cover_weight,
        cover_posture=cover_posture,
        fm_weight=fm_weight,
    )


# Board order is the honesty contract: two named passes, then the drifts.
BOARD = (
    {
        "arm": DEMO_CLAIM,
        "claim": DEMO_CLAIM,
        "cover_weight": DEMO_COVER,
        "cover_posture": DEMO_LABEL,
        "reported_drift": None,
        "pairing": "pair",
        "regularizer": "faithful",
        "fm_weight": FM_WEIGHT,
    },
    {
        "arm": MUSIC_CLAIM,
        "claim": MUSIC_CLAIM,
        "cover_weight": MUSIC_COVER,
        "cover_posture": MUSIC_LABEL,
        "reported_drift": dict(MUSIC_REPORT),
        "pairing": "pair",
        "regularizer": "faithful",
        "fm_weight": FM_WEIGHT,
    },
    {
        "arm": "mislabeled_swap",
        "claim": DEMO_CLAIM,
        "cover_weight": MUSIC_COVER,
        "cover_posture": DEMO_LABEL,
        "reported_drift": None,
        "pairing": "pair",
        "regularizer": "faithful",
        "fm_weight": FM_WEIGHT,
    },
    {
        "arm": "stranger",
        "claim": DEMO_CLAIM,
        "cover_weight": DEMO_COVER,
        "cover_posture": DEMO_LABEL,
        "reported_drift": None,
        "pairing": "stranger",
        "regularizer": "faithful",
        "fm_weight": FM_WEIGHT,
    },
    {
        "arm": "fm_on",
        "claim": DEMO_CLAIM,
        "cover_weight": DEMO_COVER,
        "cover_posture": DEMO_LABEL,
        "reported_drift": None,
        "pairing": "pair",
        "regularizer": "faithful",
        "fm_weight": 0.1,
    },
    {
        "arm": "thinned_kappa",
        "claim": DEMO_CLAIM,
        "cover_weight": DEMO_COVER,
        "cover_posture": DEMO_LABEL,
        "reported_drift": None,
        "pairing": "pair",
        "regularizer": "thinned",
        "fm_weight": FM_WEIGHT,
    },
)


def evaluate_claim(
    cfg: LockedSharedFloor,
    *,
    claim: str,
    reported_drift: dict | None = None,
    pairing: str = "pair",
    regularizer: str = "faithful",
    arm: str = "custom",
    log: bool = False,
) -> dict:
    """One floor step-loop plus the posture claim."""
    obs = run_floor(
        cfg,
        regularizer=regularizer,
        pairing=pairing,
        log=False,
        arm=arm,
    )
    row = score_claimed_posture(obs, claim=claim, reported_drift=reported_drift)
    row["arm"] = arm
    if log:
        why = row["mismatches"][0] if row["mismatches"] else "named posture"
        print(
            "cover_posture_fork arm=%s claim=%s verdict=%s cover=%s label=%s "
            "g_err=%.3e alias_gap=%.3e why=%s"
            % (
                arm,
                claim,
                row["verdict"],
                row["cover_weight"],
                row["cover_posture"],
                row["g_abs_err"],
                row["alias_gap"],
                why,
            ),
            flush=True,
        )
    return row


def evaluate_spec(spec: dict, *, log: bool = False) -> dict:
    cfg = _cfg(
        cover_weight=spec["cover_weight"],
        cover_posture=spec["cover_posture"],
        fm_weight=spec["fm_weight"],
    )
    return evaluate_claim(
        cfg,
        claim=spec["claim"],
        reported_drift=spec["reported_drift"],
        pairing=spec["pairing"],
        regularizer=spec["regularizer"],
        arm=spec["arm"],
        log=log,
    )


def run_board(*, log: bool = True) -> list[dict]:
    """Named passes first, then swap / stranger / FM / thinned kappa."""
    return [evaluate_spec(spec, log=log) for spec in BOARD]


def format_board(rows: list[dict]) -> str:
    header = "| arm | claim | cover | label | g_err | alias_gap | gate | why |"
    sep = "|---|---|---:|---|---:|---:|---|---|"
    lines = [header, sep]
    for row in rows:
        why = row["mismatches"][0] if row["mismatches"] else "named posture matches the cover pin"
        lines.append(
            "| %s | %s | %s | %s | %.3e | %.3e | %s | %s |"
            % (
                row["arm"],
                row["claim"],
                row["cover_weight"],
                row["cover_posture"],
                row["g_abs_err"],
                row["alias_gap"],
                row["verdict"],
                why,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    from pathlib import Path

    rows = run_board(log=True)
    text = format_board(rows)
    print(text, end="")
    out = Path("results/cover_posture_fork")
    out.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        item = dict(row)
        item["music_gpu_transfer"] = False
        item["anima_gpu_transfer"] = False
        payload.append(item)
    (out / "board.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out / 'board.json'}")


if __name__ == "__main__":
    main()
