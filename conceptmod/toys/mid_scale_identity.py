"""CPU mid-scale identity hold on the scale grid ``-1, 0, 0.5, 1``.

Particle-sliders formulation sampling always includes scale ``-1``.
The unipolar board
(``docs/FORMULATION_LEADERBOARD_UNIPOLAR.md``) scores ``{0, 0.5, 1}`` and
still reports ``-1`` as a canary on every cell. The bipolar board gates
both poles. This toy's eval grid is that convention made explicit:
``(-1, 0, 0.5, 1)``.

Anima smile's live sample grid is
``conceptmod/textsliders/anima_slider.py`` ``DEFAULT_SAMPLE_SCALES =
(0.0, 0.25, 0.5, 1.0)``. It drops ``-1``. That protocol is a FAIL here
even when the residual itself is fine. The other smile failure this
gate encodes: the poles may move the concept (expression), and scale 0
can still be the right person, while scale ``0.5`` swaps in a stranger.
Identity is the content retain axis from
:func:`conceptmod.toys.cover_leftover.hold_dir`. Teacher poles go
through :func:`conceptmod.toys.cover_leftover.faithful_guard_e` so
leftover ê is not part of the concept. Pole motions, when both poles
were evaluated, are summarized with
:func:`conceptmod.toys.cover_leftover.leftover_bipolar`.

Locked shape
------------
* RpGAN logistic (``particlegan.GANLoss``).
* ``GradientPenalty`` ``b_cap``, coeff=1, kappa=1, norm=l2, lazy=1.
* ``fm_weight=0``.
* Demo cover **1.5** on the train grid (the train grid always includes
  ``-1``). Cover is what pins identity at 0 and at 0.5; it is not a
  silent alias of Music cover 1.0.

Intentional drift (reported)
----------------------------
* No particle cloud. This is a residual student, not Music ``n=12`` and
  not Hub 128.
* Step budget is this gate's budget, not locked 1200.

Arms
----
* ``locked`` — concept direction at ±1, identity kept at 0 and at 0.5.
  Must PASS.
* ``mid_collapse`` — poles and scale 0 stay the person; scale 0.5 is a
  stranger. Must FAIL ``identity_mid``.
* ``missing_minus`` — same locked residual, scored on the Anima smile
  grid that omits ``-1``. Must FAIL ``missing_minus``.
* ``polarity_flipped`` — the +1 teacher is the minus concept. Must FAIL
  ``polarity``.
* ``stranger`` — the relativistic pair and the cover target are a
  different person. Must FAIL ``stranger_pairing``.

Vanilla pairing, FM-on, and a thinned ``b_cap`` are refused.

A CPU PASS is not a Music or Anima GPU transfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import GANLoss, GradientPenalty

from conceptmod.toys.cover_leftover import (
    LOCKED_COVER,
    LeftoverField,
    faithful_guard_e,
    hold_dir,
    leftover_bipolar,
)


# Formulation eval grid. ``-1`` is required, not a skipped canary.
EVAL_SCALES = (-1.0, 0.0, 0.5, 1.0)
# Anima smile ``DEFAULT_SAMPLE_SCALES``. Drops ``-1``.
ANIMA_SMILE_SCALES = (0.0, 0.25, 0.5, 1.0)

CONCEPT_COS_MIN = 0.85
CONCEPT_MAG_LO = 0.75
CONCEPT_MAG_HI = 1.25
IDENTITY_KEPT_MIN = 0.85

# Shorter than the 1200-step demo lock. Budget, not a second recipe.
# 800 matches the cover/leftover gate: demo cover 1.5 has finished
# (identity at 0 and at 0.5 above 0.85) and the drift arms have separated.
GATE_STEPS = 800

DIM = 4
N_ROWS = 8
LR = 5e-3
BETAS = (0.0, 0.99)
DELAY = 80
MIN_LR_RATIO = 0.05
CRITIC_HIDDEN = 64

FORMULATION = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "reg_lazy": 1,
    "target_anneal": "none",
    "fm_weight": 0.0,
    "cover_weight": LOCKED_COVER,
    "pairing": "matched",
    "lr": LR,
    "beta1": BETAS[0],
    "beta2": BETAS[1],
}

ARMS = (
    "locked",
    "mid_collapse",
    "missing_minus",
    "polarity_flipped",
    "stranger",
)

REASON_ORDER = (
    "missing_minus",
    "incomplete_grid",
    "polarity",
    "concept",
    "identity_0",
    "identity_mid",
    "stranger_pairing",
)


def locked_shape_report() -> dict:
    """What matches locked_shared, and the drifts this toy reports."""
    return {
        "matches": {
            "objective": "RpGAN logistic relativistic pair",
            "grad_regularizer": "b_cap coeff=1 kappa=1 norm=l2 lazy=1 anneal=none",
            "fm_weight": 0.0,
            "cover_weight": LOCKED_COVER,
            "cover_posture": "demo_1.5",
            "eval_grid": list(EVAL_SCALES),
        },
        "intentional_drift": {
            "particles": (
                "no particle cloud (mid-scale identity residual). "
                "Not Music n=12 / particle_l2=0.02, and not Hub 128 gmix."
            ),
            "steps": f"{GATE_STEPS} (this gate's budget), not locked 1200.",
            "anima_sample_grid": (
                "Anima smile DEFAULT_SAMPLE_SCALES (0, 0.25, 0.5, 1) drops -1. "
                "That eval protocol is the missing_minus FAIL, not a PASS grid."
            ),
        },
    }


def reject_unlocked(overrides: dict, *, arm: str = "locked") -> None:
    """Refuse formulation drift. The arm name is the only legal delta."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {list(ARMS)})")
    for key, value in overrides.items():
        if key in ("steps", "seed"):
            continue
        if key not in FORMULATION:
            raise ValueError(f"unknown formulation knob {key!r}")
        if value == FORMULATION[key]:
            continue
        if key in ("gan_mode", "pairing"):
            raise ValueError(
                "stranger pairing refused: locked_shared is RpGAN logistic "
                f"matched to the same person, got {key}={value!r}"
            )
        if key == "fm_weight":
            raise ValueError(
                f"FM-on under b_cap refused: fm_weight={value!r} (locked fm_weight=0)"
            )
        if key in ("reg_arm", "reg_coeff", "reg_kappa", "reg_norm", "reg_lazy", "target_anneal"):
            raise ValueError(
                f"thinned b_cap refused: {key}={value!r} != locked {FORMULATION[key]!r} "
                "(coeff=1, kappa=1, norm=l2, arm=b_cap)"
            )
        raise ValueError(
            f"formulation drift refused: {key}={value!r} != locked {FORMULATION[key]!r}"
        )


def mid_bump(scale: float) -> float:
    """One-sided smile bump. 0 at ``{-1, 0, 1}``, 1 at ``0.5``.

    The negative pole is not a mid-scale. A residual can sit on the
    concept at both poles and on the person at 0, and still replace
    the person at ``+0.5``.
    """
    s = float(scale)
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return 4.0 * s * (1.0 - s)


def _has_scale(scales, target: float) -> bool:
    return any(abs(float(scale) - float(target)) <= 1e-6 for scale in scales)


def delayed_cosine(step: int, *, total: int, delay: int = DELAY, min_ratio: float = MIN_LR_RATIO) -> float:
    """1.0 for ``delay`` steps, then cosine down to ``min_ratio``."""
    if step < int(delay):
        return 1.0
    span = max(1, int(total) - int(delay))
    t = min(1.0, float(step - int(delay)) / float(span))
    return float(min_ratio) + 0.5 * (1.0 - float(min_ratio)) * (1.0 + math.cos(math.pi * t))


@dataclass(frozen=True, eq=False)
class SmileTeacher:
    """Guarded concept, the person, and an orthogonal stranger."""

    identity: torch.Tensor
    concept: torch.Tensor
    stranger: torch.Tensor
    retain_unit: torch.Tensor
    identity_amp: float
    plus: torch.Tensor
    minus: torch.Tensor

    def train_target(self, arm: str, scale: float) -> torch.Tensor:
        """Teacher state for one train scale. Eval never reads this."""
        polarity = -1.0 if arm == "polarity_flipped" else 1.0
        concept = polarity * self.concept
        if arm == "stranger" or (arm == "mid_collapse" and abs(float(scale) - 0.5) <= 1e-6):
            base = self.stranger
        else:
            base = self.identity
        return base + float(scale) * concept


def smile_teacher(field: LeftoverField | None = None) -> SmileTeacher:
    """Identity on the content axis, concept on û, stranger on the lyric axis.

    Raw poles carry leftover ê. ``faithful_guard_e`` takes ê off the odd
    part when the blend guard admits it. The concept this toy scores is
    that guarded odd direction.
    """
    field = field or LeftoverField()
    concept_axis = field.basis(0)
    content_axis = field.basis(1)
    leak_axis = field.basis(2)
    retain = hold_dir(content_axis, concept_axis)
    if retain is None:
        raise ValueError("identity retain is parallel to the concept; hold is off")
    retain_unit = retain / retain.norm().clamp_min(1e-8)
    identity = float(field.content) * content_axis
    concept_raw = float(field.slider) * concept_axis
    leak = float(field.leak) * leak_axis
    stranger = float(field.content) * field.basis(3)
    raw_plus = identity + concept_raw + leak
    raw_minus = identity - concept_raw - leak
    plus, minus = faithful_guard_e(raw_plus, raw_minus, identity, leak_axis, concept_axis)
    concept = plus - identity
    if float(concept.norm()) <= 1e-8:
        raise ValueError("guarded concept direction vanished")
    identity_amp = float(identity @ retain_unit)
    return SmileTeacher(
        identity=identity.detach(),
        concept=concept.detach(),
        stranger=stranger.detach(),
        retain_unit=retain_unit.detach(),
        identity_amp=identity_amp,
        plus=plus.detach(),
        minus=minus.detach(),
    )


class MidScaleResidual(nn.Module):
    """``s*odd + |s|*even + origin + bump(s)*mid``.

    ``bump`` is zero on ``{-1, 0, 1}``, so pole and neutral losses cannot
    see a mid-scale identity swap. Scale ``0.5`` can.
    """

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.origin = nn.Parameter(torch.zeros(dim))
        self.mid = nn.Parameter(torch.zeros(dim))

    def state(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.odd + abs(s) * self.even + self.origin + mid_bump(s) * self.mid


class ScaleCritic(nn.Module):
    """Two-layer LeakyReLU MLP. The cap differentiates the state, not the scale."""

    def __init__(self, dim: int, teacher: torch.Tensor, hidden: int = CRITIC_HIDDEN) -> None:
        super().__init__()
        rms = teacher.detach().float().square().mean().sqrt()
        if not torch.isfinite(rms) or float(rms) <= 0.0:
            raise ValueError("teacher must have finite, nonzero RMS")
        self.register_buffer("input_scale", rms)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def score(self, z: torch.Tensor, scale: float) -> torch.Tensor:
        label = z.new_full((z.shape[0], 1), float(scale))
        return self.net(torch.cat([z, label], dim=-1)).squeeze(-1)

    def forward(self, state: torch.Tensor, scale: float) -> torch.Tensor:
        return self.score(state.float() / self.input_scale, scale)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).squeeze())


def _find(pairs: list[tuple[float, torch.Tensor]], target: float) -> torch.Tensor | None:
    for scale, state in pairs:
        if abs(scale - float(target)) <= 1e-6:
            return state
    return None


def _identity_kept(state: torch.Tensor, teacher: SmileTeacher) -> float:
    coef = float(state.float() @ teacher.retain_unit.float())
    amp = abs(float(teacher.identity_amp)) + 1e-8
    return 1.0 - min(1.0, abs(coef - float(teacher.identity_amp)) / amp)


def _concept_motion(state: torch.Tensor, origin: torch.Tensor, teacher: SmileTeacher) -> tuple[float, float]:
    motion = state.float() - origin.float()
    cos = _cos(motion, teacher.concept)
    mag = float(motion.norm()) / float(teacher.concept.norm().clamp_min(1e-8))
    return cos, mag


@torch.no_grad()
def score_hold(
    student: MidScaleResidual,
    *,
    scales=EVAL_SCALES,
    pairing: str = "matched",
    teacher: SmileTeacher | None = None,
) -> dict:
    """Score one residual on the scales the caller actually passed.

    The default grid is :data:`EVAL_SCALES` (includes ``-1``). A grid that
    omits ``-1`` is ``missing_minus`` and cannot PASS. This function does
    not probe a scale the caller left out.
    """
    if pairing not in ("matched", "stranger"):
        raise ValueError(f"pairing must be 'matched' or 'stranger', got {pairing!r}")
    teacher = teacher or smile_teacher()
    ordered: list[tuple[float, torch.Tensor]] = []
    for scale in scales:
        value = float(scale)
        if not math.isfinite(value):
            raise ValueError(f"scale must be finite, got {scale!r}")
        ordered.append((value, student.state(value).detach()))
    by_scale = ordered
    state0 = _find(by_scale, 0.0)
    state1 = _find(by_scale, 1.0)
    state_m = _find(by_scale, -1.0)
    state_mid = _find(by_scale, 0.5)

    concept_cos_plus = concept_mag_plus = None
    concept_cos_minus = concept_mag_minus = None
    if state0 is not None and state1 is not None:
        concept_cos_plus, concept_mag_plus = _concept_motion(state1, state0, teacher)
    if state0 is not None and state_m is not None:
        motion = state_m.float() - state0.float()
        concept_cos_minus = _cos(motion, -teacher.concept)
        concept_mag_minus = float(motion.norm()) / float(teacher.concept.norm().clamp_min(1e-8))
    identity_at_0 = None if state0 is None else _identity_kept(state0, teacher)
    identity_at_mid = None if state_mid is None else _identity_kept(state_mid, teacher)

    reasons: list[str] = []
    if not _has_scale(scales, -1.0):
        reasons.append("missing_minus")
    if not (_has_scale(scales, 0.0) and _has_scale(scales, 0.5) and _has_scale(scales, 1.0)):
        reasons.append("incomplete_grid")

    bad_sign = False
    bad_concept = False
    if concept_cos_plus is not None and concept_mag_plus is not None:
        if concept_cos_plus < 0.0:
            bad_sign = True
        elif concept_cos_plus < CONCEPT_COS_MIN or not (CONCEPT_MAG_LO <= concept_mag_plus <= CONCEPT_MAG_HI):
            bad_concept = True
    if concept_cos_minus is not None and concept_mag_minus is not None:
        if concept_cos_minus < 0.0:
            bad_sign = True
        elif concept_cos_minus < CONCEPT_COS_MIN or not (CONCEPT_MAG_LO <= concept_mag_minus <= CONCEPT_MAG_HI):
            bad_concept = True
    if bad_sign:
        reasons.append("polarity")
    elif bad_concept:
        reasons.append("concept")
    if identity_at_0 is not None and identity_at_0 < IDENTITY_KEPT_MIN:
        reasons.append("identity_0")
    if identity_at_mid is not None and identity_at_mid < IDENTITY_KEPT_MIN:
        reasons.append("identity_mid")
    if pairing == "stranger":
        reasons.append("stranger_pairing")
    reasons = [name for name in REASON_ORDER if name in reasons]

    bipolar = None
    if state0 is not None and state1 is not None and state_m is not None:
        bipolar = leftover_bipolar(state1 - state0, state_m - state0)

    per_scale = []
    for scale, state in by_scale:
        item = {
            "scale": float(scale),
            "identity_kept": _identity_kept(state, teacher),
            "on_concept": float(state.float() @ teacher.concept.float()) / float(teacher.concept.norm().clamp_min(1e-8)) ** 2,
        }
        per_scale.append(item)

    row = {
        "scales": [float(scale) for scale, _state in by_scale],
        "pairing": pairing,
        "concept_cos_plus": concept_cos_plus,
        "concept_cos_minus": concept_cos_minus,
        "concept_mag_plus": concept_mag_plus,
        "concept_mag_minus": concept_mag_minus,
        "identity_at_0": identity_at_0,
        "identity_at_mid": identity_at_mid,
        "pass": not reasons,
        "fail_reasons": ",".join(reasons),
        "per_scale": per_scale,
        "device": "cpu",
    }
    if bipolar is not None:
        row["same_dir"] = float(bipolar["same_dir"])
        row["leak_frac"] = float(bipolar["leak_frac"])
    else:
        row["same_dir"] = None
        row["leak_frac"] = None
    return row


def _fmt(value) -> str:
    if value is None:
        return "na"
    return f"{float(value):+.4f}"


def format_row(row: dict) -> str:
    """One line, meant to be tailed."""
    return (
        f"mid_scale arm={row['arm']} step={row.get('steps')} seed={row.get('seed')} "
        f"concept+={_fmt(row.get('concept_cos_plus'))} concept-={_fmt(row.get('concept_cos_minus'))} "
        f"id0={_fmt(row.get('identity_at_0'))} id_mid={_fmt(row.get('identity_at_mid'))} "
        f"scales={row.get('scales')} gate={'PASS' if row.get('pass') else 'FAIL'} "
        f"why={row.get('fail_reasons') or 'locked'}"
    )


def _apply_lr(opt: torch.optim.Optimizer, step: int, total: int) -> None:
    scale = delayed_cosine(step, total=total)
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * scale


def _batch(vector: torch.Tensor, rows: int = N_ROWS) -> torch.Tensor:
    return vector.detach().unsqueeze(0).expand(rows, -1)


def _train_arm_name(arm: str) -> str:
    """``missing_minus`` trains the locked residual and only drifts the eval grid."""
    if arm == "missing_minus":
        return "locked"
    return arm


def _eval_scales(arm: str) -> tuple[float, ...]:
    if arm == "missing_minus":
        return ANIMA_SMILE_SCALES
    return EVAL_SCALES


def _fit(
    arm: str,
    *,
    steps: int,
    seed: int,
    teacher: SmileTeacher,
) -> tuple[MidScaleResidual, dict]:
    """Train on :data:`EVAL_SCALES` (always includes ``-1``)."""
    if not _has_scale(EVAL_SCALES, -1.0):
        raise RuntimeError("training grid must include -1")
    train_arm = _train_arm_name(arm)
    torch.manual_seed(int(seed))
    student = MidScaleResidual(int(teacher.concept.numel()))
    if any(param.is_cuda for param in student.parameters()):
        raise RuntimeError("mid-scale identity toy is CPU only")
    targets = {scale: teacher.train_target(train_arm, scale) for scale in EVAL_SCALES}
    cloud = torch.stack([targets[scale] for scale in EVAL_SCALES], dim=0)
    critic = ScaleCritic(student.odd.numel(), cloud, hidden=CRITIC_HIDDEN)
    gan = GANLoss(loss_type=FORMULATION["loss_type"], mode=FORMULATION["gan_mode"])
    reg = GradientPenalty(
        arm=FORMULATION["reg_arm"],
        coeff=FORMULATION["reg_coeff"],
        kappa=FORMULATION["reg_kappa"],
        norm=FORMULATION["reg_norm"],
        lazy_k=FORMULATION["reg_lazy"],
        target_anneal=FORMULATION["target_anneal"],
    )
    if reg.arm != "b_cap" or reg.coeff != 1.0 or reg.kappa != 1.0 or reg.norm != "l2":
        raise RuntimeError("b_cap champion was not constructed")
    opt_g = torch.optim.Adam(student.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    for opt in (opt_g, opt_d):
        opt.param_groups[0]["initial_lr"] = LR
    reals = {scale: _batch(targets[scale]) for scale in EVAL_SCALES}
    cover_w = float(FORMULATION["cover_weight"])
    reg_calls = 0
    n_scales = float(len(EVAL_SCALES))

    for step in range(int(steps)):
        _apply_lr(opt_g, step, steps)
        _apply_lr(opt_d, step, steps)
        critic.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        d_loss = student.odd.new_zeros(())
        for scale in EVAL_SCALES:
            fake = student.state(scale).unsqueeze(0).expand(N_ROWS, -1).detach()
            cap, _stats = reg.penalty(
                lambda z, scale=scale: critic.score(z, scale),
                reals[scale] / critic.input_scale,
                fake / critic.input_scale,
                step=step + 1,
            )
            reg_calls += 1
            d_term = gan.d_loss(critic(reals[scale], scale), critic(fake, scale))
            d_loss = d_loss + (d_term + cap) / n_scales
        d_loss.backward()
        opt_d.step()

        critic.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        g_loss = student.odd.new_zeros(())
        with torch.no_grad():
            real_scores = {scale: critic(reals[scale], scale) for scale in EVAL_SCALES}
        for scale in EVAL_SCALES:
            fake = student.state(scale).unsqueeze(0).expand(N_ROWS, -1)
            g_loss = g_loss + gan.g_loss(critic(fake, scale), real_scores[scale]) / n_scales
        cover = student.odd.new_zeros(())
        for scale in EVAL_SCALES:
            cover = cover + F.mse_loss(student.state(scale), targets[scale])
        g_loss = g_loss + cover_w * cover / n_scales
        g_loss.backward()
        opt_g.step()
        critic.requires_grad_(True)

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(steps):
            preview_scales = _eval_scales(arm)
            preview = score_hold(
                student,
                scales=preview_scales,
                pairing="stranger" if arm == "stranger" else "matched",
                teacher=teacher,
            )
            preview.update(arm=arm, steps=step + 1, seed=seed)
            print(format_row(preview), flush=True)

    meta = {
        "steps": int(steps),
        "seed": int(seed),
        "train_scales": [float(scale) for scale in EVAL_SCALES],
        "loss_type": gan.loss_type if hasattr(gan, "loss_type") else FORMULATION["loss_type"],
        "gan_mode": FORMULATION["gan_mode"],
        "reg_arm": reg.arm,
        "reg_coeff": float(reg.coeff),
        "reg_kappa": float(reg.kappa),
        "reg_norm": reg.norm,
        "reg_lazy": int(reg.lazy_k),
        "reg_anneal": reg.target_anneal,
        "reg_calls": int(reg_calls),
        "reg_is_gradient_penalty": isinstance(reg, GradientPenalty),
        "cover_weight": cover_w,
        "fm_weight": float(FORMULATION["fm_weight"]),
        "device": "cpu",
    }
    return student, meta


def _finish(
    student: MidScaleResidual,
    meta: dict,
    *,
    arm: str,
    teacher: SmileTeacher,
) -> dict:
    pairing = "stranger" if arm == "stranger" else "matched"
    row = score_hold(student, scales=_eval_scales(arm), pairing=pairing, teacher=teacher)
    row.update(meta)
    row["arm"] = arm
    print(format_row(row), flush=True)
    return row


def run_arm(arm: str, *, steps: int = GATE_STEPS, seed: int = 0, **overrides) -> dict:
    """Fit one arm and score its eval grid. Prints a tailable line."""
    reject_unlocked(overrides, arm=arm)
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    teacher = smile_teacher()
    student, meta = _fit(arm, steps=steps, seed=seed, teacher=teacher)
    return _finish(student, meta, arm=arm, teacher=teacher)


def run_board(*, steps: int = GATE_STEPS, seed: int = 0) -> list[dict]:
    """Locked residual first, then the four drift arms. One seed.

    ``missing_minus`` rescores the locked weights on the Anima smile grid.
    It does not train a second residual.
    """
    teacher = smile_teacher()
    locked_student, locked_meta = _fit("locked", steps=steps, seed=seed, teacher=teacher)
    rows = [
        _finish(locked_student, locked_meta, arm="locked", teacher=teacher),
        _finish(locked_student, locked_meta, arm="missing_minus", teacher=teacher),
    ]
    for arm in ("mid_collapse", "polarity_flipped", "stranger"):
        student, meta = _fit(arm, steps=steps, seed=seed, teacher=teacher)
        rows.append(_finish(student, meta, arm=arm, teacher=teacher))
    # Board order matches the scoreboard: locked, collapse, missing -1, polarity, stranger.
    by_arm = {row["arm"]: row for row in rows}
    return [by_arm[name] for name in ("locked", "mid_collapse", "missing_minus", "polarity_flipped", "stranger")]


def format_board(rows: list[dict]) -> str:
    header = (
        "| arm | gate | concept+ | concept- | identity@0 | identity@0.5 | eval grid | why |"
    )
    sep = "|---|---|---:|---:|---:|---:|---|---|"
    lines = [header, sep]
    for row in rows:
        grid = ",".join(f"{scale:g}" for scale in row["scales"])
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["arm"],
                "PASS" if row["pass"] else "FAIL",
                _fmt(row["concept_cos_plus"]),
                _fmt(row["concept_cos_minus"]),
                _fmt(row["identity_at_0"]),
                _fmt(row["identity_at_mid"]),
                grid,
                row["fail_reasons"] or "concept at ±1, identity at 0 and 0.5",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    torch.set_num_threads(1)
    print("mid_scale_identity family=smile mid-scale hold device=cpu", flush=True)
    print(format_board(run_board()), end="", flush=True)


if __name__ == "__main__":
    main()
