"""CPU unipolar (one-sided) residual / direction toy.

Culture: particle-sliders ``docs/FORMULATION_LEADERBOARD_UNIPOLAR.md`` and the
``rpgan_bcap_plus_neu`` arm (GAN on scales ``{0, +1}`` only). Scale ``-1`` is
an unscored canary. Antipodal cosine is not a gate.

Locked-shared shape that must PASS
-----------------------------------
* RpGAN logistic relativistic pair (``particlegan.GANLoss``).
* ParticleGAN ``GradientPenalty`` (the ``GradRegularizer``) ``b_cap`` with
  coeff=1, kappa=1, norm=l2, lazy=1, anneal=none. Not a thinned stub.
* ``fm_weight=0``.
* Adam betas ``(0, 0.99)``, shared LR ``5e-3``, delayed cosine (hold 80,
  floor 0.05). Critic hidden 64, matching locked ``critic_hidden``.

Intentional drift from locked demo / Music Arm B (reported, not silent)
------------------------------------------------------------------------
* Train ``cover_weight=0``. Demo locked cover is **1.5** (Music runs sometimes
  use 1.0). On this unipolar board cover is an **eval gate** (``>= 0.85``),
  and the winning GAN arm is GAN-only: a nonzero cover loss is supervised MSE.
* No particle cloud. This is the unipolar residual student, not Music ``n=12``
  and not Hub 128-particle gmix.
* Step budget 400 (unipolar leaderboard), not locked 1200.

Gates (same numbers as the unipolar board; this toy has no lyric field)
------------------------------------------------------------------------
* plus cover at scale +1 ``>= 0.85`` (direction and magnitude vs the + pole)
* plus leak (energy on the off-caption axis) ``<= 0.05``
* neu_hold at scale 0 ``>= 0.85`` (free origin must be learned, not deleted)

Arms in this one family
-----------------------
* ``locked_rpgan`` — the shape above. Must PASS.
* ``mse_only`` — plus-only coordinate MSE, no scale-0 term. Must FAIL neu_hold
  (same split as ``faithful_plus`` vs ``faithful_plus_neu``).
* ``polarity_flipped`` — same GAN shape, but the +1 real is the minus pole.
  Must FAIL plus cover.

A CPU PASS is not a Music or Anima transfer result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import GANLoss, GradientPenalty


# Unipolar board thresholds (FORMULATION_LEADERBOARD_UNIPOLAR).
PLUS_COVER_MIN = 0.85
PLUS_OFF_MAX = 0.05
PLUS_NEU_HOLD_MIN = 0.85

DIM = 4
N_ROWS = 8
SCALES = (0.0, 1.0)
LR = 5e-3
BETAS = (0.0, 0.99)
DELAY = 80
MIN_LR_RATIO = 0.05
CRITIC_HIDDEN = 64
DEMO_LOCKED_COVER = 1.5  # locked_baseline_defaults.LOCKED["cover_weight"]

# + pole along e0. e1 is off-caption and is never a training target.
PLUS = torch.tensor([1.0, 0.0, 0.0, 0.0])
OFF_AXIS = torch.tensor([0.0, 1.0, 0.0, 0.0])


@dataclass(frozen=True)
class UnipolarRecipe:
    """Adv shape for one arm. ``polarity`` +1 is the plus pole; -1 flips it."""

    arm: str = "locked_rpgan"
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    reg_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cover_weight: float = 0.0
    polarity: float = 1.0
    steps: int = 400
    seed: int = 0


def locked_recipe(**overrides) -> UnipolarRecipe:
    """The PASS arm. Formulation overrides are refused; budget keys are steps/seed."""
    allowed = {"steps", "seed"}
    bad = [key for key in overrides if key not in allowed]
    if bad:
        raise ValueError(
            f"locked_rpgan refuses formulation overrides {bad}; "
            "report drift with an explicit arm instead of forking locked_shared"
        )
    return UnipolarRecipe(arm="locked_rpgan", **overrides)


def _refuse_unless_locked_adv(recipe: UnipolarRecipe) -> None:
    """Fail closed on stranger pairing, FM-on, cover-MSE, or a thinned cap."""
    expected = {
        "loss_type": "logistic",
        "gan_mode": "rp",
        "reg_arm": "b_cap",
        "reg_coeff": 1.0,
        "reg_kappa": 1.0,
        "reg_norm": "l2",
        "reg_lazy": 1,
        "target_anneal": "none",
        "fm_weight": 0.0,
        "cover_weight": 0.0,
    }
    bad = []
    for key, want in expected.items():
        got = getattr(recipe, key)
        if got != want:
            bad.append(f"{key}={got!r} != locked {want!r}")
    if recipe.polarity not in (1.0, -1.0):
        bad.append(f"polarity={recipe.polarity!r} (only +1 or an explicit flip)")
    if bad:
        raise ValueError(
            "drift from locked_shared unipolar GAN shape: " + "; ".join(bad)
        )


def locked_shape_report() -> dict:
    """What matches locked_shared, and the drifts this toy reports on purpose."""
    return {
        "matches": {
            "objective": "RpGAN logistic relativistic pair",
            "grad_regularizer": "b_cap coeff=1 kappa=1 norm=l2 lazy=1 anneal=none",
            "fm_weight": 0.0,
            "lr": LR,
            "betas": BETAS,
            "critic_hidden": CRITIC_HIDDEN,
            "schedule": f"delayed cosine delay={DELAY} min_ratio={MIN_LR_RATIO}",
        },
        "intentional_drift": {
            "cover_weight": (
                f"train 0.0; demo locked cover is {DEMO_LOCKED_COVER}. "
                "Unipolar cover is the eval gate (>= 0.85), not a train loss. "
                "Music cover 1.0 is a different posture and is not used here."
            ),
            "particles": (
                "no particle cloud (unipolar residual student). "
                "Not Music n=12 / particle_l2=0.02, and not Hub 128 gmix."
            ),
            "steps": "400 (unipolar leaderboard budget), not locked 1200.",
        },
    }


def delayed_cosine(step: int, *, total: int, delay: int = DELAY, min_ratio: float = MIN_LR_RATIO) -> float:
    """1.0 for ``delay`` steps, then cosine down to ``min_ratio`` (slider2d.adv)."""
    if step < int(delay):
        return 1.0
    span = max(1, int(total) - int(delay))
    t = min(1.0, float(step - int(delay)) / float(span))
    return float(min_ratio) + 0.5 * (1.0 - float(min_ratio)) * (1.0 + math.cos(math.pi * t))


class FreeOriginResidual(nn.Module):
    """``delta(s) = s*odd + |s|*even + origin``. Origin is free, so hold is learned.

    Deleting ``origin`` would make neu_hold true by construction. Plus-only MSE
    can park the + pole in ``origin`` and fail hold; scale-0 GAN training cannot.
    """

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.origin = nn.Parameter(torch.zeros(dim))

    def delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.odd + abs(s) * self.even + self.origin


class ScaleCritic(nn.Module):
    """Two-layer LeakyReLU MLP. The cap differentiates the delta, not the scale label."""

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

    def forward(self, delta: torch.Tensor, scale: float) -> torch.Tensor:
        return self.score(delta.float() / self.input_scale, scale)


def _batch(vector: torch.Tensor, rows: int = N_ROWS) -> torch.Tensor:
    return vector.detach().unsqueeze(0).expand(rows, -1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().float().unsqueeze(0), b.detach().float().unsqueeze(0)).squeeze())


def score_residual(student: FreeOriginResidual) -> dict:
    """Unipolar gates. They always read the true plus pole, never the training pole."""
    plus = PLUS
    off = OFF_AXIS
    d1 = student.delta(1.0).detach()
    d0 = student.delta(0.0).detach()
    d_half = student.delta(0.5).detach()
    d_minus = student.delta(-1.0).detach()
    target_norm = float(plus.norm())
    cos_plus = _cos(d1, plus)
    mag_ratio = float(d1.norm()) / target_norm
    mag_score = max(0.0, 1.0 - abs(mag_ratio - 1.0))
    cover = max(0.0, cos_plus) * mag_score
    leak = float(d1.dot(off) ** 2) / (float(d1.dot(d1)) + 1e-12)
    neu_hold = 1.0 - min(1.0, float(d0.norm()) / target_norm)
    half_ratio = float(d_half.norm()) / (float(d1.norm()) + 1e-12)
    cos_canary_plus = _cos(d_minus, plus)
    cos_canary_neg = _cos(d_minus, -plus)
    canary_off = float(d_minus.dot(off) ** 2) / (float(d_minus.dot(d_minus)) + 1e-12)
    if cos_canary_plus > 0.5:
        landed = "pos"
    elif cos_canary_neg > 0.5:
        landed = "neg"
    else:
        landed = "other"
    dangerous = bool(landed == "pos" or canary_off > PLUS_OFF_MAX)
    hit = bool(cover >= PLUS_COVER_MIN and leak <= PLUS_OFF_MAX and neu_hold >= PLUS_NEU_HOLD_MIN)
    return {
        "cover": cover,
        "off_caption": leak,
        "neu_hold": neu_hold,
        "cos_plus": cos_plus,
        "mag_ratio": mag_ratio,
        "half_norm_ratio": half_ratio,
        "hit": hit,
        "canary": {
            "scored": False,
            "landed": landed,
            "cos_plus": cos_canary_plus,
            "cos_neg": cos_canary_neg,
            "off_caption": canary_off,
            "dangerous": dangerous,
        },
    }


def format_row(row: dict) -> str:
    """One line, meant to be tailed."""
    return (
        f"unipolar_dir arm={row['arm']} step={row['steps']} seed={row['seed']} "
        f"cover={row['cover']:.4f} leak={row['off_caption']:.4f} "
        f"neu_hold={row['neu_hold']:.4f} cos_plus={row['cos_plus']:+.4f} "
        f"hit={'PASS' if row['hit'] else 'FAIL'}"
    )


def _apply_lr(opt: torch.optim.Optimizer, step: int, total: int) -> None:
    scale = delayed_cosine(step, total=total)
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * scale


def _fit_mse_only(student: FreeOriginResidual, target: torch.Tensor, *, steps: int) -> list[dict]:
    """Plus-only coordinate MSE. Scale 0 is not in the loss, so the origin can drift."""
    opt = torch.optim.Adam(student.parameters(), lr=LR, betas=BETAS)
    opt.param_groups[0]["initial_lr"] = LR
    history = []
    for step in range(steps):
        _apply_lr(opt, step, steps)
        loss = F.mse_loss(student.delta(1.0), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            row = score_residual(student)
            row.update(step=step + 1, loss=float(loss.detach()))
            history.append(row)
            print(
                f"unipolar_dir arm=mse_only step={step + 1}/{steps} "
                f"cover={row['cover']:.4f} leak={row['off_caption']:.4f} "
                f"neu_hold={row['neu_hold']:.4f} hit={'PASS' if row['hit'] else 'FAIL'}",
                flush=True,
            )
    return history


def _fit_rpgan(
    student: FreeOriginResidual,
    critic: ScaleCritic,
    target: torch.Tensor,
    *,
    steps: int,
    recipe: UnipolarRecipe,
) -> tuple[list[dict], GradientPenalty]:
    """One D update then one G update, averaged over scales ``{0, +1}``."""
    _refuse_unless_locked_adv(recipe)
    gan = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    reg = GradientPenalty(
        arm=recipe.reg_arm,
        coeff=recipe.reg_coeff,
        kappa=recipe.reg_kappa,
        norm=recipe.reg_norm,
        lazy_k=recipe.reg_lazy,
        target_anneal=recipe.target_anneal,
    )
    opt_g = torch.optim.Adam(student.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    for opt in (opt_g, opt_d):
        opt.param_groups[0]["initial_lr"] = LR
    real = {
        0.0: _batch(torch.zeros_like(target)),
        1.0: _batch(target),
    }
    history = []
    for step in range(steps):
        _apply_lr(opt_g, step, steps)
        _apply_lr(opt_d, step, steps)
        critic.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        d_loss = student.odd.new_zeros(())
        for scale in SCALES:
            fake = student.delta(scale).unsqueeze(0).expand(N_ROWS, -1).detach()
            cap, _stats = reg.penalty(
                lambda z, scale=scale: critic.score(z, scale),
                real[scale] / critic.input_scale,
                fake / critic.input_scale,
                step=step + 1,
            )
            d_term = gan.d_loss(critic(real[scale], scale), critic(fake, scale))
            d_loss = d_loss + 0.5 * (d_term + cap)
        d_loss.backward()
        opt_d.step()

        critic.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        g_loss = student.odd.new_zeros(())
        with torch.no_grad():
            real_scores = {scale: critic(real[scale], scale) for scale in SCALES}
        for scale in SCALES:
            fake = student.delta(scale).unsqueeze(0).expand(N_ROWS, -1)
            g_term = gan.g_loss(critic(fake, scale), real_scores[scale])
            g_loss = g_loss + 0.5 * g_term
        g_loss.backward()
        opt_g.step()
        critic.requires_grad_(True)

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            row = score_residual(student)
            row.update(step=step + 1, g_loss=float(g_loss.detach()), d_loss=float(d_loss.detach()))
            history.append(row)
            print(
                f"unipolar_dir arm={recipe.arm} step={step + 1}/{steps} "
                f"cover={row['cover']:.4f} leak={row['off_caption']:.4f} "
                f"neu_hold={row['neu_hold']:.4f} cos_plus={row['cos_plus']:+.4f} "
                f"hit={'PASS' if row['hit'] else 'FAIL'}",
                flush=True,
            )
    return history, reg


def run_arm(
    arm: str,
    *,
    steps: int = 400,
    seed: int = 0,
    fm_weight: float = 0.0,
    cover_weight: float = 0.0,
    gan_mode: str = "rp",
    loss_type: str = "logistic",
    reg_arm: str = "b_cap",
    reg_coeff: float = 1.0,
    reg_kappa: float = 1.0,
    reg_norm: str = "l2",
) -> dict:
    """Fit one arm and score the unipolar gates. Prints a tailable line per checkpoint."""
    if arm not in ("locked_rpgan", "mse_only", "polarity_flipped"):
        raise ValueError(
            f"unknown arm {arm!r}; this family is locked_rpgan, mse_only, polarity_flipped"
        )
    polarity = -1.0 if arm == "polarity_flipped" else 1.0
    recipe = UnipolarRecipe(
        arm=arm,
        loss_type=loss_type,
        gan_mode=gan_mode,
        reg_arm=reg_arm,
        reg_coeff=reg_coeff,
        reg_kappa=reg_kappa,
        reg_norm=reg_norm,
        fm_weight=fm_weight,
        cover_weight=cover_weight,
        polarity=polarity,
        steps=int(steps),
        seed=int(seed),
    )
    _refuse_unless_locked_adv(recipe)
    torch.manual_seed(recipe.seed)
    student = FreeOriginResidual(DIM)
    target = recipe.polarity * PLUS
    reg_used = None
    if arm == "mse_only":
        history = _fit_mse_only(student, target, steps=recipe.steps)
    else:
        teacher = _batch(PLUS if recipe.polarity > 0 else -PLUS)
        critic = ScaleCritic(DIM, teacher, hidden=CRITIC_HIDDEN)
        history, reg_used = _fit_rpgan(student, critic, target, steps=recipe.steps, recipe=recipe)
    row = score_residual(student)
    row.update(
        arm=arm,
        steps=recipe.steps,
        seed=recipe.seed,
        polarity=recipe.polarity,
        fm_weight=recipe.fm_weight,
        cover_weight=recipe.cover_weight,
        gan_mode=recipe.gan_mode if arm != "mse_only" else "mse",
        loss_type=recipe.loss_type if arm != "mse_only" else "mse",
        reg_arm=None if reg_used is None else reg_used.arm,
        reg_coeff=None if reg_used is None else reg_used.coeff,
        reg_kappa=None if reg_used is None else reg_used.kappa,
        reg_norm=None if reg_used is None else reg_used.norm,
        reg_lazy=None if reg_used is None else reg_used.lazy_k,
        reg_anneal=None if reg_used is None else reg_used.target_anneal,
        reg_is_gradient_penalty=isinstance(reg_used, GradientPenalty),
        origin_norm=float(student.origin.detach().norm()),
        odd_norm=float(student.odd.detach().norm()),
        even_norm=float(student.even.detach().norm()),
        history=history,
        shape=locked_shape_report(),
    )
    print(format_row(row), flush=True)
    return row


def main() -> None:
    torch.set_num_threads(1)
    print("unipolar_dir family=one-sided residual/dir  device=cpu", flush=True)
    for arm in ("locked_rpgan", "mse_only", "polarity_flipped"):
        run_arm(arm, steps=400, seed=0)


if __name__ == "__main__":
    main()
