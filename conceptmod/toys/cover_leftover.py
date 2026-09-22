"""Cover / leftover / faithful-teacher geometry toy.

CPU stand-in for the particle-sliders Field3D leftover cell. The scored
object is one shared odd+even residual. The game uses this repo's
primitives: RpGAN logistic (``GANLoss``), ParticleGAN ``GradRegularizer``
``b_cap`` (coeff=1, kappa=1, norm=l2), a 12-particle cloud, and
``particle_l2``. Feature matching stays off.

Cover posture is the demo lock ``cover_weight=1.5`` (``LOCKED_COVER`` in
particle-sliders ``locked_baseline_defaults``). Music's pole analogue is
1.0; this toy does not use that value and a PASS here is not a Music or
Anima transfer claim.

Teacher is ``faithful_guard_e``: subtract leftover ê from the odd part
only while the blend guard still prefers the caption to the midpoint.
Raw ``faithful`` poles are the teacher-drift arm (ê is copied).
``cover_weight=0`` is the undershoot arm (guard on, no pole pin).

Budget knobs are ``steps`` and ``seed`` only. Formulation drift (stranger
pairing, FM-on under b_cap, a thinned b_cap, a hub-sized particle cloud)
is refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from particlegan import GANLoss, GradientPenalty, ParticlePrior, ParticleRegularizer


# Demo cover, not Music pole_weight=1.0. See module docstring.
LOCKED_COVER = 1.5
LOCKED_TEACHER = "faithful_guard_e"
LOCKED_N_PARTICLES = 12
U_KEPT_MIN = 0.85
CONTENT_KEPT_MIN = 0.75
LEAK_RATIO_MAX = 0.20
POLE_REL_ERR_MAX = 0.20
SAME_DIR_MAX = 0.25

# Shorter than the 1200-step demo lock. ``steps`` is budget, not formulation.
# 800 is where demo cover + guard has finished (u_kept ~0.94) and cover=0
# is still an undershoot (u_kept ~0.58). At 1600 the unpinned arm creeps
# toward the pole but stays under the 0.85 floor.
GATE_STEPS = 800

FORMULATION = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "fm_weight": 0.0,
    "n_particles": LOCKED_N_PARTICLES,
    "particle_l2": 0.02,
    "vicreg_weight": 0.05,
    "vicreg_std": 0.05,
    "cover_weight": LOCKED_COVER,
    "teacher": LOCKED_TEACHER,
    "lr": 5.0e-3,
    "beta1": 0.0,
    "beta2": 0.99,
    "batch": 32,
    "cloud_std": 0.03,
    "particle_jitter": 0.01,
    "span_frac": 0.40,
    "end_margin": 0.60,
    "ema": 0.995,
    "delay": 80,
    "min_lr_ratio": 0.05,
    "critic_hidden": 64,
    "critic_n_rand": 16,
    "particle_init_std": 0.05,
}

# Named deltas from FORMULATION. Anything else is refused.
ARMS = {
    "locked": {},
    "cover_zero": {"cover_weight": 0.0},
    "teacher_drift": {"teacher": "faithful"},
}


def reject_unlocked(overrides: dict, *, arm: str = "locked") -> None:
    """Refuse formulation drift. Named arms may change only their delta."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {sorted(ARMS)})")
    allowed = set(ARMS[arm]) | {"steps", "seed"}
    for key, value in overrides.items():
        if key in ("steps", "seed"):
            continue
        if key not in FORMULATION:
            raise ValueError(f"unknown formulation knob {key!r}")
        if key in allowed:
            continue
        if value == FORMULATION[key]:
            continue
        if key == "gan_mode":
            raise ValueError(
                f"stranger pairing refused: locked_shared is RpGAN logistic, got mode={value!r}"
            )
        if key == "fm_weight":
            raise ValueError(
                f"FM-on under b_cap refused: fm_weight={value!r} (locked fm_weight=0)"
            )
        if key in ("reg_arm", "reg_coeff", "reg_kappa", "reg_norm"):
            raise ValueError(
                f"thinned b_cap refused: {key}={value!r} != locked {FORMULATION[key]!r} "
                "(coeff=1, kappa=1, norm=l2, arm=b_cap)"
            )
        if key == "n_particles":
            raise ValueError(
                f"particle-count drift refused: n_particles={value!r} "
                f"(locked tiny cloud is {LOCKED_N_PARTICLES}, not a hub gmix)"
            )
        raise ValueError(
            f"formulation drift refused: {key}={value!r} != locked {FORMULATION[key]!r}"
        )


@dataclass(frozen=True)
class CoverRecipe:
    """Locked formulation plus one named arm. Only steps and seed are budget."""

    arm: str = "locked"
    steps: int = GATE_STEPS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r} (expected one of {sorted(ARMS)})")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an int")
        reject_unlocked(ARMS[self.arm], arm=self.arm)

    def knob(self, name: str):
        if name in ARMS[self.arm]:
            return ARMS[self.arm][name]
        return FORMULATION[name]


class _Residual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_odd = nn.Parameter(torch.zeros(dim))
        self.w_even = nn.Parameter(torch.zeros(dim))

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd + abs(float(scale)) * self.w_even


class _FourierCritic(nn.Module):
    """Fourier-2 critic used as the training instrument.

    ParticleGAN does not ship a critic. The gate scores the residual, so
    the critic stays this one module for every arm (no architecture swap).
    """

    def __init__(self, dim: int, *, n_rand: int, hidden: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(int(seed) + 17)
        bank = 2.0 * torch.randn(int(n_rand), int(dim), generator=gen)
        self.register_buffer("bank", bank)
        feat = 4 * int(dim) + 2 * int(n_rand)
        self.net = nn.Sequential(
            nn.Linear(feat, int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden), int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        order1 = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        order2 = torch.cat([torch.sin(2.0 * x), torch.cos(2.0 * x)], dim=-1)
        proj = x @ self.bank.T
        rand = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(torch.cat([order1, order2, rand], dim=-1)).squeeze(-1)


class _EMA:
    def __init__(self, params: list[torch.Tensor], decay: float) -> None:
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in params]

    def update(self, params: list[torch.Tensor]) -> None:
        for shadow, param in zip(self.shadow, params):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, params: list[torch.Tensor]) -> None:
        for shadow, param in zip(self.shadow, params):
            param.data.copy_(shadow)


def _unit(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.flatten()
    return flat / flat.norm().clamp_min(1e-8)


def hold_dir(leak_dir: torch.Tensor, slider_dir: torch.Tensor) -> torch.Tensor | None:
    """ê perpendicular to û. Near-zero leftover turns the hold off."""
    axis = leak_dir.flatten()
    unit = _unit(slider_dir)
    out = axis - (axis @ unit) * unit
    if float(out.norm()) <= 1e-8:
        return None
    return out


def blend_guard(
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> bool:
    """True when the target is nearer its caption than the pair midpoint."""
    mid = 0.5 * (pos + neg)
    to_pole = max(float((tgt_plus - pos).norm()), float((tgt_minus - neg).norm()))
    to_mid = min(float((tgt_plus - mid).norm()), float((tgt_minus - mid).norm()))
    return to_pole < to_mid


def faithful_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    slider_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Caption poles with leftover ê removed from the odd part only."""
    axis = (pos - neg) / 2.0
    held = hold_dir(leak_dir, slider_dir)
    if held is not None:
        unit = _unit(held)
        axis = axis - ((axis.flatten() @ unit) * unit).view_as(axis)
    common = (pos + neg) / 2.0 - neu
    return neu + common + axis, neu + common - axis


def faithful_guard_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    slider_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract leftover ê when the blend guard admits it, else keep the captions."""
    plus, minus = faithful_sub_e(pos, neg, neu, leak_dir, slider_dir)
    if blend_guard(plus, minus, pos, neg):
        return plus, minus
    return pos, neg


@dataclass(frozen=True)
class LeftoverField:
    """One-row R^4 leftover: û, content, ê, lyric. Matches Field3D defaults."""

    slider: float = 1.0
    content: float = 0.55
    leak: float = 0.45
    lyric: float = 1.0

    @property
    def dim(self) -> int:
        return 4

    def basis(self, index: int) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[index] = 1.0
        return out

    def odd(self) -> torch.Tensor:
        return (
            float(self.slider) * self.basis(0)
            + float(self.content) * self.basis(1)
            + float(self.leak) * self.basis(2)
        )

    def poles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.basis(3)
        amplitude = self.odd()
        return neu + amplitude, neu - amplitude, neu


def teacher_poles(
    field: LeftoverField,
    teacher: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos, neg, neu = field.poles()
    mode = str(teacher).strip().lower()
    if mode == "faithful":
        return pos, neg, neu
    if mode == "faithful_guard_e":
        plus, minus = faithful_guard_e(pos, neg, neu, field.basis(2), field.basis(0))
        return plus, minus, neu
    raise ValueError(f"unsupported teacher {teacher!r}")


def leftover_bipolar(d_plus: torch.Tensor, d_minus: torch.Tensor) -> dict[str, float]:
    """``leak_frac = cos(d+, d−)``, ``same_dir`` = even / (even + odd)."""
    even = 0.5 * (d_plus + d_minus)
    odd = 0.5 * (d_plus - d_minus)
    even_n = float(even.norm())
    odd_n = float(odd.norm())
    return {
        "leak_frac": float(
            F.cosine_similarity(d_plus.flatten().unsqueeze(0), d_minus.flatten().unsqueeze(0))
        ),
        "same_dir": even_n / (even_n + odd_n + 1e-8),
        "even_norm": even_n,
        "odd_norm": odd_n,
    }


def score_geometry(
    residual: _Residual,
    field: LeftoverField,
    poles_p: torch.Tensor,
    poles_m: torch.Tensor,
    neu: torch.Tensor,
) -> dict[str, float | bool]:
    d_plus = residual.delta(1.0).detach()
    d_minus = residual.delta(-1.0).detach()
    on_u = float(d_plus @ field.basis(0))
    on_c = float(d_plus @ field.basis(1))
    on_e = float(d_plus @ field.basis(2))
    u_kept = on_u / (float(field.slider) + 1e-8)
    content_kept = on_c / (float(field.content) + 1e-8)
    leak_ratio = abs(on_e) / (abs(on_u) + 1e-8)
    pred_p = neu + d_plus
    pred_m = neu + d_minus
    err_p = float((pred_p - poles_p).norm() / poles_p.norm().clamp_min(1e-8))
    err_m = float((pred_m - poles_m).norm() / poles_m.norm().clamp_min(1e-8))
    covered = bool(err_p <= POLE_REL_ERR_MAX and err_m <= POLE_REL_ERR_MAX)
    bipolar = leftover_bipolar(d_plus, d_minus)
    reasons = []
    if u_kept < U_KEPT_MIN or not covered:
        reasons.append("undershoot")
    if content_kept < CONTENT_KEPT_MIN:
        reasons.append("content")
    if leak_ratio > LEAK_RATIO_MAX:
        reasons.append("teacher_leak")
    if bipolar["same_dir"] > SAME_DIR_MAX:
        reasons.append("even_leftover")
    return {
        "u_kept": float(u_kept),
        "content_kept": float(content_kept),
        "leak_ratio": float(leak_ratio),
        "on_u": float(on_u),
        "on_content": float(on_c),
        "on_e": float(on_e),
        "pole_rel_err_plus": err_p,
        "pole_rel_err_minus": err_m,
        "covered": covered,
        "pass": not reasons,
        "fail_reasons": ",".join(reasons),
        **bipolar,
    }


def _delayed_cosine(step: int, total: int, delay: int, min_ratio: float) -> float:
    if step < int(delay):
        return 1.0
    span = max(1, int(total) - int(delay))
    t = min(1.0, float(step - int(delay)) / float(span))
    return float(min_ratio) + 0.5 * (1.0 - float(min_ratio)) * (1.0 + math.cos(math.pi * t))


def _sample_real_cloud(
    pole: torch.Tensor,
    neu: torch.Tensor,
    n: int,
    *,
    cloud_std: float,
    span_frac: float,
    end_margin: float,
) -> torch.Tensor:
    """Pole mass plus a short lyric-span lerp. One row, so every draw shares it."""
    n_end = int(round(float(end_margin) * int(n)))
    n_span = int(n) - n_end
    chunks = []
    if n_end:
        chunks.append(pole.expand(n_end, -1))
    if n_span:
        u = torch.rand(n_span, 1).sqrt()
        lo = 1.0 - float(span_frac)
        u = lo + (1.0 - lo) * u
        chunks.append(neu + u * (pole - neu))
    out = torch.cat(chunks, dim=0)
    if float(cloud_std) > 0.0:
        out = out + float(cloud_std) * torch.randn_like(out)
    return out


def _particle_batch(prior: ParticlePrior, n: int, jitter: float) -> torch.Tensor:
    z, _idx = prior.sample(n)
    if float(jitter) > 0.0:
        z = z + float(jitter) * torch.randn_like(z)
    return z


def _log(handle, message: str) -> None:
    line = message.rstrip() + "\n"
    print(line, end="", flush=True)
    if handle is not None:
        handle.write(line)
        handle.flush()


def fit_cover_leftover(recipe: CoverRecipe, *, log=None, field: LeftoverField | None = None) -> dict:
    """Train one arm. Returns the EMA residual score plus recipe pins."""
    field = field or LeftoverField()
    torch.manual_seed(recipe.seed)
    dim = field.dim
    residual = _Residual(dim)
    prior_p = ParticlePrior(recipe.knob("n_particles"), dim, init_std=recipe.knob("particle_init_std"))
    prior_m = ParticlePrior(recipe.knob("n_particles"), dim, init_std=recipe.knob("particle_init_std"))
    critic = _FourierCritic(
        dim,
        n_rand=recipe.knob("critic_n_rand"),
        hidden=recipe.knob("critic_hidden"),
        seed=recipe.seed,
    )
    gan = GANLoss(loss_type=recipe.knob("loss_type"), mode=recipe.knob("gan_mode"))
    penalty = GradientPenalty(
        arm=recipe.knob("reg_arm"),
        coeff=recipe.knob("reg_coeff"),
        kappa=recipe.knob("reg_kappa"),
        norm=recipe.knob("reg_norm"),
        lazy_k=1,
        target_anneal="none",
    )
    if penalty.arm != "b_cap" or penalty.coeff != 1.0 or penalty.kappa != 1.0 or penalty.norm != "l2":
        raise RuntimeError("b_cap champion was not constructed")
    spread = ParticleRegularizer(target_std=recipe.knob("vicreg_std"), weight=recipe.knob("vicreg_weight"))
    lr = float(recipe.knob("lr"))
    betas = (float(recipe.knob("beta1")), float(recipe.knob("beta2")))
    opt_g = torch.optim.Adam(
        [
            {"params": residual.parameters(), "lr": lr},
            {"params": list(prior_p.parameters()) + list(prior_m.parameters()), "lr": lr},
        ],
        lr=lr,
        betas=betas,
    )
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
    ema = _EMA(list(residual.parameters()), decay=float(recipe.knob("ema")))
    poles_p, poles_m, neu = teacher_poles(field, recipe.knob("teacher"))
    half = max(1, int(recipe.knob("batch")) // 2)
    jitter = float(recipe.knob("particle_jitter"))
    cover_w = float(recipe.knob("cover_weight"))
    particle_l2 = float(recipe.knob("particle_l2"))
    _log(
        log,
        "cover_leftover start arm=%s steps=%s seed=%s teacher=%s cover=%.1f "
        "b_cap=%.1f kappa=%.1f fm=%.1f n_particles=%s"
        % (
            recipe.arm,
            recipe.steps,
            recipe.seed,
            recipe.knob("teacher"),
            cover_w,
            penalty.coeff,
            penalty.kappa,
            float(recipe.knob("fm_weight")),
            recipe.knob("n_particles"),
        ),
    )

    def fake_batch() -> tuple[torch.Tensor, torch.Tensor]:
        fake_p = neu + residual.delta(1.0) + _particle_batch(prior_p, half, jitter)
        fake_m = neu + residual.delta(-1.0) + _particle_batch(prior_m, half, jitter)
        return fake_p, fake_m

    for step in range(recipe.steps):
        scale = _delayed_cosine(step, recipe.steps, int(recipe.knob("delay")), float(recipe.knob("min_lr_ratio")))
        for group in opt_g.param_groups:
            group["lr"] = lr * scale
        for group in opt_d.param_groups:
            group["lr"] = lr * scale
        real_p = _sample_real_cloud(
            poles_p, neu, half,
            cloud_std=recipe.knob("cloud_std"),
            span_frac=recipe.knob("span_frac"),
            end_margin=recipe.knob("end_margin"),
        )
        real_m = _sample_real_cloud(
            poles_m, neu, half,
            cloud_std=recipe.knob("cloud_std"),
            span_frac=recipe.knob("span_frac"),
            end_margin=recipe.knob("end_margin"),
        )
        real = torch.cat([real_p, real_m], dim=0)
        fake_p, fake_m = fake_batch()
        fake = torch.cat([fake_p, fake_m], dim=0).detach()
        d_loss = gan.d_loss(critic(real.detach()), critic(fake))
        cap = penalty(critic, real.detach(), fake, step=step + 1)
        d_loss = d_loss + cap
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        fake_p, fake_m = fake_batch()
        fake = torch.cat([fake_p, fake_m], dim=0)
        g_loss = gan.g_loss(critic(fake), critic(real.detach()))
        parts = torch.cat([prior_p.z, prior_m.z], dim=0)
        g_loss = g_loss + spread(parts)
        if particle_l2 > 0.0:
            g_loss = g_loss + particle_l2 * parts.pow(2).mean()
        if cover_w > 0.0:
            cover = (neu + residual.delta(1.0) - poles_p).pow(2).mean()
            cover = cover + (neu + residual.delta(-1.0) - poles_m).pow(2).mean()
            g_loss = g_loss + cover_w * cover
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema.update(list(residual.parameters()))

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == recipe.steps:
            live = score_geometry(residual, field, poles_p, poles_m, neu)
            _log(
                log,
                "cover_leftover arm=%s step=%s/%s d=%.4f g=%.4f cap=%.4f "
                "u_kept=%.3f content=%.3f leak=%.3f err=%.3f covered=%s"
                % (
                    recipe.arm,
                    step + 1,
                    recipe.steps,
                    float(d_loss.detach()),
                    float(g_loss.detach()),
                    float(cap.detach()),
                    live["u_kept"],
                    live["content_kept"],
                    live["leak_ratio"],
                    max(live["pole_rel_err_plus"], live["pole_rel_err_minus"]),
                    int(live["covered"]),
                ),
            )

    ema.copy_to(list(residual.parameters()))
    scored = score_geometry(residual, field, poles_p, poles_m, neu)
    with torch.no_grad():
        particle_rms = float(torch.cat([prior_p.z, prior_m.z], dim=0).pow(2).mean().sqrt())
    row = {
        "arm": recipe.arm,
        "steps": recipe.steps,
        "seed": recipe.seed,
        "teacher": recipe.knob("teacher"),
        "cover_weight": cover_w,
        "b_cap": float(penalty.coeff),
        "kappa": float(penalty.kappa),
        "norm": penalty.norm,
        "fm_weight": float(recipe.knob("fm_weight")),
        "gan_mode": recipe.knob("gan_mode"),
        "loss_type": recipe.knob("loss_type"),
        "n_particles": int(recipe.knob("n_particles")),
        "particle_l2": particle_l2,
        "particle_rms": particle_rms,
        "cover_posture": "demo_1.5",
        **scored,
    }
    _log(
        log,
        "cover_leftover DONE arm=%s pass=%s reasons=%s u_kept=%.3f content=%.3f "
        "leak=%.3f err=%.3f same_dir=%.3f particle_rms=%.3f"
        % (
            recipe.arm,
            int(row["pass"]),
            row["fail_reasons"] or "none",
            row["u_kept"],
            row["content_kept"],
            row["leak_ratio"],
            max(row["pole_rel_err_plus"], row["pole_rel_err_minus"]),
            row["same_dir"],
            particle_rms,
        ),
    )
    return row


def run_board(*, steps: int = GATE_STEPS, seed: int = 0, log=None) -> list[dict]:
    """Locked arm first, then the two intentional bad arms. One seed."""
    rows = []
    for arm in ("locked", "cover_zero", "teacher_drift"):
        rows.append(fit_cover_leftover(CoverRecipe(arm=arm, steps=steps, seed=seed), log=log))
    return rows


def format_board(rows: list[dict]) -> str:
    header = (
        "| arm | pass | cover | teacher | u_kept | content | leak | pole_err | same_dir | particle_rms | why |"
    )
    sep = "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        err = max(row["pole_rel_err_plus"], row["pole_rel_err_minus"])
        lines.append(
            "| %s | %s | %.1f | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s |"
            % (
                row["arm"],
                "PASS" if row["pass"] else "FAIL",
                row["cover_weight"],
                row["teacher"],
                row["u_kept"],
                row["content_kept"],
                row["leak_ratio"],
                err,
                row["same_dir"],
                row["particle_rms"],
                row["fail_reasons"] or "locked demo cover + guard",
            )
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    import json
    from pathlib import Path

    out = Path("results/cover_leftover")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    with log_path.open("w") as handle:
        rows = run_board(log=handle)
    payload = []
    for row in rows:
        item = dict(row)
        item["pass"] = bool(item["pass"])
        item["covered"] = bool(item["covered"])
        payload.append(item)
    (out / "board.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(format_board(rows), end="")
    print(f"wrote {log_path} {out / 'board.json'}")


if __name__ == "__main__":
    _main()
