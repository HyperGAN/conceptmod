#!/usr/bin/env python
"""CPU particle-posture gate (one family).

Locks the particle-sliders shared recipe onto a tiny cloud, and refuses the
Hub routed cloud. No GPU, no Music or YuE weight transfer.

Sources (particle-sliders, not this repo's 20k-particle recipe):

* Demo lock ``LOCKED`` in ``analysis/slider2d/locked_baseline_defaults.py``:
  RpGAN logistic, ``b_cap`` coeff=1, kappa=1, norm=l2, FM off, ``n_particles=12``,
  ``particle_l2=0.02``, ``cover_weight=1.5``, ``cloud_std=0.03``,
  ``vicreg_weight=0.05``. Slider ``ParticlePrior`` draws that cloud at
  ``init_std=0.05``. Slider VICReg ``std_target`` is 0.05 (poles are O(1)).
* Music Arm B (``docs/music-arm-b-gates.md``, ``ARM_B``): ``--parts 0``,
  cover/pole ``1.0``, ``vicreg_weight=0``. Parts and ``n_particles`` are
  different objects; parts 0 is the empty-cloud spelling of the same tiny
  posture, not a 12-particle prior.
* Refused: YuE2/Music3 Hub bridge ``parts=128``, ``torch.randn`` init,
  ``routing=all_examples`` (``yue2_particle_bridge.py``). Not locked.

ParticleGAN ``Recipe()`` is a different prior (20_000 particles, ``prior_reg=1``,
variance target 1). This toy does not call it. Variance hinge uses
``ParticleRegularizer(target_std=0.05)`` on the tiny cloud so the library
default target of 1 cannot inflate the cloud. Critic is one fixed MLP for
every arm (no architecture swap).

Budget, not formulation: ambient dim 2, batch 32, 600 Adam steps, one seed.
``lr=5e-3`` moves a unit pole by about one step of that size, so a few
hundred steps are the travel budget; the particle cloud lags the residual
and arrives later. Shorter runs stop mid-travel.
``python -u experiments/particle_posture_toy.py`` prints a line per log step.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from particlegan import GANLoss, GradientPenalty, ParticlePrior, ParticleRegularizer


FAMILY = "particle_prior_posture"
DIM = 2
BATCH = 32
STEPS = 600
LOG_EVERY = 40
SEED = 0
LR = 5e-3
BETA1 = 0.0
BETA2 = 0.99
CLOUD_STD = 0.03
INIT_STD = 0.05
JITTER = 0.01
DEMO_N = 12
DEMO_L2 = 0.02
DEMO_COVER = 1.5
DEMO_VICREG = 0.05
DEMO_VICREG_STD = 0.05
MUSIC_COVER = 1.0
HUB_N = 128
HUB_INIT = 1.0
# Tiny-cloud second moment. Init N(0, 0.05^2) sits near 0.0025; Hub N(0, 1) sits near 1.
CLOUD_MS_MAX = 0.05
RESIDUAL_MAX = 0.05
TARGET = (1.0, 0.0)


@dataclass(frozen=True)
class Posture:
    name: str
    n_particles: int
    particle_l2: float
    init_std: float
    cover_weight: float
    vicreg_weight: float
    vicreg_std: float
    routing: str = "none"
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    grad_norm: str = "l2"
    grad_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cloud_std: float = CLOUD_STD
    particle_jitter: float = JITTER
    lr: float = LR
    seed: int = SEED


def locked_tiny() -> Posture:
    """Demo lock: n=12 and particle_l2=0.02. Cover is the 2-D pin 1.5."""
    return Posture(
        name="locked_tiny",
        n_particles=DEMO_N,
        particle_l2=DEMO_L2,
        init_std=INIT_STD,
        cover_weight=DEMO_COVER,
        vicreg_weight=DEMO_VICREG,
        vicreg_std=DEMO_VICREG_STD,
    )


def music_parts0() -> Posture:
    """Music Arm B spelling: no cloud, cover 1.0, VICReg absent."""
    return Posture(
        name="music_parts0",
        n_particles=0,
        particle_l2=0.0,
        init_std=0.0,
        cover_weight=MUSIC_COVER,
        vicreg_weight=0.0,
        vicreg_std=0.0,
        particle_jitter=0.0,
    )


def hub128_routed() -> Posture:
    """Hub bridge cloud: 128 standard-normal particles, routed, no particle_l2."""
    return Posture(
        name="hub128_routed",
        n_particles=HUB_N,
        particle_l2=0.0,
        init_std=HUB_INIT,
        cover_weight=DEMO_COVER,
        vicreg_weight=1.0,
        vicreg_std=1.0,
        routing="all_examples",
    )


def zero_particle_l2() -> Posture:
    """Same tiny cloud as the lock, with the L2 anchor removed."""
    return replace(locked_tiny(), name="zero_particle_l2", particle_l2=0.0)


ARMS = (locked_tiny, music_parts0, hub128_routed, zero_particle_l2)


class MLPCritic(nn.Module):
    """Fixed critic. Every arm uses this module; nothing swaps it."""

    def __init__(self, dim: int = DIM, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_penalty(cfg: Posture) -> GradientPenalty:
    return GradientPenalty(
        arm=cfg.reg_arm,
        coeff=cfg.reg_coeff,
        kappa=cfg.reg_kappa,
        lazy_k=cfg.grad_lazy,
        norm=cfg.grad_norm,
        target_anneal=cfg.target_anneal,
    )


def bcap_kappa_is_explicit(make_reg) -> bool:
    """True when kappa=1 leaves a 0.3-slope free and kappa=0.2 penalizes it.

    A stub that hardcodes kappa=1 returns the same penalty for both calls.
    """
    critic = nn.Linear(DIM, 1, bias=False)
    with torch.no_grad():
        weight = torch.zeros(1, DIM)
        weight[0, 0] = 0.3
        critic.weight.copy_(weight)
    batch = torch.zeros(4, DIM)
    loose = float(make_reg(1.0).penalty(critic, batch, batch, step=1)[0].detach())
    tight = float(make_reg(0.2).penalty(critic, batch, batch, step=1)[0].detach())
    return loose == 0.0 and tight > 0.0 and math.isfinite(tight)


def _close(got, want) -> bool:
    if isinstance(want, float):
        return math.isfinite(got) and math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-12)
    return got == want


def anchor_drop(particle_l2: float, n: int, dim: int = DIM, lr: float = LR) -> float:
    """One SGD step on ``particle_l2 * mean(z^2)`` from a unit cloud.

    Zero weight leaves the cloud where it is. The locked 0.02 weight contracts it.
    """
    if n <= 0:
        return 0.0
    cloud = torch.ones(n, dim, requires_grad=True)
    before = float(cloud.detach().pow(2).mean())
    (float(particle_l2) * cloud.pow(2).mean()).backward()
    with torch.no_grad():
        nxt = cloud - lr * cloud.grad
    return before - float(nxt.pow(2).mean())


def evaluate(cfg: Posture, metrics: dict) -> tuple[bool, tuple[str, ...]]:
    """Fail closed. Empty reasons means PASS."""
    reasons: list[str] = []
    locked_adv = {
        "loss_type": "logistic",
        "gan_mode": "rp",
        "reg_arm": "b_cap",
        "reg_coeff": 1.0,
        "reg_kappa": 1.0,
        "grad_norm": "l2",
        "grad_lazy": 1,
        "target_anneal": "none",
        "fm_weight": 0.0,
    }
    for key, want in locked_adv.items():
        if not _close(getattr(cfg, key), want):
            reasons.append(f"adv drift {key}={getattr(cfg, key)!r} (locked {want!r})")
    if cfg.routing != "none":
        reasons.append(f"routing {cfg.routing!r} is not none (Hub routes every example)")
    if not metrics.get("bcap_faithful", False):
        reasons.append("b_cap is not the explicit-kappa GradientPenalty")
    if not metrics.get("finite", False):
        reasons.append("non-finite toy losses")
    n = int(cfg.n_particles)
    if n < 0:
        reasons.append(f"n_particles={n}")
    elif n == 0:
        if not _close(cfg.cover_weight, MUSIC_COVER):
            reasons.append(f"parts 0 cover {cfg.cover_weight} is not Music {MUSIC_COVER}")
        if not _close(cfg.vicreg_weight, 0.0) or not _close(cfg.particle_l2, 0.0):
            reasons.append("parts 0 keeps vicreg and particle_l2 off (no cloud to anchor)")
        if metrics.get("has_cloud", True):
            reasons.append("parts 0 allocated a particle cloud")
        if float(metrics.get("residual_l2", math.inf)) > RESIDUAL_MAX:
            reasons.append("residual missed the pole")
    elif n <= DEMO_N:
        if not _close(cfg.particle_l2, DEMO_L2):
            reasons.append(f"particle_l2={cfg.particle_l2} is not the locked {DEMO_L2} anchor")
        if float(cfg.init_std) > INIT_STD + 1e-12:
            reasons.append(f"noisy init_std={cfg.init_std} (locked draw is {INIT_STD})")
        if not _close(cfg.cover_weight, DEMO_COVER):
            reasons.append(f"n<=12 cover {cfg.cover_weight} is not the demo pin {DEMO_COVER}")
        if not _close(cfg.vicreg_weight, DEMO_VICREG) or not _close(cfg.vicreg_std, DEMO_VICREG_STD):
            reasons.append("tiny-cloud VICReg is weight 0.05 at std target 0.05")
        if metrics.get("has_cloud") is not True:
            reasons.append("tiny posture has no cloud")
        if float(metrics.get("cloud_ms", math.inf)) > CLOUD_MS_MAX:
            reasons.append(f"cloud second moment {metrics.get('cloud_ms')} exceeds {CLOUD_MS_MAX}")
        if float(metrics.get("anchor_drop", 0.0)) <= 0.0:
            reasons.append("particle_l2 anchor did not contract a unit cloud")
        if float(metrics.get("residual_l2", math.inf)) > RESIDUAL_MAX:
            reasons.append("residual missed the pole")
    else:
        reasons.append(f"oversized n_particles={n} (lock is n<={DEMO_N}, not Hub {HUB_N})")
        if float(metrics.get("cloud_ms", 0.0)) > CLOUD_MS_MAX:
            reasons.append(f"noisy cloud second moment {metrics.get('cloud_ms')}")
    return (len(reasons) == 0), tuple(reasons)


def _emit(msg: str, log) -> None:
    print(msg, flush=True)
    if log is not None:
        log.write(msg + "\n")
        log.flush()


def fit_posture(cfg: Posture, *, log=None, bcap_faithful: bool | None = None) -> dict:
    """Short CPU fit. Prints progress so the run can be tailed."""
    if bcap_faithful is None:
        bcap_faithful = bcap_kappa_is_explicit(
            lambda kappa: GradientPenalty(arm="b_cap", coeff=1.0, kappa=float(kappa), norm="l2")
        )
    torch.manual_seed(cfg.seed)
    target = torch.tensor(TARGET)
    residual = nn.Parameter(torch.zeros(DIM))
    prior = None
    if cfg.n_particles > 0:
        prior = ParticlePrior(
            num_particles=cfg.n_particles,
            z_dim=DIM,
            init_std=cfg.init_std,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    critic = MLPCritic()
    adv = GANLoss(loss_type=cfg.loss_type, mode=cfg.gan_mode)
    penalty = make_penalty(cfg)
    groups = [{"params": [residual], "lr": cfg.lr}]
    if prior is not None:
        groups.append({"params": [p for p in prior.parameters() if p.requires_grad], "lr": cfg.lr})
    opt_g = torch.optim.Adam(groups, lr=cfg.lr, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(BETA1, BETA2))
    vic = None
    if prior is not None and cfg.vicreg_weight > 0.0 and cfg.n_particles >= 2:
        vic = ParticleRegularizer(target_std=cfg.vicreg_std, weight=cfg.vicreg_weight)

    _emit(
        f"particle_posture arm={cfg.name} start n={cfg.n_particles} "
        f"particle_l2={cfg.particle_l2} init_std={cfg.init_std} cover={cfg.cover_weight} "
        f"routing={cfg.routing} steps={STEPS}",
        log,
    )
    last_d = last_g = last_cap = float("nan")
    for step in range(STEPS):
        real = target + cfg.cloud_std * torch.randn(BATCH, DIM)
        if prior is None:
            fake_raw = residual.expand(BATCH, DIM)
        else:
            sampled, _ = prior.sample(BATCH)
            noise = cfg.particle_jitter * torch.randn_like(sampled) if cfg.particle_jitter else 0.0
            fake_raw = residual + sampled + noise

        opt_d.zero_grad(set_to_none=True)
        d_real = critic(real.detach())
        d_fake = critic(fake_raw.detach())
        cap, _stats = penalty.penalty(critic, real.detach(), fake_raw.detach(), step=step + 1)
        d_loss = adv.d_loss(d_real, d_fake) + cap
        d_loss.backward()
        opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        if prior is None:
            fake = residual.expand(BATCH, DIM)
        else:
            sampled, _ = prior.sample(BATCH)
            noise = cfg.particle_jitter * torch.randn_like(sampled) if cfg.particle_jitter else 0.0
            fake = residual + sampled + noise
        d_real = critic(real.detach())
        d_fake = critic(fake)
        g_loss = adv.g_loss(d_fake, d_real)
        if prior is not None and cfg.particle_l2 > 0.0:
            g_loss = g_loss + float(cfg.particle_l2) * prior.z.pow(2).mean()
        if vic is not None:
            g_loss = g_loss + vic(prior.z)
        if cfg.fm_weight > 0.0:
            g_loss = g_loss + float(cfg.fm_weight) * (d_fake.mean() - d_real.detach().mean()).pow(2)
        if cfg.cover_weight > 0.0:
            g_loss = g_loss + float(cfg.cover_weight) * (residual - target).pow(2).mean()
        g_loss.backward()
        opt_g.step()

        last_d = float(d_loss.detach())
        last_g = float(g_loss.detach())
        last_cap = float(cap.detach())
        if step == 0 or (step + 1) % LOG_EVERY == 0 or step + 1 == STEPS:
            cloud_now = 0.0 if prior is None else float(prior.z.detach().pow(2).mean())
            residual_now = float((residual.detach() - target).norm())
            _emit(
                f"particle_posture arm={cfg.name} step={step + 1}/{STEPS} "
                f"cloud_ms={cloud_now:.6f} residual_l2={residual_now:.4f} "
                f"d={last_d:.4f} g={last_g:.4f} cap={last_cap:.4f}",
                log,
            )

    cloud_ms = 0.0 if prior is None else float(prior.z.detach().pow(2).mean())
    residual_l2 = float((residual.detach() - target).norm())
    drop = anchor_drop(cfg.particle_l2, cfg.n_particles)
    finite = all(math.isfinite(v) for v in (cloud_ms, residual_l2, drop, last_d, last_g, last_cap))
    metrics = {
        "cloud_ms": cloud_ms,
        "residual_l2": residual_l2,
        "anchor_drop": drop,
        "finite": finite,
        "has_cloud": prior is not None,
        "bcap_faithful": bool(bcap_faithful),
        "d_loss": last_d,
        "g_loss": last_g,
        "cap": last_cap,
        "critic_hidden": critic.hidden,
    }
    passed, reasons = evaluate(cfg, metrics)
    row = {
        "name": cfg.name,
        "n_particles": cfg.n_particles,
        "particle_l2": cfg.particle_l2,
        "init_std": cfg.init_std,
        "cover_weight": cfg.cover_weight,
        "vicreg_weight": cfg.vicreg_weight,
        "routing": cfg.routing,
        "passed": passed,
        "reasons": reasons,
        **metrics,
    }
    reason_s = "-" if passed else "; ".join(reasons)
    _emit(
        f"particle_posture arm={cfg.name} GATE={'PASS' if passed else 'FAIL'} "
        f"cloud_ms={cloud_ms:.6f} residual_l2={residual_l2:.4f} "
        f"anchor_drop={drop:.3e} reasons={reason_s}",
        log,
    )
    return row


def run_family(names: list[str] | None = None, *, log=None) -> list[dict]:
    torch.set_num_threads(1)
    faithful = bcap_kappa_is_explicit(
        lambda kappa: GradientPenalty(arm="b_cap", coeff=1.0, kappa=float(kappa), norm="l2")
    )
    selected = ARMS if not names else tuple(fn for fn in ARMS if fn().name in set(names))
    if names and len(selected) != len(set(names)):
        known = [fn().name for fn in ARMS]
        raise SystemExit(f"unknown arm in {names}; expected a subset of {known}")
    return [fit_posture(fn(), log=log, bcap_faithful=faithful) for fn in selected]


def render_leaderboard(rows: list[dict]) -> str:
    lines = [
        "# Particle posture toy",
        "",
        "One CPU family. Locked shared adv is RpGAN logistic + ParticleGAN",
        "`GradientPenalty` `b_cap` (coeff=1, kappa=1, norm=l2, lazy=1) + FM off.",
        "The critic is one MLP, held fixed. This is not a Music or YuE GPU result.",
        "",
        "Cover: **1.5** is the demo lock for the n=12 cloud. **1.0** is Music Arm B",
        "at `--parts 0` only. Those covers are not interchangeable",
        "(`test_music_arm_b_gates.py` rejects cover 1.5 on the Music row).",
        "",
        "ParticleGAN `Recipe()` (20_000 particles, `prior_reg=1`, variance target 1)",
        "is a different prior. The tiny arm uses `ParticleRegularizer` weight 0.05",
        "at std target 0.05, matching the slider cloud rather than that default.",
        "",
        f"Budget: dim={DIM}, batch={BATCH}, steps={STEPS}, seed={SEED}, lr={LR}.",
        f"PASS requires cloud second moment ≤ {CLOUD_MS_MAX} and residual L2 ≤ {RESIDUAL_MAX}",
        "on the tiny cloud, and no cloud at all on parts 0.",
        "",
        "| arm | n | particle_l2 | init_std | cover | cloud_ms | anchor_drop | residual_l2 | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        gate = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| `{row['name']}` | {row['n_particles']} | {row['particle_l2']} | "
            f"{row['init_std']} | {row['cover_weight']} | {row['cloud_ms']:.6f} | "
            f"{row['anchor_drop']:.3e} | {row['residual_l2']:.4f} | **{gate}** |"
        )
    lines.extend(["", "## Why", ""])
    by_name = {row["name"]: row for row in rows}
    if "locked_tiny" in by_name:
        row = by_name["locked_tiny"]
        lines.append(
            f"`locked_tiny` is the demo lock (n={DEMO_N}, particle_l2={DEMO_L2}, "
            f"cover={DEMO_COVER}). cloud_ms={row['cloud_ms']:.6f}, "
            f"residual_l2={row['residual_l2']:.4f}, anchor_drop={row['anchor_drop']:.3e}. "
            f"Gate {'PASS' if row['passed'] else 'FAIL'}."
        )
    if "music_parts0" in by_name:
        row = by_name["music_parts0"]
        lines.append(
            f"`music_parts0` is Music `--parts 0` (cover={MUSIC_COVER}, no cloud, "
            f"VICReg off). residual_l2={row['residual_l2']:.4f}. "
            f"Gate {'PASS' if row['passed'] else 'FAIL'}. "
            "Same adv lock, empty cloud — not a claim that a Music GPU run passed."
        )
    if "hub128_routed" in by_name:
        row = by_name["hub128_routed"]
        lines.append(
            f"`hub128_routed` is the Hub bridge cloud (n={HUB_N}, init N(0,1), "
            f"routing=all_examples, particle_l2=0). cloud_ms={row['cloud_ms']:.6f}. "
            f"Gate FAIL. {row['reasons'][0] if row['reasons'] else ''}"
        )
    if "zero_particle_l2" in by_name:
        row = by_name["zero_particle_l2"]
        lines.append(
            "`zero_particle_l2` keeps n=12 and the 0.05 draw, and drops only "
            f"particle_l2. cloud_ms={row['cloud_ms']:.6f} can stay small "
            "(the 2026-09-09 slider sweep also passed the sheet at l2=0), "
            f"but anchor_drop={row['anchor_drop']:.3e}. Gate FAIL: the 0.02 anchor is part of the lock."
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        "- Keep the living toy cloud at n≤12, init_std 0.05, particle_l2=0.02, demo cover 1.5.",
        "- Music transfer stays `--parts 0`, cover/pole 1.0, vicreg 0. Do not copy n=12 or cover 1.5 onto that row.",
        "- Do not treat Hub 128 routed N(0,1) particles, or `Recipe()`'s 20_000-particle prior, as this lock.",
        "- Do not read a toy PASS as a Music or YuE training result.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CPU particle posture gate")
    parser.add_argument("--arm", action="append", default=None, help="arm name (repeatable); default runs the family")
    parser.add_argument(
        "--out",
        default="results/particle_posture/LEADERBOARD.md",
        help="leaderboard path",
    )
    parser.add_argument(
        "--log",
        default="results/particle_posture/toy.log",
        help="append-friendly step log",
    )
    args = parser.parse_args(argv)
    from pathlib import Path

    log_path = Path(args.log)
    out_path = Path(args.out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        rows = run_family(args.arm, log=log)
        text = render_leaderboard(rows)
        _emit(text, log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    failed = [row["name"] for row in rows if row["name"] in ("locked_tiny", "music_parts0") and not row["passed"]]
    if failed:
        raise SystemExit(f"winning posture failed: {failed}")


if __name__ == "__main__":
    main()
