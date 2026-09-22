"""Locked-shared RpGAN + faithful ``b_cap`` regression floor (CPU).

Mirrors the particle-sliders locked_shared / #94 stamp
(``analysis/slider2d/locked_baseline_defaults.py``, Music Arm B gates)
using this repo's primitives. The floor scores formulation identity.
It does not claim a Music or Anima GPU transfer.

Demo cover posture: ``cover_weight=1.5`` (the 2-D locked demo). Music
transfer cover is 1.0 and fails this stamp. There is no leftover axis
here, so Field3D leak is not scored.

Naming vs particle-sliders (same knobs, different fields):

    sliders ``b_cap`` / ``adv_b_cap``     -> ``reg_coeff`` / ``GradientPenalty.coeff``
    sliders ``kappa`` / ``adv_reg_kappa`` -> ``reg_kappa`` / ``.kappa``
    sliders ``grad_arm``                  -> ``reg_arm`` / ``.arm``
    sliders ``grad_norm`` / ``adv_norm``  -> ``reg_norm`` / ``.norm``
    sliders ``grad_lazy``                 -> ``reg_lazy`` / ``.lazy_k``
    sliders ``rp_d_loss`` / ``rp_g_loss`` -> ``GANLoss("logistic", "rp")``
    sliders ``n_particles=12``            -> this floor's cloud, not ``Recipe`` 20_000
    sliders ``particle_l2``               -> mean square on the cloud, not VICReg
    sliders ``fm_weight``                 -> toy knob; ``Recipe`` has no FM term

``Recipe("gan")`` stays the 100-Gaussians study default. This module does
not retune it. The critic is caller-owned and unscored.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import GANLoss, GradientPenalty, ParticlePrior


# Demo locked_shared cloud. Not the Hub 128-particle gmix, not Recipe's 20k.
N_PARTICLES = 12
PARTICLE_L2 = 0.02
COVER_WEIGHT = 1.5  # demo; Music Arm B transfer is 1.0
COVER_POSTURE = "demo"
FM_WEIGHT = 0.0

# Two fixed centers. Cover pins generator offsets to these (demo mode pin).
DEMO_TARGETS = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

BUDGET_KEYS = frozenset({"steps", "seed", "lr", "batch"})


@dataclass(frozen=True)
class LockedSharedFloor:
    """Formulation stamp plus budget knobs the gate ignores."""

    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    reg_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = FM_WEIGHT
    particle_l2: float = PARTICLE_L2
    n_particles: int = N_PARTICLES
    z_dim: int = 2
    cover_weight: float = COVER_WEIGHT
    cover_posture: str = COVER_POSTURE
    steps: int = 8
    seed: int = 0
    lr: float = 5.0e-3
    batch: int = 16


LOCKED = LockedSharedFloor()


class ThinnedKappaCap(GradientPenalty):
    """Intentional drift: κ stored on the object, center hardcoded to 1.

    At the locked κ=1 this matches ``GradientPenalty`` numerically. A κ
    probe (0.2 on a slope-0.4 critic) is what separates it from the
    faithful module.
    """

    def center(self, step: int) -> float:
        if self.arm in ("b_cap", "g_interp_cap"):
            return 1.0
        return super().center(step)


class ToyCritic(nn.Module):
    """Caller-owned MLP for this toy. The gate does not read its layers."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2))
        self.out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(x)).squeeze(-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


class ToyGenerator(nn.Module):
    """Data-space particles plus one offset per demo mode."""

    def __init__(self) -> None:
        super().__init__()
        self.offsets = nn.Parameter(torch.zeros(2, 2))

    def forward(self, z: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        return self.offsets[mode] + z


def cover_mse(offsets: torch.Tensor) -> torch.Tensor:
    """Demo cover analogue: pin mode offsets to ``DEMO_TARGETS``.

    Not Music leftover and not a Field3D leak term.
    """
    targets = DEMO_TARGETS.to(device=offsets.device, dtype=offsets.dtype)
    return (offsets - targets).pow(2).mean()


def feature_match(feat_real: torch.Tensor, feat_fake: torch.Tensor) -> torch.Tensor:
    """Normalized mean-feature L2. Same shape as the sliders FM term."""
    real = F.normalize(feat_real.flatten(1).mean(dim=0), dim=0)
    fake = F.normalize(feat_fake.flatten(1).mean(dim=0), dim=0)
    return (real - fake).pow(2).mean()


def make_regularizer(cfg: LockedSharedFloor, *, kind: str = "faithful") -> GradientPenalty:
    """Faithful ``GradientPenalty``, or the κ-hardcoded drift."""
    cls = GradientPenalty if kind == "faithful" else ThinnedKappaCap
    if kind not in ("faithful", "thinned"):
        raise ValueError(f"unknown regularizer kind {kind!r}")
    return cls(
        arm=cfg.reg_arm,
        coeff=cfg.reg_coeff,
        kappa=cfg.reg_kappa,
        norm=cfg.reg_norm,
        lazy_k=cfg.reg_lazy,
        target_anneal=cfg.target_anneal,
        total_steps=cfg.steps if cfg.target_anneal != "none" else 0,
    )


def _aligned(real_logits: torch.Tensor, fake_logits: torch.Tensor, pairing: str):
    if pairing == "pair":
        return real_logits, fake_logits
    if pairing == "stranger":
        # Reverse the fake row. A one-step roll hides inside a flat critic;
        # a full reversal changes the relativistic pairs.
        return real_logits, fake_logits.flip(0)
    raise ValueError(f"unknown pairing {pairing!r}")


@dataclass
class FloorObs:
    """One scored step. Tensors are detached clones from that step."""

    cfg: LockedSharedFloor
    reg_cls: type
    pairing: str
    d_real: torch.Tensor
    d_fake: torch.Tensor
    d_adv: torch.Tensor
    penalty: torch.Tensor
    critic_snap: nn.Module
    real: torch.Tensor
    fake: torch.Tensor
    g_real: torch.Tensor
    g_fake: torch.Tensor
    g_total: torch.Tensor
    parts: torch.Tensor
    offsets: torch.Tensor
    n_particles: int
    # Non-flat logits so a shuffled pair cannot hide inside a flat critic.
    spread_real: torch.Tensor
    spread_fake: torch.Tensor
    spread_adv: torch.Tensor


def _stamp_mismatches(cfg: LockedSharedFloor) -> list[str]:
    bad = []
    for field in (
        "loss_type", "gan_mode", "reg_arm", "reg_coeff", "reg_kappa", "reg_norm",
        "reg_lazy", "target_anneal", "fm_weight", "particle_l2", "n_particles",
        "z_dim", "cover_weight", "cover_posture",
    ):
        got, want = getattr(cfg, field), getattr(LOCKED, field)
        if got != want:
            bad.append(f"{field}: {got!r} != locked_shared {want!r}")
    return bad


def _kappa_probe(reg_cls: type) -> tuple[float, str | None]:
    """Slope 0.4 is under κ=1 and over κ=0.2. Hardcoded κ=1 scores 0."""
    critic = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[0.4, 0.0]]))
    batch = torch.zeros(4, 2)
    try:
        probed = reg_cls(
            arm="b_cap", coeff=1.0, kappa=0.2, norm="l2",
            lazy_k=1, target_anneal="none",
        )
        got = probed.penalty(critic, batch, batch)[0]
    except Exception as exc:
        return float("inf"), f"kappa probe refused {reg_cls.__name__}: {exc}"
    faithful = GradientPenalty(
        arm="b_cap", coeff=1.0, kappa=0.2, norm="l2",
    ).penalty(critic, batch, batch)[0]
    err = abs(float(got.detach()) - float(faithful.detach()))
    if err > 1e-5:
        return err, (
            f"thinned or non-faithful b_cap: kappa probe abs err {err:.3e} "
            "(κ=0.2 on ||g||=0.4 must match GradientPenalty)"
        )
    return err, None


def score_floor(obs: FloorObs) -> dict:
    """PASS only for the locked_shared shape wired through the real modules."""
    bad = _stamp_mismatches(obs.cfg)
    if obs.pairing != "pair":
        bad.append(f"pairing: {obs.pairing!r} != locked_shared 'pair'")
    if obs.reg_cls is not GradientPenalty:
        bad.append(
            f"regularizer: {obs.reg_cls.__name__} is not particlegan.GradientPenalty"
        )
    if obs.n_particles != obs.cfg.n_particles:
        bad.append(
            f"n_particles: cloud {obs.n_particles} != claimed {obs.cfg.n_particles}"
        )

    rp = GANLoss("logistic", "rp")
    d_want = rp.d_loss(obs.d_real, obs.d_fake)
    spread_want = rp.d_loss(obs.spread_real, obs.spread_fake)
    g_want = (
        rp.g_loss(obs.g_fake, obs.g_real)
        + PARTICLE_L2 * obs.parts.pow(2).mean()
        + COVER_WEIGHT * cover_mse(obs.offsets)
    )
    cap_want = GradientPenalty(
        arm="b_cap", coeff=1.0, kappa=1.0, norm="l2",
    ).penalty(obs.critic_snap, obs.real, obs.fake)[0]
    data_adv_err = abs(float(obs.d_adv.detach()) - float(d_want.detach()))
    spread_adv_err = abs(float(obs.spread_adv.detach()) - float(spread_want.detach()))
    adv_err = max(data_adv_err, spread_adv_err)
    g_err = abs(float(obs.g_total.detach()) - float(g_want.detach()))
    cap_err = abs(float(obs.penalty.detach()) - float(cap_want.detach()))
    if adv_err > 1e-5:
        bad.append(f"adv: abs err {adv_err:.3e} vs RpGAN logistic pair")
    if g_err > 1e-5:
        bad.append(
            f"g: abs err {g_err:.3e} vs RpGAN + particle_l2={PARTICLE_L2} "
            f"+ demo cover {COVER_WEIGHT} + FM off"
        )
    if cap_err > 1e-5:
        bad.append(f"b_cap: abs err {cap_err:.3e} vs GradientPenalty coeff=1 κ=1 l2")
    probe_err, probe_bad = _kappa_probe(obs.reg_cls)
    if probe_bad:
        bad.append(probe_bad)

    return {
        "verdict": "PASS" if not bad else "FAIL",
        "mismatches": bad,
        "adv_abs_err": adv_err,
        "g_abs_err": g_err,
        "cap_abs_err": cap_err,
        "kappa_probe_abs_err": probe_err,
        "fm_weight": obs.cfg.fm_weight,
        "cover_weight": obs.cfg.cover_weight,
        "cover_posture": obs.cfg.cover_posture,
        "gan_mode": obs.cfg.gan_mode,
        "loss_type": obs.cfg.loss_type,
        "n_particles": obs.n_particles,
        # Public name. The class statement is GradRegularizer; GradientPenalty
        # is the alias particlegan exports.
        "reg_class": "GradientPenalty" if obs.reg_cls is GradientPenalty else obs.reg_cls.__name__,
    }


def run_floor(
    cfg: LockedSharedFloor | None = None,
    *,
    regularizer: str = "faithful",
    pairing: str = "pair",
    log: bool = False,
    arm: str = "locked_shared",
) -> FloorObs:
    """Tiny CPU loop. The last step is the regression observation."""
    cfg = LOCKED if cfg is None else cfg
    torch.manual_seed(cfg.seed)
    critic = ToyCritic()
    generator = ToyGenerator()
    prior = ParticlePrior(num_particles=cfg.n_particles, z_dim=cfg.z_dim, init_std=0.05)
    gan = GANLoss(loss_type=cfg.loss_type, mode=cfg.gan_mode)
    reg = make_regularizer(cfg, kind=regularizer)
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(0.0, 0.99))
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(prior.parameters()),
        lr=cfg.lr, betas=(0.0, 0.99),
    )
    obs = None
    for step in range(1, cfg.steps + 1):
        mode = torch.randint(0, 2, (cfg.batch,))
        real = DEMO_TARGETS[mode] + 0.03 * torch.randn(cfg.batch, 2)
        z, _ = prior.sample(cfg.batch)
        fake = generator(z, mode).detach()
        d_real = critic(real)
        d_fake = critic(fake)
        paired_real, paired_fake = _aligned(d_real, d_fake, pairing)
        d_adv = gan.d_loss(paired_real, paired_fake)
        penalty, _stats = reg.penalty(critic, real, fake, step=step)
        last = step == cfg.steps
        if last:
            spread_x = torch.linspace(-2.0, 2.0, cfg.batch).unsqueeze(1).expand(cfg.batch, 2).contiguous()
            spread_real_logits = critic(spread_x)
            spread_fake_logits = critic(spread_x + 0.7)
            spread_adv = gan.d_loss(*_aligned(spread_real_logits, spread_fake_logits, pairing))
            d_snap = (
                d_real.detach().clone(),
                d_fake.detach().clone(),
                d_adv.detach().clone(),
                penalty.detach().clone(),
                copy.deepcopy(critic),
                real.detach().clone(),
                fake.detach().clone(),
                spread_real_logits.detach().clone(),
                spread_fake_logits.detach().clone(),
                spread_adv.detach().clone(),
            )
        opt_d.zero_grad(set_to_none=True)
        (d_adv + penalty).backward()
        opt_d.step()

        mode_g = torch.randint(0, 2, (cfg.batch,))
        z_g, _ = prior.sample(cfg.batch)
        fake_g = generator(z_g, mode_g)
        g_real = critic(real.detach())
        g_fake = critic(fake_g)
        pr, pf = _aligned(g_real, g_fake, pairing)
        g_adv = gan.g_loss(pf, pr)
        g_extra = cfg.particle_l2 * prior.z.pow(2).mean()
        g_extra = g_extra + cfg.cover_weight * cover_mse(generator.offsets)
        if cfg.fm_weight > 0.0:
            g_extra = g_extra + cfg.fm_weight * feature_match(
                critic.features(real.detach()), critic.features(fake_g),
            )
        g_total = g_adv + g_extra
        if last:
            obs = FloorObs(
                cfg=cfg,
                reg_cls=type(reg),
                pairing=pairing,
                d_real=d_snap[0],
                d_fake=d_snap[1],
                d_adv=d_snap[2],
                penalty=d_snap[3],
                critic_snap=d_snap[4],
                real=d_snap[5],
                fake=d_snap[6],
                g_real=g_real.detach().clone(),
                g_fake=g_fake.detach().clone(),
                g_total=g_total.detach().clone(),
                parts=prior.z.detach().clone(),
                offsets=generator.offsets.detach().clone(),
                n_particles=int(prior.z.shape[0]),
                spread_real=d_snap[7],
                spread_fake=d_snap[8],
                spread_adv=d_snap[9],
            )
        opt_g.zero_grad(set_to_none=True)
        g_total.backward()
        opt_g.step()
        if log:
            print(json.dumps({
                "event": "step",
                "arm": arm,
                "step": step,
                "d_adv": float(d_adv.detach()),
                "cap": float(penalty.detach()),
                "g": float(g_total.detach()),
                "fm_weight": cfg.fm_weight,
                "cover_weight": cfg.cover_weight,
                "gan_mode": cfg.gan_mode,
                "n_particles": cfg.n_particles,
                "regularizer": regularizer,
                "pairing": pairing,
            }), flush=True)
    assert obs is not None
    return obs


def _public_row(arm: str, score: dict) -> dict:
    return {
        "arm": arm,
        "verdict": score["verdict"],
        "adv_abs_err": score["adv_abs_err"],
        "g_abs_err": score["g_abs_err"],
        "cap_abs_err": score["cap_abs_err"],
        "kappa_probe_abs_err": score["kappa_probe_abs_err"],
        "fm_weight": score["fm_weight"],
        "cover_weight": score["cover_weight"],
        "gan_mode": score["gan_mode"],
        "loss_type": score["loss_type"],
        "n_particles": score["n_particles"],
        "reg_class": score["reg_class"],
        "mismatches": score["mismatches"],
    }


def leaderboard(*, log_steps: bool = False) -> list[dict]:
    """Locked row first, then intentional drifts. One seed (the floor's)."""
    specs = (
        ("locked_shared", LOCKED, "faithful", "pair"),
        ("thinned_kappa", LOCKED, "thinned", "pair"),
        ("fm_on", replace(LOCKED, fm_weight=0.1), "faithful", "pair"),
        ("vanilla_logistic", replace(LOCKED, gan_mode="vanilla"), "faithful", "pair"),
        ("stranger_pair", LOCKED, "faithful", "stranger"),
        ("music_cover_1", replace(LOCKED, cover_weight=1.0, cover_posture="music"), "faithful", "pair"),
        ("hub128", replace(LOCKED, n_particles=128), "faithful", "pair"),
    )
    rows = []
    for name, cfg, kind, pairing in specs:
        obs = run_floor(cfg, regularizer=kind, pairing=pairing, log=log_steps, arm=name)
        row = _public_row(name, score_floor(obs))
        rows.append(row)
        if log_steps:
            print(json.dumps({"event": "score", **row}), flush=True)
    return rows


def main() -> None:
    rows = leaderboard(log_steps=True)
    print(json.dumps({"event": "leaderboard", "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
