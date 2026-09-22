"""Keep-critic locked_shared: the host critic stays frozen (CPU).

Music Arm B (particle-sliders ``train_lm_slider_music3.ARM_B``,
``docs/music-arm-b-gates.md``) builds a fresh ``--adv_arch mlp`` head.
This port does not. The host critic is the frozen linear scorer below.
``conceptmod.toys.mlp.SimpleMLPDiscriminator`` is the ring critic, and
swapping it in is refused before any step.

Locked shape, same stamp as ``locked_baseline_defaults.LOCKED`` / the
floor (not a new adv recipe):

- RpGAN logistic (``GANLoss`` loss_type=logistic, mode=rp)
- ParticleGAN ``GradRegularizer`` ``b_cap``, coeff=1, kappa=1, norm=l2
- FM off (``fm_weight=0``)
- cover posture is documented on every row: **demo 1.5** is this lock;
  Music pole/cover **1.0** is a different posture and fails the stamp
- tiny cloud n=12, ``particle_l2=0.02`` (not Hub 128, not Music ``--parts 0``)

The critic is not stepped. ``b_cap`` is still the real module, measured
on the frozen host. A PASS is a CPU toy. It is not a Music or Anima GPU
transfer.

Tail the board with ``python -m conceptmod.toys.keep_critic``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer
from particlegan.particle_prior import ParticlePrior


# Demo locked_shared cloud. Music Arm B ``--parts 0`` is a different object.
N_PARTICLES = 12
PARTICLE_L2 = 0.02
DEMO_COVER = 1.5
MUSIC_COVER = 1.0
COVER_POSTURE = "demo"
FM_WEIGHT = 0.0
HOST_ARCH = "host"

DEMO_TARGETS = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

# Slope under kappa=1 so the locked cap is quiet. The kappa probe uses
# its own critic; this weight is only the frozen host.
_HOST_WEIGHT = torch.tensor([[0.5, 0.0]])


class KeepCriticError(ValueError):
    """A keep-critic arm tried to swap or invent a critic."""


@dataclass(frozen=True)
class KeepCritic:
    """Formulation stamp. ``steps`` / ``seed`` / ``lr`` / ``batch`` are budget."""

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
    cover_weight: float = DEMO_COVER
    cover_posture: str = COVER_POSTURE
    critic_arch: str = HOST_ARCH
    critic_frozen: bool = True
    steps: int = 8
    seed: int = 0
    lr: float = 5.0e-3
    batch: int = 16


LOCKED = KeepCritic()


class HostCritic(nn.Module):
    """Frozen host scorer. Not the ring MLP and not a trained Music head."""

    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.score.weight.copy_(_HOST_WEIGHT)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x).squeeze(-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # A linear host has no hidden MLP. FM, when someone turns it on,
        # sees the input the host actually scores.
        return x


class ToyGenerator(nn.Module):
    """One offset per demo mode. The host critic does not own this."""

    def __init__(self) -> None:
        super().__init__()
        self.offsets = nn.Parameter(torch.zeros(2, 2))

    def forward(self, z: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        return self.offsets[mode] + z


class ThinnedKappaCap(GradRegularizer):
    """κ is stored and ignored. The cap center is hardcoded at 1.

    At locked κ=1 the step penalty still matches. A κ=0.2 probe does not.
    """

    def center(self, step: int) -> float:
        if self.arm in ("b_cap", "g_interp_cap"):
            return 1.0
        return super().center(step)


def cover_mse(offsets: torch.Tensor) -> torch.Tensor:
    targets = DEMO_TARGETS.to(device=offsets.device, dtype=offsets.dtype)
    return (offsets - targets).pow(2).mean()


def feature_match(feat_real: torch.Tensor, feat_fake: torch.Tensor) -> torch.Tensor:
    real = F.normalize(feat_real.flatten(1).mean(dim=0), dim=0)
    fake = F.normalize(feat_fake.flatten(1).mean(dim=0), dim=0)
    return (real - fake).pow(2).mean()


def make_regularizer(cfg: KeepCritic, *, kind: str = "faithful") -> GradRegularizer:
    cls = GradRegularizer if kind == "faithful" else ThinnedKappaCap
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
        return real_logits, fake_logits.flip(0)
    raise ValueError(f"unknown pairing {pairing!r}")


def _refuse_mlp_swap() -> None:
    """Name the ring critic, then stop. Do not construct it and do not train."""
    from conceptmod.toys import mlp as ring

    refused = ring.SimpleMLPDiscriminator.__name__
    raise KeepCriticError(
        "forced mlp critic swap refused: host critic stays frozen "
        f"(conceptmod.toys.mlp.{refused} is the ring critic, not this lock). "
        "Music Arm B --adv_arch mlp is a different head."
    )


def _require_host_arch(arch: str) -> None:
    if arch == HOST_ARCH:
        return
    if arch == "mlp":
        _refuse_mlp_swap()
    raise KeepCriticError(
        f"unknown critic arch {arch!r}; keep-critic only runs the frozen host"
    )


@dataclass
class KeepObs:
    cfg: KeepCritic
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
    spread_real: torch.Tensor
    spread_fake: torch.Tensor
    spread_adv: torch.Tensor
    weight_delta: float
    params_require_grad: bool


def _stamp_mismatches(cfg: KeepCritic) -> list[str]:
    bad = []
    for field in (
        "loss_type", "gan_mode", "reg_arm", "reg_coeff", "reg_kappa", "reg_norm",
        "reg_lazy", "target_anneal", "fm_weight", "particle_l2", "n_particles",
        "z_dim", "cover_weight", "cover_posture", "critic_arch", "critic_frozen",
    ):
        got, want = getattr(cfg, field), getattr(LOCKED, field)
        if got != want:
            bad.append(f"{field}: {got!r} != locked_shared {want!r}")
    return bad


def _kappa_probe(reg_cls: type) -> tuple[float, str | None]:
    """Slope 0.4 is under κ=1 and over κ=0.2. A hardcoded center scores off."""
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
    faithful = GradRegularizer(
        arm="b_cap", coeff=1.0, kappa=0.2, norm="l2",
    ).penalty(critic, batch, batch)[0]
    err = abs(float(got.detach()) - float(faithful.detach()))
    if err > 1e-5:
        return err, (
            f"thinned or non-faithful b_cap: kappa probe abs err {err:.3e} "
            "(κ=0.2 on ||g||=0.4 must match GradRegularizer)"
        )
    return err, None


def _flat_weights(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().float() for p in module.parameters()])


def score_keep(obs: KeepObs) -> dict:
    """PASS only for locked_shared on the frozen host critic."""
    bad = _stamp_mismatches(obs.cfg)
    if obs.pairing != "pair":
        bad.append(f"pairing: {obs.pairing!r} != locked_shared 'pair'")
    if obs.reg_cls is not GradRegularizer:
        bad.append(
            f"regularizer: {obs.reg_cls.__name__} is not particlegan.GradRegularizer"
        )
    if obs.n_particles != obs.cfg.n_particles:
        bad.append(
            f"n_particles: cloud {obs.n_particles} != claimed {obs.cfg.n_particles}"
        )
    if not isinstance(obs.critic_snap, HostCritic):
        bad.append(
            f"critic_arch: {type(obs.critic_snap).__name__} is not the frozen host"
        )
    actually_frozen = obs.weight_delta == 0.0 and not obs.params_require_grad
    if not actually_frozen:
        bad.append(
            f"critic_moved: weight_delta={obs.weight_delta:.3e} "
            f"requires_grad={obs.params_require_grad}"
        )

    rp = GANLoss("logistic", "rp")
    d_want = rp.d_loss(obs.d_real, obs.d_fake)
    spread_want = rp.d_loss(obs.spread_real, obs.spread_fake)
    g_want = (
        rp.g_loss(obs.g_fake, obs.g_real)
        + PARTICLE_L2 * obs.parts.pow(2).mean()
        + DEMO_COVER * cover_mse(obs.offsets)
    )
    cap_want = GradRegularizer(
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
            f"+ demo cover {DEMO_COVER} + FM off"
        )
    if cap_err > 1e-5:
        bad.append(f"b_cap: abs err {cap_err:.3e} vs GradRegularizer coeff=1 κ=1 l2")
    probe_err, probe_bad = _kappa_probe(obs.reg_cls)
    if probe_bad:
        bad.append(probe_bad)

    return {
        "verdict": "PASS" if not bad else "FAIL",
        "refused": False,
        "mismatches": bad,
        "adv_abs_err": adv_err,
        "g_abs_err": g_err,
        "cap_abs_err": cap_err,
        "kappa_probe_abs_err": probe_err,
        "fm_weight": obs.cfg.fm_weight,
        "cover_weight": obs.cfg.cover_weight,
        "cover_posture": obs.cfg.cover_posture,
        "demo_cover": DEMO_COVER,
        "music_cover": MUSIC_COVER,
        "gan_mode": obs.cfg.gan_mode,
        "loss_type": obs.cfg.loss_type,
        "n_particles": obs.n_particles,
        "reg_class": "GradRegularizer" if obs.reg_cls is GradRegularizer else obs.reg_cls.__name__,
        "critic_arch": HOST_ARCH if isinstance(obs.critic_snap, HostCritic) else type(obs.critic_snap).__name__,
        "critic_frozen": actually_frozen,
        "critic_weight_delta": obs.weight_delta,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
    }


def run_keep(
    cfg: KeepCritic | None = None,
    *,
    regularizer: str = "faithful",
    pairing: str = "pair",
    log: bool = False,
    arm: str = "locked_shared",
) -> KeepObs:
    """Tiny CPU loop. The host critic is not an optimizer unless unfrozen."""
    cfg = LOCKED if cfg is None else cfg
    _require_host_arch(cfg.critic_arch)
    torch.manual_seed(cfg.seed)
    critic = HostCritic()
    if cfg.critic_frozen:
        critic.requires_grad_(False)
        opt_d = None
    else:
        # Named drift: step the same host module. Still not an MLP swap.
        critic.requires_grad_(True)
        opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(0.0, 0.99))
    weight_start = _flat_weights(critic).clone()
    generator = ToyGenerator()
    prior = ParticlePrior(num_particles=cfg.n_particles, z_dim=cfg.z_dim, init_std=0.05)
    gan = GANLoss(loss_type=cfg.loss_type, mode=cfg.gan_mode)
    reg = make_regularizer(cfg, kind=regularizer)
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
        if opt_d is not None:
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
            end = _flat_weights(critic)
            obs = KeepObs(
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
                weight_delta=float((end - weight_start).abs().max()),
                params_require_grad=any(p.requires_grad for p in critic.parameters()),
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
                "cover_posture": cfg.cover_posture,
                "critic_arch": cfg.critic_arch,
                "critic_frozen": cfg.critic_frozen,
                "pairing": pairing,
            }), flush=True)
    assert obs is not None
    return obs


def _public_row(arm: str, score: dict) -> dict:
    row = {"arm": arm}
    row.update(score)
    return row


def _refused_row(arm: str, exc: KeepCriticError, *, critic_arch: str) -> dict:
    return {
        "arm": arm,
        "verdict": "FAIL",
        "refused": True,
        "mismatches": [str(exc)],
        "adv_abs_err": None,
        "g_abs_err": None,
        "cap_abs_err": None,
        "kappa_probe_abs_err": None,
        "fm_weight": FM_WEIGHT,
        "cover_weight": DEMO_COVER,
        "cover_posture": "not_run",
        "demo_cover": DEMO_COVER,
        "music_cover": MUSIC_COVER,
        "gan_mode": "rp",
        "loss_type": "logistic",
        "n_particles": N_PARTICLES,
        "reg_class": "not_run",
        "critic_arch": critic_arch,
        "critic_frozen": None,
        "critic_weight_delta": None,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
    }


def leaderboard(*, log_steps: bool = False) -> list[dict]:
    """Locked host first, then named drifts. MLP swap is refused, not trained."""
    specs = (
        ("locked_shared", LOCKED, "faithful", "pair"),
        ("stranger_pairing", LOCKED, "faithful", "stranger"),
        ("fm_on", replace(LOCKED, fm_weight=0.1), "faithful", "pair"),
        ("thinned_kappa", LOCKED, "thinned", "pair"),
        ("music_cover_1", replace(LOCKED, cover_weight=MUSIC_COVER, cover_posture="music"), "faithful", "pair"),
        ("critic_step", replace(LOCKED, critic_frozen=False), "faithful", "pair"),
    )
    rows = []
    for name, cfg, kind, pairing in specs:
        obs = run_keep(cfg, regularizer=kind, pairing=pairing, log=log_steps, arm=name)
        row = _public_row(name, score_keep(obs))
        rows.append(row)
        if log_steps:
            print(json.dumps({"event": "score", **row}), flush=True)
    try:
        run_keep(replace(LOCKED, critic_arch="mlp"), arm="forced_mlp_swap")
    except KeepCriticError as exc:
        row = _refused_row("forced_mlp_swap", exc, critic_arch="mlp")
    else:
        row = _refused_row(
            "forced_mlp_swap",
            KeepCriticError("mlp swap was not refused"),
            critic_arch="mlp",
        )
        row["verdict"] = "FAIL"
    rows.append(row)
    if log_steps:
        print(json.dumps({"event": "score", **row}), flush=True)
    return rows


def main() -> None:
    rows = leaderboard(log_steps=True)
    print(json.dumps({
        "event": "leaderboard",
        "cpu_toy": True,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "rows": rows,
    }), flush=True)


if __name__ == "__main__":
    main()
