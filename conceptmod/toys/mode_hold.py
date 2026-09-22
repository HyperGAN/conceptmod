"""CPU ring: locked_shared RpGAN + b_cap holds modes; cap-off collapses.

Reviewed against particle-sliders (HyperGAN) before this gate existed:

- ``analysis/slider2d/locked_baseline_defaults.py`` — #94 locked_shared card
- ``tests/test_music_arm_b_gates.py`` — FM-on, kappa drift, and cover drift fail closed
- ``tests/test_gan_stability_experiment.py``, ``tests/test_gan_convergence.py``,
  ``analysis/gan_bcap/quality_audit_v2.py`` — stability / convergence / diversity
  gates are vetoes, not extra reward terms
- ``analysis/gan_bcap/gaussian_repro.py`` — 8-mode smoke. That fixture still
  covers with ``b_cap=0`` because particles start on the ring, so coverage
  there is not a diversity gate. This ring starts the cloud at the origin and
  uses a sharp host Fourier critic, which is the 100-Gaussians failure the
  cap was introduced to stop.

Locked shape used here (formulation, not a new adv recipe):

- RpGAN logistic pair (``GANLoss(loss_type='logistic', mode='rp')``)
- ParticleGAN ``GradientPenalty`` ``b_cap``, coeff=1, kappa=1, norm=l2
- ``fm_weight=0``
- demo cover pin ``cover_weight=1.5`` recorded on the card. Music pole/cover
  1.0 is a different posture and is not used. No supervised cover loss is
  added; the gate scores empirical mode cover.
- tiny cloud ``n_particles=12`` and ``particle_l2=0.02``. Not the Hub
  20k-particle table and not a 128-particle gmix.
- critic stays ``SimpleMLPDiscriminator`` (the 100-Gaussians host shape).
  ParticleGAN's installable package does not export that MLP, so the class
  lives in ``conceptmod.toys.mlp``. No architecture swap.

``b_cap`` off (``f_none``) is the intentional drift. It is refused unless the
caller passes an explicit drift report, and the diversity gate then FAILs.

This is a CPU toy. A PASS here is not a Music or Anima transfer result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from conceptmod.toys.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANLoss, GradientPenalty, ParticlePrior, ParticleRegularizer


N_MODES = 8
RADIUS = 3.0
SIGMA = 0.07
Z_DIM = 4
HIDDEN = 96
N_HIDDEN = 3
FOURIER = 3
BATCH = 128
LR = 2.0e-3
EVAL_N = 4096
LOG_EVERY = 400

# Diversity veto. Locked clears it; cap-off stays under the collapse ceiling.
PASS_MODES = 7
PASS_HQ = 0.90
COLLAPSE_MODES = 2


@dataclass(frozen=True)
class ModeHoldRecipe:
    """locked_shared card for this toy. ``steps`` is budget, not formulation."""

    gan_mode: str = "rp"
    loss_type: str = "logistic"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    fm_weight: float = 0.0
    # Demo / Field3D pin. Recorded so a silent swap to Music cover 1.0 is drift.
    # Not applied as a training loss: see module docstring.
    cover_weight: float = 1.5
    n_particles: int = 12
    particle_l2: float = 0.02
    vicreg_weight: float = 0.05
    beta1: float = 0.0
    beta2: float = 0.99
    ema: float = 0.995
    d_lr_mult: float = 1.0
    steps: int = 1200

    def replace(self, **overrides):
        return replace(self, **overrides)


def locked_recipe() -> ModeHoldRecipe:
    return ModeHoldRecipe()


def formulation_mismatches(recipe: ModeHoldRecipe) -> list[str]:
    """Formulation drift from locked_shared. ``steps`` is a budget key."""
    locked = locked_recipe()
    bad = []
    for field in fields(ModeHoldRecipe):
        if field.name == "steps":
            continue
        got, want = getattr(recipe, field.name), getattr(locked, field.name)
        if got != want:
            bad.append(f"{field.name}: {got!r} != locked {want!r}")
    return bad


def assert_locked_shared(recipe: ModeHoldRecipe) -> None:
    bad = formulation_mismatches(recipe)
    if bad:
        raise AssertionError(
            "mode-hold recipe drifted from locked_shared: " + "; ".join(bad)
        )


def require_drift_report(recipe: ModeHoldRecipe, drift: dict | None) -> None:
    """Refuse a drifted adv recipe that does not name its drift."""
    bad = formulation_mismatches(recipe)
    if not bad:
        return
    if not drift:
        raise ValueError(
            "refusing unreported drift from locked_shared: " + "; ".join(bad)
        )
    missing = [item for item in bad if item.split(":", 1)[0] not in drift]
    if missing:
        raise ValueError(
            "drift report does not name every moved knob: " + "; ".join(missing)
        )


def ring_means(n_modes: int = N_MODES, radius: float = RADIUS) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, int(n_modes) + 1)[:-1]
    return torch.stack((angles.cos(), angles.sin()), dim=1) * float(radius)


def sample_ring(means: torch.Tensor, n: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, means.shape[0], (int(n),), generator=generator)
    noise = torch.randn(int(n), means.shape[1], generator=generator)
    return means[idx] + float(sigma) * noise


def diversity(samples: torch.Tensor, means: torch.Tensor, sigma: float = SIGMA) -> dict:
    """Mode hold on the ring. HQ = within 3 sigma of a center, balls disjoint."""
    dist = torch.cdist(samples, means)
    nearest, which = dist.min(dim=1)
    hq = nearest <= 3.0 * float(sigma)
    counts = torch.bincount(which[hq], minlength=means.shape[0]).to(dtype=torch.float32)
    modes = int((counts > 0).sum())
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    positive = probs[probs > 0]
    entropy = -(positive * positive.log()).sum()
    effective = float(entropy.exp()) if modes else 0.0
    n_modes = int(means.shape[0])
    return {
        "modes": modes,
        "n_modes": n_modes,
        "hq": float(hq.float().mean()),
        "cover": modes / n_modes,
        "effective_modes": effective,
    }


def verdict(row: dict) -> str:
    """PASS holds the ring. FAIL is mode collapse. Anything between is inconclusive."""
    if row["modes"] >= PASS_MODES and row["hq"] >= PASS_HQ:
        return "PASS"
    if row["modes"] <= COLLAPSE_MODES:
        return "FAIL"
    return "INCONCLUSIVE"


def _log(arm: str, step: int, row: dict, tag: str) -> None:
    print(
        f"mode-hold arm={arm} step={step} modes={row['modes']}/{row['n_modes']} "
        f"hq={row['hq']:.3f} cover={row['cover']:.3f} "
        f"effective={row['effective_modes']:.2f} {tag}",
        flush=True,
    )


_CACHE: dict[tuple, dict] = {}


def train_mode_hold(
    recipe: ModeHoldRecipe | None = None,
    *,
    drift: dict | None = None,
    seed: int = 0,
    log: bool = True,
) -> dict:
    """Train one arm. Drifted recipes must pass ``drift={knob: reason}``."""
    recipe = locked_recipe() if recipe is None else recipe
    require_drift_report(recipe, drift)
    key = (recipe, seed)
    if key in _CACHE:
        return _CACHE[key]

    torch.manual_seed(seed)
    stream = torch.Generator().manual_seed(seed)
    means = ring_means()
    prior = ParticlePrior(recipe.n_particles, Z_DIM, init_std=0.5, generator=stream)
    generator = SimpleMLPGenerator(Z_DIM, HIDDEN, N_HIDDEN, 2)
    # Host critic shape from the 100-Gaussians toy. Fourier width is the
    # sharp-D stress, not an architecture swap.
    critic = SimpleMLPDiscriminator(2, HIDDEN, N_HIDDEN, FOURIER)
    gan = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    regularizer = GradientPenalty(
        arm=recipe.reg_arm,
        coeff=recipe.reg_coeff,
        kappa=recipe.reg_kappa,
        norm=recipe.reg_norm,
    )
    vicreg = ParticleRegularizer(weight=recipe.vicreg_weight)
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(prior.parameters()),
        lr=LR,
        betas=(recipe.beta1, recipe.beta2),
    )
    opt_d = torch.optim.Adam(
        critic.parameters(),
        lr=LR * recipe.d_lr_mult,
        betas=(recipe.beta1, recipe.beta2),
    )
    ema_g = [p.detach().clone() for p in generator.parameters()]
    ema_z = prior.z.detach().clone()
    if not formulation_mismatches(recipe):
        arm = "locked"
    elif recipe.gan_mode != "rp":
        arm = "stranger"
    elif recipe.fm_weight > 0.0:
        arm = "fm_on"
    elif recipe.reg_arm == "f_none":
        arm = "b_cap_off"
    else:
        arm = "drift"

    def snapshot(step: int) -> dict:
        saved_g = [p.detach().clone() for p in generator.parameters()]
        saved_z = prior.z.detach().clone()
        with torch.no_grad():
            for param, ema in zip(generator.parameters(), ema_g):
                param.copy_(ema)
            prior.z.copy_(ema_z)
            eval_stream = torch.Generator().manual_seed(seed + 9)
            latent, _ = prior.sample(EVAL_N, generator=eval_stream)
            row = diversity(generator(latent), means)
        with torch.no_grad():
            for param, saved in zip(generator.parameters(), saved_g):
                param.copy_(saved)
            prior.z.copy_(saved_z)
        row.update(step=step, arm=arm, seed=seed, fm_weight=recipe.fm_weight, reg_arm=recipe.reg_arm)
        return row

    for step in range(recipe.steps):
        real = sample_ring(means, BATCH, SIGMA, stream)
        latent, _ = prior.sample(BATCH, generator=stream)
        fake = generator(latent).detach()
        d_loss = gan.d_loss(critic(real), critic(fake))
        d_loss = d_loss + regularizer(critic, real, fake, step=step + 1)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        latent, _ = prior.sample(BATCH, generator=stream)
        fake = generator(latent)
        if gan.mode == "rp":
            real_g = sample_ring(means, BATCH, SIGMA, stream)
            g_loss = gan.g_loss(critic(fake), critic(real_g))
        else:
            # Stranger / unpaired pairing: real and fake are scored apart.
            g_loss = gan.g_loss(critic(fake))
        if recipe.fm_weight > 0.0:
            # Mean-feature match on coordinates. Uncapped by b_cap (FM-on drift).
            real_mean = sample_ring(means, BATCH, SIGMA, stream).detach().mean(0)
            g_loss = g_loss + recipe.fm_weight * (fake.mean(0) - real_mean).pow(2).sum()
        g_loss = g_loss + recipe.particle_l2 * prior.z.pow(2).mean()
        g_loss = g_loss + vicreg(prior.z)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        with torch.no_grad():
            for ema, param in zip(ema_g, generator.parameters()):
                ema.mul_(recipe.ema).add_(param, alpha=1.0 - recipe.ema)
            ema_z.mul_(recipe.ema).add_(prior.z, alpha=1.0 - recipe.ema)
        done = step + 1
        if log and (done % LOG_EVERY == 0 or done == recipe.steps):
            row = snapshot(done)
            _log(arm, done, row, "train")

    final = snapshot(recipe.steps)
    final["verdict"] = verdict(final)
    final["drift"] = dict(drift) if drift else {}
    if log:
        _log(arm, recipe.steps, final, f"verdict={final['verdict']}")
    _CACHE[key] = final
    return final


def b_cap_off_recipe() -> ModeHoldRecipe:
    """Same card with the cap removed. Caller must report the drift."""
    return locked_recipe().replace(reg_arm="f_none", reg_coeff=0.0)


def main() -> None:
    locked = train_mode_hold()
    collapsed = train_mode_hold(
        b_cap_off_recipe(),
        drift={"reg_arm": "b_cap off (f_none)", "reg_coeff": "0, cap removed"},
    )
    print(
        f"mode-hold leaderboard locked={locked['verdict']} "
        f"modes={locked['modes']}/{locked['n_modes']} hq={locked['hq']:.3f} | "
        f"b_cap_off={collapsed['verdict']} modes={collapsed['modes']}/{collapsed['n_modes']} "
        f"hq={collapsed['hq']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
