"""CPU toy gate for particle AE-GAN reconstruction plus a locked adversarial shape.

Winning arm: ``ae_gan`` reconstruction together with the locked_shared
adversarial recipe (RpGAN logistic, ParticleGAN ``GradRegularizer`` b_cap
coeff=1, kappa=1, norm=l2, lazy=1, anneal=none, FM off).

Cover posture is the **demo** pin ``cover_weight=1.5`` (the Field3D /
locked-baseline demo value). It is not the Music transfer pin of 1.0.
Particle cloud is the locked tiny table (n=12, ``particle_l2=0.02``), not the
Hub 128-particle mixture and not the AE-GAN study default of 400.

The host package has no critic module. This toy owns a small MLP critic; it
does not swap or unfreeze a host critic.

A PASS is a CPU toy result only. It is not a Music or Anima GPU transfer claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from particlegan import get_recipe
from particlegan.grad_regularizers import GradRegularizer


# Demo cover, not Music 1.0. See module docstring.
DEMO_COVER = 1.5
PARTICLE_L2 = 0.02
N_PARTICLES = 12
# Toy budget, not a formulation knob. One seed; do not sweep seeds.
STEPS = 250
BATCH = 64
LR = 2e-3
SEED = 0
DATA_STD = 0.05
ANCHORS = ((-1.5, 0.0), (1.5, 0.0))
# Filled after the seed-0 run. Untrained recon on this toy is ~2.
RECON_MAX = 0.05
# Mean distance from each anchor to the nearest unconditional sample.
HOLD_MAX = 0.35

# Formulation keys. Budget (steps, batch, lr, seed) is not in this map.
LOCKED_SHAPE = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "reg_lazy": 1,
    "target_anneal": "none",
    "fm_weight": 0.0,
    "cover_weight": DEMO_COVER,
    "particle_l2": PARTICLE_L2,
    "n_particles": N_PARTICLES,
    "reconstruction_weight": 1.0,
    "adversarial_weight": 1.0,
    "encoder_mode": "ae",
}


@dataclass(frozen=True)
class HoldConfig:
    name: str
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    reg_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cover_weight: float = DEMO_COVER
    particle_l2: float = PARTICLE_L2
    n_particles: int = N_PARTICLES
    reconstruction_weight: float = 1.0
    adversarial_weight: float = 1.0
    encoder_mode: str = "ae"
    steps: int = STEPS
    batch: int = BATCH
    lr: float = LR
    seed: int = SEED


def locked_config(**overrides) -> HoldConfig:
    return HoldConfig(name="locked", **overrides)


def shape_mismatches(cfg: HoldConfig) -> list[str]:
    """Formulation drift from locked_shared. Empty means the shape is intact."""
    bad = []
    for key, want in LOCKED_SHAPE.items():
        got = getattr(cfg, key)
        if isinstance(want, float):
            if float(got) != float(want):
                bad.append(f"{key}: {got!r} != locked {want!r}")
        elif got != want:
            bad.append(f"{key}: {got!r} != locked {want!r}")
    return bad


class ThinnedBCap(GradRegularizer):
    """b_cap stub that advertises kappa=1 but applies a hardcoded thin cap."""

    def __init__(self) -> None:
        super().__init__(arm="b_cap", coeff=1.0, kappa=1.0, norm="l2", lazy_k=1, target_anneal="none")

    def center(self, step: int = 0) -> float:
        return 0.1


def regularizer_mismatches(reg) -> list[str]:
    """Refuse anything that is not the faithful b_cap champion, including a thin stub."""
    bad = []
    if type(reg) is not GradRegularizer:
        bad.append(f"regularizer type {type(reg).__name__} is not GradRegularizer")
    want = {
        "arm": "b_cap",
        "coeff": 1.0,
        "kappa": 1.0,
        "norm": "l2",
        "lazy_k": 1,
        "target_anneal": "none",
    }
    for key, value in want.items():
        got = getattr(reg, key, "<missing>")
        if isinstance(value, float):
            if not isinstance(got, (int, float)) or float(got) != value:
                bad.append(f"reg.{key}: {got!r} != locked {value!r}")
        elif got != value:
            bad.append(f"reg.{key}: {got!r} != locked {value!r}")
    if bad:
        return bad
    champion = GradRegularizer(arm="b_cap", coeff=1.0, kappa=1.0, norm="l2", lazy_k=1, target_anneal="none")
    torch.manual_seed(0)
    critic = nn.Linear(2, 1)
    real = torch.randn(8, 2)
    fake = torch.randn(8, 2)
    pen, _ = reg.penalty(critic, real, fake, step=1)
    ref, _ = champion.penalty(critic, real, fake, step=1)
    if not torch.allclose(pen, ref):
        bad.append("regularizer penalty != faithful b_cap (kappa hardcoded or thinned)")
    return bad


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[1](self.net[0](x))


def _anchors() -> torch.Tensor:
    return torch.tensor(ANCHORS, dtype=torch.float32)


def sample_data(n: int) -> torch.Tensor:
    choice = torch.randint(0, len(ANCHORS), (n,))
    return _anchors()[choice] + DATA_STD * torch.randn(n, 2)


def _hold_distance(fake: torch.Tensor) -> float:
    distances = torch.cdist(_anchors(), fake)
    return float(distances.min(dim=1).values.mean())


@torch.no_grad()
def evaluate(encoder, decoder, prior, recipe) -> dict[str, float]:
    """Score reconstruction and unconditional hold without moving the train RNG."""
    state = torch.get_rng_state()
    try:
        data = sample_data(1024)
        query, offset = encoder(data).chunk(2, dim=1)
        encoded = recipe.encode(query, prior, offset=offset)
        recon = decoder(encoded.codes[:, 0])
        recon_mse = float((recon - data).square().mean())
        codes, _ = prior.sample(1024)
        fake = decoder(codes)
        return {"recon_mse": recon_mse, "hold": _hold_distance(fake)}
    finally:
        torch.set_rng_state(state)


def _log(arm: str, step: int, metrics: dict, extra: str = "") -> None:
    print(
        f"ae-gan-hold arm={arm} step={step} recon={metrics['recon_mse']:.4f} "
        f"hold={metrics['hold']:.4f}{extra}",
        flush=True,
    )


def make_recipe(cfg: HoldConfig):
    """``ae_gan`` recipe with this arm's knobs. Formulation drift is caught separately."""
    return get_recipe(
        "ae_gan",
        num_particles=cfg.n_particles,
        reg_every=cfg.reg_lazy,
        reg_arm=cfg.reg_arm,
        reg_coeff=cfg.reg_coeff,
        reg_kappa=cfg.reg_kappa,
        loss_type=cfg.loss_type,
        gan_mode=cfg.gan_mode,
        reconstruction_weight=cfg.reconstruction_weight,
        lr=cfg.lr,
        encoder_mode=cfg.encoder_mode,
    )


def train(cfg: HoldConfig) -> dict:
    """Train one arm. Drifted adversarial shapes are refused before any step."""
    if cfg.name != "ae_only":
        bad = shape_mismatches(cfg)
        if bad:
            raise ValueError("refusing drifted adversarial shape: " + "; ".join(bad))
    torch.manual_seed(cfg.seed)
    recipe = make_recipe(cfg)
    prior = recipe.make_prior()
    encoder, decoder, critic = MLP(2, 4), MLP(2, 2), MLP(2, 1)
    opt_g, opt_d = recipe.make_optimizers(decoder, critic, prior, encoder=encoder)
    gan = recipe.make_loss()
    regularizer = recipe.make_gradient_penalty(norm=cfg.reg_norm, target_anneal=cfg.target_anneal)
    if cfg.adversarial_weight > 0:
        reg_bad = regularizer_mismatches(regularizer)
        if cfg.name != "ae_only" and reg_bad:
            raise ValueError("refusing regularizer: " + "; ".join(reg_bad))

    opened = evaluate(encoder, decoder, prior, recipe)
    _log(cfg.name, 0, opened, extra=" phase=init")
    bcap_applied = 0
    adv_steps = 0
    for step in range(1, cfg.steps + 1):
        data = sample_data(cfg.batch)
        if cfg.adversarial_weight > 0:
            codes, _ = prior.sample(cfg.batch)
            fake = decoder(codes).detach()
            opt_d.zero_grad(set_to_none=True)
            d_loss = gan.d_loss(critic(data).squeeze(-1), critic(fake).squeeze(-1))
            penalty, stats = regularizer.penalty(critic, data, fake, step=step)
            if stats.get("applied"):
                bcap_applied += 1
            (d_loss + penalty).backward()
            opt_d.step()

        query, offset = encoder(data).chunk(2, dim=1)
        encoded = recipe.encode(query, prior, offset=offset)
        reconstructed = decoder(encoded.codes[:, 0])
        recon = encoded.reconstruction_loss(reconstructed[:, None], data)
        codes, _ = prior.sample(cfg.batch)
        generated = decoder(codes)
        opt_g.zero_grad(set_to_none=True)
        for param in critic.parameters():
            param.requires_grad_(False)
        loss = cfg.reconstruction_weight * recon + cfg.particle_l2 * prior.z.square().mean()
        if cfg.adversarial_weight > 0:
            real_logits = critic(data).squeeze(-1).detach()
            fake_logits = critic(generated).squeeze(-1)
            adv = gan.g_loss(fake_logits, real_logits)
            anchors = _anchors()
            cover = torch.cdist(anchors, generated).min(dim=1).values.mean()
            loss = loss + cfg.adversarial_weight * adv + cfg.cover_weight * cover
            if cfg.fm_weight > 0:
                real_feat = critic.features(data).detach().mean(0)
                fake_feat = critic.features(generated).mean(0)
                loss = loss + cfg.fm_weight * (real_feat - fake_feat).square().mean()
            adv_steps += 1
        loss.backward()
        for param in critic.parameters():
            param.requires_grad_(True)
        opt_g.step()
        if step == 1 or (step % 50 == 0 and step != cfg.steps):
            snap = evaluate(encoder, decoder, prior, recipe)
            _log(cfg.name, step, snap, extra=f" loss={float(loss.detach()):.4f}")

    final = evaluate(encoder, decoder, prior, recipe)
    row = {
        "name": cfg.name,
        "cfg": cfg,
        "recon_mse": final["recon_mse"],
        "hold": final["hold"],
        "init_recon_mse": opened["recon_mse"],
        "bcap_applied": bcap_applied,
        "adv_steps": adv_steps,
        "steps": cfg.steps,
    }
    row["reasons"] = gate_reasons(row)
    row["verdict"] = "PASS" if not row["reasons"] else "FAIL"
    _log(cfg.name, cfg.steps, final, extra=f" verdict={row['verdict']}")
    return row


def gate_reasons(row: dict) -> list[str]:
    """Why a row fails. Empty means reconstruction and the locked adv shape both held.

    ``hold`` is the mean distance from each data anchor to the nearest unconditional
    sample. On this two-blob toy, AE-only also lands there, because reconstruction
    already places codes on the data. That number is still gated so a winner whose
    prior samples leave the blobs cannot PASS. AE-only fails on the adversarial
    counters (no RpGAN step, no b_cap), not on reconstruction.
    """
    reasons = shape_mismatches(row["cfg"])
    if row["recon_mse"] > RECON_MAX:
        reasons.append(f"recon_mse {row['recon_mse']:.4f} > {RECON_MAX}")
    if row["hold"] > HOLD_MAX:
        reasons.append(f"hold {row['hold']:.4f} > {HOLD_MAX}")
    if row["bcap_applied"] != row["steps"]:
        reasons.append(f"b_cap applied {row['bcap_applied']} of {row['steps']} steps")
    if row["adv_steps"] != row["steps"]:
        reasons.append(f"RpGAN generator steps {row['adv_steps']} of {row['steps']}")
    return reasons


def ae_only_config() -> HoldConfig:
    return HoldConfig(name="ae_only", adversarial_weight=0.0, cover_weight=0.0)


def stranger_config() -> HoldConfig:
    return HoldConfig(name="stranger_pairing", gan_mode="vanilla")


def fm_on_config() -> HoldConfig:
    return HoldConfig(name="fm_on", fm_weight=0.1)


def main() -> None:
    locked = train(locked_config())
    only = train(ae_only_config())
    print(
        "ae-gan-hold board "
        f"locked recon={locked['recon_mse']:.4f} hold={locked['hold']:.4f} {locked['verdict']} | "
        f"ae_only recon={only['recon_mse']:.4f} hold={only['hold']:.4f} {only['verdict']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
