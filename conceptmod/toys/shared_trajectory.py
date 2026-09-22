"""Same-seed slow→fast pairs under the locked_shared recipe.

Each identity is one seed. The slow arc and the fast arc of that seed are the
only legal relativistic pair. A stranger or nearest-stranger fast target is a
different identity: the locked entry point refuses it, and an explicit drift
arm fails the identity gate.

The adversarial recipe is the particle-sliders locked card: RpGAN logistic,
``b_cap`` coeff=1, kappa=1, norm=l2, feature matching off, ``particle_l2`` 0.02
on a 12-particle cloud, VICReg weight 0.05. Cover is the demo weight 1.5
because this toy keeps that cloud. Music Arm B pole/cover 1.0 is a different
host and is not used. A pass here is not a Music or Anima GPU result.

The critic is this toy's own MLP. ``TrajectoryDiscriminator`` is left alone.
"""

from __future__ import annotations

import json
from typing import Callable

import torch
from torch import nn

from particlegan import GANLoss, ParticlePrior, ParticleRegularizer
from particlegan.grad_regularizers import GradRegularizer


# Budget and formulation pins. Only ``pairing`` may change, and only on a
# drift arm. ``steps`` / ``seed`` match the locked budget keys.
LOCKED = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "reg_lazy": 1,
    "target_anneal": "none",
    "fm_weight": 0.0,
    "cover_weight": 1.5,
    "cover_posture": "demo",
    "particle_l2": 0.02,
    "vicreg_weight": 0.05,
    "n_particles": 12,
    "z_dim": 4,
    "frames": 8,
    "slow_speed": 0.45,
    "fast_speed": 2.2,
    "lr": 5.0e-3,
    "beta1": 0.0,
    "beta2": 0.99,
    "steps": 400,
    "seed": 0,
    "critic_hidden": 64,
    "pairing": "shared",
}

PAIRINGS = ("shared", "stranger", "nearest_stranger")
# Shared training lands near 0. Identity error of a perfect nearest-stranger
# copy is ~0.09, so 0.02 separates the locked arm from both drift arms.
PASS_IDENTITY_MSE = 0.02
LOG_EVERY = 100


def trajectories(n: int | None = None, frames: int | None = None):
    """Slow and fast arcs that share a seed-specific phase and radius.

    Returns ``(slow, fast)`` with shape ``[n, frames * 2]``.
    """
    n = LOCKED["n_particles"] if n is None else n
    frames = LOCKED["frames"] if frames is None else frames
    index = torch.arange(n, dtype=torch.float32)
    phase = 2 * torch.pi * index / n
    radius = 0.7 + 0.25 * ((index % 3) - 1)
    time = torch.linspace(0, 1, frames)
    slow_angle = phase[:, None] + LOCKED["slow_speed"] * time[None, :]
    fast_angle = phase[:, None] + LOCKED["fast_speed"] * time[None, :]

    def pack(angle: torch.Tensor) -> torch.Tensor:
        xy = radius[:, None, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
        return xy.reshape(n, frames * 2)

    return pack(slow_angle), pack(fast_angle)


def pairing_index(mode: str, slow: torch.Tensor) -> torch.Tensor:
    """Row index of the fast target paired with each slow identity."""
    if mode not in PAIRINGS:
        raise ValueError(f"unknown pairing {mode!r} (expected one of {PAIRINGS})")
    n = slow.shape[0]
    identity = torch.arange(n)
    if mode == "shared":
        return identity
    if mode == "stranger":
        if n % 2:
            raise ValueError("stranger shift needs an even number of identities")
        return (identity + n // 2) % n
    distance = torch.cdist(slow, slow)
    distance.fill_diagonal_(float("inf"))
    return distance.argmin(dim=1)


def identity_mse(pred: torch.Tensor, fast: torch.Tensor) -> float:
    return float((pred - fast).pow(2).mean())


def passed(mse: float) -> bool:
    return mse <= PASS_IDENTITY_MSE


class _Generator(nn.Module):
    def __init__(self, slow_dim: int, z_dim: int, fast_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slow_dim + z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, fast_dim),
        )

    def forward(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, z), dim=-1))


class _Critic(nn.Module):
    """Scores a slow/fast pair. This toy owns the critic; nothing is swapped."""

    def __init__(self, pair_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, fast), dim=-1)).squeeze(-1)


class _FastView(nn.Module):
    """Gradient penalty sees the fast arc; the slow arc stays conditioning."""

    def __init__(self, critic: _Critic) -> None:
        super().__init__()
        self.critic = critic

    def forward(self, fast: torch.Tensor) -> torch.Tensor:
        return self.critic(self.slow, fast).unsqueeze(-1)


def _cover(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Demo cover: each true fast arc must sit near some generated arc."""
    return torch.cdist(real, fake).min(dim=1).values.square().mean()


def _emit(record: dict, echo: bool, log: Callable[[dict], None] | None) -> None:
    if log is not None:
        log(record)
    if echo:
        print(json.dumps(record), flush=True)


def train(*, pairing: str = "shared", locked: bool = True, echo: bool = False,
          log: Callable[[dict], None] | None = None) -> dict:
    """Train one pairing.

    ``locked=True`` accepts only ``pairing='shared'`` and raises otherwise.
    A drift arm sets ``locked=False`` and must name ``stranger`` or
    ``nearest_stranger``. The adversarial modules stay the locked ones.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"unknown pairing {pairing!r} (expected one of {PAIRINGS})")
    if locked and pairing != "shared":
        raise ValueError(
            "locked_shared refuses pairing "
            f"{pairing!r}: slow and fast arcs must share a seed. "
            "stranger and nearest-stranger are drift arms, not the locked recipe"
        )
    if not locked and pairing == "shared":
        raise ValueError("a drift arm must change pairing; shared is the locked arm")

    torch.set_num_threads(1)
    torch.manual_seed(LOCKED["seed"])
    slow, fast = trajectories()
    index = pairing_index(pairing, slow)
    paired = fast[index]
    hidden = LOCKED["critic_hidden"]
    generator = _Generator(slow.shape[1], LOCKED["z_dim"], fast.shape[1], hidden)
    critic = _Critic(slow.shape[1] + fast.shape[1], hidden)
    view = _FastView(critic)
    prior = ParticlePrior(
        LOCKED["n_particles"], LOCKED["z_dim"], init_std=0.1,
        generator=torch.Generator().manual_seed(LOCKED["seed"]),
    )
    gan = GANLoss(LOCKED["loss_type"], LOCKED["gan_mode"])
    regularizer = GradRegularizer(
        LOCKED["reg_arm"], LOCKED["reg_coeff"], kappa=LOCKED["reg_kappa"],
        norm=LOCKED["reg_norm"], lazy_k=LOCKED["reg_lazy"],
        target_anneal=LOCKED["target_anneal"],
    )
    spread = ParticleRegularizer(weight=LOCKED["vicreg_weight"])
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(prior.parameters()),
        lr=LOCKED["lr"], betas=(LOCKED["beta1"], LOCKED["beta2"]),
    )
    opt_d = torch.optim.Adam(
        critic.parameters(), lr=LOCKED["lr"], betas=(LOCKED["beta1"], LOCKED["beta2"]),
    )
    _emit({
        "event": "config",
        "pairing": pairing,
        "locked": locked,
        "fm_weight": LOCKED["fm_weight"],
        "cover_weight": LOCKED["cover_weight"],
        "cover_posture": LOCKED["cover_posture"],
        "reg_arm": regularizer.arm,
        "reg_coeff": regularizer.coeff,
        "reg_kappa": regularizer.kappa,
        "reg_norm": regularizer.norm,
        "n_particles": LOCKED["n_particles"],
        "particle_l2": LOCKED["particle_l2"],
        "vicreg_weight": LOCKED["vicreg_weight"],
        "steps": LOCKED["steps"],
        "seed": LOCKED["seed"],
        "pass_identity_mse": PASS_IDENTITY_MSE,
    }, echo, log)

    steps = LOCKED["steps"]
    for step in range(1, steps + 1):
        fake = generator(slow, prior.z)
        opt_d.zero_grad(set_to_none=True)
        d_loss = gan.d_loss(critic(slow, paired), critic(slow, fake.detach()))
        view.slow = slow.detach()
        d_loss = d_loss + regularizer(view, paired, fake.detach(), step=step)
        d_loss.backward()
        opt_d.step()

        flags = [p.requires_grad for p in critic.parameters()]
        critic.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            fake = generator(slow, prior.z)
            g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())
            # Cover matches the true fast cloud (set coverage). It does not
            # retarget identity; only the relativistic pair does that.
            # fm_weight is 0: no feature-matching term is added.
            g_loss = g_loss + LOCKED["cover_weight"] * _cover(fake, fast)
            g_loss = g_loss + LOCKED["particle_l2"] * prior.z.square().mean()
            g_loss = g_loss + spread(prior.z)
            g_loss.backward()
            opt_g.step()
        finally:
            for parameter, flag in zip(critic.parameters(), flags):
                parameter.requires_grad_(flag)

        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            with torch.no_grad():
                mse = identity_mse(generator(slow, prior.z), fast)
            _emit({
                "event": "step",
                "step": step,
                "pairing": pairing,
                "d_loss": float(d_loss.detach()),
                "g_loss": float(g_loss.detach()),
                "identity_mse": mse,
            }, echo, log)

    with torch.no_grad():
        pred = generator(slow, prior.z)
        mse = identity_mse(pred, fast)
        paired_mse = identity_mse(pred, paired)
    result = {
        "event": "done",
        "pairing": pairing,
        "locked": locked,
        "identity_mse": mse,
        "paired_target_mse": paired_mse,
        "pass": passed(mse),
        "pass_identity_mse": PASS_IDENTITY_MSE,
        "fm_weight": LOCKED["fm_weight"],
        "cover_weight": LOCKED["cover_weight"],
        "cover_posture": LOCKED["cover_posture"],
        "reg_arm": regularizer.arm,
        "reg_coeff": regularizer.coeff,
        "reg_kappa": regularizer.kappa,
        "reg_norm": regularizer.norm,
        "n_particles": prior.num_particles,
        "particle_l2": LOCKED["particle_l2"],
        "vicreg_weight": LOCKED["vicreg_weight"],
        "steps": steps,
        "seed": LOCKED["seed"],
        "delta": {} if locked else {"pairing": pairing},
    }
    _emit(result, echo, log)
    return result


def train_locked(**kwargs) -> dict:
    """Winning formulation on shared-identity pairs."""
    kwargs["pairing"] = "shared"
    kwargs["locked"] = True
    return train(**kwargs)


def train_drift(pairing: str, **kwargs) -> dict:
    """Same modules as locked_shared; the only change is the pairing index."""
    kwargs["pairing"] = pairing
    kwargs["locked"] = False
    return train(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """Locked shared-identity run, or ``--drift`` for a stranger pairing."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", default="shared", choices=PAIRINGS)
    parser.add_argument(
        "--drift", action="store_true",
        help="run stranger or nearest-stranger; locked_shared refuses those pairings",
    )
    args = parser.parse_args(argv)
    if args.drift:
        train(pairing=args.pairing, locked=False, echo=True)
        return 0
    if args.pairing != "shared":
        parser.error(
            "locked_shared refuses stranger pairing; pass --drift to run the failing arm"
        )
    result = train(pairing="shared", locked=True, echo=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
