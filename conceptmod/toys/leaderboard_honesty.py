"""Formulation-leaderboard honesty gate for one CPU toy.

A cell is not marked PASS because its logged score is high. ``claim_pass``
refuses unless the locked_shared cell wins this toy and every declared bad
arm fails the same toy. Calling it with no negative is an error.

locked_shared (slider2d #94 shape, demo cover — not a Music GPU claim):

- RpGAN logistic pair (``GANLoss`` loss_type=logistic, mode=rp)
- ParticleGAN ``GradRegularizer`` b_cap, coeff=1, kappa=1, norm=l2
- FM off (``fm_weight=0``)
- demo cover posture 1.5, logged on the score, not a second generator loss
- tiny cloud ``n_particles=12`` with ``particle_l2=0.02``
  (not a Hub 128-particle gmix)
- host critic kept; an arch swap is refused

Budget (not formulation): 80 Adam steps, seed 0, shared lr 5e-3, betas
(0, 0.99). Slider2d locked steps stay 1200; this toy does not copy them.

The toy is a 1-D two-pole cloud. Live relativistic pairing moves the
particles off the origin while b_cap keeps the median critic slope at or
below kappa. Stranger pairing drops the live cloud out of the pair, so the
particles stay at 0. A thinned b_cap that hardcodes kappa=100 can travel
farther and post a higher cover score, and still fails the cap.

CPU only. A PASS here does not transfer to Music or Anima.

Tail the log with::

    python -m conceptmod.toys.leaderboard_honesty
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer

TOY_ID = "two_pole_cloud"
TOY_STEPS = 80
TOY_SEED = 0
TOY_LR = 5e-3
TOY_BETAS = (0.0, 0.99)
TRAVEL_MIN = 0.30
# Median ||grad D|| on reals plus the live cloud must sit at or below kappa.
GRAD_MED_MAX = 1.0
POLES = (-1.0, 1.0)
THINNED_HARDCODED_KAPPA = 100.0

# Formulation knobs. Anything else is budget. Drift from this map must be
# reported before an arm is allowed to run.
LOCKED_SHARED = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "reg_impl": "grad_regularizer",
    "fm_weight": 0.0,
    "cover_weight": 1.5,
    "n_particles": 12,
    "particle_l2": 0.02,
    "pairing": "live",
    "critic_arch": "host",
}

# Seed-0 Linear(1, 32) → SiLU → Linear(32, 1), stored so the toy does not
# depend on a global RNG draw. This is the host critic; arms do not swap it.
_HOST_W1 = (
    -0.007487, 0.536444, -0.823045, -0.735939, -0.385154, 0.268157, -0.019813,
    0.792889, -0.088744, 0.264613, -0.302213, -0.196565, -0.955348, -0.662282,
    -0.412223, 0.037044, 0.395335, 0.600023, -0.677941, -0.435463, 0.363217,
    0.830388, -0.205800, 0.748312, -0.161183, 0.105814, 0.905476, -0.927670,
    -0.629538, -0.253165, -0.389800, 0.864001,
)
_HOST_B1 = (
    -0.648180, -0.460333, -0.698640, -0.936561, -0.583740, 0.859598, 0.446218,
    0.484673, 0.052592, -0.512684, 0.169185, -0.933695, -0.722566, -0.515530,
    0.630938, 0.586321, -0.443495, -0.036082, 0.639561, 0.994133, 0.396882,
    0.135093, 0.670486, -0.588802, 0.186344, -0.775306, -0.693086, -0.516584,
    0.452473, 0.402160, -0.592353, 0.302107,
)
_HOST_W2 = (
    0.097045, -0.022312, 0.006750, 0.040960, 0.109668, 0.169740, -0.136228,
    -0.064783, 0.069475, 0.146468, 0.153832, 0.155980, 0.035181, -0.153722,
    0.016262, -0.110592, -0.164748, 0.157065, 0.134414, -0.176340, 0.033088,
    -0.029780, -0.029091, -0.080921, 0.067981, -0.104705, 0.064805, 0.089397,
    0.126549, 0.066099, -0.174962, -0.114674,
)
_HOST_B2 = (0.088267,)


class HonestyError(ValueError):
    """A PASS claim broke the leaderboard honesty rule."""


@dataclass(frozen=True)
class Arm:
    name: str
    role: str
    spec: dict
    drift: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class CellResult:
    name: str
    role: str
    toy: str
    won: bool
    mean_abs: float
    grad_med: float
    nearest: float
    cover_score: float
    order: int
    steps: int
    seed: int
    drift: tuple[tuple[str, object], ...] = ()


class HostCritic(nn.Module):
    """1-D critic pinned to the stored host weights. Not an arch menu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(_HOST_W1).reshape(32, 1))
            self.fc1.bias.copy_(torch.tensor(_HOST_B1))
            self.fc2.weight.copy_(torch.tensor(_HOST_W2).reshape(1, 32))
            self.fc2.bias.copy_(torch.tensor(_HOST_B2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.silu(self.fc1(x))).squeeze(-1)


class ThinnedBCap:
    """b_cap stub that discards kappa and hardcodes the hinge center.

    The requested kappa is kept only so a caller can see it was ignored.
    """

    def __init__(self, coeff: float, kappa: float, norm: str) -> None:
        self.arm = "b_cap"
        self.coeff = float(coeff)
        self.requested_kappa = float(kappa)
        self.kappa = float(THINNED_HARDCODED_KAPPA)
        self.norm = norm
        self._inner = GradRegularizer(
            arm="b_cap",
            coeff=self.coeff,
            kappa=self.kappa,
            norm=self.norm,
        )

    def __call__(self, critic, real, fake, step: int = 1):
        return self._inner(critic, real, fake, step=step)


def drift_from_locked(spec: dict) -> dict:
    return {key: spec[key] for key in LOCKED_SHARED if spec[key] != LOCKED_SHARED[key]}


def make_arm(name: str, role: str, *, reported_drift: dict | None = None, **overrides) -> Arm:
    """Build an arm. Drift from locked_shared must be reported in full."""
    unknown = set(overrides) - set(LOCKED_SHARED)
    if unknown:
        raise HonestyError(f"unknown formulation keys {sorted(unknown)}")
    if role not in ("locked_shared", "negative"):
        raise HonestyError("role must be locked_shared or negative")
    spec = {**LOCKED_SHARED, **overrides}
    drift = drift_from_locked(spec)
    if role == "locked_shared":
        if name != "locked_shared":
            raise HonestyError("the locked_shared role is only the locked_shared cell")
        if drift or reported_drift:
            raise HonestyError("locked_shared cannot drift and has nothing to report")
    else:
        if not drift:
            raise HonestyError(
                "a declared bad arm must drift from locked_shared; "
                "an identical cell is not a negative"
            )
        if reported_drift is None:
            shown = ", ".join(f"{key}={value!r}" for key, value in drift.items())
            raise HonestyError(
                "new adv recipe without reported drift from locked_shared: " + shown
            )
        if dict(reported_drift) != drift:
            raise HonestyError(
                "reported drift does not match the arm "
                f"(reported {dict(reported_drift)!r}, actual {drift!r})"
            )
    if spec["critic_arch"] != "host":
        raise HonestyError("host critic stays; refusing a critic arch swap")
    items = tuple(sorted(drift.items(), key=lambda item: item[0]))
    return Arm(name=name, role=role, spec=spec, drift=items)


def make_regularizer(spec: dict):
    """Real GradRegularizer, or the κ-hardcoded stub when the arm says so."""
    impl = spec["reg_impl"]
    if impl == "grad_regularizer":
        return GradRegularizer(
            arm=spec["reg_arm"],
            coeff=spec["reg_coeff"],
            kappa=spec["reg_kappa"],
            norm=spec["reg_norm"],
        )
    if impl == "thinned_hardcoded_kappa":
        if spec["reg_arm"] != "b_cap":
            raise HonestyError("thinned stub is a b_cap drift, not a new arm name")
        return ThinnedBCap(spec["reg_coeff"], spec["reg_kappa"], spec["reg_norm"])
    raise HonestyError(f"unknown reg_impl {impl!r}")


def real_batch(n: int) -> torch.Tensor:
    """Balanced poles at ±1 with a fixed ±0.05 spread. No RNG."""
    half = n // 2
    offset = torch.linspace(-0.05, 0.05, half)
    return torch.cat([-1.0 + offset, 1.0 + offset]).unsqueeze(1)


def _check_toy_spec(spec: dict) -> None:
    if spec["n_particles"] != LOCKED_SHARED["n_particles"]:
        raise HonestyError(
            "this toy's cloud is n=12 with particle_l2; "
            "refusing a Hub-scale particle count"
        )
    if spec["pairing"] not in ("live", "stranger"):
        raise HonestyError(f"unknown pairing {spec['pairing']!r}")
    if spec["loss_type"] != "logistic" or spec["gan_mode"] != "rp":
        raise HonestyError("this toy's adversarial term is RpGAN logistic only")
    if spec["reg_arm"] != "b_cap" or spec["reg_norm"] != "l2":
        raise HonestyError("this toy's penalty is l2 b_cap")


def _grad_median(critic: nn.Module, real: torch.Tensor, particles: torch.Tensor) -> float:
    xs = torch.cat([real, particles.detach()]).detach().requires_grad_(True)
    grad = torch.autograd.grad(critic(xs).sum(), xs, create_graph=False)[0]
    return float(grad.flatten().abs().median())


def _nearest(particles: torch.Tensor) -> float:
    poles = particles.new_tensor(POLES)
    return float((particles.flatten().unsqueeze(1) - poles).abs().min(dim=1).values.mean())


def cell_wins(mean_abs: float, grad_med: float) -> bool:
    """Travel off the origin, and the b_cap median slope stays ≤ kappa.

    Both-pole balance and cover_score are logged elsewhere. Gating this cell
    on them would crown a thinned hinge that walks farther than locked_shared.
    """
    return mean_abs >= TRAVEL_MIN and grad_med <= GRAD_MED_MAX


def run_cell(arm: Arm, *, steps: int = TOY_STEPS, seed: int = TOY_SEED, order: int = 1) -> CellResult:
    """Train one arm on the two-pole cloud and score the honesty cell."""
    spec = arm.spec
    _check_toy_spec(spec)
    if steps < 1:
        raise HonestyError("steps must be positive")
    torch.manual_seed(seed)
    critic = HostCritic()
    particles = nn.Parameter(torch.zeros(spec["n_particles"], 1))
    opt_d = torch.optim.Adam(critic.parameters(), lr=TOY_LR, betas=TOY_BETAS)
    opt_p = torch.optim.Adam([particles], lr=TOY_LR, betas=TOY_BETAS)
    gan = GANLoss(loss_type=spec["loss_type"], mode=spec["gan_mode"])
    regularizer = make_regularizer(spec)
    real = real_batch(spec["n_particles"])
    stranger = torch.linspace(-3.0, 3.0, spec["n_particles"]).unsqueeze(1)
    for step in range(1, steps + 1):
        opt_d.zero_grad(set_to_none=True)
        fake = particles.detach() if spec["pairing"] == "live" else stranger
        d_loss = gan.d_loss(critic(real), critic(fake))
        (d_loss + regularizer(critic, real, fake, step=step)).backward()
        opt_d.step()

        opt_p.zero_grad(set_to_none=True)
        d_real = critic(real).detach()
        paired = critic(particles) if spec["pairing"] == "live" else critic(stranger)
        g_loss = gan.g_loss(paired, d_real)
        g_loss = g_loss + spec["particle_l2"] * particles.square().mean()
        if spec["fm_weight"] != 0.0:
            # FM is outside b_cap. On this symmetric toy the real mean is 0,
            # so a nonzero weight pins the cloud at the origin.
            g_loss = g_loss + spec["fm_weight"] * (particles.mean() - real.mean()).square()
        g_loss.backward()
        opt_p.step()

    with torch.no_grad():
        flat = particles.detach().flatten()
        mean_abs = float(flat.abs().mean())
        nearest = _nearest(flat)
    grad_med = _grad_median(critic, real, particles)
    cover_score = float(spec["cover_weight"]) * (1.0 - min(nearest, 1.0))
    return CellResult(
        name=arm.name,
        role=arm.role,
        toy=TOY_ID,
        won=cell_wins(mean_abs, grad_med),
        mean_abs=mean_abs,
        grad_med=grad_med,
        nearest=nearest,
        cover_score=cover_score,
        order=order,
        steps=steps,
        seed=seed,
        drift=arm.drift,
    )


def log_cell(cell: CellResult) -> None:
    drift = ",".join(f"{key}={value}" for key, value in cell.drift) or "-"
    print(
        f"honesty order={cell.order} arm={cell.name} role={cell.role} "
        f"won={int(cell.won)} mean_abs={cell.mean_abs:.4f} "
        f"grad_med={cell.grad_med:.4f} nearest={cell.nearest:.4f} "
        f"cover_score={cell.cover_score:.4f} drift={drift}",
        flush=True,
    )


def claim_pass(*, winner: CellResult, negatives: list[CellResult] | tuple[CellResult, ...]) -> dict:
    """PASS only when locked_shared won and every declared bad arm failed.

    An empty negative list is an error, including when the winner looks perfect.
    """
    if not negatives:
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    if winner.role != "locked_shared" or winner.name != "locked_shared":
        raise HonestyError("PASS requires the winning cell to be locked_shared")
    if winner.drift:
        raise HonestyError("locked_shared cell drifted; refusing PASS")
    if winner.toy != TOY_ID:
        raise HonestyError("PASS is only defined for the two_pole_cloud toy")
    if not winner.won:
        raise HonestyError("locked_shared did not win the toy; refusing PASS")
    for neg in negatives:
        if neg.role != "negative":
            raise HonestyError("PASS negatives must be declared bad arms")
        if neg.toy != winner.toy:
            raise HonestyError("declared bad arm was not scored on the same toy")
        if neg.won:
            raise HonestyError(
                f"declared bad arm {neg.name!r} did not fail; refusing PASS"
            )
    verdict = {
        "verdict": "PASS",
        "winner": winner.name,
        "negatives": [neg.name for neg in negatives],
        "toy": TOY_ID,
        "music_gpu_transfer": False,
        "anima_gpu_transfer": False,
        "cpu_toy": True,
        "cover_posture": "demo_1.5",
    }
    return verdict


def log_verdict(verdict: dict) -> None:
    names = ",".join(verdict["negatives"])
    print(
        f"honesty verdict={verdict['verdict']} winner={verdict['winner']} "
        f"negatives={names} music_gpu_transfer={int(verdict['music_gpu_transfer'])} "
        f"anima_gpu_transfer={int(verdict['anima_gpu_transfer'])}",
        flush=True,
    )


def demo_arms() -> tuple[Arm, Arm, Arm]:
    """locked_shared first, then the two declared bad arms."""
    locked = make_arm("locked_shared", "locked_shared")
    stranger = make_arm(
        "stranger_pairing",
        "negative",
        reported_drift={"pairing": "stranger"},
        pairing="stranger",
    )
    thinned = make_arm(
        "thinned_b_cap",
        "negative",
        reported_drift={"reg_impl": "thinned_hardcoded_kappa"},
        reg_impl="thinned_hardcoded_kappa",
    )
    return (locked, stranger, thinned)


def run_honesty_board(
    arms: tuple[Arm, ...] | list[Arm],
    *,
    steps: int = TOY_STEPS,
    seed: int = TOY_SEED,
) -> dict:
    """Run locked_shared first. Refuse to claim PASS without a bad arm."""
    if not arms or arms[0].role != "locked_shared" or arms[0].name != "locked_shared":
        raise HonestyError("FIRST cell must be locked_shared")
    if not any(arm.role == "negative" for arm in arms[1:]):
        raise HonestyError(
            "PASS cannot be claimed without a declared negative on the same toy"
        )
    cells = []
    for order, arm in enumerate(arms, start=1):
        cell = run_cell(arm, steps=steps, seed=seed, order=order)
        log_cell(cell)
        cells.append(cell)
    verdict = claim_pass(winner=cells[0], negatives=cells[1:])
    log_verdict(verdict)
    return {"verdict": verdict, "cells": cells}


def main() -> int:
    board = run_honesty_board(demo_arms())
    return 0 if board["verdict"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
