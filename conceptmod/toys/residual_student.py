"""Slow→fast residual student on same-seed both-land pairs.

Composes with :mod:`conceptmod.toys.shared_trajectory`. The arcs, the pairing
index, and the locked_shared pins live there. This module does not restate
them.

The head does not emit a fast arc. It emits a residual on the slow arc:

    fast_hat = slow + head(slow, z)

Supervised residual loss is applied only on rows where both teachers succeed
for that seed (both-land):

* the slow arc's touchdown impact is at most ``SLOW_IMPACT_MAX`` (a gentle
  landing; a fast arc impacts harder and does not count)
* the paired fast arc ends within ``LAND_TOL`` of this seed's pad, the
  endpoint of the same-seed fast arc

A stranger or nearest-stranger fast target ends on a different pad, so those
rows are not both-land pairs and the residual term stays closed. The
relativistic pair still uses the requested index. That is the drift. Matching
the stranger pad is a wrong-pad touchdown: identity MSE against the same-seed
fast arc and the own-pad success rate both fail. That is the Lunar crash
catch. A pass here is a CPU toy score.
"""

from __future__ import annotations

import json
from typing import Callable

import torch
from torch import nn

from particlegan import GANLoss, ParticlePrior, ParticleRegularizer
from particlegan.grad_regularizers import GradRegularizer

from .shared_trajectory import (
    LOCKED,
    PAIRINGS,
    PASS_IDENTITY_MSE,
    identity_mse,
    pairing_index,
    trajectories,
)


# Touchdown protocol. Slow impacts on the shared arcs top out near 0.061;
# fast impacts start near 0.141. 0.10 sits between them. Pad gaps start near
# 0.383, so a tol of 0.25 cannot count another seed's pad as this landing.
# The locked arm's worst endpoint sits near 0.14, which leaves a gap under
# that tol. A point between pads can still be inside the tol and nearer to
# a neighbor; ``wrong_pad_rate`` is that crash, and a pass requires it to be 0.
SLOW_IMPACT_MAX = 0.10
LAND_TOL = 0.25
SUCCESS_MIN = 1.0
RESIDUAL_WEIGHT = 1.0
LOG_EVERY = 100


def _xy(arc: torch.Tensor) -> torch.Tensor:
    frames = int(LOCKED["frames"])
    if arc.shape[-1] != frames * 2:
        raise ValueError(f"expected arc dim {frames * 2}, got {arc.shape[-1]}")
    return arc.reshape(arc.shape[0], frames, 2)


def endpoints(arc: torch.Tensor) -> torch.Tensor:
    """Last (x, y) of each packed arc."""
    return _xy(arc)[:, -1]


def impact(arc: torch.Tensor) -> torch.Tensor:
    """Chord of the last step. This is the touchdown speed proxy."""
    xy = _xy(arc)
    return (xy[:, -1] - xy[:, -2]).norm(dim=-1)


def both_land_mask(slow: torch.Tensor, fast: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows whose slow teacher and paired fast teacher both succeed here.

    The slow teacher must make a gentle touchdown. The paired fast teacher
    must finish on this seed's pad. Only the same-seed fast arc does that.
    """
    if index.shape != (slow.shape[0],):
        raise ValueError("pairing index must be one entry per identity")
    slow_lands = impact(slow) <= SLOW_IMPACT_MAX
    own_pad = endpoints(fast)
    paired_pad = own_pad[index]
    fast_lands = (paired_pad - own_pad).norm(dim=-1) <= LAND_TOL
    return slow_lands & fast_lands


def landing_stats(pred: torch.Tensor, fast: torch.Tensor) -> dict:
    """Own-pad success and wrong-pad crashes for a predicted fast arc."""
    pad = endpoints(fast)
    end = endpoints(pred)
    dist = (end - pad).norm(dim=-1)
    landed = dist <= LAND_TOL
    nearest = torch.cdist(end, pad).argmin(dim=1)
    own = torch.arange(end.shape[0])
    wrong_pad = nearest != own
    return {
        "success_rate": float(landed.float().mean()),
        "wrong_pad_rate": float(wrong_pad.float().mean()),
        "endpoint_l2": float(dist.mean()),
    }


def passed(mse: float, success_rate: float, wrong_pad_rate: float) -> bool:
    """Identity, every seed on its own pad, and no wrong-pad touchdown."""
    return (
        mse <= PASS_IDENTITY_MSE
        and success_rate >= SUCCESS_MIN
        and wrong_pad_rate == 0.0
    )


class ResidualHead(nn.Module):
    """``slow + head(slow, z)``. A zero head leaves the slow arc untouched."""

    def __init__(self, slow_dim: int, z_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slow_dim + z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, slow_dim),
        )

    def delta(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, z), dim=-1))

    def forward(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return slow + self.delta(slow, z)


class _Critic(nn.Module):
    """Scores a slow/fast pair. This toy owns the critic."""

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


def _check_pairing(pairing: str, locked: bool) -> None:
    if pairing not in PAIRINGS:
        raise ValueError(f"unknown pairing {pairing!r} (expected one of {PAIRINGS})")
    if locked and pairing != "shared":
        raise ValueError(
            "locked_shared refuses pairing "
            f"{pairing!r}: the residual learns only from same-seed both-land pairs. "
            "stranger and nearest-stranger are drift arms, not the locked recipe"
        )
    if not locked and pairing == "shared":
        raise ValueError("a drift arm must change pairing; shared is the locked arm")


def train(*, pairing: str = "shared", locked: bool = True, echo: bool = False,
          log: Callable[[dict], None] | None = None) -> dict:
    """Train the residual head on one pairing.

    ``locked=True`` accepts only ``pairing='shared'``. A drift arm sets
    ``locked=False`` and must name ``stranger`` or ``nearest_stranger``.
    Adv modules stay the locked_shared ones from :data:`LOCKED`.
    """
    _check_pairing(pairing, locked)
    torch.set_num_threads(1)
    torch.manual_seed(LOCKED["seed"])
    slow, fast = trajectories()
    index = pairing_index(pairing, slow)
    paired = fast[index]
    mask = both_land_mask(slow, fast, index)
    hidden = LOCKED["critic_hidden"]
    head = ResidualHead(slow.shape[1], LOCKED["z_dim"], hidden)
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
        list(head.parameters()) + list(prior.parameters()),
        lr=LOCKED["lr"], betas=(LOCKED["beta1"], LOCKED["beta2"]),
    )
    opt_d = torch.optim.Adam(
        critic.parameters(), lr=LOCKED["lr"], betas=(LOCKED["beta1"], LOCKED["beta2"]),
    )
    _emit({
        "event": "config",
        "family": "residual_student",
        "pairing": pairing,
        "locked": locked,
        "both_land_rows": int(mask.sum()),
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
        "residual_weight": RESIDUAL_WEIGHT,
        "land_tol": LAND_TOL,
        "slow_impact_max": SLOW_IMPACT_MAX,
        "steps": LOCKED["steps"],
        "seed": LOCKED["seed"],
        "pass_identity_mse": PASS_IDENTITY_MSE,
        "success_min": SUCCESS_MIN,
    }, echo, log)

    steps = LOCKED["steps"]
    both = int(mask.sum())
    for step in range(1, steps + 1):
        fake = head(slow, prior.z)
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
            fake = head(slow, prior.z)
            g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())
            # Cover matches the true fast cloud (set coverage). It does not
            # retarget identity. The residual term does, and only on both-land
            # rows. fm_weight is 0: no feature-matching term is added.
            g_loss = g_loss + LOCKED["cover_weight"] * _cover(fake, fast)
            g_loss = g_loss + LOCKED["particle_l2"] * prior.z.square().mean()
            g_loss = g_loss + spread(prior.z)
            if both:
                residual = (fake[mask] - fast[mask]).pow(2).mean()
            else:
                residual = fake.new_zeros(())
            g_loss = g_loss + RESIDUAL_WEIGHT * residual
            g_loss.backward()
            opt_g.step()
        finally:
            for parameter, flag in zip(critic.parameters(), flags):
                parameter.requires_grad_(flag)

        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            with torch.no_grad():
                pred = head(slow, prior.z)
                mse = identity_mse(pred, fast)
                stats = landing_stats(pred, fast)
            _emit({
                "event": "step",
                "step": step,
                "pairing": pairing,
                "d_loss": float(d_loss.detach()),
                "g_loss": float(g_loss.detach()),
                "residual_mse": float(residual.detach()),
                "identity_mse": mse,
                "success_rate": stats["success_rate"],
                "wrong_pad_rate": stats["wrong_pad_rate"],
            }, echo, log)

    with torch.no_grad():
        pred = head(slow, prior.z)
        mse = identity_mse(pred, fast)
        paired_mse = identity_mse(pred, paired)
        stats = landing_stats(pred, fast)
    ok = passed(mse, stats["success_rate"], stats["wrong_pad_rate"])
    result = {
        "event": "done",
        "family": "residual_student",
        "pairing": pairing,
        "locked": locked,
        "both_land_rows": both,
        "identity_mse": mse,
        "paired_target_mse": paired_mse,
        "success_rate": stats["success_rate"],
        "wrong_pad_rate": stats["wrong_pad_rate"],
        "endpoint_l2": stats["endpoint_l2"],
        "pass": ok,
        "pass_identity_mse": PASS_IDENTITY_MSE,
        "success_min": SUCCESS_MIN,
        "land_tol": LAND_TOL,
        "slow_impact_max": SLOW_IMPACT_MAX,
        "residual_weight": RESIDUAL_WEIGHT,
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
    """Winning formulation: locked_shared adv, shared-trajectory residual."""
    kwargs["pairing"] = "shared"
    kwargs["locked"] = True
    return train(**kwargs)


def train_drift(pairing: str, **kwargs) -> dict:
    """Same modules as the locked arm. The pairing index is the only change."""
    kwargs["pairing"] = pairing
    kwargs["locked"] = False
    return train(**kwargs)


def run_family(*, echo: bool = True) -> list[dict]:
    """Locked shared residual, then nearest-stranger and opposite-seed drifts."""
    rows = [train_locked(echo=echo)]
    rows.append(train_drift("nearest_stranger", echo=echo))
    rows.append(train_drift("stranger", echo=echo))
    return rows


def _board(rows: list[dict]) -> str:
    lines = [
        "RESIDUAL family=slow_fast_student device=cpu",
        "| arm | both_land | identity_mse | success | wrong_pad | gate |",
    ]
    for row in rows:
        gate = "PASS" if row["pass"] else "FAIL"
        lines.append(
            f"| {row['pairing']} | {row['both_land_rows']} | "
            f"{row['identity_mse']:.5f} | {row['success_rate']:.3f} | "
            f"{row['wrong_pad_rate']:.3f} | {gate} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Locked shared residual, or ``--drift`` for one stranger pairing."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", default="shared", choices=PAIRINGS)
    parser.add_argument(
        "--drift", action="store_true",
        help="run stranger or nearest-stranger; locked_shared refuses those pairings",
    )
    parser.add_argument(
        "--family", action="store_true",
        help="run locked_shared and both drift arms",
    )
    args = parser.parse_args(argv)
    if args.family:
        rows = run_family(echo=True)
        print(_board(rows), flush=True)
        locked_ok = rows[0]["pass"]
        drifts_fail = all(not row["pass"] for row in rows[1:])
        return 0 if locked_ok and drifts_fail else 1
    if args.drift:
        result = train(pairing=args.pairing, locked=False, echo=True)
        return 0 if not result["pass"] else 1
    if args.pairing != "shared":
        parser.error(
            "locked_shared refuses stranger pairing; pass --drift to run the failing arm"
        )
    result = train(pairing="shared", locked=True, echo=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
