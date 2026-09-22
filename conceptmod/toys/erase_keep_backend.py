"""Backend-agnostic erase/keep geometry on a host latent.

DSL erase/keep is scored by :func:`conceptmod.ops_erase.erase_keep_geometry`,
which calls ``hold_dir``, ``faithful_guard_e``, and ``leftover_bipolar``.
Those helpers are the cover / leftover source of truth. This module does
not restate their thresholds and does not train the 800-step cover GAN.

What it does prove: the same locked geometry PASSes when the axes are
read back from a host ``predict_v``, not from a private tensor and not
from a video latent with a singleton time axis. Hosts are

* ``load_backend("cpu")`` and the ``dummy`` alias (``CpuBackend``)
* a tiny Supra2-IMG-shaped stub: image latent ``(4, 32, 32)``, flow time
  ``t = i/K`` in ``[0, 1]``. No CUDA, no Hub checkpoint. The real supra
  backend is not required; this stub is the stand-in.

``cover_weight`` on a row is the posture being scored. Demo cover 1.5
means the residual sits on the guarded pole. ``cover_zero`` is an
unpinned residual at half the pole (undershoot). Neither arm runs Adam.

Drift that must FAIL
-------------------
* ``teacher_leak`` — keep axis restates the erase axis, so the faithful
  guard has no perpendicular ê to peel off.
* ``cover_zero`` — odd residual undershoots the guarded pole.
* ``wrong_poles`` — the minus residual copies the plus pole (even leftover).

A PASS here is a CPU toy. It is not a Music or Anima transfer.
"""

from __future__ import annotations

import torch

from conceptmod import ops_erase
from conceptmod.toys.cover_leftover import (
    LEAK_RATIO_MAX,
    LOCKED_COVER,
    LOCKED_TEACHER,
    POLE_REL_ERR_MAX,
    SAME_DIR_MAX,
    U_KEPT_MIN,
)

FAMILY = "erase_keep_backend"
ARMS = ("locked", "teacher_leak", "cover_zero", "wrong_poles")

# Unpinned residual. Half the guarded pole is under U_KEPT_MIN and over
# POLE_REL_ERR_MAX. Not the 800-step cover_zero digit from the GAN toy.
COVER_ZERO_SCALE = 0.5

# Supra2-IMG at 256px is a 4-channel 32×32 SD-VAE latent. The real
# backend, when it exists, is CUDA plus a Hub checkpoint. This rank is
# the CPU stand-in. It is not a (C, 1, H, W) video latent.
SUPRA_LATENT = (4, 32, 32)
SUPRA_FLOW_STEPS = 50
SUPRA_CFG = 3.0

# Legal on the supra clock (t = i/K) and on CpuBackend (which divides
# the scheduler timestep by 1000). One probe time for every host.
PROBE_T = 0.5
PROBE_SEED = 0


def require_image_latent(shape) -> tuple[int, int, int]:
    """``(C, H, W)`` only. A leading singleton time axis is refused."""
    got = tuple(int(s) for s in shape)
    if len(got) != 3 or any(s < 1 for s in got):
        raise ValueError(
            f"erase/keep latent must be (C, H, W), got {got}. "
            "A singleton time axis is a different backend shape."
        )
    if got[0] * got[1] * got[2] < 2:
        raise ValueError(f"erase/keep latent needs two orthogonal axes, got {got}")
    return got


def _basis(shape: tuple[int, int, int], index: int) -> torch.Tensor:
    flat = torch.zeros(shape[0] * shape[1] * shape[2])
    flat[int(index)] = 1.0
    return flat


def arm_plants(shape, arm: str) -> dict[str, torch.Tensor]:
    """Unit erase û at index 0, leftover ê at index 1, plus the two poles."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")
    shape = require_image_latent(shape)
    erase = _basis(shape, 0)
    keep = _basis(shape, 1)
    plus = erase.clone()
    minus = -erase
    if arm == "teacher_leak":
        keep = erase.clone()
    elif arm == "cover_zero":
        plus = COVER_ZERO_SCALE * erase
        minus = -plus
    elif arm == "wrong_poles":
        minus = erase.clone()
    return {"erase": erase, "keep": keep, "plus": plus, "minus": minus}


class SupraShapedStub:
    """Weight-free velocity with Supra2-IMG's image latent and flow clock.

    ``predict_v`` is ``0.1 * z`` for every prompt, so a CFG delta is zero
    until :class:`PlantedHost` adds a concept. Timesteps outside ``[0, 1]``
    are refused: this is ``t = i/K``, not a 0–1000 sigma clock.
    """

    def __init__(self) -> None:
        self.device = "cpu"
        self.latent_shape = SUPRA_LATENT
        self.generate_steps = SUPRA_FLOW_STEPS
        self.generate_guidance = SUPRA_CFG

    def predict_v(self, prompt, z, timestep, frozen=False):
        del prompt, frozen
        t = timestep.detach().float().reshape(-1)[0].item() if torch.is_tensor(timestep) else float(timestep)
        if not 0.0 <= t <= 1.0:
            raise ValueError(
                f"supra flow time is t=i/K in [0, 1], got {t}"
            )
        if tuple(z.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"supra stub latent is {self.latent_shape}, got {tuple(z.shape)}"
            )
        return 0.1 * z


class PlantedHost:
    """Add one arm's concept velocities on top of a host ``predict_v``.

    The host is always called. Cpu/dummy map these prompt strings onto the
    empty class, so the measured CFG delta is the plant. A supra stub's
    base field cancels the same way.
    """

    def __init__(self, backend, plants: dict[str, torch.Tensor], *, name: str) -> None:
        self.backend = backend
        self.plants = plants
        self.name = name
        self.latent_shape = require_image_latent(backend.latent_shape)
        self.device = getattr(backend, "device", "cpu")
        self.backend_calls = 0

    def predict_v(self, prompt, z, timestep, frozen=False):
        base = self.backend.predict_v(prompt, z, timestep, frozen)
        self.backend_calls += 1
        extra = self.plants.get(prompt)
        if extra is None:
            return base
        if base.ndim != 4:
            raise ValueError(
                f"{self.name} velocity rank is BCHW, got {tuple(base.shape)}"
            )
        if tuple(base.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"{self.name} velocity shape {tuple(base.shape)} "
                f"does not match latent {self.latent_shape}"
            )
        add = extra.to(device=base.device, dtype=base.dtype).reshape(base.shape)
        return base + add


def _probe_batch(host: PlantedHost, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn((1, *host.latent_shape), generator=g)
    t = torch.tensor([PROBE_T])
    return z, t


def _cfg_delta(host: PlantedHost, prompt: str, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    v = host.predict_v(prompt, z, t, True)
    v0 = host.predict_v("", z, t, True)
    if v.ndim != 4 or v0.ndim != 4:
        raise ValueError(
            f"{host.name} erase/keep velocity rank is BCHW, "
            f"got {tuple(v.shape)} and {tuple(v0.shape)}"
        )
    return (v - v0).reshape(-1).float()


def _rel_err(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm() / target.norm().clamp_min(1e-8))


def score_backend(backend, arm: str, *, name: str, seed: int = PROBE_SEED) -> dict:
    """Read one arm off ``backend.predict_v`` and score it with the live helper.

    Axes that do not round-trip through the host are a hard error: this
    toy does not score a private copy of the field.
    """
    plants = arm_plants(getattr(backend, "latent_shape", None), arm)
    host = PlantedHost(backend, plants, name=name)
    z, t = _probe_batch(host, seed)
    measured = {prompt: _cfg_delta(host, prompt, z, t) for prompt in plants}
    for prompt, plant in plants.items():
        got = measured[prompt]
        if not torch.allclose(got, plant, atol=1e-5, rtol=0.0):
            err = float((got - plant).norm())
            raise RuntimeError(
                f"{name} cfg delta for {prompt!r} is not the planted leftover "
                f"(err={err:.3e}). Erase/keep would be scoring a private tensor."
            )
    erase = measured["erase"]
    keep = measured["keep"]
    plus = measured["plus"]
    minus = measured["minus"]
    # Live path. Monkeypatches on ops_erase.erase_keep_geometry / hold_dir /
    # faithful_guard_e / leftover_bipolar land here.
    report = ops_erase.erase_keep_geometry(erase, keep, plus, minus)
    neu = torch.zeros_like(erase)
    pole_p, pole_m = ops_erase.faithful_guard_e(erase, -erase, neu, keep, erase)
    odd_p = (pole_p - neu).reshape(-1)
    odd_m = (pole_m - neu).reshape(-1)
    err_p = _rel_err(plus, odd_p)
    err_m = _rel_err(minus, odd_m)
    pole_rel_err = max(err_p, err_m)
    covered = bool(err_p <= POLE_REL_ERR_MAX and err_m <= POLE_REL_ERR_MAX)
    u_hat = erase / erase.norm().clamp_min(1e-8)
    target = float(odd_p @ u_hat)
    u_kept = float(plus @ u_hat) / (abs(target) + 1e-8)
    reasons = []
    if not report["teacher_leak_ok"]:
        reasons.append("teacher_leak")
    if u_kept < U_KEPT_MIN or not covered:
        reasons.append("undershoot")
    if report.get("same_dir_ok") is False:
        reasons.append("wrong_poles")
    axis_dot = float(erase @ keep / (erase.norm() * keep.norm()).clamp_min(1e-8))
    return {
        "family": FAMILY,
        "arm": arm,
        "backend": name,
        "backend_cls": type(backend).__name__,
        "latent_shape": host.latent_shape,
        "teacher": report["teacher"],
        "cover_weight": 0.0 if arm == "cover_zero" else LOCKED_COVER,
        "teacher_leak": float(report["teacher_leak"]),
        "teacher_leak_ok": bool(report["teacher_leak_ok"]),
        "hold_cos": report["hold_cos"],
        "leak_frac": float(report["leak_frac"]),
        "same_dir": float(report["same_dir"]),
        "same_dir_ok": bool(report["same_dir_ok"]),
        "u_kept": float(u_kept),
        "pole_rel_err": float(pole_rel_err),
        "covered": covered,
        "axis_dot": axis_dot,
        "backend_calls": int(host.backend_calls),
        "pass": not reasons,
        "fail_reasons": ",".join(reasons),
        "leak_ratio_max": LEAK_RATIO_MAX,
        "locked_teacher": LOCKED_TEACHER,
    }


def run_board(*, seed: int = PROBE_SEED) -> list[dict]:
    """Locked arm and three drift arms on cpu, dummy, and the supra stub."""
    from conceptmod.backends import load_backend

    hosts: list[tuple[str, object]] = []
    for alias in ("cpu", "dummy"):
        hosts.append((alias, load_backend(alias, device="cpu", lora_rank=4, seed=seed)))
    hosts.append(("supra_stub", SupraShapedStub()))
    rows = []
    for name, backend in hosts:
        for arm in ARMS:
            rows.append(score_backend(backend, arm, name=name, seed=seed))
    return rows


def format_board(rows: list[dict]) -> str:
    header = (
        "| backend | arm | pass | cover | leak | hold | same_dir | u_kept | pole_err | why |"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        hold = "—" if row["hold_cos"] is None else f"{row['hold_cos']:.3f}"
        lines.append(
            "| %s | %s | %s | %.1f | %.3f | %s | %.3f | %.3f | %.3f | %s |"
            % (
                row["backend"],
                row["arm"],
                "PASS" if row["pass"] else "FAIL",
                row["cover_weight"],
                row["teacher_leak"],
                hold,
                row["same_dir"],
                row["u_kept"],
                row["pole_rel_err"],
                row["fail_reasons"] or "guarded leftover on the host latent",
            )
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    print(format_board(run_board()), end="")


if __name__ == "__main__":
    _main()
