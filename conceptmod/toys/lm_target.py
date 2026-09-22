"""Image UNI lm_target: trajectory vs direct vs cfg_delta.

Culture reviewed: particle-sliders Anima image UNI
(``conceptmod/textsliders/anima_slider.py``). Live smile locks
``--lm_target trajectory`` because a 1-step velocity gap cannot carry
expression. The v4 diagnostic on real Anima is
``cos(v(plus), v(neu)) ≈ 0.99993`` and ``MSE ≈ 0.00037`` at one
timestep, while the images still differ. ``direct`` and ``cfg_delta``
stay in that CLI; they are not the lock. Music 3 stays ``--lm_target
v9`` and is not this toy.

This CPU field is that fact with the high-σ expression gate shut, so
the 1-step gradient is exactly zero instead of a step-budget race.
Expression is written only at late σ (``σ ≤ 0.5``). A K-step FlowMatch
Euler trajectory (``σ = linspace(1, 1/K, K) ∪ {0}``,
``x ← x + (σ_next − σ) v``) accumulates it. The student is a neu/infer
residual: the plus caption is teacher only, matching Anima UNI.

Same metrics on every arm
-------------------------
* ``expr_gain`` — student short-traj expression vs frozen plus, ``≥ 0.85``
* ``struct_hold`` — shared structure axis stays on the neu traj, ``≥ 0.95``
* ``identity_mse`` — scale 0 matches frozen neu, ``≤ 1e-8``

``trajectory`` must PASS. ``direct`` and ``cfg_delta`` train at infer
noise (``σ = 1``), where plus and neu velocities agree, and FAIL
``expr_gain``. Structure hold stays healthy on all three; the concept
axis is the gate.

Adversarial loss does not apply. Image UNI is target MSE, same as the
Anima trainer. The locked_shared stamp (RpGAN logistic, ``b_cap``
coeff=1 κ=1, FM off) is still the only legal adv posture: stranger
pairing, FM-on, and a thinned cap are refused rather than trained.
A PASS here is not an Anima or Music GPU transfer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from conceptmod.toys.locked_shared_floor import LOCKED


LOCKED_TARGET = "trajectory"
ARMS = ("trajectory", "direct", "cfg_delta")

# K-step flow. Same contract as Anima's short trajectory (default K=4).
TRAJ_STEPS = 4
LATE_SIGMA = 0.5
ONE_STEP_SIGMA = 1.0
IDENTITY_WEIGHT = 0.25

# Shared structure rate, and the plus-only late-σ expression rate.
STRUCT = 1.0
EXPR = 2.0
Z_T = (0.25, -0.40)

EXPR_GAIN_MIN = 0.85
STRUCT_HOLD_MIN = 0.95
IDENTITY_MSE_MAX = 1e-8

# Fixture budget, not a second adv recipe. Slider locked steps stay 1200.
STEPS = 80
LR = 0.5
BETAS = (0.0, 0.99)
SEED = 0

# Adv knobs this board will not train. Drift is refused against LOCKED.
_ADV_KEYS = (
    "loss_type",
    "gan_mode",
    "reg_arm",
    "reg_coeff",
    "reg_kappa",
    "reg_norm",
    "reg_lazy",
    "target_anneal",
    "fm_weight",
)


def reject_unlocked(overrides: dict) -> None:
    """Refuse adv drift. Image UNI does not train a critic.

    Passing the locked_shared value is a no-op. A different value is a
    second recipe and is refused, including stranger pairing, FM-on,
    and a thinned ``b_cap``.
    """
    for key, value in overrides.items():
        if key in ("steps", "seed"):
            continue
        if key not in _ADV_KEYS:
            raise ValueError(
                f"unknown formulation knob {key!r}; "
                "this board's only arm delta is lm_target"
            )
        want = getattr(LOCKED, key)
        if value == want:
            continue
        if key == "gan_mode":
            raise ValueError(
                "stranger pairing refused: locked_shared is RpGAN logistic, "
                f"got mode={value!r}"
            )
        if key == "fm_weight":
            raise ValueError(
                f"FM-on under b_cap refused: fm_weight={value!r} "
                "(locked fm_weight=0). Adversarial loss does not apply "
                "on this image-UNI board."
            )
        if key in ("reg_arm", "reg_coeff", "reg_kappa", "reg_norm", "reg_lazy", "target_anneal"):
            raise ValueError(
                f"thinned b_cap refused: {key}={value!r} != locked {want!r} "
                "(coeff=1, kappa=1, norm=l2, arm=b_cap)"
            )
        raise ValueError(
            f"formulation drift refused: {key}={value!r} != locked {want!r}"
        )


def adv_posture() -> dict:
    """locked_shared adv stamp. Reported, not trained."""
    return {key: getattr(LOCKED, key) for key in _ADV_KEYS}


def flow_sigmas(num_steps: int = TRAJ_STEPS) -> torch.Tensor:
    """``linspace(1, 1/K, K)`` plus terminal 0. Length ``K+1``."""
    steps = int(num_steps)
    if steps < 1:
        raise ValueError(f"traj_steps must be >= 1, got {num_steps!r}")
    vals = torch.linspace(1.0, 1.0 / steps, steps)
    return torch.cat([vals, vals.new_zeros(1)])


def _late(sigma: float) -> float:
    return 1.0 if float(sigma) <= LATE_SIGMA + 1e-8 else 0.0


def frozen_velocity(kind: str, sigma: float) -> torch.Tensor:
    """Teacher velocity. Expression exists only on plus, and only at late σ."""
    if kind not in ("plus", "neu", "empty"):
        raise ValueError(f"unknown prompt kind {kind!r}")
    expr = EXPR if kind == "plus" and _late(sigma) == 1.0 else 0.0
    return torch.tensor([STRUCT, expr], dtype=torch.float32)


class NeuResidual(nn.Module):
    """Additive write on the neu/infer prompt. Scale 0 disables it.

    The gate matches the field: nothing is added at high σ, so a 1-step
    target at infer noise has no gradient. Late σ is where trajectory
    can see expression. Empty and plus prompts do not carry the residual
    (plus is teacher only; empty stays the CFG anchor).
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(2))

    def delta(self, kind: str, sigma: float, scale: float) -> torch.Tensor:
        if float(scale) == 0.0 or kind != "neu":
            return self.a * 0.0
        return _late(sigma) * float(scale) * self.a


def adapted_velocity(
    kind: str,
    sigma: float,
    residual: NeuResidual,
    scale: float,
) -> torch.Tensor:
    return frozen_velocity(kind, sigma) + residual.delta(kind, sigma, scale)


def _euler(x: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    return x + (sigma_next - sigma) * velocity


def rollout_frozen(kind: str, *, num_steps: int = TRAJ_STEPS) -> torch.Tensor:
    x = torch.tensor(Z_T, dtype=torch.float32)
    sigmas = flow_sigmas(num_steps)
    for i in range(num_steps):
        sigma = float(sigmas[i])
        v = frozen_velocity(kind, sigma)
        x = _euler(x, v, sigmas[i], sigmas[i + 1])
    return x


def rollout_student(
    residual: NeuResidual,
    scale: float,
    *,
    num_steps: int = TRAJ_STEPS,
) -> torch.Tensor:
    """Neu/infer prompt. ``scale=1`` is the adapter; ``scale=0`` is frozen neu."""
    x = torch.tensor(Z_T, dtype=torch.float32)
    sigmas = flow_sigmas(num_steps)
    for i in range(num_steps):
        sigma = float(sigmas[i])
        v = adapted_velocity("neu", sigma, residual, scale)
        x = _euler(x, v, sigmas[i], sigmas[i + 1])
    return x


def one_step_loss(arm: str, residual: NeuResidual, sigma: float = ONE_STEP_SIGMA) -> torch.Tensor:
    """1-step UNI loss at one σ. ``direct`` is raw velocity; ``cfg_delta`` is ``v(c)−v('')``."""
    if arm == "direct":
        s_plus = adapted_velocity("neu", sigma, residual, 1.0)
        t_plus = frozen_velocity("plus", sigma).detach()
        s_zero = adapted_velocity("neu", sigma, residual, 0.0)
        t_zero = frozen_velocity("neu", sigma).detach()
        return F.mse_loss(s_plus, t_plus) + F.mse_loss(s_zero, t_zero)
    if arm == "cfg_delta":
        s_plus = adapted_velocity("neu", sigma, residual, 1.0) - adapted_velocity(
            "empty", sigma, residual, 1.0
        )
        t_plus = (frozen_velocity("plus", sigma) - frozen_velocity("empty", sigma)).detach()
        s_zero = adapted_velocity("neu", sigma, residual, 0.0) - adapted_velocity(
            "empty", sigma, residual, 0.0
        )
        t_zero = (frozen_velocity("neu", sigma) - frozen_velocity("empty", sigma)).detach()
        return F.mse_loss(s_plus, t_plus) + F.mse_loss(s_zero, t_zero)
    raise ValueError(f"one_step_loss is direct|cfg_delta, got {arm!r}")


def trajectory_loss(residual: NeuResidual) -> torch.Tensor:
    """``MSE(x_student, x_plus) + λ_id MSE(x_zero, x_neu)``."""
    x_student = rollout_student(residual, 1.0)
    x_plus = rollout_frozen("plus").detach()
    x_zero = rollout_student(residual, 0.0)
    x_neu = rollout_frozen("neu").detach()
    return F.mse_loss(x_student, x_plus) + IDENTITY_WEIGHT * F.mse_loss(x_zero, x_neu)


def arm_loss(arm: str, residual: NeuResidual) -> torch.Tensor:
    if arm == "trajectory":
        return trajectory_loss(residual)
    if arm in ("direct", "cfg_delta"):
        return one_step_loss(arm, residual, ONE_STEP_SIGMA)
    raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")


def field_diagnostics() -> dict:
    """Geometry shared by every arm. High-σ gap is shut; late-σ gap is not."""
    v_plus = frozen_velocity("plus", ONE_STEP_SIGMA)
    v_neu = frozen_velocity("neu", ONE_STEP_SIGMA)
    one_step_mse = float(F.mse_loss(v_plus, v_neu))
    one_step_cos = float(
        F.cosine_similarity(v_plus.reshape(1, -1), v_neu.reshape(1, -1), dim=1, eps=1e-8)
    )
    late = min(LATE_SIGMA, 0.25)
    late_expr_gap = float(frozen_velocity("plus", late)[1] - frozen_velocity("neu", late)[1])
    x_plus = rollout_frozen("plus")
    x_neu = rollout_frozen("neu")
    return {
        "one_step_mse": one_step_mse,
        "one_step_cos": one_step_cos,
        "late_expr_gap": late_expr_gap,
        "traj_expr_gap": float(x_plus[1] - x_neu[1]),
        "traj_struct": float(x_plus[0] - torch.tensor(Z_T)[0]),
    }


def score_residual(arm: str, residual: NeuResidual) -> dict:
    """Same gates for every lm_target. ``passed`` is the conjunction."""
    with torch.no_grad():
        x_student = rollout_student(residual, 1.0)
        x_zero = rollout_student(residual, 0.0)
        x_plus = rollout_frozen("plus")
        x_neu = rollout_frozen("neu")
        diag = field_diagnostics()
        expr_span = diag["traj_expr_gap"]
        if abs(expr_span) <= 1e-8:
            raise RuntimeError("trajectory expression gap vanished; the field is not UNI")
        expr_gain = float((x_student[1] - x_neu[1]) / expr_span)
        struct_span = abs(diag["traj_struct"])
        struct_err = abs(float(x_student[0] - x_neu[0]))
        struct_hold = 1.0 - struct_err / struct_span
        identity_mse = float(F.mse_loss(x_zero, x_neu))
        param_l2 = float(residual.a.detach().pow(2).sum().sqrt())
    reasons = []
    if expr_gain < EXPR_GAIN_MIN:
        reasons.append("expr_gain")
    if struct_hold < STRUCT_HOLD_MIN:
        reasons.append("struct_hold")
    if identity_mse > IDENTITY_MSE_MAX:
        reasons.append("identity")
    posture = adv_posture()
    return {
        "arm": arm,
        "lm_target": arm,
        "locked_target": LOCKED_TARGET,
        "locked": arm == LOCKED_TARGET,
        "passed": not reasons,
        "expr_gain": expr_gain,
        "struct_hold": struct_hold,
        "identity_mse": identity_mse,
        "param_l2": param_l2,
        "fail_reasons": reasons,
        "adv_applies": False,
        "adv_posture": "locked_shared",
        **diag,
        **posture,
    }


def _format_row(row: dict, *, step: int | None = None) -> str:
    step_bit = ""
    if step is not None:
        step_bit = f" step={step}/{row['steps']}"
    why = ",".join(row["fail_reasons"]) or "locked trajectory"
    return (
        f"lm_target arm={row['arm']}{step_bit} seed={row['seed']} "
        f"expr_gain={row['expr_gain']:.4f} struct_hold={row['struct_hold']:.4f} "
        f"identity={row['identity_mse']:.3e} one_step_mse={row['one_step_mse']:.3e} "
        f"pass={int(row['passed'])} why={why}"
    )


def run_arm(arm: str, *, steps: int = STEPS, seed: int = SEED, **overrides) -> dict:
    """Fit one lm_target on the shared field. Budget is steps and seed."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    reject_unlocked(overrides)
    torch.manual_seed(seed)
    residual = NeuResidual()
    opt = torch.optim.Adam(residual.parameters(), lr=LR, betas=BETAS)
    last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = arm_loss(arm, residual)
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
            row = score_residual(arm, residual)
            row["steps"] = steps
            row["seed"] = seed
            row["lr"] = LR
            print(_format_row(row, step=step + 1), flush=True)
            last = row
    assert last is not None
    return last


def run_board(*, steps: int = STEPS, seed: int = SEED) -> list[dict]:
    """Locked trajectory, then the two 1-step targets that must fail."""
    print(
        "lm_target family=image-UNI device=cpu "
        f"locked={LOCKED_TARGET} adv_applies=0 posture=locked_shared",
        flush=True,
    )
    return [run_arm(arm, steps=steps, seed=seed) for arm in ARMS]


def main() -> None:
    torch.set_num_threads(1)
    rows = run_board()
    print(
        "| arm | expr_gain | struct_hold | one_step_mse | traj_expr_gap | pass |",
        flush=True,
    )
    for row in rows:
        print(
            f"| {row['arm']} | {row['expr_gain']:.4f} | {row['struct_hold']:.4f} | "
            f"{row['one_step_mse']:.3e} | {row['traj_expr_gap']:.4f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
