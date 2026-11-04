"""Translate conceptmod's recorded formulation knobs to ParticleGAN 0.9.4.

The DSL and archived toy rows call paired logistic loss ``logistic/rp`` and
record ``norm=l2, target_anneal=none``. ParticleGAN now names that loss
``relativistic`` and its fixed penalties only implement L2 without annealing.
These factories keep those records readable while returning the actual
upstream classes. All loss and gradient-penalty arithmetic stays in ParticleGAN.
"""
from particlegan import GANLoss
from particlegan.grad_regularizers import GradientPenalty


def loss_name(loss_type: str = "logistic", mode: str = "rp") -> str:
    if loss_type == "logistic" and mode in ("rp", "vanilla"):
        return "relativistic" if mode == "rp" else "non_saturating"
    if loss_type == "hinge" and mode in ("rp", "vanilla"):
        # The orbit toy uses this upstream objective only as a declared drift.
        return "hinge"
    raise ValueError(f"unsupported conceptmod adversary: {loss_type!r}/{mode!r}")


def make_gan_loss(loss_type: str = "logistic", mode: str = "rp") -> GANLoss:
    loss = GANLoss(loss=loss_name(loss_type, mode))
    # Reporting aliases only. The upstream `loss` field owns the computation.
    loss.loss_type = loss_type
    loss.mode = mode
    return loss


def make_gradient_penalty(
    arm: str = "b_cap", coeff: float = 1.0, *, kappa: float = 1.0,
    norm: str = "l2", lazy_k: int = 1, target_anneal: str = "none",
    total_steps: int = 0, cls=GradientPenalty,
) -> GradientPenalty:
    if norm != "l2" or target_anneal != "none":
        raise ValueError("ParticleGAN fixed penalties require norm='l2', target_anneal='none'")
    if arm == "f_none" and coeff == 0:
        # The cap-off negative control is an upstream fixed penalty at zero weight.
        arm = "b_cap"
    if arm not in ("a_r1r2", "b_cap"):
        raise ValueError(f"unsupported conceptmod fixed penalty: {arm!r}")
    penalty = cls(arm=arm, coeff=coeff, kappa=kappa, lazy_k=lazy_k)
    # Fixed L2/no-annealing are properties of the upstream implementation.
    # Keep these aliases for existing formulation records and drift checks.
    penalty.norm = norm
    penalty.target_anneal = target_anneal
    return penalty
