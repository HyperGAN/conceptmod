"""CPU formulation toys for the locked_shared / Arm B shape.

These gates were opened on 255BITS/ParticleGAN by mistake (pulls #26–#34).
#26 (shared-trajectory) and #27 (orbit radius hold) were merged there and are
being reverted. Both families stay in this package; a ParticleGAN revert does
not remove them. Adversarial
primitives come from the ``particlegan`` package (``GANLoss``,
``GradientPenalty`` / ``GradRegularizer``, ``ParticlePrior``,
``ParticleRegularizer``, ``get_recipe``). This package does not vendor
``GradRegularizer``.

A PASS is a CPU toy result. It is not a Music or Anima GPU transfer.

Scoreboard: ``docs/formulation-toys.md``.
"""

from __future__ import annotations

__all__ = [
    "ae_gan_hold",
    "cover_leftover",
    "leaderboard_honesty",
    "locked_shared_floor",
    "mode_hold",
    "orbit_hold",
    "particle_posture",
    "shared_trajectory",
    "unipolar",
]
