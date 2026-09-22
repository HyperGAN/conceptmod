"""CPU formulation toys for the locked_shared / Arm B shape.

HyperGAN/conceptmod is the home for these gates. ParticleGAN is core
primitives only. This package depends on the published ``particlegan``
distribution for ``GANLoss`` (RpGAN) and ``GradRegularizer`` /
``GradientPenalty``, plus ``ParticlePrior``, ``ParticleRegularizer``, and
``get_recipe``. It does not import toy modules from that package and does
not vendor ``GradRegularizer``.

The gates were opened on 255BITS/ParticleGAN by mistake (pulls #26–#34).
#26 (shared-trajectory) and #27 (orbit radius hold) were merged there and
are being reverted. Both families stay here.

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
