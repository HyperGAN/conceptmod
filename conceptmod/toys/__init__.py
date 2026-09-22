"""CPU formulation toys for the locked_shared / Arm B shape.

HyperGAN/conceptmod is the home for these gates. ParticleGAN is core
primitives only. This package depends on the published ``particlegan``
distribution for ``GANLoss`` (RpGAN) and ``GradRegularizer`` /
``GradientPenalty``, plus ``ParticlePrior``, ``ParticleRegularizer``, and
``get_recipe``. It does not import toy modules from that package and does
not vendor ``GradRegularizer``.

This package is the formulation source of truth. Live conceptmod
(``analysis_2d``, ``analysis_dsl``, ``ops_erase``) calls
:func:`locked_adv_defaults`, the cover / leftover helpers, and
:func:`claim_pass`. It does not restate the adv recipe.

The gates were opened on 255BITS/ParticleGAN by mistake (pulls #26–#34).
#26 (shared-trajectory) and #27 (orbit radius hold) were merged there and
are being reverted. Both families stay here.

A PASS is a CPU toy result. It is not a Music or Anima GPU transfer.

Scoreboard: ``docs/formulation-toys.md``.
"""

from __future__ import annotations

from . import (
    ae_gan_hold,
    cover_leftover,
    leaderboard_honesty,
    locked_shared_floor,
    mode_hold,
    orbit_hold,
    particle_posture,
    shared_trajectory,
    unipolar,
)
from .cover_leftover import (
    CONTENT_KEPT_MIN,
    FORMULATION,
    LEAK_RATIO_MAX,
    LOCKED_COVER,
    LOCKED_TEACHER,
    POLE_REL_ERR_MAX,
    SAME_DIR_MAX,
    U_KEPT_MIN,
    CoverRecipe,
    LeftoverField,
    blend_guard,
    faithful_guard_e,
    faithful_sub_e,
    hold_dir,
    leftover_bipolar,
    reject_unlocked,
    run_board as run_cover_board,
    score_geometry,
    teacher_poles,
)
from .leaderboard_honesty import (
    LOCKED_SHARED,
    HonestyError,
    claim_pass,
    make_arm,
    run_honesty_board,
)
from .locked_shared_floor import (
    COVER_WEIGHT,
    FM_WEIGHT,
    LOCKED,
    N_PARTICLES,
    PARTICLE_L2,
    LockedSharedFloor,
    leaderboard as run_locked_floor,
    make_regularizer,
)

# Formulation fields on the floor stamp. ``steps`` / ``seed`` / ``lr`` /
# ``batch`` are budget, not a second recipe.
_ADV_FIELDS = (
    "loss_type",
    "gan_mode",
    "reg_arm",
    "reg_coeff",
    "reg_kappa",
    "reg_norm",
    "reg_lazy",
    "target_anneal",
    "fm_weight",
    "particle_l2",
    "n_particles",
    "z_dim",
    "cover_weight",
    "cover_posture",
)


def locked_adv_defaults() -> dict:
    """Winning locked_shared adv stamp, read off :data:`LOCKED`.

    Not a new recipe. Keys that also live on :data:`LOCKED_SHARED` are
    checked so the floor and the honesty gate cannot fork.
    """
    stamp = {name: getattr(LOCKED, name) for name in _ADV_FIELDS}
    for key, value in stamp.items():
        if key in LOCKED_SHARED and LOCKED_SHARED[key] != value:
            raise RuntimeError(
                "locked_shared stamps diverged on "
                f"{key}: floor {value!r} vs honesty {LOCKED_SHARED[key]!r}"
            )
    return stamp


train_locked = shared_trajectory.train_locked
train_drift = shared_trajectory.train_drift
run_orbit_family = orbit_hold.run_family
run_unipolar_arm = unipolar.run_arm
train_ae_gan = ae_gan_hold.train
train_mode_hold = mode_hold.train_mode_hold
run_posture_family = particle_posture.run_family

__all__ = [
    "CONTENT_KEPT_MIN",
    "COVER_WEIGHT",
    "FORMULATION",
    "FM_WEIGHT",
    "HonestyError",
    "LEAK_RATIO_MAX",
    "LOCKED",
    "LOCKED_COVER",
    "LOCKED_SHARED",
    "LOCKED_TEACHER",
    "LeftoverField",
    "LockedSharedFloor",
    "N_PARTICLES",
    "PARTICLE_L2",
    "POLE_REL_ERR_MAX",
    "SAME_DIR_MAX",
    "U_KEPT_MIN",
    "ae_gan_hold",
    "blend_guard",
    "claim_pass",
    "cover_leftover",
    "faithful_guard_e",
    "faithful_sub_e",
    "hold_dir",
    "leaderboard_honesty",
    "leftover_bipolar",
    "locked_adv_defaults",
    "locked_shared_floor",
    "make_arm",
    "make_regularizer",
    "mode_hold",
    "orbit_hold",
    "particle_posture",
    "reject_unlocked",
    "run_cover_board",
    "run_honesty_board",
    "run_locked_floor",
    "run_orbit_family",
    "run_posture_family",
    "run_unipolar_arm",
    "score_geometry",
    "shared_trajectory",
    "teacher_poles",
    "train_ae_gan",
    "train_drift",
    "train_locked",
    "train_mode_hold",
    "unipolar",
]
