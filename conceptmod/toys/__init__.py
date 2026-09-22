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
``erase_keep_backend`` scores that same erase/keep geometry on the
cpu/dummy backend (and a supra-shaped latent stub).

The gates were opened on 255BITS/ParticleGAN by mistake (pulls #26–#34).
#26 (shared-trajectory) and #27 (orbit radius hold) were merged there and
are being reverted. Both families stay here. Late-collapse selection is a
later local gate (Lunar #23); it was not one of those pulls. Keep-critic
is a later family: the same locked_shared stamp with the host critic
frozen, and the Music ``mlp`` head refused. Image UNI ``lm_target``
(trajectory vs direct vs cfg_delta), unused-token UNI hold (an embed
slot pinned while the concept slot moves), and field lift (the 2D sheet
recipe on a tilted Field3D plane) are later conceptmod families. They
are not those ParticleGAN pulls.

The slow→fast residual student composes with shared_trajectory and lives in
this tree.

A PASS is a CPU toy result. It is not a Music or Anima GPU transfer.

Scoreboard: ``docs/formulation-toys.md``.
"""

from __future__ import annotations

from . import (
    ae_gan_hold,
    cover_leftover,
    erase_keep_backend,
    field_lift,
    keep_critic,
    leaderboard_honesty,
    late_collapse,
    lm_target,
    locked_shared_floor,
    mode_hold,
    orbit_hold,
    particle_posture,
    residual_student,
    shared_trajectory,
    unipolar,
    unused_token_hold,
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
from .late_collapse import (
    claim_selection_pass,
    run_selection_board,
)
from .leaderboard_honesty import (
    LOCKED_SHARED,
    HonestyError,
    claim_pass,
    make_arm,
    run_honesty_board,
)
from .keep_critic import leaderboard as run_keep_critic_board
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
run_residual_family = residual_student.run_family
run_orbit_family = orbit_hold.run_family
run_field_board = field_lift.run_board
run_unipolar_arm = unipolar.run_arm
train_ae_gan = ae_gan_hold.train
train_mode_hold = mode_hold.train_mode_hold
run_posture_family = particle_posture.run_family
run_unused_token_family = unused_token_hold.run_family
run_erase_keep_board = erase_keep_backend.run_board

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
    "claim_selection_pass",
    "cover_leftover",
    "erase_keep_backend",
    "faithful_guard_e",
    "faithful_sub_e",
    "field_lift",
    "hold_dir",
    "keep_critic",
    "late_collapse",
    "leaderboard_honesty",
    "leftover_bipolar",
    "lm_target",
    "locked_adv_defaults",
    "locked_shared_floor",
    "make_arm",
    "make_regularizer",
    "mode_hold",
    "orbit_hold",
    "particle_posture",
    "reject_unlocked",
    "residual_student",
    "run_cover_board",
    "run_erase_keep_board",
    "run_field_board",
    "run_honesty_board",
    "run_keep_critic_board",
    "run_locked_floor",
    "run_selection_board",
    "run_orbit_family",
    "run_posture_family",
    "run_residual_family",
    "run_unipolar_arm",
    "score_geometry",
    "shared_trajectory",
    "teacher_poles",
    "train_ae_gan",
    "train_drift",
    "train_locked",
    "train_mode_hold",
    "unipolar",
    "unused_token_hold",
    "run_unused_token_family",
]
