"""CPU formulation toys for the locked_shared / Arm B shape.

HyperGAN/conceptmod is the home for these gates. ParticleGAN is core
primitives only. This package depends on the published ``particlegan``
distribution for ``GANLoss`` (RpGAN) and ``GradRegularizer`` /
``GradientPenalty``, plus ``ParticlePrior``, ``ParticleRegularizer``, and
``get_recipe``. It does not import toy modules from that package and does
not vendor ``GradRegularizer``.

This package is the formulation source of truth. Live conceptmod
research (``analysis_2d`` and ``analysis_dsl``) calls
:func:`locked_adv_defaults`, and
:func:`claim_pass`. It does not restate the adv recipe. Shared tensor geometry lives in
``conceptmod.geometry``; this package re-exports those exact definitions.
Imports are lazy: importing one helper never loads the whole research suite.
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
are not those ParticleGAN pulls. Path-suffix LoRA honesty
(``path_suffix_lora``) is a later CPU gate in this package. Mid-scale
identity hold is a later family on the scale grid ``-1, 0, 0.5, 1``; it
is not the lm_target teacher split. Cover-posture fork is a later family
on that stamp: demo cover 1.5 and Music cover 1.0 are named postures,
and a silent swap fails. Replace-macro expand honesty
(``dsl_macro_expand``) is a Wave-2 CPU gate: ``a~b`` must expand to the
documented triple, and geometric ``right`` for bare ``red~blue`` fails.
Phrase-DSL jobs (``dsl_phrase_jobs``) are a later family on the
``docs/dsl.md`` scoreboard. Geometric ``right`` stays that verdict.
Formulation PASS is honesty that the phrase is the documented job.
The DSL game-geometry bridge (``dsl_game_geometry``) PASSes only when a
locked_shared ``loss_game`` step and geometric ``right`` from the 2-D
phrase scoreboard are both true. A game PASS alone is not that gate.
The cross-toy suite leaderboard (``suite_leaderboard``) scores candidate
adv configs across stamp and DSL families and names which config(s) PASS
every applicable stamp toy. Cover-posture fork stays its own toy; the
suite does not score a config under the other claim. It is not the
single-toy honesty gate (``leaderboard_honesty``).

The slow→fast residual student composes with shared_trajectory and lives in
this tree.

A PASS is a CPU toy result. It is not a Music or Anima GPU transfer.

Scoreboard: ``docs/formulation-toys.md``.
"""

from __future__ import annotations

from importlib import import_module

# Public research aliases resolve only when requested.
_EXPORTS = {
    'CONTENT_KEPT_MIN': ('conceptmod.geometry', 'CONTENT_KEPT_MIN'),
    'COVER_WEIGHT': ('conceptmod.research.toys.locked_shared_floor', 'COVER_WEIGHT'),
    'CoverRecipe': ('conceptmod.research.toys.cover_leftover', 'CoverRecipe'),
    'FM_WEIGHT': ('conceptmod.research.toys.locked_shared_floor', 'FM_WEIGHT'),
    'FORMULATION': ('conceptmod.research.toys.cover_leftover', 'FORMULATION'),
    'HonestyError': ('conceptmod.research.toys.leaderboard_honesty', 'HonestyError'),
    'LEAK_RATIO_MAX': ('conceptmod.geometry', 'LEAK_RATIO_MAX'),
    'LOCKED': ('conceptmod.research.toys.locked_shared_floor', 'LOCKED'),
    'LOCKED_COVER': ('conceptmod.research.toys.cover_leftover', 'LOCKED_COVER'),
    'LOCKED_SHARED': ('conceptmod.research.toys.leaderboard_honesty', 'LOCKED_SHARED'),
    'LOCKED_TEACHER': ('conceptmod.geometry', 'LOCKED_TEACHER'),
    'LeftoverField': ('conceptmod.geometry', 'LeftoverField'),
    'LockedSharedFloor': ('conceptmod.research.toys.locked_shared_floor', 'LockedSharedFloor'),
    'N_PARTICLES': ('conceptmod.research.toys.locked_shared_floor', 'N_PARTICLES'),
    'PARTICLE_L2': ('conceptmod.research.toys.locked_shared_floor', 'PARTICLE_L2'),
    'POLE_REL_ERR_MAX': ('conceptmod.geometry', 'POLE_REL_ERR_MAX'),
    'SAME_DIR_MAX': ('conceptmod.geometry', 'SAME_DIR_MAX'),
    'U_KEPT_MIN': ('conceptmod.geometry', 'U_KEPT_MIN'),
    'ae_gan_hold': ('conceptmod.research.toys.ae_gan_hold', None),
    'blend_guard': ('conceptmod.geometry', 'blend_guard'),
    'claim_expand_pass': ('conceptmod.research.toys.dsl_macro_expand', 'claim_expand_pass'),
    'claim_pass': ('conceptmod.research.toys.leaderboard_honesty', 'claim_pass'),
    'claim_selection_pass': ('conceptmod.research.toys.late_collapse', 'claim_selection_pass'),
    'cover_leftover': ('conceptmod.research.toys.cover_leftover', None),
    'cover_posture_fork': ('conceptmod.research.toys.cover_posture_fork', None),
    'dsl_game_geometry': ('conceptmod.research.toys.dsl_game_geometry', None),
    'dsl_macro_expand': ('conceptmod.research.toys.dsl_macro_expand', None),
    'dsl_phrase_jobs': ('conceptmod.research.toys.dsl_phrase_jobs', None),
    'erase_keep_backend': ('conceptmod.research.toys.erase_keep_backend', None),
    'faithful_guard_e': ('conceptmod.geometry', 'faithful_guard_e'),
    'faithful_sub_e': ('conceptmod.geometry', 'faithful_sub_e'),
    'field_lift': ('conceptmod.research.toys.field_lift', None),
    'hold_dir': ('conceptmod.geometry', 'hold_dir'),
    'keep_critic': ('conceptmod.research.toys.keep_critic', None),
    'late_collapse': ('conceptmod.research.toys.late_collapse', None),
    'leaderboard_honesty': ('conceptmod.research.toys.leaderboard_honesty', None),
    'leftover_bipolar': ('conceptmod.geometry', 'leftover_bipolar'),
    'lm_target': ('conceptmod.research.toys.lm_target', None),
    'locked_shared_floor': ('conceptmod.research.toys.locked_shared_floor', None),
    'make_arm': ('conceptmod.research.toys.leaderboard_honesty', 'make_arm'),
    'make_regularizer': ('conceptmod.research.toys.locked_shared_floor', 'make_regularizer'),
    'mid_scale_identity': ('conceptmod.research.toys.mid_scale_identity', None),
    'mode_hold': ('conceptmod.research.toys.mode_hold', None),
    'orbit_hold': ('conceptmod.research.toys.orbit_hold', None),
    'particle_posture': ('conceptmod.research.toys.particle_posture', None),
    'path_suffix_lora': ('conceptmod.research.toys.path_suffix_lora', None),
    'reject_unlocked': ('conceptmod.research.toys.cover_leftover', 'reject_unlocked'),
    'residual_student': ('conceptmod.research.toys.residual_student', None),
    'run_cover_board': ('conceptmod.research.toys.cover_leftover', 'run_board'),
    'run_cover_posture_board': ('conceptmod.research.toys.cover_posture_fork', 'run_board'),
    'run_dsl_game_board': ('conceptmod.research.toys.dsl_game_geometry', 'run_board'),
    'run_dsl_phrase_board': ('conceptmod.research.toys.dsl_phrase_jobs', 'run_board'),
    'run_erase_keep_board': ('conceptmod.research.toys.erase_keep_backend', 'run_board'),
    'run_field_board': ('conceptmod.research.toys.field_lift', 'run_board'),
    'run_honesty_board': ('conceptmod.research.toys.leaderboard_honesty', 'run_honesty_board'),
    'run_keep_critic_board': ('conceptmod.research.toys.keep_critic', 'leaderboard'),
    'run_locked_floor': ('conceptmod.research.toys.locked_shared_floor', 'leaderboard'),
    'run_macro_expand_board': ('conceptmod.research.toys.dsl_macro_expand', 'run_board'),
    'run_mid_scale_board': ('conceptmod.research.toys.mid_scale_identity', 'run_board'),
    'run_orbit_family': ('conceptmod.research.toys.orbit_hold', 'run_family'),
    'run_posture_family': ('conceptmod.research.toys.particle_posture', 'run_family'),
    'run_residual_family': ('conceptmod.research.toys.residual_student', 'run_family'),
    'run_selection_board': ('conceptmod.research.toys.late_collapse', 'run_selection_board'),
    'run_suite_board': ('conceptmod.research.toys.suite_leaderboard', 'run_suite'),
    'run_unipolar_arm': ('conceptmod.research.toys.unipolar', 'run_arm'),
    'run_unused_token_family': ('conceptmod.research.toys.unused_token_hold', 'run_family'),
    'score_geometry': ('conceptmod.research.toys.cover_leftover', 'score_geometry'),
    'shared_trajectory': ('conceptmod.research.toys.shared_trajectory', None),
    'suite_leaderboard': ('conceptmod.research.toys.suite_leaderboard', None),
    'teacher_poles': ('conceptmod.geometry', 'teacher_poles'),
    'train_ae_gan': ('conceptmod.research.toys.ae_gan_hold', 'train'),
    'train_drift': ('conceptmod.research.toys.shared_trajectory', 'train_drift'),
    'train_locked': ('conceptmod.research.toys.shared_trajectory', 'train_locked'),
    'train_mode_hold': ('conceptmod.research.toys.mode_hold', 'train_mode_hold'),
    'unipolar': ('conceptmod.research.toys.unipolar', None),
    'unused_token_hold': ('conceptmod.research.toys.unused_token_hold', None),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


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
    from .locked_shared_floor import LOCKED
    from .leaderboard_honesty import LOCKED_SHARED

    stamp = {name: getattr(LOCKED, name) for name in _ADV_FIELDS}
    for key, value in stamp.items():
        if key in LOCKED_SHARED and LOCKED_SHARED[key] != value:
            raise RuntimeError(
                "locked_shared stamps diverged on "
                f"{key}: floor {value!r} vs honesty {LOCKED_SHARED[key]!r}"
            )
    return stamp



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
    "claim_expand_pass",
    "claim_pass",
    "claim_selection_pass",
    "cover_leftover",
    "cover_posture_fork",
    "dsl_game_geometry",
    "dsl_macro_expand",
    "dsl_phrase_jobs",
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
    "mid_scale_identity",
    "mode_hold",
    "orbit_hold",
    "particle_posture",
    "path_suffix_lora",
    "reject_unlocked",
    "residual_student",
    "run_cover_board",
    "run_cover_posture_board",
    "run_dsl_game_board",
    "run_dsl_phrase_board",
    "run_erase_keep_board",
    "run_field_board",
    "run_honesty_board",
    "run_keep_critic_board",
    "run_locked_floor",
    "run_macro_expand_board",
    "run_mid_scale_board",
    "run_selection_board",
    "run_suite_board",
    "run_orbit_family",
    "run_posture_family",
    "run_residual_family",
    "run_unipolar_arm",
    "score_geometry",
    "shared_trajectory",
    "suite_leaderboard",
    "teacher_poles",
    "train_ae_gan",
    "train_drift",
    "train_locked",
    "train_mode_hold",
    "unipolar",
    "unused_token_hold",
    "run_unused_token_family",
]
