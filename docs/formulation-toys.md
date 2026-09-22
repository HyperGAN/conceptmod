# Formulation toys (CPU scoreboard)

**HyperGAN/conceptmod is the home for these gates.** ParticleGAN is
core primitives only. Install the PyPI package (`pip install particlegan`,
`particlegan>=0.5.0` in `pyproject.toml` and `requirements.txt`). Toys call
that package for `GANLoss` / RpGAN, `GradRegularizer` / `b_cap`
(`GradientPenalty`), and `ParticlePrior`. Those modules are not copied into
conceptmod. The only local geometry is the leftover / cover / teacher field
in `conceptmod/toys/cover_leftover.py` (and the per-toy fixtures around it).
Toy code is not forked back into ParticleGAN.

The nine Wave-1 families were first opened on
[255BITS/ParticleGAN](https://github.com/255BITS/ParticleGAN) by mistake.
**#26 (shared-trajectory) and #27 (orbit radius hold) were merged there and are being reverted.** This tree keeps both families. #28–#34 were closed unmerged and are ported from those PR heads.
Late-collapse selection is a later local gate (Lunar #23). Keep-critic is a new family in this tree. Image UNI lm_target is a later conceptmod family. The residual student is a later family in this tree and composes with shared_trajectory. Field lift is a later CPU gate in this tree. None of these was a ParticleGAN pull, and none is a fork of those toys.

| family | module | ParticleGAN source |
|---|---|---|
| shared slow→fast vs stranger | `conceptmod/toys/shared_trajectory.py` | [#26](https://github.com/255BITS/ParticleGAN/pull/26) (merged by mistake, being reverted; kept here) |
| slow→fast residual student | `conceptmod/toys/residual_student.py` | this tree; composes with shared_trajectory |
| orbit radius hold | `conceptmod/toys/orbit_hold.py` | [#27](https://github.com/255BITS/ParticleGAN/pull/27) (merged by mistake, being reverted; kept here) |
| locked_shared RpGAN + `b_cap` floor | `conceptmod/toys/locked_shared_floor.py` | [#28](https://github.com/255BITS/ParticleGAN/pull/28) |
| unipolar residual | `conceptmod/toys/unipolar.py` | [#29](https://github.com/255BITS/ParticleGAN/pull/29) |
| AE-GAN + locked hold | `conceptmod/toys/ae_gan_hold.py` | [#30](https://github.com/255BITS/ParticleGAN/pull/30) |
| cover / leftover / faithful teacher | `conceptmod/toys/cover_leftover.py` | [#31](https://github.com/255BITS/ParticleGAN/pull/31) |
| leaderboard honesty | `conceptmod/toys/leaderboard_honesty.py` | [#32](https://github.com/255BITS/ParticleGAN/pull/32) |
| tiny particle posture | `conceptmod/toys/particle_posture.py` | [#33](https://github.com/255BITS/ParticleGAN/pull/33) |
| mode-hold ring | `conceptmod/toys/mode_hold.py` | [#34](https://github.com/255BITS/ParticleGAN/pull/34) |
| late-collapse selection | `conceptmod/toys/late_collapse.py` | local (Lunar #23); not a ParticleGAN pull |
| keep-critic locked_shared | `conceptmod/toys/keep_critic.py` | new. Music Arm B handoff keeps `--adv_arch mlp`; this row freezes the host critic and refuses that swap |
| image UNI lm_target | `conceptmod/toys/lm_target.py` | conceptmod (Anima image UNI; not a ParticleGAN pull) |
| 2D→3D field lift | `conceptmod/toys/field_lift.py` | particle-sliders `tests/test_field3d.py` (reviewed; not a ParticleGAN pull) |

Allowed `particlegan` imports are the primitives: `GANLoss`, `GradientPenalty`
/ `GradRegularizer`, `ParticlePrior`, `ParticleRegularizer`, and
`get_recipe` (the AE-GAN study preset). This tree does not copy
`GradRegularizer`. It also does not call `particlegan.orbit_hold`,
`particlegan.leaderboard_honesty`, or any other toy that briefly lived in
that repo. The ring critic is `conceptmod.toys.mlp` because the published
package does not export the 100-Gaussians MLP; that class is host critic
shape, not a second penalty.

Culture reviewed: HyperGAN/particle-sliders
`analysis/slider2d/locked_baseline_defaults.py` (locked_shared / #94),
`analysis/slider2d/field3d.py` and `tests/test_field3d.py` (2D recipe on an
R³ leftover field), Music Arm B (`docs/music-arm-b-gates.md`, trainer
`ARM_B`: `adv_arch=mlp`, cover/pole 1.0), and this repo's
`conceptmod/analysis_2d.py` (DSL geometry, a different fixture).
Keep-critic does not adopt the Music `mlp` head.

A PASS here is a CPU toy. It is **not** a Music or Anima GPU transfer.

## Live conceptmod consumes this package

These modules are the formulation source of truth. Live scoring does not
keep a second copy of the adv recipe.

| live path | what it calls |
|---|---|
| `conceptmod.analysis_2d` / `conceptmod.analysis_dsl` | `locked_adv_defaults()` — the `locked_shared_floor.LOCKED` stamp (RpGAN logistic, `b_cap` coeff=1 κ=1, FM off, demo cover 1.5, n=12). Keep/leak flags use `U_KEPT_MIN` and `SAME_DIR_MAX`. |
| `conceptmod.game.loss_game` | The same `locked_adv_defaults()` stamp. FM-on and thinned κ go through `reject_unlocked` when the game claims locked. Stranger pairing (flipped batch, `GANLoss` mode stays `rp`) is refused on that claim. The game does not sample `n_particles` or apply cover. |
| `conceptmod.ops_erase.erase_keep_geometry` | `hold_dir`, `faithful_guard_e`, and `leftover_bipolar` from the cover / leftover toy. |
| formulation `PASS` | `claim_pass`. An empty negative list raises `HonestyError`. Geometric verdicts stay `right` / `needs help` / `recipe` and are not a PASS. |

The 2-D Adam budget (40 steps, lr 8e-2) is fixture budget. It is not a new
adv recipe, and it does not replace locked_shared as the winning default.
`tests/test_live_toys.py` imports the analysis and erase paths and checks
that they execute these helpers.

## Locked shape

Demo posture, from the slider lock, unless a row below names a drift:

| knob | value |
|---|---|
| pairing | RpGAN logistic (`GANLoss`, `loss_type=logistic`, `mode=rp`) |
| penalty | `GradRegularizer` `b_cap`, coeff=1, kappa=1, norm=l2 |
| FM | off (`fm_weight=0`) |
| cover | **demo 1.5** (Music pole/cover 1.0 is a different posture) |
| cloud | n=12, `particle_l2=0.02` (not Hub 128, not `Recipe("gan")` 20_000) |
| VICReg | 0.05 when the toy has a latent prior |
| critic | the toy's host critic; an architecture swap is refused |

Named omissions (reported, not silent aliases):

- **Unipolar** trains with cover weight 0. Cover is the eval gate (`>= 0.85`). No particle cloud. Budget 400 steps.
- **Orbit** does not apply VICReg. Those particles are the orbit state, not a latent prior. The critic is a frozen linear probe.
- **Mode-hold** records cover 1.5 and does not add a supervised cover loss. Learning rate is this toy's budget (2e-3), not the slider 5e-3.
- **Music `--parts 0`** on the posture toy is the empty-cloud spelling: cover **1.0**, VICReg off. It is a second PASS, not a substitute for the n=12 demo lock.
- **Late-collapse** does not train. The curve is eight synthetic checkpoints. Train loss falls through the collapse; the val gate is the only export rule. The locked_shared stamp is read and not retuned.
- **Image UNI lm_target** trains target MSE only. Adversarial loss does not apply (Anima image UNI is trajectory MSE). locked_shared is the only legal adv posture and is refused if drifted, not trained. Step budget is 80, lr 0.5.

## Scoreboard

Seed 0. Numbers are the reference runs these gates were written against.
Pytest re-checks the PASS/FAIL split (and, where the original test pinned a
digit, that digit). Lower error is better unless a column says otherwise.

### Shared trajectory (identity MSE ≤ 0.02)

| arm | pairing | identity MSE | gate |
|---|---|---:|---|
| locked_shared | shared | 0.00167 | **PASS** |
| nearest_stranger | nearest other slow arc | 0.297 | FAIL |
| stranger | opposite seed | 0.567 | FAIL |

400 steps. Locked entry point refuses stranger pairing.

### Residual student (identity MSE ≤ 0.02 and success rate = 1)

Same arcs and locked_shared pins as the shared-trajectory toy. The head is a residual on the slow arc, `fast = slow + head(slow, z)`. Supervised residual loss runs only on same-seed both-land rows: slow touchdown impact ≤ 0.10, and the paired fast arc ends within 0.25 of this seed's pad (the same-seed fast endpoint). Pad gaps start near 0.383, so a touchdown on another seed's pad falls outside this landing. A pass also requires `wrong_pad_rate = 0`.

| arm | pairing | both-land | identity MSE | success | wrong-pad | gate |
|---|---|---:|---:|---:|---:|---|
| locked_shared | shared | 12 | 0.00197 | 1.00 | 0 | **PASS** |
| nearest_stranger | nearest other slow arc | 0 | 0.143 | 0.083 | 0.917 | FAIL |
| stranger | opposite seed | 0 | 0.511 | 0 | 1 | FAIL |

400 steps, seed 0. Locked entry point refuses stranger pairing. Copying the paired fast arc lands on the wrong pad (success 0, wrong-pad 1) for both drift indexes; that floor is the Lunar crash catch. One nearest-stranger seed can graze its own pad after training and the arm still fails. A pass here is a CPU toy score.

### Orbit radius hold

| arm | radius_rel | direction | speed_rel | b_cap | gate |
|---|---:|---:|---:|---:|---|
| locked_shared | ~0 | 0.992 | 8.0e-3 | 4 | **PASS** |
| ablate_residual | **0.410** | 1.000 | ~0 | 4 | FAIL `radius_drift` |
| stranger_vanilla / shuffle | ~0 | 0.992 | 8.0e-3 | 4 | FAIL `stranger_pairing` |
| fm_on | ~0 | 0.992 | 8.0e-3 | 4 | FAIL `fm_on` |
| hinge_drift | ~0 | 0.992 | 8.0e-3 | 4 | FAIL `loss_not_logistic` |
| thin_bcap | ~0 | 0.992 | 8.0e-3 | **9** | FAIL `thinned_bcap` |

Heading and speed stay healthy on the spiral. Radius is the gate.

### Locked-shared floor (absolute error vs the real modules)

| arm | gate | adv err | g err | cap err | κ probe |
|---|---|---:|---:|---:|---:|
| locked_shared | **PASS** | 0 | 0 | 0 | 0 |
| thinned_kappa | FAIL | 0 | 0 | 0 | **0.040** |
| fm_on | FAIL | 0 | 7.28e-4 | 0 | 0 |
| vanilla_logistic | FAIL | 0.693 | 0.0231 | 0 | 0 |
| stranger_pair | FAIL | 2.44e-4 | 1.19e-6 | 0 | 0 |
| music_cover_1 | FAIL | 0 | 0.233 | 0 | 0 |
| hub128 | FAIL | 0 | 0 | 0 | 0 |

The κ=0.2 probe is what catches a cap that hardcodes center 1. At locked κ=1
that stub still matches the step penalty.

### Unipolar residual (cover ≥ 0.85, leak ≤ 0.05, neu_hold ≥ 0.85)

| arm | cover | leak | neu_hold | gate |
|---|---:|---:|---:|---|
| locked_rpgan | 0.964 | 0.0005 | 0.956 | **PASS** |
| mse_only | 1.000 | 0 | 0.667 | FAIL neu_hold |
| polarity_flipped | 0 | 0 | 0.963 | FAIL plus cover |

FM-on, vanilla pairing, thinned κ, and putting demo cover 1.5 back on as a train loss are refused.

### AE-GAN hold (recon ≤ 0.05, hold ≤ 0.35, `b_cap` and RpGAN every step)

| arm | recon | hold | b_cap | RpGAN | gate |
|---|---:|---:|---|---|---|
| locked | 0.0038 | 0.024 | 250/250 | 250/250 | **PASS** |
| ae_only | 0.0024 | 0.005 | 0/250 | 0/250 | FAIL |
| stranger / FM-on / thin κ | — | — | refused | refused | FAIL |

AE-only can still reconstruct. It fails because the locked adversary never ran.
`get_recipe("ae_gan")` stays the study preset (K=400, lazy every 4); this gate
overrides to n=12 and lazy=1.

### Cover / leftover / faithful teacher

| arm | cover | teacher | u_kept | leak | gate |
|---|---:|---|---:|---:|---|
| locked | 1.5 | faithful_guard_e | 0.936 | 0.000 | **PASS** |
| cover_zero | 0 | faithful_guard_e | 0.577 | 0.060 | FAIL undershoot |
| teacher_drift | 1.5 | faithful | 0.935 | 0.468 | FAIL teacher_leak |

800 steps (budget). Stranger pairing, FM-on, thinned `b_cap`, and n=128 are refused.

### Leaderboard honesty (mean_abs ≥ 0.30 and grad_med ≤ 1)

| order | arm | mean_abs | grad_med | cover_score | gate |
|---|---|---:|---:|---:|---|
| 1 | locked_shared | 0.514 | 0.420 | 0.772 | **won** |
| 2 | stranger_pairing | 0 | 0.073 | 0 | fail |
| 3 | thinned_b_cap (κ hardcoded at 100) | 0.694 | 2.342 | **1.041** | fail |

PASS requires the locked cell to win and every declared bad arm to fail.
An empty negative list raises `HonestyError`. Cover alone would crown the thinned hinge.

### Tiny particle posture

| arm | n | particle_l2 | cloud_ms | anchor_drop | residual_l2 | gate |
|---|---:|---:|---:|---:|---:|---|
| locked_tiny | 12 | 0.02 | 0.00246 | 1.66e-5 | 0.0037 | **PASS** |
| music_parts0 | 0 | 0 | 0 | 0 | 0.0116 | **PASS** |
| hub128_routed | 128 | 0 | 0.999 | 0 | 0.0455 | FAIL |
| zero_particle_l2 | 12 | 0 | 0.00211 | 0 | 0.0023 | FAIL |

600 steps. `zero_particle_l2` can keep a small cloud; it fails because the 0.02 anchor does not contract a unit cloud.

### Mode-hold ring (8 Gaussians, modes ≥ 7 and HQ ≥ 0.90)

| arm | drift | modes | HQ | effective | gate |
|---|---|---:|---:|---:|---|
| locked | none | 8/8 | 1.000 | 7.48 | **PASS** |
| b_cap off | `f_none`, coeff 0 | 1/8 | 0.090 | 1.00 | FAIL |

1200 steps, EMA, 4096 samples. Stranger pairing, FM-on, and a κ-hardcoded stub are refused unless the drift is named. The stub is not trained.

### Late-collapse selection (export the earlier good checkpoint)

Eight checkpoints. No Adam budget. Train loss falls on every step, so the best train loss is the last step. Val error is best at step 3 (`0.06`), still inside the gate at step 4 (`0.11`), then collapsed. The val gate refuses `val_error > 0.25` and any collapsed flag, then keeps the minimum val error (earlier step on a tie).

| arm | rule | step | train_loss | val_error | collapsed | gate |
|---|---|---:|---:|---:|---|---|
| locked_val_gate | val gate | 3 | 0.32 | 0.06 | no | **PASS** |
| last_step | final index | 7 | 0.01 | 1.40 | yes | FAIL |
| best_train_loss | argmin train, no val | 7 | 0.01 | 1.40 | yes | FAIL |

PASS requires the locked rule to export step 3 and every declared bad arm to export a collapsed checkpoint. An empty negative list raises `HonestyError`. A won flag that disagrees with the export is a dishonest board and raises `HonestyError`. `claim_pass` stays the two-pole gate.

### Keep-critic (frozen host, no MLP swap)

The host critic is a frozen linear scorer. It is not
`conceptmod.toys.mlp.SimpleMLPDiscriminator`, and it is not stepped.
Music Arm B's `--adv_arch mlp` is refused before any step. Cover posture
is on every row: demo **1.5** is the lock, Music pole/cover **1.0** fails
this stamp. A PASS does not transfer to a Music or Anima GPU run.

| arm | gate | adv err | g err | cap err | κ probe | critic |
|---|---|---:|---:|---:|---:|---|
| locked_shared | **PASS** | 0 | 0 | 0 | 0 | frozen host, Δw=0 |
| stranger_pairing | FAIL | 0.167 | 2.22e-4 | 0 | 0 | frozen host |
| fm_on | FAIL | 0 | 0.200 | 0 | 0 | frozen host |
| thinned_kappa | FAIL | 0 | 0 | 0 | **0.040** | frozen host |
| music_cover_1 | FAIL | 0 | 0.233 | 0 | 0 | frozen host, cover 1.0 |
| critic_step | FAIL | 0 | 0 | 0 | 0 | same host, Δw=0.0283 |
| forced_mlp_swap | FAIL | — | — | — | — | **refused** (`mlp` not trained) |

8 steps (budget). The κ=0.2 probe is what catches a cap that hardcodes center 1.

### Image UNI lm_target (expr_gain ≥ 0.85, struct_hold ≥ 0.95)

Locked target: **trajectory**. Particle-sliders Anima image UNI
(`--lm_target trajectory` in `anima_slider.py`). Live smile does not
lock `direct` or `cfg_delta`: a 1-step velocity gap cannot carry
expression (`cos(v(plus), v(neu)) ≈ 0.99993`, `MSE ≈ 0.00037` on the v4
diagnostic). This CPU field shuts that high-σ gate, so the 1-step
gradient is zero. Expression is a late-σ write (`σ ≤ 0.5`) and shows up
on the K=4 FlowMatch Euler trajectory (`traj_expr_gap = -1`).

| arm | expr_gain | struct_hold | one_step_mse | traj_expr_gap | gate |
|---|---:|---:|---:|---:|---|
| trajectory | 1.000 | 1.000 | 0 | -1 | **PASS** |
| direct | 0 | 1.000 | 0 | -1 | FAIL `expr_gain` |
| cfg_delta | 0 | 1.000 | 0 | -1 | FAIL `expr_gain` |

80 steps, lr 0.5 (budget). Same field and the same three gates on every
arm. Structure hold stays 1 on the failing arms; the concept axis is
the split. Adversarial loss does not apply. Stranger pairing, FM-on,
and a thinned `b_cap` are refused. Not an Anima GPU transfer.

### 2D→3D field lift (plane gates survive the tangent pushforward)

Reviewed against particle-sliders Field3D (`analysis/slider2d/field3d.py`,
`tests/test_field3d.py`): the locked 2D recipe transfers with no 3D-only
hack. κ stays 1. The sheet is the cover toy's guarded odd residual
(slider 1, content 0.55, unused ê stripped by `faithful_guard_e`). On the
coordinate plane that vector is the paste `(1, 0.55, 0)`. The lift tilts
û toward ê by π/4; the correction is `slider·û(θ) + content·ĉ`.

| arm | frame | u_kept | content | leak | pole_err | gate |
|---|---|---:|---:|---:|---:|---|
| locked_2d | plane | 1 | 1 | 0 | 0 | **PASS** |
| locked_lift | tilted | 1 | 1 | 0 | 0 | **PASS** |
| naive_copy | tilted, no z correction | 0.707 | 1 | 1 | 0.504 | FAIL `naive_lift`, undershoot |
| stranger_pairing | tilted, flipped pair | 1 | 1 | 0 | 0 | FAIL `stranger_pairing` (adv err 4.92e-4) |
| fm_on | tilted, `fm_weight=0.1` | 1 | 1 | 0 | 0 | FAIL `fm_on` (fm term 6.36e-3) |
| thinned_kappa | tilted, center hardcoded at 1 | 1 | 1 | 0 | 0 | FAIL `thinned_kappa` (κ probe **0.040**) |

One scored step. The paste's cover gap is g err 0.439. At locked κ=1 the
thinned center still matches the step penalty; the κ=0.2 probe is what
fails it. Stranger pairing, FM-on, and the stub keep the lifted geometry.

## Run

```bash
pytest tests/test_live_toys.py tests/test_2d_analysis.py tests/test_erase_cpu.py tests/test_dsl_jobs.py -q

pytest tests/test_toy_shared_trajectory.py \
       tests/test_toy_residual_student.py \
       tests/test_toy_orbit_hold.py \
       tests/test_toy_locked_shared_floor.py \
       tests/test_toy_unipolar.py \
       tests/test_toy_ae_gan_hold.py \
       tests/test_toy_cover_leftover.py \
       tests/test_toy_leaderboard_honesty.py \
       tests/test_toy_particle_posture.py \
       tests/test_toy_mode_hold.py \
       tests/test_toy_late_collapse.py \
       tests/test_toy_keep_critic.py \
       tests/test_toy_lm_target.py \
       tests/test_toy_field_lift.py -q
```

One family at a time, with a tailable log:

```bash
python -m conceptmod.toys.shared_trajectory   # via the module's train(); see tests
python -m conceptmod.toys.residual_student --family
python -m conceptmod.toys.orbit_hold
python -m conceptmod.toys.locked_shared_floor
python -m conceptmod.toys.unipolar
python -m conceptmod.toys.ae_gan_hold
python -m conceptmod.toys.cover_leftover
python -m conceptmod.toys.leaderboard_honesty
python -m conceptmod.toys.particle_posture
python -m conceptmod.toys.mode_hold
python -m conceptmod.toys.late_collapse
python -m conceptmod.toys.keep_critic
python -m conceptmod.toys.lm_target
python -m conceptmod.toys.field_lift
```

`shared_trajectory` and `residual_student` are libraries (`train_locked` /
`train_drift`). `python -m conceptmod.toys.residual_student --family` prints
the residual board. The others print a one-line board when run as `__main__`.
