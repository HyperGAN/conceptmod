# Formulation toys (CPU scoreboard)

Home for the Wave-1 particle-sliders-style toy gates. They were opened on
[255BITS/ParticleGAN](https://github.com/255BITS/ParticleGAN) by mistake.
**#26 (shared-trajectory) and #27 (orbit radius hold) were merged there and are being reverted.** This tree keeps both families. Dropping them because they once landed on ParticleGAN would lose the gates. #28–#34 were closed unmerged and are ported from those PR heads. All nine live here.

| family | module | ParticleGAN source |
|---|---|---|
| shared slow→fast vs stranger | `conceptmod/toys/shared_trajectory.py` | [#26](https://github.com/255BITS/ParticleGAN/pull/26) (merged by mistake, being reverted; kept here) |
| orbit radius hold | `conceptmod/toys/orbit_hold.py` | [#27](https://github.com/255BITS/ParticleGAN/pull/27) (merged by mistake, being reverted; kept here) |
| locked_shared RpGAN + `b_cap` floor | `conceptmod/toys/locked_shared_floor.py` | [#28](https://github.com/255BITS/ParticleGAN/pull/28) |
| unipolar residual | `conceptmod/toys/unipolar.py` | [#29](https://github.com/255BITS/ParticleGAN/pull/29) |
| AE-GAN + locked hold | `conceptmod/toys/ae_gan_hold.py` | [#30](https://github.com/255BITS/ParticleGAN/pull/30) |
| cover / leftover / faithful teacher | `conceptmod/toys/cover_leftover.py` | [#31](https://github.com/255BITS/ParticleGAN/pull/31) |
| leaderboard honesty | `conceptmod/toys/leaderboard_honesty.py` | [#32](https://github.com/255BITS/ParticleGAN/pull/32) |
| tiny particle posture | `conceptmod/toys/particle_posture.py` | [#33](https://github.com/255BITS/ParticleGAN/pull/33) |
| mode-hold ring | `conceptmod/toys/mode_hold.py` | [#34](https://github.com/255BITS/ParticleGAN/pull/34) |

Adversarial primitives come from the `particlegan` package
(`GANLoss`, `GradientPenalty` which is `GradRegularizer`, `ParticlePrior`,
`ParticleRegularizer`, `get_recipe("ae_gan")`). This tree does not copy
`GradRegularizer`. The ring critic is `conceptmod.toys.mlp` because
ParticleGAN does not export the 100-Gaussians MLP on the installable package.

Culture reviewed: HyperGAN/particle-sliders
`analysis/slider2d/locked_baseline_defaults.py` (locked_shared / #94) and
this repo's `conceptmod/analysis_2d.py` (DSL geometry, a different fixture).

A PASS here is a CPU toy. It is **not** a Music or Anima GPU transfer.

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

## Run

```bash
pytest tests/test_toy_shared_trajectory.py \
       tests/test_toy_orbit_hold.py \
       tests/test_toy_locked_shared_floor.py \
       tests/test_toy_unipolar.py \
       tests/test_toy_ae_gan_hold.py \
       tests/test_toy_cover_leftover.py \
       tests/test_toy_leaderboard_honesty.py \
       tests/test_toy_particle_posture.py \
       tests/test_toy_mode_hold.py -q
```

One family at a time, with a tailable log:

```bash
python -m conceptmod.toys.shared_trajectory   # via the module's train(); see tests
python -m conceptmod.toys.orbit_hold
python -m conceptmod.toys.locked_shared_floor
python -m conceptmod.toys.unipolar
python -m conceptmod.toys.ae_gan_hold
python -m conceptmod.toys.cover_leftover
python -m conceptmod.toys.leaderboard_honesty
python -m conceptmod.toys.particle_posture
python -m conceptmod.toys.mode_hold
```

`shared_trajectory` is a library (`train_locked` / `train_drift`); the others
print a one-line board when run as `__main__`.
