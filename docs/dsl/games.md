# GAN games

A game is a phrase plus two players on the CPU 2-D fixture
(`TwoAxisBackend`, color `red`/`blue`, pattern `stripe`/`dot`).

| Player | Role | Objective |
|---|---|---|
| student | generator (the LoRA residual) | `Σ alpha * rule_loss(rule)` + RpGAN `g_loss` |
| critic | `PlaneCritic` on CFG plane points | RpGAN `d_loss` + `GradRegularizer` `b_cap` |

The phrase is the concept objective. RpGAN and `b_cap` are the
locked_shared adversary from `particlegan`, parameterized by
`locked_adv_defaults()`. Feature matching is added to the student
only when `fm_weight != 0`, and that build cannot claim locked.
The toys' `feature_match` is called as-is; it is not reimplemented
here.

```python
from conceptmod.game import game, loss_game

loss_game("red=blue")
game("""
phrase red=blue
pairing locked_shared
claim locked
""")
```

`score()` runs one step if you have not called `step()` yet, then
returns a row. A later `score()` re-reads the last step. Loop
`step()` yourself for a longer CPU run. Student Adam uses the 2-D
fixture learning rate (`analysis_2d.DEFAULT_LR`). Critic Adam uses
`LOCKED.lr`. Both are budget knobs, not a new recipe. Betas stay at
the Adam defaults.

## Stamp

`adv` must be `"locked_shared"`. The game copies
`locked_adv_defaults()` into `Game.stamp` and checks it again at
score time. Fields you do not see moving in the step:

| Stamp field | Locked value | What this game does with it |
|---|---|---|
| `loss_type`, `gan_mode` | `logistic`, `rp` | constructs `GANLoss` |
| `reg_arm`, `reg_coeff`, `reg_kappa`, `reg_norm` | `b_cap`, `1`, `1`, `l2` | constructs `GradRegularizer` |
| `fm_weight` | `0` | no feature-match term |
| `n_particles`, `cover_weight`, `cover_posture` | `12`, `1.5`, `demo` | recorded only — no cloud, no cover loss |

Pairing `locked_shared` feeds each frozen probe to the critic beside
the trained probe in the same order. Pairing `stranger` flips the
fake row and keeps `GANLoss` in `mode="rp"`. It does not switch the
loss to vanilla. Vanilla logistic is a different drift, owned by the
formulation toys, and this stanza does not offer it.

## Claim locked vs claim drift

| Build | Result |
|---|---|
| `pairing=locked_shared`, `fm_weight=0`, `kappa=1`, `claim=locked` | `score()["verdict"] == "PASS"` when the saved logits match a fresh `GANLoss` and the penalty matches a fresh `GradRegularizer` |
| `pairing=stranger` and `claim=locked` | `LockedClaimError`: stranger pairing refused |
| `fm_weight=0.1` and `claim=locked` | `LockedClaimError` from `reject_unlocked`: FM-on under `b_cap` refused |
| `kappa=0.2` and `claim=locked` | `LockedClaimError` from `reject_unlocked`: thinned `b_cap` refused |
| any of those drifts with `claim=drift` | one CPU step, `verdict == "FAIL"`, mismatch named `stranger_pairing`, `fm_on`, or `thinned_kappa` |
| `claim=drift` with no drift | `GameError` |
| `adv` other than `locked_shared` | `LockedClaimError` |

`^` and `;` are `GameError` before any step. See
[operators/pixel.md](operators/pixel.md) and
[operators/reward.md](operators/reward.md).

## What PASS is

Formulation wiring on this fixture. The row's `probe` is the color /
pattern measurement after that step so you can see the field. It is
not a `right` / `needs help` / `recipe` label. Those stay in
[../dsl.md](../dsl.md), which trains the phrase alone for 40 Adam
steps and does not run this critic.

A toy `PASS` is not a Music or Anima GPU transfer.

## Stanza keys that exist

`phrase`, `pairing`, `claim`, `fm`, `kappa`, `adv`, `seed`. A line
that starts with `#` is a comment. So is the tail after ` #`. Freeze
inside `phrase` is written `stripe#stripe` with no spaces.
Any other key is an error. That includes `player`, `schedule`, and
`cover` — they are named in [SPEC.md](SPEC.md) as not implemented,
and the parser will not accept them quietly.
