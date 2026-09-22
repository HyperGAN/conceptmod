# SPEC — loss DSL vs formulation toys

This note is the boundary for the current game cut. It is also the
list of stanza features that are **not implemented**. The parser will
reject unknown keys. Do not document them as if `game()` accepted them.

## What the loss DSL owns

- Phrase syntax and the parse tree (`conceptmod/dsl.py`): `++`, `--`,
  `=`, `#`, `%`, the `~` macro, `^`, `;`, `@`, `|`, `:alpha`,
  `:key=value`, `{random_prompt}`.
- The velocity loss for each live op (`conceptmod/ops.rule_loss`) and
  the sentences in `describe_phrase`.
- The 2-D job scoreboard ([dsl.md](../dsl.md)): which jobs are already
  recipes, which glyphs are dead on `TwoAxisBackend`, and the
  geometric words `right` / `needs help` / `recipe`.
- Compiling one phrase into a `Game` (`conceptmod/game.py`) whose
  student concept objective is those rules.

## What the formulation toys own

- The locked_shared stamp: `locked_adv_defaults()` /
  `locked_shared_floor.LOCKED`. RpGAN logistic (`GANLoss`,
  `loss_type=logistic`, `mode=rp`), `GradRegularizer` `b_cap`
  coeff=1 κ=1 norm=l2, FM off, demo cover 1.5, `n_particles=12`.
- Refusals of FM-on and thinned κ: `reject_unlocked`. The game calls
  that helper when a locked claim carries those knobs.
- Stranger-as-vanilla, cover, leftover, particle clouds, and
  `claim_pass`. A geometric `right` is not a formulation `PASS`.
- `feature_match` in `locked_shared_floor`. The drift arm of the game
  calls that function. It does not redefine the term.

`particlegan` supplies `GANLoss` and `GradRegularizer`. Conceptmod
does not vendor them.

## What a game step actually does

1. Critic: frozen vs trained CFG plane points for the fixture probes,
   same order if pairing is `locked_shared`, fake row flipped if
   pairing is `stranger`. Loss is `GANLoss.d_loss` + `GradRegularizer.penalty`.
2. Student: each rule's `alpha * rule_loss`, then `GANLoss.g_loss` on
   the updated critic. FM is added only when `fm_weight != 0`.

The critic batch is the probe list (`red`, `blue`, `stripe`, `dot`,
`red stripe`, empty). It is not a draw from `ParticlePrior`. The
stamp still records `n_particles=12` and cover `1.5` so a forked
default is visible in `score()["adv"]`. The step does not train them.

Student learning rate is the 2-D fixture budget. Critic learning rate
is `LOCKED.lr`. Neither replaces the stamp.

`score()["verdict"]` is `PASS` only for a locked claim whose modules
and saved logits match that stamp, with finite phrase and adversary
losses. Drift claims score `FAIL` with `stranger_pairing`, `fm_on`,
or `thinned_kappa`. A `PASS` is a CPU toy. It is not a Music or Anima
GPU transfer, and it is not the 40-step geometric verdict in
[dsl.md](../dsl.md).

## Not implemented

These are the next cuts. They are not accepted stanza keys today.

| Idea | Status |
|---|---|
| `player` blocks with a different phrase per player | not implemented — one phrase, and it is the student's |
| A critic phrase (the critic has no concept rules) | not implemented |
| `schedule` / lazy critic steps other than `reg_lazy` from the stamp | not implemented — `reg_lazy` stays 1 because the stamp says so |
| `cover` as a stanza knob | not implemented — cover stays on the toys |
| Sampling `n_particles` inside the game | not implemented |
| Vanilla logistic as a pairing name | not implemented — stranger here is the flipped `rp` batch, matching `locked_shared_floor` |
| GEM / EA selected by the stanza | not implemented — those hooks stay on `analysis_2d` / `ops_erase` |
| `^` and `;` inside a CPU game | refused — pixel has no `render` here; reward has no scorer |
| A backend other than `TwoAxisBackend` | not implemented — DiT training is still `train.py` plus `rule_loss`, not `Game` |
| Multi-critic or more than two players | not implemented |

When one of those lands, it has to call the toy stamp rather than
grow a second `LOCKED` inside `game.py`, and the catalog page has to
match the parser that actually shipped.
