# Adding a loss operator

A new glyph is a parser arm plus a `rule_loss` branch plus a catalog
page plus a 2-D row. It is not a new adversarial recipe. Pairing,
`b_cap`, feature matching, cover, and the particle cloud stay in
`conceptmod.toys` (`locked_adv_defaults`, `reject_unlocked`). Call
those. Do not copy `GANLoss` or `GradRegularizer` into conceptmod, and
do not retune κ, FM, or the demo cover inside the op.

Pages in `docs/dsl/operators/` are written by hand. This repo does not
generate them from a decorator.

## Before you add a glyph

Read [docs/dsl.md](../dsl.md) "Jobs that look missing — already recipes".
If the job is already `c--`, `a=b`, `c++`, or `c--|k#k`, do not add
`c!!`, `+`, or a keep+erase token. The tests in
`tests/test_dsl_jobs.py` exist to stop that.

Also check the dead list. `;` stays unimplemented until there is a
real scorer. `^` already has a `rule_loss` branch; the 2-D gap is the
missing `render`, not a missing parser arm.

## Files

| File | What changes |
|---|---|
| `conceptmod/dsl.py` | One arm in `_parse_concept`, in the same order the original dispatcher used. A constant next to `EXAGGERATE`. `describe_phrase` / `_describe_rule` get one sentence that matches the loss. |
| `conceptmod/ops.py` | One branch in `rule_loss`. Defaults live on `OpDefaults`, not as literals scattered through the branch. |
| `tests/test_dsl.py` | The string parses, and the strings that must fail still fail. |
| `tests/test_ops.py` | The velocity target on a tiny backend, including the default guidance. |
| `docs/dsl/operators/<name>.md` | Syntax, velocity target, defaults, composition, 2-D verdict, failure modes, a snippet that calls the live parser. |
| `docs/dsl/operators/index.md` | One row. If the op cannot run on `TwoAxisBackend`, say so. |
| `docs/dsl.md` | One scoreboard row after you have actually run the 2-D fixture. `right` / `needs help` / `recipe` / `dead`. Do not write the verdict first. |
| `conceptmod/game.py` | Only if the new op can run on `TwoAxisBackend`. If it cannot, refuse it in `loss_game` the way `^` and `;` are refused. Do not stub a fake loss. |

`conceptmod/toys/` is not in this list. A loss op does not get its
own `gan_mode`, κ, or cover weight.

## Checklist

1. Parse the glyph in `dsl.py`. Keep `|`, `:alpha`, and `:key=value`
   working. New options are floats, same as `guidance`.
2. Implement `rule_loss` with the frozen/trained velocity notation
   already used in `ops.py` (`v_f` / `v_t`, CFG as `v(p) − v("")`).
3. If the 2-D fixture can express it, add a job in
   `conceptmod/analysis_dsl.py` that goes through `rule_loss` and
   extend [dsl.md](../dsl.md) from the run. If it cannot, say **dead
   here** and why (missing `render`, missing scorer, uncond embedding
   is zero). Do not invent a probe that hides the gap.
4. Write the catalog page from the code you just ran, including the
   default that `OpDefaults` actually ships.
5. Decide whether `loss_game` can step it. Refuse with `GameError`
   when the fixture cannot. A formulation `PASS` must not be
   reachable for a loss that never ran.
6. `pytest tests/test_dsl.py tests/test_ops.py tests/test_dsl_jobs.py tests/test_game.py tests/test_dsl_catalog.py -q`

GPU proofs (SANA, Z-Image, Anima, Krea) are a separate run. A green
2-D job is not that run. Say so on the page.

## What you do not fork

- `locked_adv_defaults()` — the game and the 2-D suite already call
  it. A new op inherits that stamp by running under `loss_game` or
  under `analysis_2d`. It does not take an `adv=` recipe of its own.
- `reject_unlocked` — stranger pairing, FM-on, and thinned `b_cap`
  are already refused when a game claims locked. Do not re-encode
  those checks inside the op.
- Erase variants GEM / EA — `conceptmod/ops_erase.py`. They are not
  phrase glyphs. Wiring a new erase mode means extending that hook
  and the 2-D method list, not a new token, and still not a new adv
  stamp.
