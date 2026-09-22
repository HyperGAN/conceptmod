# Loss DSL

**Status.** The parser in `conceptmod/dsl.py` is the live phrase language.
The velocity truth table is [../dsl.md](../dsl.md) — a 2-D job scoreboard,
not a promise that every glyph wins on a GPU. This folder is the catalog,
the worked phrases, and a GAN-game layer that runs on the same 2-D fixture.
A formulation `PASS` from `loss_game` is a CPU toy. It is not a Music or
Anima GPU transfer.

Install from a checkout (Python 3.11+). `particlegan` is a dependency; the
game calls it for `GANLoss` and `GradRegularizer` and does not vendor them.

```bash
python -m pip install -e ".[dev]"
pytest tests/test_dsl.py tests/test_dsl_examples.py tests/test_game.py tests/test_dsl_catalog.py -q
```

## Write a phrase, parse it, see the loss, score it

`red=blue` is one write rule. The loss is `ops.rule_loss`: trained
velocity at `red` is pulled toward frozen `blue`. On the CPU fixture
that is a remap of the color axis.

```python
from conceptmod.dsl import describe_phrase, parse_phrase
from conceptmod.game import loss_game

phrase = "red=blue"
rules = parse_phrase(phrase)
print(describe_phrase(phrase))

match = loss_game(phrase)     # student vs critic, adv="locked_shared"
print(match)
row = match.score()           # one CPU step on the 2-D fixture
assert row["verdict"] == "PASS"
```

`parse_phrase` returns a `Rule` (`op="write"`, `a="red"`, `b="blue"`,
`alpha=1`). `describe_phrase` says this is a remap, not a global boost
of blue. `loss_game` builds a `Game`: the student minimizes that phrase
plus the locked RpGAN generator term; the critic minimizes RpGAN
logistic `d_loss` plus `GradRegularizer` `b_cap`. The stamp is
`locked_adv_defaults()` — RpGAN logistic, `b_cap` coeff=1 κ=1, FM off.
The step does not sample the toys' 12-particle cloud and does not apply
demo cover. Those fields are recorded so the game cannot quietly fork
them.

`row["verdict"] == "PASS"` means that wiring matched. It does not
re-score the geometric table in [../dsl.md](../dsl.md). Run that table
with:

```bash
pytest tests/test_dsl_jobs.py tests/test_2d_analysis.py -q
python scripts/analyze_2d.py --jobs --out outputs/2d_analysis
```

The same phrase as a stanza:

```python
from conceptmod.game import game

match = game("""
phrase red=blue
pairing locked_shared
claim locked
""")
```

`print(match)` is the shape table for this cut:

```text
Game  phrase='red=blue'  claim=locked  pairing=locked_shared
fixture  TwoAxisBackend  color=red/blue  pattern=stripe/dot
stamp  loss=logistic mode=rp arm=b_cap coeff=1 kappa=1 fm=0  source=locked_adv_defaults
recorded  n_particles=12 cover=1.5 posture=demo  (toys own these; this step does not sample a cloud)
applied  pairing=locked_shared kappa=1 fm=0
player   role       objective
student  generator  phrase rule_loss + RpGAN g_loss
critic   critic     RpGAN d_loss + GradRegularizer b_cap
index  op           a                 b                 alpha
0      write        red               blue              1
```

Stranger pairing, `fm_weight != 0`, and `kappa != 1` raise
`LockedClaimError` when the game claims locked. Built with
`claim="drift"` they take one step and score `FAIL`. Details:
[games.md](games.md). What the next stanza keys are not allowed to
pretend: [SPEC.md](SPEC.md).

## Where to read next

| | |
|---|---|
| Velocity truth table | [../dsl.md](../dsl.md) |
| Operator catalog | [operators/index.md](operators/index.md) |
| Worked phrases | [examples/bipolar.md](examples/bipolar.md), [keep+erase](examples/keep_erase.md), [remap](examples/remap.md), [replace](examples/replace_macro.md), [locked game](examples/locked_game.md) |
| GAN game | [games.md](games.md) |
| Add an operator | [ADDING_OPERATORS.md](ADDING_OPERATORS.md) |
| What this layer owns | [SPEC.md](SPEC.md) |
| 2-D pictures | [../2d-analysis.md](../2d-analysis.md) |
| Formulation stamp | [../formulation-toys.md](../formulation-toys.md) |

Glyphs the parser accepts: `c++`, `c--`, `a=b`, `a#b`, `a%b`, `a~b`,
`a^b`, `;c`, plus `:alpha` / `:key=value`, `|`, and `{random_prompt}`.
`@` is stripped and ignored. Do not add a synonym for a job the
scoreboard already calls a recipe.
