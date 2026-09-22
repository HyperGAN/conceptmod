# `c--` erase

Neutralize concept `c`: prompts that ask for it match the empty
prompt. Default guidance does **not** write the opposite concept.

**Category:** velocity · **Identity:** `erase`

## Syntax

```text
c--
c--:0.5
c--:guidance=0
c--:guidance=1
```

`c` is required. A bare `--` raises `DSLError`.

## Velocity target

ESD, in a templated context `c'` (the GPU trainer samples
`ERASE_TEMPLATES`; the 2-D fixture and `loss_game` pin that list to
`"{}"` so the story is the axis, not the template):

```text
v0 = v_f("")          # in context c'
vc = v_f(c')
v_t(c') → v0 − g (vc − v0)
loss = MSE
```

`g` is `options["guidance"]` if set, otherwise
`OpDefaults.erase_guidance`.

## Defaults

| Knob | Value | Meaning |
|---|---|---|
| `alpha` | `1` | scales the MSE |
| `guidance` | `0` | neutralize: target is `v0` |
| `guidance=1` | phrase override | classic ESD overshoot, `−CFG(c)` |

`describe_phrase("red--")` says "Neutralize". `describe_phrase("red--:guidance=1")`
says ESD overshoot. There is no `c!!` or `c/` synonym.

GEM and EA are not this glyph. They are `ops_erase.erase_loss`, selected
by `analysis_2d` via `erase_mode`, not by the phrase. `loss_game` always
uses `rule_loss` for `--`.

## Composition

Keep+erase is already `|`:

```text
red--|stripe#stripe
red--|#|stripe#stripe
```

`#` on the empty prompt is a no-op on this fixture (`e("") = 0`). The
keep pin is the rule that names the concept you want held.
See [freeze](freeze.md) and [../examples/keep_erase.md](../examples/keep_erase.md).

## 2-D verdict

[../../dsl.md](../../dsl.md): **right.** Bare `red--` lands on the
origin. It does not write blue. `red--:guidance=1` matches `red=blue`
on this antipodal field, because `−CFG(red) = CFG(blue)`. That
identity is the fixture, not a reason to add an operator.

## Failure modes

- Treating default `--` as "write the antipode" is the old `g=1`
  reading. The live default is `g=0`.
- A freeze whose prompt shares tokens with `c` can suppress the erase
  on a real encoder. Pick a token-disjoint keep.
- `;` is not an erase and is not implemented ([reward](reward.md)).

## Example

```python
from conceptmod.dsl import parse_phrase
bare = parse_phrase("red--")[0]
over = parse_phrase("red--:guidance=1")[0]
assert bare.options == {}
assert over.options["guidance"] == 1.0
```
