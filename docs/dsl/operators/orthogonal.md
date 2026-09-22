# `a%b` orthogonal

Decorrelate the trained concept direction of `b` from the frozen
direction of `a`. Negative alpha aligns them (a blend). Only `b` is
trained.

**Category:** velocity · **Identity:** `orthogonal`

## Syntax

```text
a%b
a%b:2
a%b:-0.1
a%b:--0.1
```

Both sides are required. The extra leading `-` on an alpha (`--0.1`)
is accepted and means `-0.1`, matching the original parser.

`{random_prompt}` is legal on either side. `materialize` (and
`loss_game(..., random_prompt=)`) substitutes it. The placeholder is
not a loss by itself.

## Velocity target

```text
da = CFG_f(a) = v_f(a) − v_f("")
db = CFG_t(b) = v_t(b) − v_t("")
loss = orthogonal_scale * mean(|cos(da, db)|)
```

The value is non-negative. The trainer multiplies by `rule.alpha`, so
a negative alpha maximizes `|cos|` (align) instead of minimizing it.
The loss does not flip its own sign.

## Defaults

| Knob | Trainer `OpDefaults` | 2-D fixture and `loss_game` |
|---|---|---|
| `alpha` | `1` (decorrelate) | `1` |
| `orthogonal_scale` | `0.01` | `1` |

The fixture scale is larger so a non-zero cosine would actually move
the tiny field. On this field the cosine starts at 0, so the scale
does not matter: the gradient is zero either way. The game copies the
fixture pin. It does not define a third scale.

## Composition

`%` is one-directional. A mutual blend is two rules:

```text
red%stripe:-1|stripe%red:-1
```

A small negative `%` next to a write is a regularizer, not the remap:

```text
#:0.4|human=robot:0.8|robot%human:-0.1
```

That composite is the original-repo phrase. The human/robot words are
the historical example; the 2-D scoreboard uses color and pattern.

`%` does not project a mix onto one axis. `red%red stripe` inflates
the perpendicular component. The isolate job is the write
`red stripe=stripe`.

## 2-D verdict

[../../dsl.md](../../dsl.md): **right** no-op. The color and pattern
axes start orthogonal, so `|cos| = 0` and the gradient vanishes.
Negative `%` therefore cannot *create* a mix from a perpendicular
pair. Use `red=red stripe` for that job.

The SANA proof `anime%hyperrealistic:-3|hyperrealistic%anime:-3` is
the same op on concepts that are already correlated. That is a
different geometry from this fixture. Do not read the 2-D no-op as
"blend is broken."

## Failure modes

- Expecting `%` to subtract a concept the way a projection would. It
  minimizes absolute cosine, which can grow the part of `b` that was
  already perpendicular.
- Expecting negative `%` to mix `red` with `stripe` from a cold start
  on this fixture. It will not move.
- Leaving `orthogonal_scale` at `0.01` on a GPU run and wondering why
  a mild `|cos|` barely shows up. Raise alpha, or add a `#` anchor,
  before inventing a new glyph.

## Example

```python
from conceptmod.dsl import parse_phrase
rule = parse_phrase("red%stripe:-0.1")[0]
assert (rule.op, rule.a, rule.b, rule.alpha) == ("orthogonal", "red", "stripe", -0.1)
```
