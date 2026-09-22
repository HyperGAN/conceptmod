# Operator catalog

One page per glyph the parser actually builds. Semantics below are the
live `rule_loss` path. The 2-D column points at the scoreboard in
[../../dsl.md](../../dsl.md). Pages are written by hand against that
parser. Nothing here is generated, and nothing here is a second adv
recipe.

| Glyph | Page | Kind | 2-D fixture |
|---|---|---|---|
| `c++` | [exaggerate](exaggerate.md) | loss | **right** — bipolar slider |
| `c--` | [erase](erase.md) | loss | **right** — neutralize at `g=0` |
| `a=b` | [write](write.md) | loss | **right** — remap |
| `a#b` | [freeze](freeze.md) | loss | **right** — retain |
| `a%b` | [orthogonal](orthogonal.md) | loss | **right** no-op on perpendicular axes |
| `a~b` | [replace](replace.md) | macro, expands | **recipe**, not an independent win |
| `a^b` | [pixel](pixel.md) | loss | **dead** — `TwoAxisBackend` has no `render` |
| `;c` | [reward](reward.md) | parsed | **not implemented** |
| `@` | — | stripped | **ignored** before parse |

## Composition

Rules join with `|`. Each rule's loss is multiplied by `alpha` (default
1). A bare `:0.4` sets alpha. `:key=value` sets a float option
(`guidance` on `++`, `--`, and `=`). The concept text itself cannot
contain `:`; `sanitize_prompt` rewrites the characters the parser uses
as syntax before `{random_prompt}` is substituted.

```text
red--|stripe#stripe:0.5
red++:0.4:guidance=5
```

`a~b` is expanded before that split. It is not a fourth loss.

## What the 2-D game will not run

`loss_game` / `game` call `rule_loss` for every rule they keep.
`^` and `;` raise `GameError` on this fixture instead of pretending to
train. See [../games.md](../games.md).
