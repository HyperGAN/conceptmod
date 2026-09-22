# Replace macro

`red~blue` is not a loss. Default `λ = 0.1` expands to exaggerate
blue, write blue into red, and a small align of red toward blue.

On this fixture those first two terms oppose each other. The
scoreboard marks the macro **recipe**, not **right**. A formulation
`PASS` from `loss_game("red~blue")` only means the three expanded
rules stepped under locked_shared.

Source: [`examples/dsl/replace_macro.py`](../../../examples/dsl/replace_macro.py).
Scoreboard: [../../dsl.md](../../dsl.md). Operator:
[../operators/replace.md](../operators/replace.md).

## Phrase

```text
red~blue
```

## Parse tree (after expansion)

| op | a | b | alpha | options |
|---|---|---|---:|---|
| `exaggerate` | `blue` | | `0.2` | `{}` |
| `write` | `red` | `blue` | `0.4` | `{}` |
| `orthogonal` | `blue` | `red` | `-0.1` | `{}` |

```python
from conceptmod.dsl import parse_phrase
rules = parse_phrase("red~blue")
assert [r.op for r in rules] == ["exaggerate", "write", "orthogonal"]
assert [r.alpha for r in rules] == [0.2, 0.4, -0.1]
```
