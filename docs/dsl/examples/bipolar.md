# Bipolar slider

`red++` is the signed scale between red and blue on this fixture.
Classic exaggerate stretches the color axis; the antipode moves the
other way because the LoRA is a linear map. `blue++` is the opposite
sign. There is no extra slider glyph.

Source: [`examples/dsl/bipolar.py`](../../../examples/dsl/bipolar.py).
Scoreboard: [../../dsl.md](../../dsl.md) (**right**). Operator:
[../operators/exaggerate.md](../operators/exaggerate.md).

## Phrase

```text
red++
```

## Parse tree

| op | a | b | alpha | options |
|---|---|---|---:|---|
| `exaggerate` | `red` | | `1` | `{}` |

```python
from conceptmod.dsl import parse_phrase
rule = parse_phrase("red++")[0]
assert rule.op == "exaggerate" and rule.a == "red" and rule.alpha == 1.0
```
