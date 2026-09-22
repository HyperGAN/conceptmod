# Remap

`red=blue` trains red to behave like blue. Blue prompts are not
boosted. That global half of a swap is `++` or the `~` macro, not
this rule.

Source: [`examples/dsl/remap.py`](../../../examples/dsl/remap.py).
Scoreboard: [../../dsl.md](../../dsl.md) (**right** for `red=blue`).
Operator: [../operators/write.md](../operators/write.md).

## Phrase

```text
red=blue
```

## Parse tree

| op | a | b | alpha | options |
|---|---|---|---:|---|
| `write` | `red` | `blue` | `1` | `{}` |

```python
from conceptmod.dsl import parse_phrase
rule = parse_phrase("red=blue")[0]
assert (rule.op, rule.a, rule.b) == ("write", "red", "blue")
```
