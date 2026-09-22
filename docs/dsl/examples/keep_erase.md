# Keep + erase

Neutralize red and pin stripe to the frozen model. Two live ops, one
phrase. The longer preserve recipe `red--|#|stripe#stripe` adds an
empty-prompt freeze that does not move this fixture.

Source: [`examples/dsl/keep_erase.py`](../../../examples/dsl/keep_erase.py).
Scoreboard: [../../dsl.md](../../dsl.md) (keep+erase / preserve,
**right**). Operators: [erase](../operators/erase.md),
[freeze](../operators/freeze.md).

## Phrase

```text
red--|stripe#stripe
```

## Parse tree

| op | a | b | alpha | options |
|---|---|---|---:|---|
| `erase` | `red` | | `1` | `{}` |
| `freeze` | `stripe` | `stripe` | `1` | `{}` |

```python
from conceptmod.dsl import parse_phrase
rules = parse_phrase("red--|stripe#stripe")
assert [r.op for r in rules] == ["erase", "freeze"]
```
