# `;c` reward

Parsed so old phrases do not crash the splitter. Not a loss.

**Category:** unimplemented · **Identity:** `reward`

## Syntax

```text
;c
;a bright sunset
```

The text after `;` is stored on `Rule.a`. Options after `:` still
parse. Nothing reads them.

## Velocity target

None. `ops.rule_loss` raises `NotImplementedError`. `model_train`
raises the same error before the training loop if any rule has this
op. `loss_game` / `game` raise `GameError` and do not step.

There is no scorer, no default guidance, and no 2-D number.

## Defaults

| Knob | Value |
|---|---|
| implementation | absent |
| alpha | parsed, unused |

## Composition

A phrase that mixes `;` with a live rule still parses
(`red++|;a bright sunset` is exaggerate plus reward). The CPU game
rejects the whole phrase. The GPU trainer rejects it too. Do not
ship a phrase that contains `;` and expect the other rules to run.

## 2-D verdict

[../../dsl.md](../../dsl.md): **parsed, not implemented.** It needs a
GPU scorer (the original ImageReward hook). Leaving it unimplemented
is the current decision, not a forgotten branch.

## Failure modes

- Calling `rule_loss` on a reward rule. `NotImplementedError`.
- Calling `loss_game` on a phrase that contains `;`. `GameError`.
- Documenting a score, a default, or a 2-D arrow for this glyph.

## Example

```python
from conceptmod.dsl import parse_phrase
from conceptmod.game import GameError, loss_game

rule = parse_phrase(";a bright sunset")[0]
assert rule.op == "reward"
assert rule.a == "a bright sunset"
try:
    loss_game(";a bright sunset")
except GameError as exc:
    assert "not implemented" in str(exc)
```
