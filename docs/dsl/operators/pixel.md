# `a^b` pixel

Pixelwise L2 between a render of `a` and a frozen render of `b`, plus
a velocity freeze-anchor so `b` does not drift through shared weights.

**Category:** velocity + pixels · **Identity:** `pixel`

## Syntax

```text
a^b
a^b:0.5
```

Both concepts are required. A one-sided `^` raises `DSLError`.

## Velocity target

On a backend that implements `render`:

```text
image loss = MSE(render(a), render_frozen(b))     # same start noise
anchor     = pixel_anchor_weight * MSE(v_t(b), v_f(b))
loss       = image loss + anchor
```

Gradients flow through the last `pixel_grad_steps` Euler steps and the
VAE decode. The anchor is there because token overlap would otherwise
move `b`.

## Defaults

| Knob | `OpDefaults` |
|---|---|
| `alpha` | `1` |
| `pixel_render_steps` | `10` |
| `pixel_grad_steps` | `1` |
| `pixel_anchor_weight` | `20` |
| `sample_guidance` | `4.5` (the render CFG) |

No `guidance` option on the rule. Strength is alpha, the anchor
weight, the learning rate, and how many steps carry gradient.

## Composition

`^` is a full sample, so it is expensive next to a stack of velocity
rules. The GPU notes in the README keep it on a short run (low lr,
few iterations) with the built-in anchor. Adding an explicit `b#b`
on top double-counts the anchor.

## 2-D verdict

[../../dsl.md](../../dsl.md): **dead here.** `TwoAxisBackend` has no
`render`. The fixture is a velocity field, not pixels. `rule_loss`
would call `backend.render` and fail. `loss_game` / `game` refuse the
rule up front with `GameError` instead of starting a step.

This is not "pixel is unimplemented." `ops.rule_loss` has the branch.
The CPU game and the 2-D scoreboard cannot execute it.

## Failure modes

- Calling `loss_game` on a `^` phrase. Refused.
- Turning the anchor weight down and watching `b` wash out with `a`.
- Judging a 2-D job table by whether `^` moved `red`. It cannot.

## Example

```python
from conceptmod.dsl import parse_phrase
from conceptmod.game import GameError, loss_game

rule = parse_phrase("a painting of a house^a photo of a house")[0]
assert rule.op == "pixel"
try:
    loss_game("a painting of a house^a photo of a house")
except GameError as exc:
    assert "render" in str(exc)
```
