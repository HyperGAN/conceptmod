# `c++` exaggerate

Push concept `c` past the frozen model's own direction so generations
show more of it.

**Category:** velocity · **Identity:** `exaggerate`

## Syntax

```text
c++
c++:0.4
c++:guidance=5
c++:0.4:guidance=5
```

`c` is required. A bare `++` raises `DSLError`.

## Velocity target

Classic path (`StepContext.probe is None`), which is what the 2-D
fixture and `loss_game` run:

```text
v0 = v_f("")
vc = v_f(c)          # both in the empty-prompt context
v_t(c) → v0 + g (vc − v0)
loss = MSE
```

`g` is `options["guidance"]` if the phrase set it, otherwise
`OpDefaults.exaggerate_guidance`.

Probe path, used by `model_train` when a random prompt is drawn: with
probability `probe_p` (default **0.7**) the step trains an unrelated
prompt `p` instead of `c`:

```text
v_t(p) → v_f(p) + g (v_f("p, c") − v_f(p))
```

That is the globalization the GPU proofs relied on. The 2-D fixture
does not set `probe`, so `red++` here is the classic target only.

## Defaults

| Knob | Value | Where |
|---|---|---|
| `alpha` | `1` | parser |
| `guidance` | `3` | `OpDefaults.exaggerate_guidance` and the 2-D fixture |
| `probe_p` | `0.7` | GPU trainer only; off on the 2-D fixture and in `loss_game` |

Alpha multiplies the MSE after `rule_loss` returns. Negative alpha
flips the step.

## Composition

`++` trains the concept prompt (classic) or the probe (GPU). It does
not pin other concepts. Pair it with `#` when a neighbor must stay
put. It does not remap a different word onto `c`; that is `=`.

## 2-D verdict

[../../dsl.md](../../dsl.md): **right.** `red++` moves color from +1
to about +3. Stripe holds. Because the residual is a linear map of
antipodes, `blue` goes to about −3. That *is* the bipolar slider —
`blue++` is the other sign. Not a missing operator.
Worked phrase: [../examples/bipolar.md](../examples/bipolar.md).

## Failure modes

- Empty concept (`++`) is a parse error.
- Reading the probe formula into a 2-D result overstates what this
  fixture measured.
- On a real DiT, classic `++` without the probe stays local to the
  literal prompt. That is a trainer choice (`model_train`), not a
  different glyph.

## Example

```python
from conceptmod.dsl import parse_phrase
rules = parse_phrase("red++")
assert rules[0].op == "exaggerate"
assert rules[0].a == "red"
assert rules[0].options == {}
```
