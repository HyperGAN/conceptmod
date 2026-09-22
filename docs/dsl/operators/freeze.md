# `a#b` freeze

Pin the trained velocity at prompt `a` to the frozen model's velocity
at `b`.

**Category:** velocity · **Identity:** `freeze`

## Syntax

```text
a#b
a#a
a#b:0.5
#
#:0.4
```

No `guidance` key. The target is the frozen prediction itself, not a
scaled CFG step. Bare `#` is `a=""` and `b=""` (the empty prompt).

In a game stanza, write the operator without spaces (`stripe#stripe`).
A ` #` with a leading space is a comment, not a freeze. See
[../games.md](../games.md).

## Velocity target

```text
v_t(a) → v_f(b)     # both read in b's context
loss = MSE
```

Only `a` is trained. `b` is the frozen reference.

## Defaults

| Knob | Value |
|---|---|
| `alpha` | `1` |

There is no guidance default because the op has no guidance.

## Composition

Freeze is the retain you add beside an edit, not a substitute for it:

```text
red--|stripe#stripe
red--|stripe#stripe:0.5
```

`%` only trains its right-hand side; a half-weight `#` on that side
is how the orthogonal proofs kept anatomy while stripping a feature.
Several `c#c` rules in a row are still separate losses.
`describe_phrase` collapses a run of self-freezes into one sentence.

## 2-D verdict

[../../dsl.md](../../dsl.md): **right** as a retain. `stripe#stripe`
is the identity on the pattern axis. `red--|stripe#stripe` is
keep+erase and matches the preserve geometry of
`red--|#|stripe#stripe`. Bare `#` does not move this fixture
(`e("") = 0`).

Worked phrase: [../examples/keep_erase.md](../examples/keep_erase.md).

## Failure modes

- Freezing a prompt that shares tokens with the concept you are
  erasing can cancel the erase on a real text encoder. The 2-D axes
  are single tokens, so they do not show that collision.
- `#` does not orthogonalize. It copies a frozen velocity.
- Spaces around `#` inside `game("""...""")` start a comment and drop
  the rest of the line.

## Example

```python
from conceptmod.dsl import parse_phrase
keep = parse_phrase("stripe#stripe:0.5")[0]
assert (keep.op, keep.a, keep.b, keep.alpha) == ("freeze", "stripe", "stripe", 0.5)
assert parse_phrase("#")[0].a == ""
```
