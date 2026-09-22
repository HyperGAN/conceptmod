# `a=b` write

Remap prompt `a` so its velocity behaves like concept `b`. One loss.
It does not turn `b` up everywhere.

**Category:** velocity · **Identity:** `write`

## Syntax

```text
a=b
a=b:0.8
a=b:guidance=2
=b
b=
```

The target `b` is required. A bare `=` raises `DSLError`. `=b` and
`b=` both write `b` into the empty prompt (`a=""`).

## Velocity target

In `b`'s sampling context. When `a` is non-empty the GPU trainer wraps
`a` and `b` in one shared template from `WRITE_TEMPLATES`. The 2-D
fixture and `loss_game` pin that list to `"{}"`.

```text
v0 = v_f("")
vb = v_f(b)
v_t(a) → v0 + g (vb − v0)
loss = MSE
```

`g` is `options["guidance"]` if set, otherwise
`OpDefaults.write_guidance`.

## Defaults

| Knob | Trainer `OpDefaults` | 2-D fixture and `loss_game` |
|---|---|---|
| `alpha` | `1` | `1` |
| `guidance` | `2` (overshoot past an exact copy) | `1` (exact remap) |

The game uses the fixture pin on purpose. It does not invent a third
guidance. GPU training still constructs `OpDefaults()` unless a script
overrides it, so a DiT run of `red=blue` uses guidance 2 unless you
set `:guidance=`.

## Composition

`=` does not boost `b` on prompts that never said `a`. The swap recipe
that also exaggerates `b` and lightly aligns is `a~b`
([replace](replace.md)), or the same three rules written out.

Mix on this fixture is `red=red stripe` (write toward the concatenated
embedding). Isolate is `red stripe=stripe`. Both are writes. A `+`
glyph would be a synonym here; it is not a separate op.
See [../../dsl.md](../../dsl.md).

## 2-D verdict

[../../dsl.md](../../dsl.md): **right** for `red=blue`. Trained red
aligns with frozen blue; stripe hold stays high.

Uncond `=b` cannot move this fixture: `e("")` is the zero vector, so
the shared LoRA has nothing to grab. That is a fixture limit, not a
parser bug. Worked phrase:
[../examples/remap.md](../examples/remap.md).

## Failure modes

- Expecting `cat=dog` to boost dogs that were never cats. It will not.
  Use `~` or add `dog++`.
- `guidance` left at the trainer default (2) is a stronger write than
  the 2-D pictures, which use 1.
- Prompts that contain `=`, `:`, `|`, `#`, `%`, `^`, or `~` must go
  through `sanitize_prompt` before they are pasted into a rule.

## Example

```python
from conceptmod.dsl import parse_phrase
rule = parse_phrase("red=blue")[0]
assert (rule.op, rule.a, rule.b) == ("write", "red", "blue")
assert parse_phrase("=blue")[0].a == ""
```
