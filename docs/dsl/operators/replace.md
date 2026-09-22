# `a~b` replace

A swap recipe. Not a loss, and not a fourth operator. The parser
expands it, then parses the three rules it already knows.

**Category:** macro · **Identity:** expands to `exaggerate`, `write`,
`orthogonal`

## Syntax

```text
a~b
a~b:0.35
```

`a` is the prompt you want to replace. `b` is what should stand there
instead. The optional float is `λ`, not an alpha on a `~` loss — there
is no `~` loss.

## Expansion

Default `λ = 0.1` (`DEFAULT_REPLACE_LAMBDA`):

```text
b++:2λ | a=b:4λ | b%a:-λ
```

So `red~blue` is exactly:

```text
blue++:0.2 | red=blue:0.4 | blue%red:-0.1
```

`parse_phrase` returns those three `Rule`s. `describe_phrase` does
**not** expand: it describes one swap, so a caption can say "replace"
without listing three losses. `_phrase_items` is what keeps that
split honest.

## Defaults

| Knob | Value |
|---|---|
| `λ` | `0.1` |
| exaggerate alpha | `2λ` (`0.2`) |
| write alpha | `4λ` (`0.4`) |
| orthogonal alpha | `−λ` (`−0.1`) |

Guidance on the expanded `++` and `=` rules is still those ops'
defaults. The macro does not set `:guidance=`.

## Composition

You can write the three rules yourself and add a freeze. `~` only
packages the common swap. It composes with `|` as a single stanza
piece: `#|red~blue` freezes the empty prompt and then expands the
macro.

## 2-D verdict

[../../dsl.md](../../dsl.md): **recipe, not independently right
here.** On a linear antipodal LoRA, `blue++` and `red=blue` are
opposite motions, so the macro fights itself. Use `=` or `++` alone
when you want a clean 2-D story. The game will still *run* the
expansion — `loss_game("red~blue")` steps three real rules under
locked_shared — and a formulation `PASS` on that step is not the
geometric verdict "right".

Worked phrase: [../examples/replace_macro.md](../examples/replace_macro.md).

## Failure modes

- Treating `~` as a loss you can weight independently of the three
  expansions. The float is `λ` and hits all three with the factors
  above.
- Expecting the 2-D quiver for `red~blue` to look like a clean remap.
  The scoreboard marks it `recipe` and leaves it off the quiver for
  that reason.
- Nesting `~` inside a concept that already expanded. The parser
  rejects a leftover `~`.

## Example

```python
from conceptmod.dsl import parse_phrase
rules = parse_phrase("red~blue")
assert [(r.op, r.a, r.b, r.alpha) for r in rules] == [
    ("exaggerate", "blue", "", 0.2),
    ("write", "red", "blue", 0.4),
    ("orthogonal", "blue", "red", -0.1),
]
```
