# Locked game

One remap, compiled into a student-versus-critic step. The phrase is
the student's concept objective. The critic is RpGAN logistic plus
`b_cap`, stamped from `locked_adv_defaults()`.

Source: [`examples/dsl/locked_game.py`](../../../examples/dsl/locked_game.py).
Game layer: [../games.md](../games.md).

## Stanza

```text
phrase red=blue
pairing locked_shared
claim locked
```

## What one score call checks

```python
from conceptmod.game import game

match = game("""
phrase red=blue
pairing locked_shared
claim locked
""")
row = match.score()
assert row["verdict"] == "PASS"
assert row["rules"][0]["op"] == "write"
```

`PASS` means the step's adversary matched the locked_shared modules
(same-order pairing, FM off, κ=1). It is a CPU toy on the 2-D fixture.
It is not a geometric re-run of the write quiver, and it is not a
Music or Anima GPU transfer.

The recorded `n_particles=12` and demo cover `1.5` stay on the stamp.
This step does not sample that cloud.
