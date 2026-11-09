# Explicit observation and research state

Velocity metrics observe an explicit, reusable probe set. Capture it before the
first training update; reuse those tensors as weights change. Reusing only
seeds is insufficient because partial denoising itself depends on current model
weights.

```python
from conceptmod.metrics import capture_probes, eval_phrase
from conceptmod.ops import OpDefaults

cfg = OpDefaults()
probes = capture_probes(backend, cfg, seeds=(0, 1), stop_fracs=(0.35, 0.7))
# Perform explicitly requested training updates here.
report = eval_phrase(backend, "cat=dog", cfg, probes=probes)
```

`ProbeSet` owns detached copies of its latent and timestep tensors. Treat their
contents as read-only. `eval_phrase` never samples probes, updates weights, or
uses a backend-attached cache. Sampling and evaluation set directly exposed
PyTorch modules to evaluation mode first, then restore each module's previous
mode, including mixed child modes, on success or failure.

Swap ratio, hold, remain, and the `done` threshold flag are velocity-geometry
diagnostics. They do not rank candidate edits or prove intended effect in
finished outputs. Intended-effect comparisons use matched finished-output
intensity ON minus OFF, equally weighted across cases.

Template policy is local to an `OpDefaults` instance:

```python
cfg = OpDefaults(write_templates=("{}",), erase_templates=("{}",))
```

Templates are validated and normalized to tuples. The research two-axis fixture
passes this bare-word policy explicitly. It never temporarily replaces a module
global or changes another session's policy. `WRITE_TEMPLATES` and
`ERASE_TEMPLATES` remain immutable defaults, rather than mutable runtime controls.

The CPU `Game` is a research formulation fixture. `score()` observes its latest
committed step and raises before the first step. `step_and_score()` explicitly
performs one step and returns its score. A failed critic/student update does not
increment `steps_run` or append history. Because one player may already have
changed, the game enters a failed state; rebuild it before further stepping or
scoring. It does not silently replay the failed update.

Shared pole/leftover operations live in `conceptmod.geometry`, which imports no
research trainers. Existing `conceptmod.toys` geometry names alias those exact
functions and constants. Other research exports resolve lazily, so importing an
erase loss or a geometry helper does not load the research suite. Historical
formulation outcomes remain research evidence, rather than production ranking
objectives.
