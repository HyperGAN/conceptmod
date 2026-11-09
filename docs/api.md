# Core API

The implementation checklist and verification record are in [plan.md](plan.md).

## Compile without loading

`compile_phrase(source, stage="model", capabilities=...) -> EditPlan` is pure.
Plans are immutable, serializable with `to_dict()` and validated on
`EditPlan.from_dict()`. Macro expansion preserves its signed coefficients.
Known options, finite values, stage compatibility and rendering support are
checked before loading. The parser can describe historical syntax without
claiming that the runtime implements it.

Model and encoder objectives are separate stages. Compile distinct plans and
compose sessions to perform both. No unsupported rule is silently skipped.

## Load an explicitly selected provider

`BackendRegistry().resolve(name)` returns a lightweight descriptor. Its
`capabilities` and `training_defaults` are available before model construction.
`provider.create(**product_options)` constructs a runtime and binds its declared
capabilities. See [providers.md](providers.md) for entry points and product homes.

Runtime contracts are narrow: velocity prediction, trainable encoding,
differentiable rendering, declared training state and artifact serialization.
`backend.original_view` and `backend.current_view` apply the same role to the
encoder and velocity model. Products own model-specific data layouts, loading
and device placement.

`training_modules()` includes current, encoder and any separate original modules.
`training_identity()` records non-tensor behavior which restoring a state dict
cannot reproduce. The base class includes artifact metadata and module graph
configuration; products must declare additional tokenizer, stochastic,
conditioning or numerical behavior. Changing behavior changes identity and is
not exact continuation. `invalidate_caches()` is called after encoder commits
and checkpoint loads.

## Execute and observe

```python
from conceptmod import compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession

plan = compile_phrase("red=blue", capabilities=backend.capabilities)
config = TrainingConfig(iterations=5, accumulation_steps=2, lr=0.01)
with TrainingSession(backend, plan, config) as session:
    session.step()                       # commits two microsteps
    session.save_checkpoint("saved.pt")
    session.run()                        # commits groups of two and one
    assert session.completed_updates == 3
```

`iterations` counts microsteps; `step()` commits at most `accumulation_steps`.
The final partial group is normalized by its actual size. `StepReport` contains
committed microstep/update counts, group size and detached scalar losses.
Observers are callables `(session, report)` invoked after commit. An observer exception does not undo
that update. `run(stop_when=...)` accepts an explicit caller decision function;
the default never selects a metric or stops early.

`prompt_source` is an explicit sequence of nonempty strings for model-stage
exaggeration and random-prompt rules. Core does not download prompt data.
Presentation, logging and scoring belong to observers/callers rather than loss
computation. `train_model()` and `train_encoder()` are convenience functions
which execute the same session API.

Session states are `ready -> running -> ready/complete`, with `failed` after an
unsuccessful optimization. Checkpoints are allowed only at committed boundaries.
A failed optimizer may have partly changed weights; restore a known checkpoint
before continuing. Closing clears owned gradients and prevents further work.
Callbacks do not run while an uncommitted update is in progress.

## Save state or export an edit

`state_dict()`/`save_checkpoint(path)` include the validated plan, effective
configuration, explicit backend identity, raw current/original tensors,
optimizer state, random streams, prompt sequence, reports and committed counts.
`load_checkpoint(path)` uses `weights_only=True`; it verifies the complete
checkpoint before applying it. Save files are atomically replaced. Restore into
the same declared architecture/configuration and continue missing updates;
do not replay a completed prefix.

`save_trained(directory)` exports all edited components for inference.
`save_transformer(path)` exports one component intentionally. Bundle details are
in [edit-bundles.md](edit-bundles.md). Core does not enable model/adapter/critic/
particle-bank weight EMA. Official ParticleGAN optimizer scalar statistics are
unchanged.

## Research observation

Shared geometry is in `conceptmod.geometry`; importing it does not import the
research suite. `OpDefaults.write_templates` and `erase_templates` are explicit
per-run policy; research never temporarily replaces global loss templates.

`metrics.capture_probes()` captures a detached `ProbeSet` once. Reuse that exact
set with `metrics.eval_phrase(..., probes=...)` to observe changes. Capturing
probes and evaluating them are distinct operations. Research `Game.score()`
observes a committed step; `step_and_score()` explicitly performs both actions.
See [observation.md](observation.md).
