# Conceptmod core and model-package migration

This is the implementation and acceptance plan for the architecture review of
`ea3422450152eb42fd09ae0a36964b6fe0af78e8` (2026-11-09). Keep this document
current as changes merge. The owner authorized parallel implementation, fixes
first, and discretionary merges. A completed phase requires working code and
verification, not just an interface or a passing mock.

## Intended result

Conceptmod is the model-independent language and execution library for concept
editing. Following the separation used by HyperGAN/particle-sliders, pretrained
model architectures, Hub IDs, tokenizer/latent conventions, loaders, memory
placement and model-specific adapter mappings live in separately named product
repositories. Installing/importing core does not load model libraries, download
weights or datasets, select a GPU, or change process-wide numerical settings.

The public path is:

```
phrase -> validated EditPlan -> TrainingSession -> complete EditBundle
                 |                   |
          backend capabilities    explicit observers / evaluation reports
```

Research formulation toys remain available in this repository as research, not
as a dependency of the production runtime. Historical evidence stays intact in
Git and remains discoverable. No scientific results or completed training are
replayed to prove a packaging change.

## Phase 1 — merge concrete correctness fixes

- [x] Export all edited components (encoder and transformer) together, including
  effective adapter configuration and base-model identity; verify a CPU
  roundtrip. Keep resumable training state distinct from inference artifacts.
- [x] Anima's original-model generation disables both encoder and transformer
  adapters, with exception-safe restoration.
- [x] Commit the final partial accumulation group with correct normalization;
  handle early stop, invalid budgets and stale gradients explicitly. Count
  microbatches separately from committed optimizer updates.
- [x] Keep particle routing valid through activation-checkpoint backward
  recomputation and restore context on failure. Preserve frozen host weights and
  the official ParticleGAN optimizer/checkpoint formula.
- [x] Give StepContext ownership of gradient-cache invalidation; stop editing
  private dictionaries from trainers.
- [ ] Verify raw-current model/adapter/critic/prior weights and explicitly disable
  weight EMA. Do not change the upstream scalar optimizer running statistics.

## Phase 2 — define and use the core contracts

- [ ] Compile phrases once into immutable, serializable EditPlans. Validate
  finite option values, known option names, supported operators, stage policy
  and backend capabilities before expensive loading. Unsupported rules cannot
  silently disappear. Preserve documented macro semantics.
- [ ] Define backend capabilities and narrow protocols for velocity prediction,
  trainable text encoding, differentiable rendering and artifact I/O. Distinguish
  original/current model views consistently across encoder and transformer.
- [ ] Introduce a TrainingSession that owns optimizer state, RNG state, progress,
  accumulation and lifecycle. Expose committed-update checkpoints and exact
  continuation; an exception cannot report a completed update. Restore modes
  and clear transient graphs safely.
- [ ] Inject prompt sources. No hidden Hub dataset downloads from training.
- [ ] Separate evaluation from stepping, stopping and candidate selection.
  Game.score must observe state; an explicitly named convenience method may
  step-and-score. Freeze probe tensors for repeated diagnostics.
- [ ] Move samples, grids and logging to explicit observers; train code does not
  classify fruit prompts, write presentation files or call CUDA housekeeping.
- [ ] Replace module-global template mutation with per-plan/per-step policy.
- [ ] Complete EditBundle save/load with encoder+transformer+manifest, effective
  configuration, raw-weight provenance and safe loading. Invalid/incomplete
  bundles fail before partially mutating a runtime.

For intended-effect/LR decisions, use the equally weighted mean of matched
finished-output intensity ON minus OFF. Velocity SSE, preservation drift, beauty
and formulation PASS remain diagnostics and do not become ranking gates or
tiebreaks. Core supplies reports and explicit policy hooks rather than silently
choosing a scientific objective.

## Phase 3 — extract model products and remove model specifics from core

- [ ] Inspect particle-sliders' actual package/product contracts and adopt the
  useful boundary, without copying unrelated implementation.
- [ ] Discover installed model providers using one documented entry-point group;
  support explicit local registration for embedding and tests. Static provider
  metadata permits preflight without importing heavyweight model modules.
- [ ] Extract SANA, Z-Image, Anima, Krea, Qwen-Image, FLUX.2 Klein and Supra into
  independently installable, properly named repositories/packages. Proposed names
  are `sana-conceptmod`, `zimage-conceptmod`, `anima-conceptmod`,
  `krea2-conceptmod`, `qwen-image-conceptmod`, `flux2-klein-conceptmod` and
  `supra-conceptmod`; verify existing organization names before creating them.
  Each owns its model dependencies, loader, defaults, original/current view,
  serialization implementation, relevant tests and model-specific docs.
- [ ] Preserve the small CPU fixture for core contract tests without making it a
  pretrained product or coupling core installation to PEFT/diffusers.
- [ ] Extract model-format converters into the relevant product or a separately
  installable conversion tool. Preserve Music/YuE conversion functionality and
  evidence; remove conversion mappings and deployment strength policy from core.
- [ ] Replace bundled backend selection and model-specific root CLI flags with a
  provider-aware core CLI/API. Missing products yield actionable install guidance;
  no compatibility shim may keep pretrained implementations bundled in core.
- [ ] Remove heavyweight model dependencies and floating Git dependencies from
  core packaging. Build/install/import core without those extras.
- [ ] Update README, examples and model documentation links to the new homes;
  preserve historical proofs through clearly labeled archive links.

## Phase 4 — refactor remaining dependencies and polish

- [ ] Extract shared geometry/formulation definitions from the eager toy package;
  runtime imports cannot import the whole research suite. Keep toys usable and
  distinguish their historical objective from production training.
- [ ] Consolidate duplicated sampler/adapter/cache lifecycle code at real shared
  boundaries, keeping model-specific math in model products.
- [ ] Remove dead/unwired runtime paths, implicit unsupported operators and
  obsolete compatibility layers; document intentional breaking changes and the
  migration from 2.x.
- [ ] Make tests run with the default command; remove cross-test fixture import
  fragility, ensure optional dependency tests skip explicitly, and add CI for
  core plus product contract compatibility.
- [ ] Verify clean builds, installed-package smoke examples, safe checkpoint and
  bundle loading, exact continuation, deterministic independent sessions,
  exception cleanup and provider isolation. Update release/version metadata and
  public API documentation to the resulting behavior.

## Merge and completion evidence

Use small reviewed commits/PRs. Merge the correctness fixes before the breaking
architecture work. Root coordinates worktrees and integration; workers do not
overwrite one another's branches. Run focused checks for each change, then the
appropriate complete core/product suites and isolated wheel tests. CPU fixtures
prove contracts, not pretrained model quality. GPU/model downloads are not
required for this code migration and are not a reason to replay existing work.

Record merged PRs/commits, published product locations, exact test commands and
results below. Final completion requires every item above to be implemented or
explicitly reconciled with the owner request and verified against current code.
Do not call scaffolding, pending repositories, or unmerged work complete.

## Execution record

- 2026-11-09: Review baseline matches origin/main `ea34224`. Reproduced missing
  encoder export, partial accumulation tail, activation-checkpoint backward
  failure, Anima original-view encoder contamination and implicit score update.
  Focused baseline: 48 CPU tests pass with `--import-mode=importlib`; default
  collection fails importing `tests.test_ops` in the existing environment.
- 2026-11-09: Parallel fix workers started for training state, complete exports
  and reference views, and particle routing. Root owns planning, integration,
  provider architecture and extraction coordination.

- 2026-11-09: Correctness integration includes training state `ee94411`, particle
  routing `bc3a7b7`, complete exports/reference `9a8a0c7`, and unambiguous test
  imports. Integration regression command: existing CPU environment, CUDA hidden,
  two threads, `python -m pytest -q -p no:cacheprovider` over model-train state,
  particle API/LoRA, bundles, Anima reference, CPU sample, Supra, Klein, operators,
  metrics, game and 2-D analysis: **136 passed**. Full pre-export integration suite:
  **473 passed, 6 skipped, 3 failed**. The two unchanged research assertions
  (mode-hold PASS and shared-trajectory PASS) also fail on the pre-fix baseline;
  the optional ComfyUI checkout lacks `comfy_aimdo.host_buffer`. These are recorded
  separately and remain work for the research/tool boundary; no thresholds were
  changed to hide them.
