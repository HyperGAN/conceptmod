# Opt-in shared game: CPU conformance and native backend gates

This preview implements a real shared-game route for the repository's tiny
`CpuBackend`. Its two PEFT LoRA matrices execute through actual HyperGAN-built
HNDL, its Conv2d critic executes the shared HNDL graph, and
`ConditionalPG094` owns the GAN loss, public BISPF penalty, optimizer and
updates. Shared `SliderSession` owns checkpoints, EMA, evaluation and export;
shared native inference loads the adapter without constructing a game, critic
or prepared training rows. This is the same `conceptmod` distribution with a
`modern` optional dependency extra, intended for a new environment.

TinyFlowDiT is an affine MLP: its flat 256-coordinate output is reshaped into
`[B,4,8,8]`. The critic itself has 1x1 and 3x3 Conv2d layers followed by
FiLM, spatial mean and a linear readout. Native
field shape here does not qualify an image model or learned spatial geometry.

This is a **distinct cohort using the existing shared GAN objective**. It does not transparently
migrate `model_train.train_model`'s DSL `ops.rule_loss`, `encoder_train`'s
embedding MSE/cosine/curriculum, or `game.py`'s 2-D locked_shared RpGAN/b_cap
and local PlaneCritic. Those files and all eight historical backend source
files remain unchanged. The modern CPU fixture admits exactly `red=blue` at
DSL rule weight 1. Its two rows use slider alpha 0.25 and 1 with the same native
latent and raw timestep 500. Shared CFG computes both sides:
`teacher = frozen red + alpha*(frozen blue - frozen red)` and
`student = frozen red + alpha*(adapted red - frozen red)`.
The critic condition is that actual alpha, derived from each response row.
Only `linear2` is adapted, so the class-path residual is independent of latent
and time. Different z/t rows would duplicate the residual; no timestep- or
latent-conditional learning is claimed. It uses the shared conditional residual
GAN objective, not the historical write-rule CFG squared error. There is no
new local GAN mathematics or optimizer loop. The critic is initialized with
CPU seed `seed + 1` in a reversible RNG scope, independently of caller RNG.

## Installation and reviewed source boundary

Use a new environment. Keep a working historical environment/checkpoint with
its original dependency versions. In particular, PG 0.9.4 does not export the
historical `GradRegularizer`, so importing the old locked game in the modern
environment is unsupported. Do not reinterpret an old PASS, state dict or
PEFT artifact as a modern checkpoint.

```sh
python3.12 -m venv .venv-modern
.venv-modern/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -c requirements/modern-cpu-constraints.txt '.[modern]'
```

The unchanged base dependencies include unpinned diffusers Git HEAD and other
model packages. The modern extra adds a compatible exact shared source
constraint, rather than creating an old-core/new-core resolver conflict.
The installed clean CPU test environment resolved diffusers to
`86ef420b17363c28e36ebad6b58555e9aa8feda2`; this observation does not make the
historical default install reproducible. Native model imports or downloads
are not needed by this CPU path.

The clean CPU dependency solve and install also succeeded with uv. However,
uv stamped its exact HyperGAN VCS build metadata as `dirty: true`; the shared
gate rejected that self-reported flag. A comparison of the retained uv build
and pip build found every HyperGAN Python source file byte-identical despite
the differing flags. `dirty` is self-reported build metadata, not proof of
differing source bytes or a security verdict. The independent source hash
checks and required metadata gate are retained. A pip rebuild produced the
required `dirty: false` metadata in this environment. Do not edit the stamp
or relax verification. The tested pip command is:

```sh
.venv-modern/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps \
  'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@5f368a3770cf5bdae0448995c9f76e29af8049f7'
```

Modern source pins:

- Shared core 0.12: `a794e3b4a8813b012f2c4faa3da82b78d4a90214`, package
  `packages/particle-sliders-core` in HyperGAN/particle-sliders. Upstream evidence for this
  exact source includes CPU and actual-source review plus a bounded synthetic
  CUDA software fixture. That upstream fixture is not native model, width, audio or Flux
  admission. This product remains a CPU software fixture and awaits its own
  post-pin Claude RETURN review; publication is not a package release.
- PG 0.9.4, official `release-0.9.4/auto`, verified by the shared manifest.
- HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, required
  self-reported `dirty:false` build metadata plus verified source hashes;
  HNDL 0.10.0; shared builder profile `hypergan-hndl-v2`.
- `conceptmod/modern/sources.json` checks the complete installed core Python,
  JSON and HNDL closure plus original CPU/base sources and this product's
  `lora.hndl`. Shared `verified_builder()` additionally verifies the actual
  HG builder and all HNDL Python files. Version strings alone do not admit a
  runtime. No explicit-domain upstream overlay is admitted.

The new cohort is
`conceptmod-cpu-alpha-velocity-pg094-hg-hndl010-v3`. Shared signatures and
native exports bind actual implementation source, graph source/plan/shapes,
base model bytes, target/rank/alpha, fresh A/B initialization hash/policy,
native response geometry and runtime. The v1 and v2 preview packets retain their
original identities and are refused by v3; they are not silently rebound.
The v3 backend/cohort/native protocol identifies the cleared core source.
Training v2 config and native v2 protocol refusals precede borrowed hook
attachment. No packet metadata or historical checkpoint is rewritten.
Cross-source/runtime continuation must be qualified as a separate cohort.

## Executable CPU route

```python
import torch
from conceptmod.modern import open_game, load_native

with open_game('red=blue', steps=4, seed=7, ema_decay=.9) as session:
    session.train(stop_after=2)
    packet = session.checkpoint('modern-training.pt')

with open_game('red=blue', steps=4, seed=7, ema_decay=.9, resume=packet) as session:
    session.train()
    velocity = session.evaluate(index=0, strength=.5)  # [1,4,8,8]
    session.export_inference('modern-native.pt', selection='ema')

# This reconstructs only the deterministic tiny base and the HNDL adapter.
with load_native('modern-native.pt') as native:
    velocity = native.response(prompt='red', latent=torch.ones(1,4,8,8),
                               timestep=torch.tensor([500.]), strength=.5)
```

`open_game(runtime=existing_cpu_backend)` borrows the exact CPU class path,
checks deterministic base bytes/rank/shape/device and **fresh seed-matched
A/B initialization** before attachment. Both A and B must exactly match the
seeded rank 4 / alpha 4 CpuBackend. Arbitrary existing/trained PEFT initialization
is refused, including a changed A with B still zero. The initialization hash
and explicit policy are bound in model, graph and native export identities.
The binding copies these fresh A/B values to HNDL and temporarily owns only
`linear2.forward`. Resume uses the modern HNDL checkpoint, not borrowed
legacy trained PEFT values. Original
native weights, PEFT artifacts, mode flags and gradient flags are restored on
close and preparation/update failures. Training modifies only the separate
HNDL adapter leaves and critic. The active binding is exclusive; a second
owner or replacement of its forward method is refused, and every remaining
cleanup still runs. Failed cleanup leaves the session and CPU binding open for
an explicit retry; a successfully removed hook is recorded independently so a
later restoration failure can be retried without removing another owner.
Product cleanup call sites pass the actual operation exception explicitly to
the cleared core, so a second cleanup failure is attached as a note while every
remaining restoration is attempted.
Complete resume validation is staged on an owned tiny
instance before attaching to a borrowed model. Native artifact identity is
checked on an unattached candidate before native attachment.
The native payload's tensor schema and finiteness are also validated on that
unattached candidate, before the shared inference factory can install a hook.

Strength 0 returns the untouched native base exactly. Intermediate strength
scales the LoRA delta and strength 1 applies the trained delta. Training-row
evaluation applies the row alpha times the requested strength; data-independent
native inference uses the requested strength directly. Render/image
capability is deliberately absent: this cohort exports native velocity, not a
new sampler. Other DSL rules/alphas and backend names refuse before loading
any pretrained model. Training checkpoints and raw/EMA native exports use
their different shared schemas and cannot substitute for one another.

## All eight DSL backends

These are source-level findings at ConceptMod base
`13271a8b3ac963832a8bdfd1e9a8c74eaf0344aa`, not refreshed model quality claims.
All seven native model backends remain **unmigrated** and refuse in `open_game`; no loader is
called when one is selected. Their historical DSL/encoder routes stay intact.

| Backend | Existing native velocity contract | Modern qualification / remaining gate |
|---|---|---|
| `cpu` | `[B,4,8,8]`; `t/1000` inside TinyFlowDiT; PEFT `linear2`, rank 4 / alpha 4 | **CPU software fixture qualified.** Exact ordinary two-linear LoRA HNDL graph; shared Conv2d critic/game/session/inference over the reshaped affine MLP output. No routed bank or pretrained-model claim. |
| `sana` | `[B,C,H,W]`; backend multiplies scheduler time by `timestep_scale`; learned-variance output may split channels | **Gated.** Native field is compatible in principle, but projection ownership, timestep/output split, masks, target selection and VJP/second-derivative evidence are not supplied here. |
| `zimage` | `[B,C,H,W]` externally, list inputs with singleton time internally; `(1000-t)/1000`; output sign negated | **Gated.** Needs its own list/text-layout, reversed-time/sign and checkpoint-scale adapter conformance. The CPU class path cannot stand in for this architecture. |
| `anima` | `[B,C,1,H,W]`, singleton time retained; normalized scheduler time and padding mask | **Gated.** Preserve 5D at game/penalty boundary. Core candidate's HNDLFieldCritic is 4D; a reviewed internal singleton-time wrapper with matched derivatives is still needed for this backend. |
| `krea` | Packed `[B,N,C]`, position IDs/grid, masks, normalized time; Raw/Turbo distinction | **Gated.** No admitted packed-token response/domain/critic binding here. Do not flatten into a wide vector or pool away native token geometry. |
| `qwen` | Packed `[B,N,C]`, explicit `img_shapes`, text mask, normalized time; Edit loader can be used on historical T2I path | **Gated.** Needs packed geometry and exact target-layout conformance; Edit reference-image behavior is not established by its T2I loader. |
| `klein` | Packed `[B,N,C]`, native image/text IDs, normalized time; distilled versus base identity | **Gated.** Needs packed response/critic admission, IDs and model variant evidence. Do not relabel distilled/base artifacts. |
| `supra` | `[B,C,H,W]`, native context/mask, increasing flow time, custom repository transformer/PEFT targets | **Gated.** Its existing dummy/live branches need their own HNDL target and response parity. A different repository's Supra integration does not qualify this implementation. |

The release-auto shared domain guard still refuses continuous flat vectors
width>=170. Its genuine native 4D/5D field branch is not permission to reshape
packed vectors into images. A packed-token API/critic and non-singleton
5D/temporal graphs are shared admission work, not hidden fallback paths.
No shared core contract is changed in this repository.

## Validation and limits

`tests/test_modern_game.py` executes the actual HG verified builder's
`hndl.torch.Network` objects, not mocks. It checks nonzero matched PEFT/HNDL
outputs and input/A/B VJPs; shared spatial critic logits/VJPs and second
parameter derivatives of the actual public BISPF penalty; real updates;
exact uninterrupted/split resume including optimizer/RNG/EMA; zero/half/full
strength; raw/EMA native-only inference in a fresh interpreter; source,
builder, base-model, shape, cohort and old-artifact rejection; borrowed
restoration; observation/update/preparation failures; late-load rollback;
and continuation of cleanup when projection ownership has changed. It also
asserts Conv2d types, alpha-dependent teacher/student values and VJPs,
nonfresh borrowed A/B rejection, and that 128 shared-game updates move
the student toward the frozen teacher at both training alphas and held-out
alpha 0.6. Teacher MSE is an evaluation metric only; it is not added to training.
The four-step example above illustrates lifecycle wiring, not convergence.

The focused historical PG 0.5 suite passes separately. Wheel and sdist builds,
installed artifact tests, public CPU dependency resolution and actual source
identity checks accompany the draft review packet. Actual installed-test
commands/stdout, source snapshot hashes, behavior values and artifact hashes
are published in `docs/validation/modern-cpu-return-review.json`. No model weights, GPU
training, model-quality benchmark, merge, release, deployment or new compute
is part of these CPU receipts. Root owns Claude review/corrections/return
review, GPU qualification routing and merge decisions.

Post-pin installed receipts, actual old-v2 packet refusal results and recursive
source identity are recorded separately in
`docs/validation/modern-cpu-core-a794.json`. The earlier
`modern-cpu-return-review.json` is retained as historical v2 evidence.
