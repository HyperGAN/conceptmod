# Particle LoRA tensor integration

Core pins ParticleGAN 0.10.0. `ParticleLoRAGenerator` binds sampled particle
codes to explicit low-rank branches on a caller-owned frozen host. The official `GANTrainer`
owns EMPF, DualNorm, prior updates and checkpoint arithmetic. Fresh
trainers explicitly use upstream's `empf` preset; Conceptmod does not replace
its loss, regularizer, optimizer or numerical profile. The pinned
[upstream release](https://github.com/HyperGAN/ParticleGAN/releases/tag/v0.10.0)
supplies that implementation; CPU integration checks are not a new
pretrained-model qualification or learning-rate comparison.

Dimensions, budget and explicit no-weight-EMA settings are supplied. Raw current
weights are used; no generator, critic or prior weight averaging is enabled.
Official scalar optimizer statistics are unchanged. Historical paired-logistic
research games remain separately under `conceptmod.research`.

Regularization runs on **every update**: `reg_every=1` in the recipe becomes
`lazy_k=1` in the actual upstream penalty kernel. An explicit recipe with a
larger interval is rejected before trainer construction. Tests compare every
recipe field against `get_recipe("empf", ...)`, check actual optimizer-group
rates, and observe a regularizer application on each consecutive update.

| EMPF setting | Value |
| --- | --- |
| Initial generator / critic / particle-bank learning rates | 0.012 / 0.018 / 0.03 |
| Schedule | Cosine, using the declared total budget |
| Critic regularizer interval / coefficient / kappa | 1 / 1 / 1 |
| Optimizer / momentum / smoothing | DualNorm / 0 / 0.001 |
| Convolution / convolution rule | Per offset / quintic |
| Dense profile / SVD backend | `newton_schulz_hybrid_v1` / batched |
| EMPF critic `ebi_beta` | 6 |
| Prior regularizer / weight | EMPF / 0.01 |
| `ucd_weight` / routing temperature | 0.02 / 0.25 |
| Training precision profile | `bf16_channels_last` (upstream support rules apply) |
| Weight EMA / served averaging | Disabled / disabled |

These are the stock ParticleGAN 0.10.0 EMPF hyperparameters. Task dimensions
and budget are caller-owned; a model-specific tuned recipe must be supplied
explicitly and retains its saved fields. No previous search or measured rate is
replaced by these starting values.

```python
import torch
from torch import nn
from conceptmod.particle_lora import ParticleLoRAGenerator

# The caller loads/fixes the model and chooses verified projection targets.
host = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
host.requires_grad_(False)
fixed_input = torch.ones(1, 3)
generator = ParticleLoRAGenerator(host, ("0", "2"), fixed_input, rank=8, z_dim=4)
critic = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
trainer = generator.make_trainer(critic, num_particles=128, batch_size=32)
for _ in range(10):
    real_targets = torch.randn(32, 2)  # supply actual targets for this context
    trainer.step(real_targets)
codes, _ = trainer.prior.sample(4)
adapted = generator(codes, strength=1.0)
base = generator(codes, strength=0.0)  # exact frozen-host output
checkpoint = trainer.state_dict()     # model, prior, optimizers, recipe and RNG
```

To resume, reconstruct the same host/generator/critic and use the **saved recipe**:

```python
from particlegan import Recipe

restored = generator.make_trainer(critic, recipe=Recipe.from_dict(checkpoint["recipe"]))
restored.load_state_dict(checkpoint)
```

An explicit `recipe` owns the particle dimensions, batch size and budget; the
factory's dimension/budget keywords are used only when it constructs a fresh
default. The saved latent dimension must match the generator. Explicit upstream
`bispf_optimized`, `empf` and `rhbi` recipes are supported with `ema_decay=0.0`
and `serve_average=0.0`. Recipes enabling weight EMA, averaged serving or a
legacy EMA critic anchor are rejected before trainer construction. The official
zero-decay checkpoint copies mirror raw current weights; no moving-weight
history is used.

The EMPF default must not reinterpret an older run. A mismatched recipe is
rejected by the upstream checkpoint loader. Its existing schema, execution and
optimizer validation stays intact: checkpoints that upstream cannot exactly
restore must use their original pinned runtime. Saved weights, initialization,
completed updates and earlier learning-rate results are not migrated or replayed.

The host is modified in place and must already be frozen. Adapters remain
FP32; residuals are cast to the host output dtype. One sampled particle code
gates all selected layers coherently. This sampled-code gate differs from the
input-dependent softmax bank attention used by some particle-slider projects.
Changing strength scales the residual;
zero bypasses it exactly. The host and fixed input are included in the full
trainer checkpoint, so it is not a small PEFT export.

This API learns the output distribution **for one fixed input context**. A
multi-input model can be wrapped as a Tensor-to-Tensor module with remaining
arguments bound as buffers. This is not an assertion of conditional training
across changing prompts, E22 routed-row evidence, or token-specific routing.
The caller owns target tensors, output-compatible critic architecture, model
loading, and evaluation. Native image backends are not wired into this API or
qualified by the small CPU tests. CPU tensor integration does not qualify a pretrained model or establish
new image controls.


The host must access layers through its registered module tree. Each invocation
binds its own codes without mutating live-layer routing state or copying frozen
weight tensors. Reentrant and nonreentrant activation-checkpoint recomputation,
nested/concurrent forwards and exact next-update trainer replay have executable
CPU regressions in `tests/test_particle_lora.py`. Replay checks compare the
complete next-step state, including optimizers, formulation records and random
streams, for RHBI, BISPF and EMPF.

Verification results and the two unchanged historical scientific outcome
assertions are recorded in [plan.md](plan.md). Original 2.x measurements remain
in the [historical documentation](https://github.com/HyperGAN/conceptmod/blob/70f0cb2632190388eb4028ef80066558259494fc/docs/particle-lora.md).
