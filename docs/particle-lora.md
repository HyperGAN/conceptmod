# Particle LoRA tensor integration

Core pins the published ParticleGAN 0.9.4 recipe used by its executable
regressions. `ParticleLoRAGenerator` binds sampled particle codes to explicit
low-rank branches on a caller-owned frozen host. The official `GANTrainer`
owns BISPF, SBNP, DualNorm, prior updates and checkpoint arithmetic.

Dimensions, budget and explicit no-weight-EMA settings are supplied. Raw current
weights are used; no generator, critic or prior weight averaging is enabled.
Official scalar optimizer statistics are unchanged. Historical paired-logistic
research games remain separately under `conceptmod.research`.

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
# Reconstruct the same host/generator/critic, then trainer.load_state_dict(checkpoint).
```

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
CPU regressions in `tests/test_particle_lora.py`.

Verification results and the two unchanged historical scientific outcome
assertions are recorded in [plan.md](plan.md). Original 2.x measurements remain
in the [historical documentation](https://github.com/HyperGAN/conceptmod/blob/70f0cb2632190388eb4028ef80066558259494fc/docs/particle-lora.md).
