# ParticleGAN 0.9.4 and particle LoRAs

Install `conceptmod` with `particlegan>=0.9.4,<0.10`. This range was tested
against the PyPI 0.9.4 wheel. Two intentionally separate training paths exist:

* The existing DSL velocity trainer and its PEFT exports remain unchanged.
  The `locked_shared` CPU games preserve their historical paired logistic
  RpGAN / fixed L2 `b_cap` formulation. They do not claim to use the current
  ParticleGAN default. `conceptmod.particle_api` translates their recorded
  knobs to actual upstream primitives; it contains no copied GAN arithmetic.
* `ParticleLoRAGenerator` is an explicit modern tensor integration: a real
  learned particle prior routes low-rank residuals into named frozen Linear
  layers. Its public `GANTrainer` uses the current `bispf_optimized` recipe,
  including non-saturating loss, Born–Infeld penalty, SBNP prior regularization
  and DualNorm. Only dimensions and training budget are overridden.

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
qualified by the small CPU tests. No new pretrained-model controls or published
weights are established by this migration; the existing image examples were
not retrained with ParticleGAN 0.9.4.

## Validation on 2026-11-04

The CPU validation environment used Python 3.11.17, Torch 2.14.1+cu130,
ParticleGAN 0.9.4, PEFT 0.21.2, Transformers 4.57.6, Accelerate 1.15.0 and
Diffusers 0.39.0. The seven new primitive/particle-LoRA tests passed, including
an actual 128×4 learned bank with rank-8 branches, exact next-update checkpoint
replay, frozen BatchNorm/Dropout behavior, and strength-zero identity.

Two existing scientific tests remain failing without relaxed thresholds:

| Gate | Required | Measured with original and migrated code |
| --- | --- | --- |
| `test_toy_mode_hold.py::test_locked_shared_holds_ring_modes` | At least 7/8 modes and HQ ≥0.90 | 3/8 modes, HQ 0.495; `INCONCLUSIVE` after 1,200 updates |
| `test_toy_shared_trajectory.py::test_locked_shared_passes` | Identity MSE ≤0.02 | 0.2338901311159134 after 400 updates; `pass=false` |

Both failures were reproduced with pristine conceptmod
`13271a8` and the original ParticleGAN 0.5.0 wheel in the same environment.
The shared-trajectory final MSE and discriminator/generator losses are exactly
identical across the comparison. This upgrade neither claims those gates pass
nor retunes their training to change the evidence.

The optional live ComfyUI tests require a separate functional ComfyUI install.
The available checkout lacked `comfy_aimdo.host_buffer`, so these integration
tests were excluded. Ordinary conversion and CPU LoRA/backend tests were run.
The broad migration check keeps those exclusions explicit:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --import-mode=importlib \
  --ignore=tests/test_comfyui_lora_apply.py \
  --deselect=tests/test_convert_yue2.py::test_actual_comfyui_yue2_loader \
  --deselect=tests/test_toy_mode_hold.py::test_locked_shared_holds_ring_modes \
  --deselect=tests/test_toy_shared_trajectory.py::test_locked_shared_passes
```

The two scientific gates can be reproduced separately without exclusions:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --import-mode=importlib \
  tests/test_toy_mode_hold.py::test_locked_shared_holds_ring_modes \
  tests/test_toy_shared_trajectory.py::test_locked_shared_passes
```

The broad command above completed with **446 passed, 1 skipped and
3 deselected** (the live YuE2/ComfyUI test and the two known scientific gates).
The independent new-API checks completed with **7 passed**. The wheel built
successfully, was installed into the isolated environment, and ran an official
BISPF update from outside the source checkout with a 128×4 prior and exact
strength-zero replay. These are source/package checks, not image-quality or
pretrained-model training results.
