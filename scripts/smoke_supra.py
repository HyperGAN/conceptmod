"""Smoke test Supra2-IMG: baseline generation + one DSL op step.

Runs only when CUDA and ``model_final_ema.pt`` are both available (local
path, ``SUPRA_CKPT``, or the Hugging Face cache). Otherwise it prints a
skip and exits 0. Weights are not vendored and this script does not
download them.

CPU contract (no Hub): ``pytest tests/test_supra.py`` with ``model_id=dummy``.
"""
import os
import time

import torch

from conceptmod import dsl, ops
from conceptmod.backends import load_backend
from conceptmod.backends.supra import resolve_checkpoint

DEVICE = os.environ.get("CONCEPTMOD_DEVICE", "cuda:0")
PROMPTS = [
    "a sunlit kitchen in neutral daylight, a bowl of fruit on a wooden table",
    "a cat on a windowsill, sunlit room, neutral daylight",
    "a ceramic vase on a table, sunlit still life, neutral daylight",
]


def main() -> int:
    ckpt = resolve_checkpoint(download=False)
    missing = []
    if not torch.cuda.is_available():
        missing.append("CUDA")
    if ckpt is None:
        missing.append(
            "model_final_ema.pt (SUPRA_CKPT, ./model_final_ema.pt, or the HF cache)"
        )
    if missing:
        print("skip supra smoke: " + "; ".join(missing))
        print("CPU contract: pytest tests/test_supra.py (model_id='dummy', no download)")
        return 0

    t0 = time.time()
    backend = load_backend("supra", device=DEVICE, lora_rank=16, ckpt=ckpt)
    print(f"loaded in {time.time() - t0:.0f}s; latent shape {backend.latent_shape}")
    print("steps", backend.generate_steps, "cfg", backend.generate_guidance,
          "ckpt", backend.ckpt_path)

    os.makedirs("outputs/supra_baseline", exist_ok=True)
    for i, prompt in enumerate(PROMPTS):
        t1 = time.time()
        img = backend.generate(prompt, seed=42 + i)
        slug = prompt.replace(" ", "_")[:36]
        img.save(f"outputs/supra_baseline/{i}_{slug}.png")
        print("saved", prompt, f"{time.time() - t1:.1f}s")

    params = backend.trainable_parameters("lora")
    print("lora params:", sum(p.numel() for p in params) / 1e6, "M")

    cfg = ops.OpDefaults(**backend.training_defaults())
    rules = dsl.parse_phrase("sunlit++")
    ctx = ops.StepContext(backend, stop_index=2, seed=7, cfg=cfg)
    t1 = time.time()
    loss = sum(r.alpha * ops.rule_loss(r, ctx) for r in rules)
    loss.backward()
    grad = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None) ** 0.5
    print(f"op step: loss={loss.item():.5f} gradnorm={grad:.2e} t={time.time() - t1:.1f}s")
    print("max vram GiB:", torch.cuda.max_memory_allocated(DEVICE) / 2**30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
