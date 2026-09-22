# Supra2-IMG backend

`conceptmod` owns this testbed. It drives [SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG) the same way as Anima: DSL ops, `predict_v`, and a smoke script. It is not a product repo and it does not import particle-sliders.

Hub `inference.py` is the architecture and sampler. Weights are not in this repo.

## Hub pin

| | |
|---|---|
| Repo | `SupraLabs/Supra2-IMG` |
| Checkpoint | `model_final_ema.pt` |
| DiT | 104,094,736 parameters |
| Width | `D_MODEL=576`, `DEPTH=14`, `N_HEADS=9`, `HEAD_DIM=64`, `MLP_RATIO=4.0` |
| Text | frozen `google/flan-t5-base`, `D_CTX=768`, `ctx_len=128` |
| VAE | `stabilityai/sd-vae-ft-mse`, `VAE_SCALE=0.18215` |
| Grid | image 256², latent 32²×4, patch 2 |
| Sampler | Euler `t = i/K`, `z <- z + dt * v`, CFG 3.0, 50 steps |

Context is projected 768 → 576 before the blocks. Cross-attention `kv` is `Linear(576, 1152)`, matching Hub (the `ctx_dim` argument on `DiTBlock` is unused).

The DiT stays fp32 here so LoRA training matches the CPU tests. Hub inference autocasts the same module to bf16; the Euler update does not change.

Training samples use a shorter budget than generate: 8 steps, CFG 3.0.

## Checkpoint

Looked up in this order, with no download unless you opt in:

1. `ckpt=` passed to `load_backend` / `SupraBackend`
2. `SUPRA_CKPT`
3. `./model_final_ema.pt`
4. The Hugging Face cache for `SupraLabs/Supra2-IMG`

`SUPRA_DOWNLOAD=1` (or `download=True`) fetches `model_final_ema.pt`. Flan-T5-Base and SD-VAE-FT-MSE load from their own Hub repos on the real path. None of those blobs are vendored.

```bash
export SUPRA_CKPT=/path/to/model_final_ema.pt
python train.py --phrase "sunlit++" --backend supra --lora 16 --stage model \
    --out outputs/supra_sunlit
```

A missing checkpoint raises `FileNotFoundError` before any encoder download. `--resolution` other than 256 is rejected. The backend is LoRA-only; `--train-method xattn` is ignored.

## LoRA

DiT targets are qualified suffixes so self- and cross-attention do not collapse into one name:

- `self_attn.qkv`, `self_attn.proj`
- `cross_attn.q`, `cross_attn.kv`, `cross_attn.proj`

Bare `proj` matches both output projections. Bare `q` matches `cross_attn.q` and not `self_attn.qkv`. MLP and AdaLN stay frozen.

Encoder LoRA (stage 1) targets Flan-T5 `q` / `k` / `v` / `o`.

## Dummy path

`model_id="dummy"` builds a tiny DiT and a byte-embedding stand-in for Flan-T5. It does not call `require_cuda` and does not touch the Hub. Use it for CI:

```bash
pytest tests/test_supra.py
```

`load_backend("supra", device="cpu")` without `dummy` still goes through `require_cuda` and fails closed. Unit tests construct `SupraDiT()` directly for the 104,094,736 count.

## Smoke

```bash
python scripts/smoke_supra.py
```

With CUDA and a checkpoint: three neutral-daylight generates, then one `sunlit++` step with LoRA grads. Without either, the script prints `skip supra smoke:` and exits 0. It does not download weights.
