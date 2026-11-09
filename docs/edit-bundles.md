# Saving and restoring edits

`backend.save_trained("run/edit")` saves the complete current edit: transformer
weights and any attached text-encoder adapters. The CLI's `--save-model` writes
this bundle to `<out>/edit`, using the backend's effective configuration. This
works when a LoRA-only backend selected its default rank without `--lora`.

The destination must be new or empty. Export completes in a temporary sibling
directory before the bundle becomes visible. Use a new destination for each
export.

A bundle contains:

- `edit_bundle.json`: format/version, effective backend initialization and
  sampling settings, component kinds/paths, and actual adapter ranks.
- Transformer PEFT files at the bundle root, or `transformer.safetensors` for
  full transformer weights.
- `text_encoder/`: PEFT weights and configuration when encoder LoRA is attached.

Weights retain their current precision. A bundle is an inference artifact;
optimizer state, random-number-generator state and unfinished gradient
accumulation are part of a training checkpoint instead.

Initialize a backend with the same base model, settings and adapter
configuration, then load the bundle. Attach encoder LoRA with its saved rank
before loading if the bundle lists a `text_encoder` component. For example,
the download-free CPU fixture roundtrip is:

```python
from conceptmod.artifacts import read_edit_bundle
from conceptmod.backends.cpu import CpuBackend

manifest = read_edit_bundle("run/edit")
settings = manifest["metadata"]
backend = CpuBackend(device="cpu", seed=settings["base_seed"],
                     lora_rank=settings["lora_rank"])
encoder = manifest["components"].get("text_encoder")
if encoder is not None:
    backend.attach_encoder_lora(rank=encoder["adapters"]["default"]["rank"])
backend.load_trained("run/edit")
image = backend.generate("red", seed=11)
```

The loader checks metadata, component names, PEFT configurations, weight keys
and shapes for the whole bundle before changing weights. It loads safetensors
files, clears cached embeddings after success and preserves the untouched
reference model. For downloaded backends, use the same base-model revision or
local checkpoint used during training; the bundle stores the model identifier,
not a copy of those base weights.

Use `backend.save_transformer(path)` when an individual transformer export is
needed for an existing consumer: a PEFT directory for LoRA, or a safetensors
file for full weights. This explicitly omits the encoder. Existing transformer
PEFT consumers can also read the adapter files at a complete bundle's root.
