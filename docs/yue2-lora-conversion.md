# Ordinary YuE2 LoRAs in ComfyUI

The shared converter accepts native YuE2 AR LoRA files whose module keys begin
with `adapters.model-layers-`. Run the existing command:

```bash
python scripts/convert_lora_comfyui.py path/to/control_distilled_rank8.safetensors
```

It writes `control_distilled_rank8_comfyui.safetensors` beside the source.
Original weights are preserved. Place the converted file in
`ComfyUI/models/loras/`, use **Load LoRA**, connect both MODEL and CLIP, and set
MODEL strength to **0** and CLIP strength to **1**. The recommended experimental
CLIP range is 0–1. These are YuE2 text-encoder / AR attention updates; they do
not patch the acoustic denoiser or VAE.

YuE2's ComfyUI text encoder fuses Q/K/V into `qkv_proj`, with output widths
2048, 1024 and 1024. The converter concatenates the three down matrices and
block-diagonalizes their up matrices, preserving independent alpha/rank
scales. Three rank-8 branches become one rank-24 fused branch; O stays rank 8.
Output keys use the generic `text_encoders.model.layers.N.self_attn.*` form.

Routed-particle checkpoints contain nonlinear bridge weights and cannot be
converted by discarding those tensors. Automatic detection rejects them and
requests distillation first. This converter only rearranges existing ordinary
linear updates; it does not train or distill a model.

The standard ComfyUI LoRA remains active when the text encoder constructs the
acoustic-prefix conditioning, which is a broader scope than the original
particle runtime's semantic-only hooks. Compare both scopes when assessing
a distilled slider's audio fidelity. Key binding alone does not prove that
two generation pipelines produce the same song.

Validation covers GQA widths, independent alpha scales, incomplete QKV groups,
particle rejection, and actual `comfy.lora` key binding and weight application
on `YuE2TEModel`. Set `COMFYUI_ROOT` to a current ComfyUI checkout to run the
integration test:

```bash
COMFYUI_ROOT=/path/to/ComfyUI pytest tests/test_convert_yue2.py
```
