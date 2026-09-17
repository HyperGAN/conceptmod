"""Ordinary native YuE2 AR LoRA mapping for the shared ComfyUI converter.

ComfyUI's YuE2TEModel has fused QKV with output widths 2048/1024/1024.
Particle checkpoints must be distilled first; their bridge cannot be dropped.
"""
import re

MODULE = re.compile(r'^adapters\.model-layers-(\d+)-self_attn-(q_proj|k_proj|v_proj|o_proj)$')
QKV = re.compile(r'^text_encoders\.(model\.layers\.\d+\.self_attn\.)(q_proj|k_proj|v_proj)\.(lora_A|lora_B)\.weight$')


def map_yue2(module):
    match = MODULE.fullmatch(module)
    return f'model.layers.{match[1]}.self_attn.{match[2]}' if match else None


def detect(keys, stems):
    if not any(MODULE.fullmatch(stem) for stem in stems):
        return None
    if 'particles' in keys or any('.bridge.' in key for key in keys):
        return 'yue2_particle', 'nonlinear particle checkpoint; distillation required'
    return 'yue2', 'native YuE2 AR LoRA projection keys'


def fuse_qkv(out, alphas, block_diag_up, dtype):
    groups = {}
    for key in out:
        match = QKV.fullmatch(key)
        if match:
            groups.setdefault(match[1], {}).setdefault(match[2], {})[match[3]] = key
    errors = []
    for stem, projs in groups.items():
        order = ('q_proj', 'k_proj', 'v_proj')
        if set(projs) != set(order) or any(set(p) != {'lora_A', 'lora_B'} for p in projs.values()):
            errors.append(f'incomplete QKV under {stem}')
            continue
        parts = []
        for proj in order:
            down, up = (out[projs[proj][side]].float() for side in ('lora_A', 'lora_B'))
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                errors.append(f'invalid LoRA dimensions under {stem}{proj}')
                break
            parts.append((down, up, alphas.get(stem+proj, float(down.shape[0]))))
        if len(parts) != 3:
            continue
        if len({p[0].shape[1] for p in parts}) != 1:
            errors.append(f'inconsistent QKV input widths under {stem}')
            continue
        down, up, alpha = block_diag_up(parts)
        for proj in order:
            for key in projs[proj].values():
                del out[key]
            alphas.pop(stem+proj, None)
        dest = stem + 'qkv_proj'
        out[f'text_encoders.{dest}.lora_A.weight'] = down.to(dtype).contiguous()
        out[f'text_encoders.{dest}.lora_B.weight'] = up.to(dtype).contiguous()
        alphas[dest] = alpha
    return errors
