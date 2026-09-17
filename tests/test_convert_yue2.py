"""YuE2 GQA fusion, nonlinear rejection and real ComfyUI key binding."""
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file
from conceptmod.convert import convert, detect_backend, sidecar_metadata


def fixture(path, width=32, q=32, kv=16):
    torch.manual_seed(17)
    state = {}
    for proj, outdim in [('q_proj', q), ('k_proj', kv), ('v_proj', kv), ('o_proj', width)]:
        stem = 'adapters.model-layers-0-self_attn-' + proj
        state[stem+'.lora_down.weight'] = torch.randn(8, q if proj=='o_proj' else width)
        state[stem+'.lora_up.weight'] = torch.randn(outdim, 8)
        state[stem+'.alpha'] = torch.tensor(4. if proj=='k_proj' else 8.)
    save_file(state, str(path))
    return state


def test_fused_gqa_preserves_three_independent_updates(tmp_path):
    path = tmp_path/'native.safetensors'
    source = fixture(path)
    assert detect_backend({}, list(source))[0] == 'yue2'
    out, dropped, errors = convert(str(path), 'yue2', {})
    assert not errors and not dropped
    stem = 'text_encoders.model.layers.0.self_attn.qkv_proj'
    a, b, alpha = [out[stem+s].float() for s in ('.lora_A.weight', '.lora_B.weight', '.alpha')]
    assert a.shape == (24, 32) and b.shape == (64, 24)
    wanted = []
    for proj in ('q_proj', 'k_proj', 'v_proj'):
        prefix = 'adapters.model-layers-0-self_attn-'+proj
        down, up = [source[prefix+s].bfloat16().float() for s in ('.lora_down.weight', '.lora_up.weight')]
        wanted.append(up @ down * source[prefix+'.alpha'] / 8)
    torch.testing.assert_close(b @ a * alpha / 24, torch.cat(wanted), atol=1e-5, rtol=1e-5)
    assert sidecar_metadata('yue2', {})['recommended_range'] == [0., 1.]


def test_particle_and_incomplete_qkv_are_rejected(tmp_path):
    path = tmp_path/'particle.safetensors'
    state = fixture(path)
    state['particles'] = torch.randn(128, 4)
    assert detect_backend({}, list(state))[0] == 'yue2_particle'
    save_file(state, str(path))
    assert 'particles' in convert(str(path), 'yue2', {})[2]
    del state['particles']
    del state['adapters.model-layers-0-self_attn-k_proj.lora_up.weight']
    save_file(state, str(path))
    assert any('incomplete QKV' in e for e in convert(str(path), 'yue2', {})[2])


def test_actual_comfyui_yue2_loader(tmp_path):
    root = Path(os.environ.get('COMFYUI_ROOT', '/tmp/comfyui'))
    if not (root/'comfy/lora.py').exists():
        pytest.skip('Set COMFYUI_ROOT for actual ComfyUI integration')
    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.lora
    from comfy.text_encoders.yue2 import YuE2TEModel
    te = YuE2TEModel(device='meta', dtype=torch.float32)
    path = tmp_path/'real-shapes.safetensors'
    fixture(path, width=2048, q=2048, kv=1024)
    out, dropped, errors = convert(str(path), 'yue2', {})
    assert not dropped and not errors
    mapping = comfy.lora.model_lora_keys_clip(te, {})
    patches = comfy.lora.load_lora(out, mapping)
    assert set(patches) == {'model.layers.0.self_attn.qkv_proj.weight', 'model.layers.0.self_attn.o_proj.weight'}
    for key, patch in patches.items():
        stem = 'text_encoders.'+key[:-7]
        down, up = out[stem+'.lora_A.weight'].float(), out[stem+'.lora_B.weight'].float()
        want = up @ down * out[stem+'.alpha'] / down.shape[0]
        zero = torch.zeros_like(want)
        for scale in (0., .5, 1.):
            got = comfy.lora.calculate_weight([(scale, patch, 1., None, None)], zero.clone(), key,
                                              intermediate_dtype=torch.float32)
            torch.testing.assert_close(got, want*scale)
