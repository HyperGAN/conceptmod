"""Supra2-IMG contract tests. No Hub download, no CUDA.

The full DiT is built on CPU from the Hub architecture constants. The
dummy backend is the CI path for LoRA, Euler, and one DSL step.
"""
from __future__ import annotations

import os

import pytest
import torch

from conceptmod.backends import BACKENDS, load_backend
from conceptmod.backends.supra import (
    CKPT_FILENAME,
    DEFAULT_CFG,
    DEFAULT_MODEL,
    DEFAULT_STEPS,
    DEPTH,
    D_CTX,
    D_MODEL,
    DUMMY_SPEC,
    EXPECTED_DIT_PARAMS,
    FULL_SPEC,
    HEAD_DIM,
    LATENT_CH,
    LATENT_SIZE,
    MAX_CTX_LEN,
    MLP_RATIO,
    N_HEADS,
    NUM_TOKENS,
    PATCH,
    TRAIN_CFG,
    TRAIN_STEPS,
    VAE_SCALE,
    SupraBackend,
    SupraDiT,
    _LORA_TARGETS,
    build_dit,
    count_parameters,
    euler_schedule,
    load_supra_checkpoint,
    resolve_checkpoint,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.delenv("SUPRA_DOWNLOAD", raising=False)


def test_supra_is_registered():
    assert BACKENDS[-1] == "supra"
    assert "supra" in BACKENDS
    assert DEFAULT_MODEL == "SupraLabs/Supra2-IMG"
    assert CKPT_FILENAME == "model_final_ema.pt"
    assert VAE_SCALE == 0.18215
    assert DEFAULT_STEPS == 50
    assert DEFAULT_CFG == 3.0
    assert TRAIN_STEPS == 8
    assert TRAIN_CFG == 3.0
    assert HEAD_DIM == D_MODEL // N_HEADS == 64
    assert "proj" not in _LORA_TARGETS
    assert "q" not in _LORA_TARGETS
    assert _LORA_TARGETS == [
        "self_attn.qkv",
        "self_attn.proj",
        "cross_attn.q",
        "cross_attn.kv",
        "cross_attn.proj",
    ]


def test_unknown_backend_lists_supra():
    with pytest.raises(ValueError, match="supra"):
        load_backend("not-a-backend", device="cpu")


def test_real_supra_rejects_cpu_before_checkpoint_lookup(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("checkpoint lookup on the CPU rejection path")

    monkeypatch.setattr("conceptmod.backends.supra.resolve_checkpoint", boom)
    with pytest.raises(ValueError, match="CUDA"):
        load_backend("supra", device="cpu")


def test_full_dit_matches_hub_contract():
    model = SupraDiT()
    assert count_parameters(model) == EXPECTED_DIT_PARAMS
    assert len(model.blocks) == DEPTH == 14
    assert model.num_tokens == NUM_TOKENS == 256
    assert model.patch == PATCH == 2
    assert tuple(model.pos_embed.shape) == (1, NUM_TOKENS, D_MODEL)
    assert model.x_embed.in_features == LATENT_CH * PATCH * PATCH
    assert model.x_embed.out_features == D_MODEL
    assert model.ctx_proj.in_features == D_CTX == 768
    assert model.ctx_proj.out_features == D_MODEL == 576
    block = model.blocks[0]
    assert block.self_attn.qkv.in_features == D_MODEL
    assert block.self_attn.qkv.out_features == D_MODEL * 3
    assert block.self_attn.head_dim == HEAD_DIM
    assert block.self_attn.proj.in_features == D_MODEL
    # Context is projected first, so cross-attn kv is d_model, not 768.
    assert block.cross_attn.q.in_features == D_MODEL
    assert block.cross_attn.kv.in_features == D_MODEL
    assert block.cross_attn.kv.out_features == D_MODEL * 2
    assert block.cross_attn.proj.out_features == D_MODEL
    assert not block.norm1.elementwise_affine
    assert block.mlp[0].out_features == int(D_MODEL * MLP_RATIO)
    assert block.mlp[1].approximate == "tanh"
    assert block.adaln[1].out_features == 6 * D_MODEL
    assert model.final.linear.out_features == LATENT_CH * PATCH * PATCH
    assert model.final.adaln[1].out_features == 2 * D_MODEL
    assert model.t_embed.mlp[0].in_features == 256
    assert FULL_SPEC.image_size == 256
    assert FULL_SPEC.latent_shape == (LATENT_CH, LATENT_SIZE, LATENT_SIZE)


def test_full_dit_forward_shape():
    model = SupraDiT().eval()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, LATENT_CH, LATENT_SIZE, LATENT_SIZE, generator=g)
    t = torch.tensor([0.0])
    ctx = torch.randn(1, MAX_CTX_LEN, D_CTX, generator=g)
    mask = torch.ones(1, MAX_CTX_LEN)
    mask[:, 64:] = 0
    with torch.no_grad():
        out = model(z, t, ctx, mask)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()
    assert out.abs().sum().item() > 0


def test_euler_schedule_matches_hub():
    steps, dt = euler_schedule(50, device="cpu")
    assert dt == pytest.approx(1.0 / 50)
    assert steps.shape == (50,)
    assert steps[0].item() == pytest.approx(0.0)
    assert steps[1].item() == pytest.approx(1.0 / 50)
    assert steps[-1].item() == pytest.approx(49.0 / 50)


def test_resolve_checkpoint_is_local_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SUPRA_CKPT", raising=False)

    def no_cache(*_args, **_kwargs):
        return None

    monkeypatch.setattr("conceptmod.backends.supra._cached_hub_file", no_cache)
    assert resolve_checkpoint(download=False) is None

    local = tmp_path / CKPT_FILENAME
    local.write_bytes(b"placeholder")
    found = resolve_checkpoint(download=False)
    assert found is not None and os.path.samefile(found, local)

    other = tmp_path / "elsewhere.pt"
    other.write_bytes(b"x")
    monkeypatch.setenv("SUPRA_CKPT", str(other))
    found = resolve_checkpoint(download=False)
    assert found is not None and os.path.samefile(found, other)


def test_checkpoint_ema_model_and_prefix_roundtrip(tmp_path):
    spec = DUMMY_SPEC
    source = build_dit(spec)
    raw = source.state_dict()

    ema_path = tmp_path / "ema.pt"
    torch.save({"ema": raw, "config": {"patch": 2, "ctx_len": spec.ctx_len}}, ema_path)
    restored = build_dit(spec)
    cfg = load_supra_checkpoint(restored, str(ema_path))
    assert cfg["ctx_len"] == spec.ctx_len
    for key, value in raw.items():
        assert torch.equal(restored.state_dict()[key], value)

    model_path = tmp_path / "model.pt"
    prefixed = {f"module.{key}": value for key, value in raw.items()}
    torch.save({"model": prefixed, "config": {"patch": PATCH}}, model_path)
    via_model = build_dit(spec)
    load_supra_checkpoint(via_model, str(model_path))
    for key, value in raw.items():
        assert torch.equal(via_model.state_dict()[key], value)

    bare_path = tmp_path / "bare.pt"
    torch.save(raw, bare_path)
    via_bare = build_dit(spec)
    load_supra_checkpoint(via_bare, str(bare_path))
    for key, value in raw.items():
        assert torch.equal(via_bare.state_dict()[key], value)

    bad = tmp_path / "bad_patch.pt"
    torch.save({"ema": raw, "config": {"patch": 4}}, bad)
    with pytest.raises(ValueError, match="PATCH"):
        load_supra_checkpoint(build_dit(spec), str(bad))


def _dummy(**kwargs):
    rank = kwargs.pop("lora_rank", 4)
    return load_backend("supra", device="cpu", model_id="dummy", lora_rank=rank, **kwargs)


def test_dummy_backend_contract_without_hub():
    backend = _dummy()
    assert isinstance(backend, SupraBackend)
    assert backend.dummy
    assert backend.device == "cpu"
    assert backend.latent_shape == DUMMY_SPEC.latent_shape
    assert backend.generate_guidance == DEFAULT_CFG
    assert backend.training_defaults() == {
        "sample_steps": 4,
        "sample_guidance": TRAIN_CFG,
    }
    text = backend.encode_text("sunlit room, neutral daylight")
    assert text.embeds.shape == (1, backend.ctx_len, DUMMY_SPEC.ctx_dim)
    assert text.mask.shape == (1, backend.ctx_len)
    assert float(text.mask.sum()) > 0
    empty = backend.encode_text("")
    assert float(empty.mask.sum()) > 0

    z = torch.randn(1, *backend.latent_shape)
    t = torch.tensor([0.0])
    params = backend.trainable_parameters("xattn")
    assert params
    assert all(
        ".self_attn." in name or ".cross_attn." in name
        for name, param in backend.transformer.named_parameters()
        if param.requires_grad
    )
    velocity = backend.predict_v("sunlit room, neutral daylight", z, t, frozen=False)
    assert velocity.shape == z.shape
    assert torch.isfinite(velocity).all()
    velocity.float().pow(2).mean().backward()
    grad = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None) ** 0.5
    assert grad > 0

    frozen = backend.predict_v("sunlit room, neutral daylight", z, t, frozen=True)
    assert frozen.shape == z.shape
    assert not frozen.requires_grad


def test_dummy_euler_and_dsl_step():
    from conceptmod import dsl, ops

    backend = _dummy()
    generator = torch.Generator(device="cpu").manual_seed(0)
    latent, timestep = backend.partial_denoise(
        "a sunlit kitchen, neutral daylight",
        stop_index=2,
        num_steps=4,
        guidance=3.0,
        generator=generator,
    )
    assert timestep.item() == pytest.approx(0.5)
    assert latent.shape == (1, *backend.latent_shape)
    assert torch.isfinite(latent).all()

    image = backend.generate(
        "a ceramic vase, sunlit still life, neutral daylight",
        seed=1,
        num_steps=2,
        guidance=3.0,
    )
    assert image.size == (backend.resolution, backend.resolution)

    params = backend.trainable_parameters("lora")
    cfg = ops.OpDefaults(**backend.training_defaults())
    rules = dsl.parse_phrase("sunlit++")
    ctx = ops.StepContext(backend, stop_index=2, seed=7, cfg=cfg)
    loss = sum(rule.alpha * ops.rule_loss(rule, ctx) for rule in rules)
    loss.backward()
    grad = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None) ** 0.5
    assert torch.isfinite(loss)
    assert grad > 0


def test_dummy_encoder_lora_and_save(tmp_path):
    backend = _dummy()
    params = backend.attach_encoder_lora(4)
    assert params
    embeds = backend.encode_text_grad("neutral daylight")
    embeds.embeds.float().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    out = tmp_path / "adapter"
    backend.save_trained(str(out))
    assert (out / "adapter_config.json").is_file()


def test_stored_uncond_is_the_empty_prompt():
    from conceptmod.backends.base import TextEmbeds

    backend = _dummy()
    stored = TextEmbeds(
        torch.ones(1, backend.ctx_len, DUMMY_SPEC.ctx_dim),
        torch.ones(1, backend.ctx_len),
    )
    backend._stored_uncond = stored
    got = backend.encode_text("")
    assert torch.equal(got.embeds, stored.embeds)
    prompted = backend.encode_text("sunlit room, neutral daylight")
    assert not torch.equal(prompted.embeds, stored.embeds)

    backend.attach_encoder_lora(2)
    trained_empty = backend.encode_text("", frozen=False)
    assert not torch.equal(trained_empty.embeds, stored.embeds)
    frozen_empty = backend.encode_text("", frozen=True)
    assert torch.equal(frozen_empty.embeds, stored.embeds)


def test_dummy_rejects_wrong_resolution():
    with pytest.raises(ValueError, match="resolution"):
        _dummy(resolution=256)
