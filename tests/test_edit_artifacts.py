"""Complete edit roundtrips, with real tiny PEFT models and no downloads."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from conceptmod.artifacts import (
    MANIFEST_NAME,
    load_edit_bundle,
    read_edit_bundle,
    save_edit_bundle,
)
from conceptmod.testing.peft_backend import CpuBackend


def _trained(stage):
    backend = CpuBackend(device="cpu", lora_rank=3, seed=17)
    if stage != "model":
        backend.attach_encoder_lora(rank=2)
    parameters = []
    if stage != "encoder":
        parameters += list(backend.transformer.parameters())
    if stage != "model":
        parameters += list(backend.text_encoder.parameters())
    optimizer = torch.optim.SGD([p for p in parameters if p.requires_grad], lr=0.1)
    z = torch.zeros(1, *backend.latent_shape)
    t = torch.tensor([500.0])
    text = backend.encode_text_grad("red")
    if stage == "encoder":
        loss = text.embeds.square().sum()
    else:
        loss = backend._forward(backend.transformer, z, t, text).square().mean()
    loss.backward()
    optimizer.step()
    backend._text_cache.clear()
    return backend


def _state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


@pytest.mark.parametrize("stage", ["encoder", "model", "both"])
def test_complete_backend_roundtrip(stage, tmp_path):
    source = _trained(stage)
    z = torch.zeros(1, *source.latent_shape)
    t = torch.tensor([500.0])
    expected = source.predict_v("red", z, t, frozen=False)
    reference = source.predict_v("red", z, t, frozen=True)
    assert not torch.equal(expected, reference), "the fixture must contain a real edit"
    path = tmp_path / "edit"
    source.save_trained(str(path))
    manifest = read_edit_bundle(path)
    assert manifest["components"]["transformer"]["adapters"]["default"]["rank"] == 3
    assert (path / "transformer" / "adapter_config.json").is_file()
    assert (path / "transformer" / "adapter_model.safetensors").is_file()
    assert not (path / "adapter_config.json").exists()

    restored = CpuBackend(device="cpu", lora_rank=3, seed=17)
    if stage != "model":
        encoder_rank = manifest["components"]["text_encoder"]["adapters"]["default"]["rank"]
        assert encoder_rank == 2
        restored.attach_encoder_lora(rank=encoder_rank)
    # Loading must clear old embeddings computed before the encoder changed.
    restored.encode_text("red")
    restored.load_trained(str(path))
    assert not restored._text_cache
    torch.testing.assert_close(restored.predict_v("red", z, t, frozen=False), expected,
                               rtol=0, atol=0)
    torch.testing.assert_close(restored.predict_v("red", z, t, frozen=True), reference,
                               rtol=0, atol=0)
    assert restored.generate("red", 11).tobytes() == source.generate("red", 11).tobytes()


def test_missing_encoder_adapter_is_rejected_before_loading(tmp_path):
    source = _trained("both")
    path = tmp_path / "edit"
    source.save_trained(str(path))
    restored = CpuBackend(device="cpu", lora_rank=3, seed=17)
    before = _state(restored.transformer)
    with pytest.raises(ValueError, match="attach the saved encoder adapters"):
        restored.load_trained(str(path))
    for key, value in before.items():
        assert torch.equal(restored.transformer.state_dict()[key], value)


@pytest.mark.parametrize("corruption", ["shape", "key", "config"])
def test_later_component_failure_does_not_mutate_earlier_weights(corruption, tmp_path):
    from safetensors.torch import load_file, save_file

    source = _trained("both")
    path = tmp_path / "edit"
    source.save_trained(str(path))
    if corruption == "config":
        config_path = path / "text_encoder" / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["lora_alpha"] += 1
        config_path.write_text(json.dumps(config))
    else:
        weights_path = path / "text_encoder" / "adapter_model.safetensors"
        state = load_file(str(weights_path))
        key = next(iter(state))
        if corruption == "shape":
            state[key] = state[key][:1].clone()
        else:
            del state[key]
        save_file(state, str(weights_path))
    restored = CpuBackend(device="cpu", lora_rank=3, seed=17)
    restored.attach_encoder_lora(rank=2)
    before = {name: _state(module) for name, module in restored.edit_components().items()}
    with pytest.raises(ValueError, match="incompatible"):
        restored.load_trained(str(path))
    for name, module in restored.edit_components().items():
        for key, value in before[name].items():
            assert torch.equal(module.state_dict()[key], value)


def test_wrong_base_seed_is_rejected(tmp_path):
    source = _trained("model")
    source.save_trained(str(tmp_path / "edit"))
    restored = CpuBackend(device="cpu", lora_rank=3, seed=18)
    with pytest.raises(ValueError, match="metadata"):
        restored.load_trained(str(tmp_path / "edit"))


def test_full_weights_keep_precision_and_buffers(tmp_path):
    source = torch.nn.Linear(3, 2, dtype=torch.float64)
    source.register_buffer("counter", torch.tensor(7))
    path = tmp_path / "full"
    save_edit_bundle(path, {"transformer": source}, metadata={"base": "test"})
    from safetensors.torch import load_file

    saved = load_file(str(path / "transformer" / "weights.safetensors"))
    assert saved["weight"].dtype == torch.float64
    assert saved["counter"].dtype == torch.int64
    restored = torch.nn.Linear(3, 2, dtype=torch.float64)
    restored.register_buffer("counter", torch.tensor(0))
    load_edit_bundle(path, {"transformer": restored}, expected_metadata={"base": "test"})
    for key, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_legacy_transformer_export_is_explicit(tmp_path):
    source = _trained("both")
    path = tmp_path / "transformer-only"
    source.save_transformer(str(path))
    assert (path / "adapter_config.json").is_file()
    assert not (path / MANIFEST_NAME).exists()
    assert not (path / "text_encoder").exists()


def test_bundles_do_not_overwrite_existing_artifacts(tmp_path):
    path = tmp_path / "edit"
    source = _trained("model")
    source.save_trained(str(path))
    original = (path / MANIFEST_NAME).read_bytes()
    with pytest.raises(FileExistsError, match="new or empty"):
        source.save_trained(str(path))
    assert (path / MANIFEST_NAME).read_bytes() == original


def test_failed_export_does_not_publish_a_partial_bundle(tmp_path):
    class BrokenAdapter(torch.nn.Module):
        peft_config = {"default": object()}

        def save_pretrained(self, path, **kwargs):
            from pathlib import Path

            (Path(path) / "partial").write_text("incomplete")
            raise RuntimeError("could not export adapter")

    path = tmp_path / "edit"
    with pytest.raises(RuntimeError, match="could not export"):
        save_edit_bundle(path, {"transformer": BrokenAdapter()})
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_manifest_paths_cannot_escape_bundle(tmp_path):
    source = _trained("model")
    path = tmp_path / "edit"
    source.save_trained(str(path))
    manifest = read_edit_bundle(path)
    manifest["components"]["transformer"]["adapters"]["default"]["path"] = "../outside"
    (path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        source.load_trained(str(path))



def test_named_transformer_adapter_cannot_collide_with_encoder_component(tmp_path):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    def adapter(name):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        return get_peft_model(model, LoraConfig(r=1, lora_alpha=1, target_modules=["0"]),
                              adapter_name=name)

    transformer, encoder = adapter("text_encoder"), adapter("default")
    for module, value in ((transformer, 1.0), (encoder, 2.0)):
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.data.fill_(value)
    path = tmp_path / "bundle"
    manifest = save_edit_bundle(path, {"transformer": transformer, "text_encoder": encoder})
    assert manifest["components"]["transformer"]["adapters"]["text_encoder"]["path"] == "transformer/text_encoder"
    assert manifest["components"]["text_encoder"]["adapters"]["default"]["path"] == "text_encoder"
    restored, restored_encoder = adapter("text_encoder"), adapter("default")
    load_edit_bundle(path, {"transformer": restored, "text_encoder": restored_encoder})
    for source, actual, name in ((transformer, restored, "text_encoder"),
                                 (encoder, restored_encoder, "default")):
        expected = get_peft_model_state_dict(source, adapter_name=name)
        loaded = get_peft_model_state_dict(actual, adapter_name=name)
        assert all(torch.equal(value, loaded[key]) for key, value in expected.items())


def test_symlinked_adapter_config_cannot_escape_bundle(tmp_path):
    source = _trained("model")
    path = tmp_path / "edit"
    source.save_trained(path)
    config = path / "transformer" / "adapter_config.json"
    outside = tmp_path / "outside.json"
    config.replace(outside)
    config.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        source.load_trained(path)


@pytest.mark.parametrize("corruption", ["dtype", "nan", "inf"])
def test_numeric_preflight_rejects_later_component_before_loading_any_weights(corruption, tmp_path):
    from safetensors.torch import load_file, save_file

    source = {"transformer": torch.nn.Linear(2, 2), "text_encoder": torch.nn.Linear(2, 2)}
    path = tmp_path / "bundle"
    save_edit_bundle(path, source)
    weights = path / "text_encoder" / "weights.safetensors"
    state = load_file(str(weights))
    if corruption == "dtype":
        state["weight"] = state["weight"].double()
    else:
        state["weight"][0, 0] = float(corruption)
    save_file(state, str(weights))
    destination = {"transformer": torch.nn.Linear(2, 2), "text_encoder": torch.nn.Linear(2, 2)}
    before = {name: _state(module) for name, module in destination.items()}
    with pytest.raises(ValueError, match="dtypes|non-finite"):
        load_edit_bundle(path, destination)
    for name, module in destination.items():
        assert all(torch.equal(value, module.state_dict()[key]) for key, value in before[name].items())


def test_provider_load_hook_failure_rolls_back_all_component_weights(tmp_path):
    class Rejecting(torch.nn.Linear):
        def _load_from_state_dict(self, *args, **kwargs):
            # Simulate a provider hook failing after it already changed weights.
            super()._load_from_state_dict(*args, **kwargs)
            raise RuntimeError("provider rejected partially applied weights")

    source = {"transformer": torch.nn.Linear(2, 2), "text_encoder": torch.nn.Linear(2, 2)}
    path = tmp_path / "bundle"
    save_edit_bundle(path, source)
    destination = {"transformer": torch.nn.Linear(2, 2), "text_encoder": Rejecting(2, 2)}
    before = {name: _state(module) for name, module in destination.items()}
    with pytest.raises(RuntimeError, match="provider rejected"):
        load_edit_bundle(path, destination)
    for name, module in destination.items():
        assert all(torch.equal(value, module.state_dict()[key]) for key, value in before[name].items())


def test_rollback_failure_does_not_replace_primary_provider_error(tmp_path):
    class BrokenRollback(torch.nn.Linear):
        broken = False

        def _load_from_state_dict(self, *args, **kwargs):
            self.broken = True
            raise RuntimeError("primary provider error")

        def state_dict(self, *args, **kwargs):
            if self.broken:
                raise RuntimeError("cleanup state unavailable")
            return super().state_dict(*args, **kwargs)

    path = tmp_path / "bundle"
    save_edit_bundle(path, {"transformer": torch.nn.Linear(2, 2)})
    with pytest.raises(RuntimeError, match="primary provider error") as raised:
        load_edit_bundle(path, {"transformer": BrokenRollback(2, 2)})
    assert any("rollback failed" in note and "cleanup state unavailable" in note
               for note in raised.value.__notes__)
