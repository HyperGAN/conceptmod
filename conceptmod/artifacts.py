"""Complete, model-neutral inference bundles for concept edits.

An edit can change more than one model component. A bundle records every
changed component; an individual adapter export is a separate operation.
These are inference artifacts, not optimizer/RNG checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import nn

MANIFEST_NAME = "edit_bundle.json"
_FORMAT = "conceptmod.edit-bundle"
_VERSION = 1


def _adapter_names(module: nn.Module) -> list[str]:
    names = list(getattr(module, "peft_config", {}))
    for name in names:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"unsafe adapter name {name!r}")
    return names


def save_transformer_weights(module: nn.Module, path: str | Path) -> None:
    """Export one transformer in the legacy PEFT-directory / safetensors format."""
    if _adapter_names(module):
        module.save_pretrained(str(path), safe_serialization=True)
    else:
        from safetensors.torch import save_file

        # Preserve the current weights and their precision, including buffers.
        state = {k: v.detach().contiguous().cpu().clone()
                 for k, v in module.state_dict().items()}
        save_file(state, str(path))


def save_edit_bundle(path: str | Path, components: Mapping[str, nn.Module],
                     *, metadata: dict | None = None) -> dict:
    """Save all edited components to a new or empty directory.

    Every component owns a separate directory, including the transformer.
    Adapter names cannot collide with other components. Write to a sibling
    temporary directory first so a failed export cannot publish a partial edit.
    """
    path = Path(path)
    if not components or "transformer" not in components:
        raise ValueError("an edit bundle requires a transformer component")
    for name in components:
        if not name.isidentifier():
            raise ValueError(f"invalid component name {name!r}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"bundle destination must be new or empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format": _FORMAT, "version": _VERSION,
                "metadata": metadata or {}, "components": {}}
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as tmp:
        root = Path(tmp)
        for name, module in components.items():
            adapter_names = _adapter_names(module)
            if adapter_names:
                relative = name
                target = root / relative
                target.mkdir()
                module.save_pretrained(str(target), safe_serialization=True)
                adapters = {}
                for adapter in adapter_names:
                    folder = target if adapter == "default" else target / adapter
                    config = json.loads((folder / "adapter_config.json").read_text())
                    adapters[adapter] = {
                        "path": folder.relative_to(root).as_posix(),
                        "rank": config.get("r"),
                    }
                entry = {"kind": "peft", "path": relative, "adapters": adapters,
                         "active_adapters": list(module.active_adapters)}
            else:
                relative = f"{name}/weights.safetensors"
                (root / name).mkdir()
                save_transformer_weights(module, root / relative)
                entry = {"kind": "state_dict", "path": relative}
            manifest["components"][name] = entry
        (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        if path.exists():
            path.rmdir()  # Only an empty destination was accepted above.
        os.replace(root, path)
    return manifest


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"invalid bundle-relative path {relative!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"bundle path escapes its directory: {relative!r}")
    return target


def read_edit_bundle(path: str | Path) -> dict:
    """Read the versioned manifest without loading model weights."""
    root = Path(path).resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    if manifest.get("format") != _FORMAT or manifest.get("version") != _VERSION:
        raise ValueError("unsupported conceptmod edit bundle format or version")
    if not isinstance(manifest.get("metadata"), dict):
        raise ValueError("bundle metadata must be an object")
    components = manifest.get("components")
    if not isinstance(components, dict) or "transformer" not in components:
        raise ValueError("bundle must list a transformer component")
    return manifest


def _normalized_config(config: dict) -> dict:
    # These are loader hints, not adapter behavior. PEFT infers auto_mapping
    # only while saving; this loader receives already initialized modules.
    config = {k: v for k, v in config.items()
              if k not in {"inference_mode", "auto_mapping"}}
    return json.loads(json.dumps(config, default=lambda value: sorted(value)))


def _validate_state(name: str, saved: dict, expected: dict) -> None:
    missing = expected.keys() - saved.keys()
    extra = saved.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"{name}: incompatible weight keys; "
                         f"missing={sorted(missing)}, extra={sorted(extra)}")
    bad = [key for key in expected if saved[key].shape != expected[key].shape]
    if bad:
        raise ValueError(f"{name}: incompatible weight shapes: {bad}")
    bad = [key for key in expected if saved[key].dtype != expected[key].dtype]
    if bad:
        raise ValueError(f"{name}: incompatible weight dtypes: {bad}")
    bad = [key for key, value in saved.items() if not torch.isfinite(value).all()]
    if bad:
        raise ValueError(f"{name}: non-finite weights: {bad}")


def load_edit_bundle(path: str | Path, components: Mapping[str, nn.Module],
                     *, expected_metadata: dict | None = None) -> dict:
    """Load into matching initialized components, after validating the whole edit.

    Adapters must already be attached with their saved configuration. All
    component names, PEFT configurations, weight keys and shapes are checked
    before any component's weights are changed. No pickle files are loaded.
    """
    from safetensors.torch import load_file

    root = Path(path).resolve()
    manifest = read_edit_bundle(root)
    if expected_metadata is not None and manifest["metadata"] != expected_metadata:
        raise ValueError("bundle metadata does not match the initialized backend")
    if set(manifest["components"]) != set(components):
        raise ValueError("bundle components do not match the initialized backend; "
                         "attach the saved encoder adapters before loading")
    pending = []
    for name, module in components.items():
        entry = manifest["components"][name]
        if entry["kind"] == "peft":
            from peft import PeftConfig, get_peft_model_state_dict

            adapters = entry["adapters"]
            if set(adapters) != set(_adapter_names(module)):
                raise ValueError(f"{name}: saved and initialized adapters differ")
            if entry["active_adapters"] != list(module.active_adapters):
                raise ValueError(f"{name}: saved and initialized active adapters differ")
            for adapter, info in adapters.items():
                config_path = _inside(root, info["path"] + "/adapter_config.json")
                config = PeftConfig.from_peft_type(
                    **json.loads(config_path.read_text())
                ).to_dict()
                actual = module.peft_config[adapter].to_dict()
                if _normalized_config(config) != _normalized_config(actual):
                    raise ValueError(f"{name}.{adapter}: incompatible PEFT configuration")
                state = load_file(str(_inside(root, info["path"] + "/adapter_model.safetensors")))
                expected = get_peft_model_state_dict(module, adapter_name=adapter)
                _validate_state(f"{name}.{adapter}", state, expected)
                pending.append((module, adapter, state, expected))
        elif entry["kind"] == "state_dict":
            if _adapter_names(module):
                raise ValueError(f"{name}: full weights require an unwrapped module")
            state = load_file(str(_inside(root, entry["path"])))
            expected = module.state_dict()
            _validate_state(name, state, expected)
            pending.append((module, None, state, expected))
        else:
            raise ValueError(f"{name}: unsupported component kind {entry['kind']!r}")
    # Snapshot the live tensors that a load can change. PEFT state normally
    # aliases its underlying adapter tensors, so frozen multi-billion-parameter
    # hosts do not need a second copy. Unusual serializers that compute new
    # tensors fall back to a full snapshot. Keep rollback copies off the GPU.
    targets = {}
    for module, adapter, state, expected in pending:
        live = module.state_dict()
        def storage_key(value):
            return (value.device, value.untyped_storage().data_ptr())
        expected_storage = {storage_key(value) for value in expected.values()}
        live_storage = {storage_key(value) for value in live.values()}
        keys = (live.keys() if adapter is None or not expected_storage <= live_storage else
                [key for key, value in live.items() if storage_key(value) in expected_storage])
        targets.setdefault(module, {}).update({key: live[key].detach().cpu().clone() for key in keys})
    try:
        for module, adapter, state, expected in pending:
            if adapter is None:
                module.load_state_dict(state, strict=True)
            else:
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(module, state, adapter_name=adapter)
    except BaseException as error:
        # Restore through tensor copies, so a failing provider load hook cannot
        # prevent rollback by raising a second time. Preserve the primary error.
        with torch.no_grad():
            for module, previous in targets.items():
                try:
                    current = module.state_dict()
                    for key, value in previous.items():
                        current[key].copy_(value)
                except BaseException as cleanup_error:
                    error.add_note(f"bundle rollback failed: {cleanup_error!r}")
        raise
    return manifest
