"""Backend protocol: what a diffusion model must expose for conceptmod.

All models are flow-matching: the network predicts a velocity v(z, t, text)
and an Euler step is ``z_next = z + (sigma_next - sigma) * v``. Everything the
DSL does is built from velocity predictions of a *trained* and a *frozen*
copy of the same network.
"""

from __future__ import annotations

import abc

import torch


def require_cuda(device: str) -> torch.device:
    """All training / sampling runs on one CUDA device. CPU is not a fallback."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"backends require a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return dev


def pin_modules(root, device: str) -> list[str]:
    """Move every nn.Module hanging off ``root`` onto ``device`` and check it stuck."""
    dev = require_cuda(device)
    moved = []
    for name in dir(root):
        if name.startswith("_"):
            continue
        try:
            mod = getattr(root, name)
        except Exception:
            continue
        if isinstance(mod, torch.nn.Module):
            mod.to(dev)
            _assert_cuda(mod, dev, name)
            moved.append(name)
    return moved


def _assert_cuda(module: torch.nn.Module, device: torch.device, name: str) -> None:
    want = device.index
    for n, p in module.named_parameters():
        if p.device.type != "cuda" or (want is not None and p.device.index != want):
            raise RuntimeError(f"{name}.{n} is on {p.device}, expected {device}")
    for n, b in module.named_buffers():
        if b.device.type != "cuda" or (want is not None and b.device.index != want):
            raise RuntimeError(f"{name}.{n} buffer is on {b.device}, expected {device}")


class TextEmbeds:
    """Prompt embeddings plus attention mask (mask may be None)."""

    def __init__(self, embeds: torch.Tensor | list[torch.Tensor], mask: torch.Tensor | None):
        self.embeds = embeds
        self.mask = mask


class Backend(abc.ABC):
    device: str
    latent_shape: tuple  # (C, H, W) for one sample

    @abc.abstractmethod
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        """Encode a prompt (cached). '' is the unconditional prompt."""

    @abc.abstractmethod
    def predict_v(self, prompt: str, z: torch.Tensor, timestep: torch.Tensor,
                  frozen: bool) -> torch.Tensor:
        """Velocity prediction for latents z at a scheduler timestep."""

    @abc.abstractmethod
    def partial_denoise(self, prompt: str, stop_index: int, num_steps: int,
                        guidance: float, generator: torch.Generator
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Euler steps from pure noise down to ``timesteps[stop_index]``
        using the trained model (no grad). Returns (z_t, timestep_t)."""

    @abc.abstractmethod
    def generate(self, prompt: str, seed: int, num_steps: int, guidance: float,
                 frozen: bool = False):
        """Full generation -> PIL.Image. ``frozen=True`` uses the untouched
        reference model (the 'before' image)."""

    @abc.abstractmethod
    def trainable_parameters(self, train_method: str) -> list[torch.nn.Parameter]:
        """Select the parameter group to finetune (xattn/selfattn/full/...)."""

    def save_trained(self, path: str) -> None:
        """Save a complete inference bundle, including any edited encoder."""
        from conceptmod.artifacts import save_edit_bundle

        save_edit_bundle(path, self.edit_components(), metadata=self.artifact_metadata())

    def save_transformer(self, path: str) -> None:
        """Export only the transformer in the legacy adapter / safetensors format."""
        from conceptmod.artifacts import save_transformer_weights

        save_transformer_weights(self.transformer, path)

    def edit_components(self) -> dict[str, torch.nn.Module]:
        """Explicit component boundary for complete inference artifacts."""
        return {"transformer": self.transformer}

    def artifact_metadata(self) -> dict:
        """Effective initialization and sampling settings, independent of CLI flags."""
        metadata = {"backend": type(self).__name__, "weight_provenance": "raw-current"}
        for name in ("model_id", "base_seed", "resolution", "lora_rank",
                     "generate_steps", "generate_guidance", "ckpt_path"):
            if hasattr(self, name):
                metadata[name] = getattr(self, name)
        metadata["latent_shape"] = list(self.latent_shape)
        return metadata

    def load_trained(self, path: str) -> None:
        """Restore a bundle into an identically initialized backend and adapters.

        Attach encoder LoRA with the saved rank first, if the bundle has one.
        Loading preserves the original reference model and invalidates embeddings.
        """
        from conceptmod.artifacts import load_edit_bundle

        load_edit_bundle(path, self.edit_components(),
                         expected_metadata=self.artifact_metadata())
        self.invalidate_caches()

    def invalidate_caches(self) -> None:
        """Invalidate model-dependent embeddings after an update or restore.

        Products with other caches override this hook. Callers never reach into
        a product's private dictionaries or retain an autograd graph in them.
        """
        cache = getattr(self, "_text_cache", None)
        if cache is not None:
            cache.clear()

    def training_modules(self) -> dict[str, torch.nn.Module]:
        """Declare current and original modules needed for exact continuation."""
        modules = {}
        for name in ("transformer", "text_encoder", "frozen"):
            module = getattr(self, name, None)
            if module is None and name == "text_encoder":
                module = getattr(getattr(self, "pipe", None), name, None)
            if isinstance(module, torch.nn.Module):
                modules[name] = module
        return modules

    def training_identity(self) -> dict:
        """Declare behavior which a checkpoint cannot change by loading tensors.

        Products add any non-module behavior settings to artifact_metadata or
        override this method. A state dict alone cannot describe arbitrary
        Python control flow, tokenizer conventions or stochastic policies.
        """
        return {
            "backend_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "metadata": self.artifact_metadata(),
            "module_graph": {
                root_name: [(name, f"{type(module).__module__}.{type(module).__qualname__}",
                             module.extra_repr()) for name, module in root.named_modules()]
                for root_name, root in self.training_modules().items()
            },
        }

    @property
    def original_view(self):
        from conceptmod.contracts import ModelView

        return ModelView(self, True)

    @property
    def current_view(self):
        from conceptmod.contracts import ModelView

        return ModelView(self, False)

    def training_defaults(self) -> dict:
        """Backend-specific OpDefaults overrides (sample_steps / sample_guidance)."""
        return {}
