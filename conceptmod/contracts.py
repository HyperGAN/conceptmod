"""Model-independent capabilities and explicit original/current model views.

Provider metadata must be importable without constructing models or importing
their optional inference libraries. Runtime implementations live in products.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable


VELOCITY_OPERATORS = frozenset({"write", "freeze", "erase", "exaggerate", "orthogonal"})
MODEL_OPERATORS = VELOCITY_OPERATORS | {"pixel"}


@dataclass(frozen=True)
class BackendCapabilities:
    operators: frozenset[str] = MODEL_OPERATORS
    stages: frozenset[str] = frozenset({"model", "encoder"})
    differentiable_render: bool = True
    api_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "operators", frozenset(self.operators))
        object.__setattr__(self, "stages", frozenset(self.stages))
        if self.api_version != 1:
            raise ValueError(f"unsupported backend API version {self.api_version}")
        if not self.stages or not self.stages <= {"model", "encoder"}:
            raise ValueError("backend stages must contain model and/or encoder")


@runtime_checkable
class VelocityBackend(Protocol):
    device: Any
    latent_shape: tuple

    def encode_text(self, prompt: str, frozen: bool = False): ...
    def predict_v(self, prompt, z, timestep, frozen: bool): ...
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator): ...
    def trainable_parameters(self, train_method: str): ...


@runtime_checkable
class TrainableEncoder(Protocol):
    def attach_encoder_lora(self, rank: int = 8): ...
    def encode_text_grad(self, prompt: str): ...
    def encode_text(self, prompt: str, frozen: bool = False): ...


@runtime_checkable
class DifferentiableRenderer(Protocol):
    def render(self, prompt, generator, num_steps, guidance,
               grad_steps: int = 0, frozen: bool = False): ...


@dataclass(frozen=True)
class ModelView:
    """One role applies to both the encoder and the velocity model.

    Products must honor ``frozen`` consistently in every implementation. A view
    does not clone weights or acquire implicit ownership of a runtime.
    """
    backend: Any
    frozen: bool

    @contextmanager
    def _reference_context(self):
        if not self.frozen:
            yield
            return
        import torch

        declared = getattr(self.backend, "training_modules", None)
        roots = list(declared().values()) if callable(declared) else []
        modes = {id(module): (module, module.training)
                 for root in roots for module in root.modules()}
        try:
            for root in roots:
                root.eval()
            with torch.no_grad():
                yield
        finally:
            for module, training in modes.values():
                module.training = training

    def encode_text(self, prompt: str):
        with self._reference_context():
            return self.backend.encode_text(prompt, frozen=self.frozen)

    def predict_v(self, prompt, z, timestep):
        with self._reference_context():
            return self.backend.predict_v(prompt, z, timestep, frozen=self.frozen)

    def render(self, prompt, generator, num_steps, guidance, *, grad_steps=0):
        with self._reference_context():
            return self.backend.render(prompt, generator, num_steps, guidance,
                                       grad_steps=grad_steps, frozen=self.frozen)

    def generate(self, prompt, seed, num_steps=None, guidance=None):
        with self._reference_context():
            return self.backend.generate(prompt, seed, num_steps, guidance,
                                         frozen=self.frozen)


def model_views(backend) -> tuple[ModelView, ModelView]:
    """Return (original, current), with no hidden adapter or mode transition."""
    return ModelView(backend, True), ModelView(backend, False)
