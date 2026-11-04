"""Fixed-context particle LoRAs trained by the official ParticleGAN recipe.

This is a tensor/model integration, not a pretrained-image training CLI. The
caller supplies a frozen host, explicit Linear targets, one fixed input context,
a critic on the host's output tensors, and real target samples for that context.
No GAN losses, regularizers or optimizer updates are reimplemented here.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from particlegan import GANTrainer, get_recipe


class RoutedLoRALinear(nn.Module):
    """Frozen Linear plus a rank-r residual gated by each sampled particle."""

    def __init__(self, base: nn.Linear, z_dim: int, rank: int):
        super().__init__()
        self.base = base
        device = base.weight.device
        self.down = nn.Linear(base.in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.route = nn.Linear(z_dim, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        self._codes = None
        self._strength = 1.0

    def forward(self, value):
        result = self.base(value)
        if self._strength == 0:
            return result
        if self._codes is None or value.shape[0] != self._codes.shape[0]:
            raise RuntimeError("particle LoRA must run through its generator with a matching batch")
        routed = self.route(self._codes).tanh()
        routed = routed.reshape(routed.shape[0], *([1] * (value.ndim - 2)), routed.shape[-1])
        delta = self.up(self.down(value.float()) * routed)
        return result + (self._strength * delta).to(result.dtype)


class ParticleLoRAGenerator(nn.Module):
    """Take ownership of a frozen host and install explicitly named adapters.

    ``context`` has a singleton leading batch dimension and is held constant
    across the GAN's samples. Wrap a multi-argument model in a Tensor-to-Tensor
    module with its remaining context bound before handing it to this class.
    The host is modified in place; create a separate host for each generator.
    Its frozen weights are included in the official trainer checkpoint.
    """

    def __init__(self, host: nn.Module, targets: tuple[str, ...], context: torch.Tensor,
                 *, z_dim: int = 4, rank: int = 8):
        super().__init__()
        targets = tuple(targets)
        if not targets or any(not name for name in targets) or len(set(targets)) != len(targets):
            raise ValueError("explicit unique Linear target names are required")
        if z_dim < 1 or rank < 1:
            raise ValueError("z_dim and rank must be positive")
        if any(p.requires_grad for p in host.parameters()):
            raise ValueError("host must already be frozen")
        if context.ndim < 2 or context.shape[0] != 1:
            raise ValueError("fixed context must have singleton batch dimension")
        modules = dict(host.named_modules())
        selected = [modules.get(name) for name in targets]
        if any(not isinstance(module, nn.Linear) for module in selected):
            raise ValueError("every target must name a Linear in the frozen host")
        if len({id(module) for module in selected}) != len(selected):
            raise ValueError("aliased targets are not supported")
        if any(module.weight.device != context.device for module in selected):
            raise ValueError("host targets and context must share a device")
        self.host = host
        self.targets, self.z_dim, self.rank = targets, z_dim, rank
        self.register_buffer("context", context.detach().clone())
        # Validate everything before mutating the caller's host.
        for name, base in zip(targets, selected):
            parent_name, _, child = name.rpartition(".")
            parent = host.get_submodule(parent_name) if parent_name else host
            setattr(parent, child, RoutedLoRALinear(base, z_dim, rank))
        self.host.eval()

    def train(self, mode: bool = True):
        # GANTrainer toggles G.train()/eval(); pretrained dropout and running
        # statistics must stay frozen along with the base parameters.
        super().train(mode)
        self.host.eval()
        return self

    @property
    def adapters(self):
        return tuple(self.host.get_submodule(name) for name in self.targets)

    def forward(self, codes: torch.Tensor, *, strength: float = 1.0):
        if codes.ndim != 2 or codes.shape[1] != self.z_dim:
            raise ValueError(f"codes must have shape (batch, {self.z_dim})")
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
        adapters = self.adapters
        if any(layer._codes is not None for layer in adapters):
            raise RuntimeError("particle generator is already evaluating a context")
        try:
            for layer in adapters:
                object.__setattr__(layer, "_codes", codes)
                layer._strength = float(strength)
            inputs = self.context.expand(codes.shape[0], *self.context.shape[1:])
            output = self.host(inputs)
            if not isinstance(output, torch.Tensor) or output.shape[0] != codes.shape[0]:
                raise ValueError("host must return a tensor with the same leading batch dimension")
            return output
        finally:
            for layer in adapters:
                object.__setattr__(layer, "_codes", None)
                layer._strength = 1.0

    def make_trainer(self, critic: nn.Module, *, num_particles: int = 128,
                     batch_size: int = 32, total_steps: int = 1000, seed: int = 0):
        """Use the unmodified current official recipe and its learned prior.

        Only task dimensions and budget are overridden. The public trainer
        owns BISPF, SBNP, DualNorm, prior updates, random streams and checkpoints.
        Call ``trainer.step(real_targets)`` and save ``trainer.state_dict()``.
        """
        recipe = get_recipe("gan", z_dim=self.z_dim, num_particles=num_particles,
                            batch_size=batch_size, total_steps=total_steps)
        return GANTrainer(recipe, self, critic, seed=seed)
