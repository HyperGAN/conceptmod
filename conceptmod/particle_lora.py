"""Fixed-context particle LoRAs trained by the official ParticleGAN recipe.

This is a tensor/model integration, not a pretrained-image training CLI. The
caller supplies a frozen host, explicit Linear targets, one fixed input context,
a critic on the host's output tensors, and real target samples for that context.
No GAN losses, regularizers or optimizer updates are reimplemented here.
"""
from __future__ import annotations

import math
from copy import copy

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

    def forward(self, value, codes=None, *, strength=1.0):
        result = self.base(value)
        if strength == 0:
            return result
        if codes is None or value.shape[0] != codes.shape[0]:
            raise RuntimeError("particle LoRA must run through its generator with a matching batch")
        routed = self.route(codes).tanh()
        routed = routed.reshape(routed.shape[0], *([1] * (value.ndim - 2)), routed.shape[-1])
        delta = self.up(self.down(value.float()) * routed)
        return result + (strength * delta).to(result.dtype)


class _BoundLoRA(nn.Module):
    """An invocation's routing, retained by its checkpoint recompute closures."""

    def __init__(self, adapter, codes, strength):
        super().__init__()
        self.adapter = adapter
        # A sampled prior may be an nn.Parameter. It belongs to the prior,
        # never to this temporary module or the generator's state_dict.
        object.__setattr__(self, "codes", codes)
        self.strength = strength
        self.training = adapter.training

    def forward(self, value):
        return self.adapter(value, codes=self.codes, strength=self.strength)


def _bind_host(host, codes, strength):
    """Copy module containers, sharing all parameter and buffer tensors.

    Each checkpoint must capture its invocation's module objects rather than
    look up mutable routing state on the live host during backward. A shallow
    module-tree copy gives ordinary registered children that lifetime without
    copying frozen model weights, altering the live tree, or retaining routing
    on it. Memoization preserves shared child modules within the invocation.
    """
    memo = {}

    def bind(module):
        if module is None:
            return None
        if id(module) not in memo:
            if isinstance(module, RoutedLoRALinear):
                memo[id(module)] = _BoundLoRA(module, codes, strength)
            else:
                bound = copy(module)
                memo[id(module)] = bound
                bound._modules = {name: bind(child) for name, child in module._modules.items()}
        return memo[id(module)]

    return bind(host)


class ParticleLoRAGenerator(nn.Module):
    """Take ownership of a frozen host and install explicitly named adapters.

    ``context`` has a singleton leading batch dimension and is held constant
    across the GAN's samples. Wrap a multi-argument model in a Tensor-to-Tensor
    module with its remaining context bound before handing it to this class.
    The host is modified in place; create a separate host for each generator.
    Its frozen weights are included in the official trainer checkpoint.

    Hosts must access layers through their registered module tree, including
    inside activation-checkpoint closures. External closures or cached bound
    methods that capture the original host's layers bypass invocation binding.
    Reentrant and nonreentrant PyTorch activation checkpoints are supported;
    PyTorch's own restrictions on each checkpoint variant still apply.
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
        host = _bind_host(self.host, codes, float(strength))
        inputs = self.context.expand(codes.shape[0], *self.context.shape[1:])
        if torch.is_grad_enabled() and strength != 0:
            # Reentrant checkpoint needs a grad-requiring input even though
            # the fixed context itself is frozen. A fresh leaf belongs only
            # to this graph; its gradient cannot accumulate on the buffer.
            inputs = inputs.detach().requires_grad_(True)
        output = host(inputs)
        if not isinstance(output, torch.Tensor) or output.shape[0] != codes.shape[0]:
            raise ValueError("host must return a tensor with the same leading batch dimension")
        return output

    def make_trainer(self, critic: nn.Module, *, num_particles: int = 128,
                     batch_size: int = 32, total_steps: int = 1000, seed: int = 0):
        """Use the unmodified current official recipe and its learned prior.

        Only task dimensions, budget and explicit no-weight-EMA settings are
        overridden. ParticleGAN 0.9.x uses zero (not None) to disable its EMA.
        The public trainer owns BISPF, SBNP, DualNorm, prior updates, random
        streams and checkpoints.
        Call ``trainer.step(real_targets)`` and save ``trainer.state_dict()``.
        """
        recipe = get_recipe("gan", z_dim=self.z_dim, num_particles=num_particles,
                            batch_size=batch_size, total_steps=total_steps,
                            ema_decay=0.0, serve_average=0.0)
        if recipe.effective_critic_formulation != "bispf":
            raise RuntimeError("particle LoRA requires an official recipe without critic weight EMA")
        return GANTrainer(recipe, self, critic, seed=seed)
