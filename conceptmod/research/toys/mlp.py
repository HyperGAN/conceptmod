"""Host MLP for the ring mode-hold toy.

ParticleGAN's installable package does not export the 100-Gaussians
generator and discriminator (``lib/toy_models.py``). The ring gate keeps
that critic shape here so the toy does not swap architectures. This is
not a copy of ``GradRegularizer``; the penalty still comes from
``particlegan.GradientPenalty``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLPGenerator(nn.Module):
    """Small MLP generator: z -> x in R^2."""

    def __init__(
        self,
        z_dim: int = 4,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SimpleMLPDiscriminator(nn.Module):
    """MLP critic: x in R^2 -> scalar score.

    ``fourier=K`` appends sin/cos features at frequencies ``pi * 2^i``
    (i < K) per input dimension.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        fourier: int = 2,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        dim = in_dim + (2 * fourier * in_dim if fourier > 0 else 0)
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs
            h = torch.cat([h, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        return self.net(h).squeeze(-1)
