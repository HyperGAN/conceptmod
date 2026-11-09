"""Model-independent pole, hold-direction and leftover geometry.

These tensor operations have no training or research-suite dependency. The
thresholds are historical CPU diagnostic defaults, not a candidate-selection
objective or a claim about finished-output edit quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LOCKED_TEACHER = "faithful_guard_e"
U_KEPT_MIN = 0.85
CONTENT_KEPT_MIN = 0.75
LEAK_RATIO_MAX = 0.20
POLE_REL_ERR_MAX = 0.20
SAME_DIR_MAX = 0.25


def _unit(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.flatten()
    return flat / flat.norm().clamp_min(1e-8)


def hold_dir(leak_dir: torch.Tensor, slider_dir: torch.Tensor) -> torch.Tensor | None:
    """ê perpendicular to û. Near-zero leftover turns the hold off."""
    axis = leak_dir.flatten()
    unit = _unit(slider_dir)
    out = axis - (axis @ unit) * unit
    if float(out.norm()) <= 1e-8:
        return None
    return out


def blend_guard(
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> bool:
    """True when the target is nearer its caption than the pair midpoint."""
    mid = 0.5 * (pos + neg)
    to_pole = max(float((tgt_plus - pos).norm()), float((tgt_minus - neg).norm()))
    to_mid = min(float((tgt_plus - mid).norm()), float((tgt_minus - mid).norm()))
    return to_pole < to_mid


def faithful_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    slider_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Caption poles with leftover ê removed from the odd part only."""
    axis = (pos - neg) / 2.0
    held = hold_dir(leak_dir, slider_dir)
    if held is not None:
        unit = _unit(held)
        axis = axis - ((axis.flatten() @ unit) * unit).view_as(axis)
    common = (pos + neg) / 2.0 - neu
    return neu + common + axis, neu + common - axis


def faithful_guard_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    slider_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract leftover ê when the blend guard admits it, else keep the captions."""
    plus, minus = faithful_sub_e(pos, neg, neu, leak_dir, slider_dir)
    if blend_guard(plus, minus, pos, neg):
        return plus, minus
    return pos, neg


@dataclass(frozen=True)
class LeftoverField:
    """One-row R^4 leftover: û, content, ê, lyric. Matches Field3D defaults."""

    slider: float = 1.0
    content: float = 0.55
    leak: float = 0.45
    lyric: float = 1.0

    @property
    def dim(self) -> int:
        return 4

    def basis(self, index: int) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[index] = 1.0
        return out

    def odd(self) -> torch.Tensor:
        return (
            float(self.slider) * self.basis(0)
            + float(self.content) * self.basis(1)
            + float(self.leak) * self.basis(2)
        )

    def poles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.basis(3)
        amplitude = self.odd()
        return neu + amplitude, neu - amplitude, neu


def teacher_poles(
    field: LeftoverField,
    teacher: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos, neg, neu = field.poles()
    mode = str(teacher).strip().lower()
    if mode == "faithful":
        return pos, neg, neu
    if mode == "faithful_guard_e":
        plus, minus = faithful_guard_e(pos, neg, neu, field.basis(2), field.basis(0))
        return plus, minus, neu
    raise ValueError(f"unsupported teacher {teacher!r}")


def leftover_bipolar(d_plus: torch.Tensor, d_minus: torch.Tensor) -> dict[str, float]:
    """``leak_frac = cos(d+, d−)``, ``same_dir`` = even / (even + odd)."""
    even = 0.5 * (d_plus + d_minus)
    odd = 0.5 * (d_plus - d_minus)
    even_n = float(even.norm())
    odd_n = float(odd.norm())
    return {
        "leak_frac": float(
            F.cosine_similarity(d_plus.flatten().unsqueeze(0), d_minus.flatten().unsqueeze(0))
        ),
        "same_dir": even_n / (even_n + odd_n + 1e-8),
        "even_norm": even_n,
        "odd_norm": odd_n,
    }


