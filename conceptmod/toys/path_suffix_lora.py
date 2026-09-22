"""Path-suffix LoRA honesty on a CPU block with colliding ``proj`` leaves.

PEFT matches a ``target_modules`` list entry when the module key is exactly
that string or when the key ends with ``"." + suffix``. A bare ``proj``
therefore attaches to every leaf named ``proj``. On this block those leaves
are ``self_attn.proj``, ``cross_attn.proj``, and a bare ``proj`` head. The
head is the wrong tiny module: it takes a real ``lora_B`` gradient, and the
coverage gate fails.

The locked targets name the attention paths:

* suffix list ``self_attn.proj`` and ``cross_attn.proj``
* the equivalent child regex ``(?:self_attn|cross_attn)\\..+``

That regex is the ``self_attn.*`` / ``cross_attn.*`` spelling that still
lands on a ``Linear``. The literal pattern ``self_attn.*`` fullmatches the
parent ``self_attn`` module (the ``.*`` may be empty). PEFT then refuses
because that parent is not a ``Linear``, and the trainable set is empty.

Anima's diffusers attach (``conceptmod.backends.anima``) uses the leaf
suffixes ``to_q``, ``to_k``, ``to_v``, ``to_out.0``. ``conceptmod.convert``
renames those to ``self_attn.q_proj`` / ``cross_attn.q_proj`` and
``self_attn.output_proj`` / ``cross_attn.output_proj``. A widened target
also reaches ``ff.net.0.proj``; roots such as ``proj_out`` are dropped
rather than guessed (``ANIMA_DROP``). Encoder LoRA uses ``q_proj`` and the
other exact leaf names, which do not match a module named ``proj``. This
toy is that collision with no Hub weights and no GPU.

Coverage is the fraction of the two attention leaves whose mean absolute
``lora_B`` gradient is at least ``GRAD_ABS_MIN``. ``lora_A`` is overwritten
to a fixed matrix and ``lora_B`` starts at 0, so only ``lora_B`` carries
the signal. Any non-intended leaf at or above the floor (the bare head)
zeroes coverage. An empty match raises :class:`EmptyTrainableError`.

``python -m conceptmod.toys.path_suffix_lora`` prints one line per arm.
A PASS here is not an Anima GPU transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

DIM = 4
RANK = 2
BATCH = 2
BASE_WEIGHT = 0.25
# Mean |lora_B grad| on a locked attention leaf, and on the bare head when
# that head is the only module (or one of the modules) in the adapter.
# dim 4, rank 2, alpha 2, base weight 0.25, lora_A = arange/numel, lora_B = 0,
# batch of ones, loss = mean square. Both values are exact in float32.
LOCKED_LEAF_GRAD = 2.25
BARE_LEAF_GRAD = 4.5
GRAD_ABS_MIN = 1.0

INTENDED = ("self_attn.proj", "cross_attn.proj")
BARE = "proj"
LOCKED_SUFFIXES = ("self_attn.proj", "cross_attn.proj")
# Child of self_attn or cross_attn. Does not fullmatch the parent module.
LOCKED_REGEX = r"(?:self_attn|cross_attn)\..+"

_LORA_B = ".lora_B."


class EmptyTrainableError(RuntimeError):
    """``target_modules`` matched no trainable Linear.

    A silent empty adapter would report no bad gradient and look like a pass.
    """


@dataclass(frozen=True)
class Arm:
    name: str
    targets: tuple[str, ...] | str
    refuse_empty: bool = False


@dataclass(frozen=True)
class SuffixReport:
    name: str
    targets: tuple[str, ...] | str
    passed: bool
    reasons: tuple[str, ...]
    coverage: float | None
    self_abs: float | None
    cross_abs: float | None
    bare_abs: float | None
    refused: bool = False
    detail: str = ""

    def line(self) -> str:
        if self.refused or self.coverage is None:
            return (
                f"{self.name} coverage=refused empty_trainable "
                f"REFUSE {self.detail}"
            )
        flag = "PASS" if self.passed else "FAIL"
        why = (" " + " ".join(self.reasons)) if self.reasons else ""
        return (
            f"{self.name} coverage={self.coverage:.2f} "
            f"self={self.self_abs:.2f} cross={self.cross_abs:.2f} "
            f"bare={self.bare_abs:.2f} {flag}{why}"
        )


class _Attn(nn.Module):
    """One attention site whose only Linear is named ``proj``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CollidingBlock(nn.Module):
    """``self_attn.proj``, ``cross_attn.proj``, and a bare ``proj`` head."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.self_attn = _Attn(dim)
        self.cross_attn = _Attn(dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.self_attn(x) + self.cross_attn(x))


def linear_paths(module: nn.Module) -> tuple[str, ...]:
    """Module paths of every ``nn.Linear``, in ``named_modules`` order."""
    return tuple(
        name for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    )


def arms() -> tuple[Arm, ...]:
    """Locked suffixes pass. A bare ``proj``, a wrong suffix, and an empty set do not."""
    return (
        Arm("locked_suffix", LOCKED_SUFFIXES),
        Arm("locked_regex", LOCKED_REGEX),
        Arm("bare_proj", ("proj",)),
        Arm("head_only", "proj"),
        Arm("partial_suffix", ("self_attn.proj",)),
        Arm("overbroad_star", r"self_attn.*", refuse_empty=True),
        Arm("empty_q_proj", ("q_proj",), refuse_empty=True),
    )


def _pin_base(block: CollidingBlock) -> None:
    with torch.no_grad():
        for layer in (block.self_attn.proj, block.cross_attn.proj, block.proj):
            layer.weight.fill_(BASE_WEIGHT)


def _pin_lora(model: nn.Module) -> None:
    """Fixed ``lora_A`` and zero ``lora_B``, independent of PEFT's init."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.copy_(
                    torch.arange(1, param.numel() + 1, dtype=param.dtype).reshape_as(param)
                    / param.numel()
                )
            elif "lora_B" in name:
                param.zero_()


def _leaf_of_lora_b(name: str) -> str | None:
    if _LORA_B not in name:
        return None
    head = name.split(_LORA_B, 1)[0]
    prefix = "base_model.model."
    if head.startswith(prefix):
        head = head[len(prefix):]
    return head


def _mean_abs(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().abs().mean())


def _coverage(leaf_abs: dict[str, float]) -> tuple[float, tuple[str, ...]]:
    intended_vals = [leaf_abs.get(name, 0.0) for name in INTENDED]
    hits = sum(value >= GRAD_ABS_MIN for value in intended_vals)
    raw = hits / len(INTENDED)
    bare_abs = leaf_abs.get(BARE, 0.0)
    other = {
        name: value for name, value in leaf_abs.items()
        if name not in INTENDED and name != BARE and value >= GRAD_ABS_MIN
    }
    reasons: list[str] = []
    leak = bare_abs >= GRAD_ABS_MIN or bool(other)
    if bare_abs >= GRAD_ABS_MIN or other:
        reasons.append("bare_proj" if bare_abs >= GRAD_ABS_MIN and not other else "unintended")
    if raw < 1.0:
        reasons.append("coverage")
    # A colliding head still moves the attention leaves. Coverage counts only
    # a locked set that is fully covered and alone.
    coverage = 0.0 if leak else raw
    return coverage, tuple(reasons)


def attach(block: CollidingBlock, targets: tuple[str, ...] | str) -> nn.Module:
    """Wrap ``block`` with PEFT LoRA. An empty or non-Linear match is refused."""
    from peft import LoraConfig, NoMatchingPeftModuleError, get_peft_model

    spec: list[str] | str = list(targets) if isinstance(targets, tuple) else targets
    try:
        model = get_peft_model(
            block,
            LoraConfig(
                r=RANK,
                lora_alpha=RANK,
                target_modules=spec,
                bias="none",
                lora_dropout=0.0,
                use_rslora=False,
            ),
        )
    except NoMatchingPeftModuleError as exc:
        raise EmptyTrainableError(
            f"empty trainable set for targets {spec!r}: {exc}"
        ) from exc
    except ValueError as exc:
        text = str(exc)
        if "not supported" in text or "Target modules" in text:
            raise EmptyTrainableError(
                f"empty trainable set for targets {spec!r}: {exc}"
            ) from exc
        raise
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise EmptyTrainableError(
            f"empty trainable set for targets {spec!r}: LoRA added no parameters"
        )
    base = [name for name in trainable if "lora_" not in name]
    if base:
        raise RuntimeError(
            f"targets {spec!r} left non-LoRA parameters trainable: {base}"
        )
    return model


def probe(arm: Arm) -> SuffixReport:
    """One CPU backward. Measures mean |lora_B grad| on each colliding leaf."""
    block = CollidingBlock()
    _pin_base(block)
    model = attach(block, arm.targets)
    _pin_lora(model)
    x = torch.ones(BATCH, DIM)
    loss = model(x).square().mean()
    loss.backward()
    leaf_abs: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type != "cpu":
            raise RuntimeError(f"path-suffix toy left {name} on {param.device}")
        leaf = _leaf_of_lora_b(name)
        if leaf is None:
            continue
        leaf_abs[leaf] = _mean_abs(param)
    coverage, reasons = _coverage(leaf_abs)
    return SuffixReport(
        name=arm.name,
        targets=arm.targets,
        passed=not reasons,
        reasons=reasons,
        coverage=coverage,
        self_abs=leaf_abs.get(INTENDED[0], 0.0),
        cross_abs=leaf_abs.get(INTENDED[1], 0.0),
        bare_abs=leaf_abs.get(BARE, 0.0),
    )


def _refused(arm: Arm, exc: EmptyTrainableError) -> SuffixReport:
    return SuffixReport(
        name=arm.name,
        targets=arm.targets,
        passed=False,
        reasons=("empty_trainable",),
        coverage=None,
        self_abs=None,
        cross_abs=None,
        bare_abs=None,
        refused=True,
        detail=str(exc),
    )


def run_board() -> list[SuffixReport]:
    """Score every arm. Empty matches are refused, not recorded as a quiet pass."""
    rows: list[SuffixReport] = []
    for arm in arms():
        try:
            row = probe(arm)
        except EmptyTrainableError as exc:
            if not arm.refuse_empty:
                raise
            row = _refused(arm, exc)
        else:
            if arm.refuse_empty:
                raise RuntimeError(
                    f"{arm.name} matched a Linear; expected an empty trainable set"
                )
        print(row.line(), flush=True)
        rows.append(row)
    passed = sum(row.passed for row in rows)
    print(
        f"PATH_SUFFIX family=path_suffix_lora pass={passed}/{len(rows)}",
        flush=True,
    )
    return rows


def _main() -> None:
    run_board()


if __name__ == "__main__":
    _main()
