"""CPU gate: path-suffix LoRA hits the attention projs, a bare proj does not.

Locked suffixes ``self_attn.proj`` and ``cross_attn.proj`` (and the child
regex) put mean |lora_B grad| on both attention leaves and leave the bare
head alone. A bare ``proj`` suffix collides with that head. A wrong suffix,
a one-sided suffix, and an empty target fail or are refused.

No Hub download. CPU only.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from conceptmod.toys.path_suffix_lora import (
    BARE_LEAF_GRAD,
    GRAD_ABS_MIN,
    LOCKED_LEAF_GRAD,
    CollidingBlock,
    EmptyTrainableError,
    arms,
    linear_paths,
    probe,
    run_board,
)


def _arm(name: str):
    return next(arm for arm in arms() if arm.name == name)


def test_block_has_three_colliding_proj_leaves():
    block = CollidingBlock()
    assert linear_paths(block) == ("self_attn.proj", "cross_attn.proj", "proj")
    leaves = {
        name: child for name, child in block.named_modules()
        if isinstance(child, nn.Linear)
    }
    assert len({id(leaf.weight) for leaf in leaves.values()}) == 3
    assert all(param.device.type == "cpu" for param in block.parameters())


def test_locked_suffix_covers_attention_and_not_the_head():
    row = probe(_arm("locked_suffix"))
    assert row.passed, row.reasons
    assert row.coverage == pytest.approx(1.0)
    assert row.self_abs == pytest.approx(LOCKED_LEAF_GRAD)
    assert row.cross_abs == pytest.approx(LOCKED_LEAF_GRAD)
    assert row.self_abs == pytest.approx(2.25)
    assert row.cross_abs == pytest.approx(2.25)
    assert row.self_abs >= GRAD_ABS_MIN
    assert row.cross_abs >= GRAD_ABS_MIN
    assert row.bare_abs == pytest.approx(0.0)
    assert row.reasons == ()


def test_locked_child_regex_matches_the_suffix_list():
    row = probe(_arm("locked_regex"))
    assert row.passed, row.reasons
    assert row.coverage == pytest.approx(1.0)
    assert row.self_abs == pytest.approx(2.25)
    assert row.cross_abs == pytest.approx(2.25)
    assert row.bare_abs == pytest.approx(0.0)


def test_bare_proj_suffix_collides_and_fails_coverage():
    row = probe(_arm("bare_proj"))
    assert row.passed is False
    assert row.coverage == pytest.approx(0.0)
    assert row.self_abs == pytest.approx(2.25)
    assert row.cross_abs == pytest.approx(2.25)
    assert row.bare_abs == pytest.approx(BARE_LEAF_GRAD)
    assert row.bare_abs == pytest.approx(4.5)
    assert row.bare_abs >= GRAD_ABS_MIN
    assert "bare_proj" in row.reasons


def test_head_only_string_trains_the_bare_proj():
    row = probe(_arm("head_only"))
    assert row.passed is False
    assert row.coverage == pytest.approx(0.0)
    assert row.self_abs == pytest.approx(0.0)
    assert row.cross_abs == pytest.approx(0.0)
    assert row.bare_abs == pytest.approx(4.5)
    assert "bare_proj" in row.reasons
    assert "coverage" in row.reasons


def test_partial_suffix_misses_cross_attn():
    row = probe(_arm("partial_suffix"))
    assert row.passed is False
    assert row.coverage == pytest.approx(0.5)
    assert row.self_abs == pytest.approx(2.25)
    assert row.cross_abs == pytest.approx(0.0)
    assert row.bare_abs == pytest.approx(0.0)
    assert row.reasons == ("coverage",)


def test_overbroad_star_and_q_proj_are_refused():
    with pytest.raises(EmptyTrainableError, match="self_attn"):
        probe(_arm("overbroad_star"))
    with pytest.raises(EmptyTrainableError, match="q_proj"):
        probe(_arm("empty_q_proj"))


def test_board_passes_only_the_locked_arms():
    rows = {row.name: row for row in run_board()}
    assert rows["locked_suffix"].passed
    assert rows["locked_regex"].passed
    for name in ("bare_proj", "head_only", "partial_suffix", "overbroad_star", "empty_q_proj"):
        assert rows[name].passed is False
    assert rows["overbroad_star"].refused
    assert rows["empty_q_proj"].refused
    assert rows["overbroad_star"].reasons == ("empty_trainable",)
    assert rows["empty_q_proj"].reasons == ("empty_trainable",)
    assert torch.zeros(1).device.type == "cpu"
