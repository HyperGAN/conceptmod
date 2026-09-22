"""Cross-toy formulation suite leaderboard.

Scores candidate adv configs across the CPU toy families. Distinct from
``leaderboard_honesty`` (the single-toy two-pole gate).

See also: ``docs/suite-leaderboard.md``, ``docs/formulation-toys.md``.
"""

from __future__ import annotations

import pytest

from conceptmod.toys.leaderboard_honesty import HonestyError
from conceptmod.toys.suite_leaderboard import (
    KIND_POSTURE,
    KIND_STAMP,
    VERDICT_FAIL,
    VERDICT_NA,
    VERDICT_PASS,
    claim_stamp_sweep_pass,
    default_candidates,
    format_matrix,
    run_suite,
    score_cover_posture_demo,
    score_cover_posture_music,
    score_locked_shared_floor,
)


# Core stamp + posture columns: enough to decide the sweep without every
# long train toy. Full matrix is ``python -m conceptmod.toys.suite_leaderboard``.
_CORE_TOYS = (
    "locked_shared_floor",
    "keep_critic",
    "leaderboard_honesty",
    "orbit_hold",
    "field_lift",
    "late_collapse",
    "path_suffix_lora",
    "erase_keep_backend",
    "lm_target",
    "cover_posture_fork_demo",
    "cover_posture_fork_music",
)


@pytest.fixture(scope="module")
def suite():
    return run_suite(toys=_CORE_TOYS, log=False)


def test_locked_shared_wins_stamp_sweep(suite):
    assert suite["stamp_winners"] == ["locked_shared"]
    claim = claim_stamp_sweep_pass(
        winner="locked_shared",
        negatives=("stranger_pair", "fm_on", "thinned_kappa"),
        suite=suite,
    )
    assert claim["verdict"] == VERDICT_PASS
    assert claim["music_gpu_transfer"] is False
    assert claim["anima_gpu_transfer"] is False
    assert claim["cpu_toy"] is True


def test_stranger_fm_thinned_fail_stamp_sweep(suite):
    for name in ("stranger_pair", "fm_on", "thinned_kappa"):
        assert name not in suite["stamp_winners"]
        fails = [
            cell
            for toy, cell in suite["matrix"][name].items()
            if cell.kind == KIND_STAMP and cell.verdict == VERDICT_FAIL
        ]
        assert fails, f"{name} must FAIL at least one stamp toy"


def test_music_fails_demo_stamp_but_can_pass_music_posture(suite):
    """Music cover 1.0 is not a silent substitute for the demo stamp sweep."""
    assert "music_cover_1_0" not in suite["stamp_winners"]
    assert suite["matrix"]["music_cover_1_0"]["locked_shared_floor"].verdict == VERDICT_FAIL
    assert "music_cover_1_0" in suite["posture_music_pass"]
    assert "music_cover_1_0" not in suite["posture_demo_pass"]
    assert "locked_shared" in suite["posture_demo_pass"]
    assert "locked_shared" not in suite["posture_music_pass"]


def test_empty_negative_honesty_raises(suite):
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_stamp_sweep_pass(winner="locked_shared", negatives=(), suite=suite)


def test_claim_refuses_when_declared_bad_still_wins(suite):
    with pytest.raises(HonestyError, match="did not fail"):
        claim_stamp_sweep_pass(
            winner="locked_shared",
            negatives=("locked_shared",),
            suite=suite,
        )


def test_posture_fork_rows_are_separate_claims():
    cands = {c.name: c for c in default_candidates()}
    locked = cands["locked_shared"]
    music = cands["music_cover_1_0"]
    assert score_cover_posture_demo(locked).verdict == VERDICT_PASS
    assert score_cover_posture_music(locked).verdict == VERDICT_FAIL
    assert score_cover_posture_demo(music).verdict == VERDICT_FAIL
    assert score_cover_posture_music(music).verdict == VERDICT_PASS
    # Floor stamp stays demo-only.
    assert score_locked_shared_floor(locked).verdict == VERDICT_PASS
    assert score_locked_shared_floor(music).verdict == VERDICT_FAIL


def test_format_matrix_mentions_no_gpu_transfer(suite):
    text = format_matrix(suite)
    assert "locked_shared" in text
    assert "stamp_winners=locked_shared" in text
    assert "music_gpu_transfer=0" in text
    assert KIND_POSTURE  # catalog exposes posture kind
    assert suite["cpu_toy"] is True


def test_vanilla_and_hub_fail_or_na_sweep(suite):
    for name in ("vanilla_logistic", "hub128"):
        assert name not in suite["stamp_winners"]
        cells = suite["matrix"][name]
        assert any(
            c.verdict == VERDICT_FAIL for c in cells.values() if c.kind == KIND_STAMP
        )
        assert not any(
            c.verdict == VERDICT_NA for c in cells.values() if c.toy == "locked_shared_floor"
        )
