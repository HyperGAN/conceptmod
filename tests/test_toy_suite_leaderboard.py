"""Cross-toy formulation suite leaderboard.

Scores candidate adv configs across the CPU toy families. Distinct from
``leaderboard_honesty`` (the single-toy two-pole gate).

Cover posture is not a suite column. ``cover_posture_fork`` stays its own
toy (import/smoke below). The suite does not score a candidate under the
other claim.

See also: ``docs/suite-leaderboard.md``, ``docs/formulation-toys.md``.
"""

from __future__ import annotations

import pytest

from conceptmod.toys.leaderboard_honesty import HonestyError
from conceptmod.toys.suite_leaderboard import (
    KIND_DSL,
    KIND_STAMP,
    TOY_CATALOG,
    VERDICT_FAIL,
    VERDICT_NA,
    VERDICT_PASS,
    claim_stamp_sweep_pass,
    default_candidates,
    format_matrix,
    run_suite,
    score_cell,
    score_locked_shared_floor,
    toy_names,
)


# Core stamp columns: enough to decide the sweep without every long train toy.
# Full matrix is ``python -m conceptmod.toys.suite_leaderboard``.
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


def test_music_fails_demo_stamp(suite):
    """Music cover 1.0 is not a silent substitute for the demo stamp sweep."""
    assert "music_cover_1_0" not in suite["stamp_winners"]
    assert suite["matrix"]["music_cover_1_0"]["locked_shared_floor"].verdict == VERDICT_FAIL


def test_suite_has_no_opposite_claim_posture_cells(suite):
    """Config identity is not a suite FAIL for the stamp winner."""
    catalog = toy_names()
    assert "cover_posture_fork_demo" not in catalog
    assert "cover_posture_fork_music" not in catalog
    assert toy_names(kinds=("posture",)) == ()
    for config in ("locked_shared", "music_cover_1_0"):
        cells = suite["matrix"][config]
        assert "cover_posture_fork_demo" not in cells
        assert "cover_posture_fork_music" not in cells
    locked_fails = [
        toy
        for toy, cell in suite["matrix"]["locked_shared"].items()
        if cell.verdict == VERDICT_FAIL
    ]
    assert locked_fails == []
    assert "posture_demo_pass" not in suite
    assert "posture_music_pass" not in suite
    assert "posture_toys" not in suite
    assert {kind for _name, kind, _scorer in TOY_CATALOG} == {KIND_STAMP, KIND_DSL}
    locked = default_candidates()[0]
    with pytest.raises(KeyError, match="cover_posture_fork_music"):
        score_cell(locked, "cover_posture_fork_music")
    with pytest.raises(KeyError, match="cover_posture_fork_demo"):
        score_cell(locked, "cover_posture_fork_demo")


def test_cover_posture_fork_remains_its_own_toy():
    """Smoke import only. Own-claim gates live in test_toy_cover_posture_fork."""
    import conceptmod.toys.cover_posture_fork as fork

    assert callable(fork.run_board)
    assert fork.DEMO_CLAIM == "demo_1_5"
    assert fork.MUSIC_CLAIM == "music_1_0"


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


def test_floor_stamp_stays_demo_only():
    cands = {c.name: c for c in default_candidates()}
    assert score_locked_shared_floor(cands["locked_shared"]).verdict == VERDICT_PASS
    assert score_locked_shared_floor(cands["music_cover_1_0"]).verdict == VERDICT_FAIL


def test_format_matrix_mentions_no_gpu_transfer(suite):
    text = format_matrix(suite)
    assert "locked_shared" in text
    assert "stamp_winners=locked_shared" in text
    assert "music_gpu_transfer=0" in text
    assert "python -m conceptmod.toys.cover_posture_fork" in text
    assert "posture_demo_pass" not in text
    assert "posture_music_pass" not in text
    assert "posture_demo" not in text
    assert "posture_music" not in text
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
