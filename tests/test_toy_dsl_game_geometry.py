"""Bridge toy: locked_shared game PASS and geometric right, both required.

A game PASS with skipped or faked geometry FAILs. Geometric right with an
unlocked claim, FM-on, or stranger pairing FAILs. Dead ``^`` / ``;`` still
refuse inside ``loss_game``; claiming PASS on that refusal FAILs. ``red~blue``
stays recipe. This is a CPU toy, not a Music or Anima GPU transfer.
"""

from __future__ import annotations

import pytest

import conceptmod.analysis_2d as analysis_2d
import conceptmod.analysis_dsl as analysis_dsl
from conceptmod.analysis_2d import DEFAULT_LR, DEFAULT_SEED, DEFAULT_STEPS
from conceptmod.game import GameError, LockedClaimError, loss_game
from conceptmod.toys import locked_adv_defaults
from conceptmod.toys.dsl_game_geometry import (
    HonestyError,
    arms,
    claim_board,
    claim_bridge_pass,
    evaluate,
    fixture_budget,
    run_board,
    score_geometry,
)
from conceptmod.toys.leaderboard_honesty import HonestyError as HonestyErrorExported


PASS_ARMS = (
    ("write", "red=blue", "write"),
    ("erase_bare", "red--", "erase_esd"),
    ("erase_freeze", "red--|stripe#stripe", "erase_esd_freeze"),
    ("exaggerate", "red++", "exaggerate"),
)


@pytest.fixture(scope="module")
def board():
    return {row["arm"]: row for row in run_board(log=False)}


def _arm(name: str):
    return next(arm for arm in arms() if arm.name == name)


def test_honesty_error_is_the_leaderboard_type():
    assert HonestyError is HonestyErrorExported


def test_budget_is_the_2d_fixture_not_an_adv_recipe():
    budget = fixture_budget()
    assert budget == {"steps": DEFAULT_STEPS, "lr": DEFAULT_LR, "seed": DEFAULT_SEED}
    assert budget["steps"] == 40
    assert budget["lr"] == 8e-2
    stamp = locked_adv_defaults()
    assert "steps" not in stamp
    assert "lr" not in stamp
    assert stamp["fm_weight"] == 0.0
    assert stamp["reg_kappa"] == 1.0
    assert stamp["gan_mode"] == "rp"


def test_pass_arms_are_locked_and_geometrically_right(board):
    passed = []
    for name, phrase, scorer_name in PASS_ARMS:
        row = board[name]
        passed.append((row["arm"], row["phrase"]))
        assert row["phrase"] == phrase
        assert row["verdict"] == "PASS", row["mismatches"]
        assert row["mismatches"] == []
        assert row["game_verdict"] == "PASS"
        assert row["claim"] == "locked"
        assert row["pairing"] == "locked_shared"
        assert row["game_steps"] == 1
        assert row["geometry"] == "right"
        assert row["geometry_source"] == "analysis_2d.run_method"
        assert row["geometry_name"] == scorer_name
        assert row["stamp_ok"] is True
        assert row["steps"] == 40
        assert row["lr"] == 8e-2
        assert row["seed"] == DEFAULT_SEED
        assert row["budget"] == "2d_fixture"
        assert row["cpu_toy"] is True
        assert row["music_gpu_transfer"] is False
        assert row["anima_gpu_transfer"] is False
        assert row["game_pass_is_sufficient"] is False
        assert "Music" in row["note"]
        claim = claim_bridge_pass(row)
        assert claim["verdict"] == "PASS"
        assert claim["game_pass_is_sufficient"] is False
    assert passed == [(name, phrase) for name, phrase, _scorer in PASS_ARMS]
    other = [row["arm"] for row in board.values() if row["arm"] not in {n for n, _, _ in PASS_ARMS}]
    assert other
    assert all(board[name]["verdict"] == "FAIL" for name in other)


def test_family_pass_needs_the_mismatch_arms(board):
    family = claim_board(list(board.values()))
    assert family["verdict"] == "PASS"
    assert family["winner"] == [name for name, _, _ in PASS_ARMS]
    assert "wiring_only" in family["negatives"]
    assert "replace_macro" in family["negatives"]
    assert family["game_pass_is_sufficient"] is False
    assert family["music_gpu_transfer"] is False
    assert family["toy"] == "dsl_game_geometry"


def test_wiring_only_game_pass_is_not_geometry(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("wiring_only must not train a geometric score")

    monkeypatch.setattr(analysis_2d, "run_method", boom)
    row = evaluate(_arm("wiring_only"))
    assert row["game_verdict"] == "PASS"
    assert row["claim"] == "locked"
    assert row["geometry"] == "skipped"
    assert row["geometry_source"] == "skipped"
    assert row["verdict"] == "FAIL"
    assert "wiring_only" in row["mismatches"]
    assert "claimed_pass" in row["mismatches"]
    with pytest.raises(HonestyError, match="not geometric right"):
        claim_bridge_pass(row)


def test_faked_right_fails_the_bridge(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("faked geometry must not train")

    monkeypatch.setattr(analysis_2d, "run_method", boom)
    row = evaluate(_arm("faked_geometry"))
    assert row["game_verdict"] == "PASS"
    assert row["geometry"] == "right"
    assert row["geometry_source"] == "faked"
    assert row["verdict"] == "FAIL"
    assert "wiring_only" in row["mismatches"]
    assert "faked_geometry" in row["mismatches"]
    assert "claimed_geometry_right" in row["mismatches"]
    with pytest.raises(HonestyError, match="not geometric right"):
        claim_bridge_pass(row)


def test_geometry_right_without_a_locked_game_fails(board):
    stranger = board["geometry_only_stranger"]
    assert stranger["geometry"] == "right"
    assert stranger["geometry_source"] == "analysis_2d.run_method"
    assert stranger["game_verdict"] == "FAIL"
    assert stranger["claim"] == "drift"
    assert stranger["pairing"] == "stranger"
    assert stranger["verdict"] == "FAIL"
    assert "unlocked_claim" in stranger["mismatches"]
    assert "stranger_pairing" in stranger["mismatches"]
    assert "geometry_only" in stranger["mismatches"]

    fm = board["geometry_only_fm"]
    assert fm["geometry"] == "right"
    assert fm["game_verdict"] == "FAIL"
    assert fm["claim"] == "drift"
    assert fm["verdict"] == "FAIL"
    assert "unlocked_claim" in fm["mismatches"]
    assert "fm_on" in fm["mismatches"]
    assert "geometry_only" in fm["mismatches"]

    locked_stranger = board["locked_stranger_refuse"]
    assert locked_stranger["geometry"] == "right"
    assert locked_stranger["game_verdict"] == "REFUSED"
    assert locked_stranger["claim"] == "locked"
    assert locked_stranger["verdict"] == "FAIL"
    assert "locked_claim_refused" in locked_stranger["mismatches"]
    assert "stranger_pairing" in locked_stranger["mismatches"]
    assert "geometry_only" in locked_stranger["mismatches"]

    locked_fm = board["locked_fm_refuse"]
    assert locked_fm["game_verdict"] == "REFUSED"
    assert locked_fm["verdict"] == "FAIL"
    assert "locked_claim_refused" in locked_fm["mismatches"]
    assert "fm_on" in locked_fm["mismatches"]
    assert "geometry_only" in locked_fm["mismatches"]
    for row in (stranger, fm, locked_stranger, locked_fm):
        with pytest.raises(HonestyError):
            claim_bridge_pass(row)


def test_dead_ops_refuse_and_a_pass_claim_fails():
    pixel = evaluate(_arm("dead_pixel"))
    reward = evaluate(_arm("dead_reward"))
    for row, tag in ((pixel, "dead_pixel"), (reward, "dead_reward")):
        assert row["game_verdict"] == "REFUSED"
        assert row["geometry"] == "dead"
        assert row["geometry_source"] == "analysis_dsl.dead_ops"
        assert row["verdict"] == "FAIL"
        assert "dead_phrase" in row["mismatches"]
        assert "dead_phrase_refused" in row["mismatches"]
        assert tag in row["mismatches"]
        assert "claimed_pass" in row["mismatches"]
        assert row["game_steps"] == 0
        with pytest.raises(HonestyError):
            claim_bridge_pass(row)


def test_replace_macro_recipe_is_not_silently_right(board):
    row = board["replace_macro"]
    assert row["phrase"] == "red~blue"
    assert row["game_verdict"] == "PASS"
    assert row["claim"] == "locked"
    assert row["geometry"] == "recipe"
    assert row["geometry_source"] == "analysis_dsl.run_job"
    assert row["geometry_name"] == "replace_macro"
    assert row["verdict"] == "FAIL"
    assert "recipe_not_independently_right" in row["mismatches"]
    assert "claimed_geometry_right" in row["mismatches"]
    assert "claimed_pass" in row["mismatches"]
    assert "wiring_only" not in row["mismatches"]
    with pytest.raises(HonestyError, match="not geometric right"):
        claim_bridge_pass(row)


def test_game_refuse_paths_are_unchanged():
    """The bridge catches refusals. ``loss_game`` itself still raises."""
    with pytest.raises(LockedClaimError, match="stranger pairing refused"):
        loss_game("red=blue", pairing="stranger")
    with pytest.raises(LockedClaimError, match="FM-on under b_cap refused"):
        loss_game("red=blue", fm_weight=0.1)
    with pytest.raises(LockedClaimError, match="thinned b_cap refused"):
        loss_game("red=blue", kappa=0.2)
    with pytest.raises(GameError, match="render"):
        loss_game("a painting of a house^a photo of a house")
    with pytest.raises(GameError, match="not implemented"):
        loss_game(";a bright sunset")


def test_score_geometry_delegates_to_the_live_scorers(monkeypatch):
    method_calls = []
    job_calls = []
    real_method = analysis_2d.run_method
    real_job = analysis_dsl.run_job

    def wrapped_method(name, phrase, **kwargs):
        method_calls.append((name, phrase, kwargs.get("steps"), kwargs.get("lr")))
        return real_method(name, phrase, **kwargs)

    def wrapped_job(name, phrase, **kwargs):
        job_calls.append((name, phrase, kwargs.get("steps"), kwargs.get("lr")))
        return real_job(name, phrase, **kwargs)

    monkeypatch.setattr(analysis_2d, "run_method", wrapped_method)
    monkeypatch.setattr(analysis_dsl, "run_job", wrapped_job)
    budget = fixture_budget()
    geo = score_geometry(_arm("write"), steps=1, lr=budget["lr"], seed=budget["seed"])
    recipe = score_geometry(
        _arm("replace_macro"), steps=1, lr=budget["lr"], seed=budget["seed"],
    )
    assert method_calls == [("write", "red=blue", 1, budget["lr"])]
    assert job_calls == [("replace_macro", "red~blue", 1, budget["lr"])]
    assert geo["source"] == "analysis_2d.run_method"
    assert recipe["source"] == "analysis_dsl.run_job"
    assert recipe["verdict"] == "recipe"


def test_empty_negative_list_refuses_a_family_pass():
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_board([])
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_board([{
            "arm": "write",
            "role": "bridge",
            "verdict": "PASS",
            "mismatches": [],
        }])


def test_a_mismatch_arm_marked_pass_refuses_the_family(board):
    rows = [dict(row) for row in board.values()]
    for row in rows:
        if row["arm"] == "wiring_only":
            row["verdict"] = "PASS"
    with pytest.raises(HonestyError, match="did not fail"):
        claim_board(rows)


def test_runner_is_exported():
    from conceptmod import toys

    assert toys.run_dsl_game_board is run_board
    assert toys.dsl_game_geometry.run_board is run_board
