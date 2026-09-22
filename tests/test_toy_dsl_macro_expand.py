"""Replace-macro expansion honesty: ``red~blue`` is the documented triple.

PASS is expand honesty. Geometric ``right`` on this fixture is a FAIL
(the scoreboard says recipe). The optional game row is locked_shared
wiring, not a geometry win and not a Music or Anima GPU transfer.

CPU toy. Tail the same lines with::

    python -m conceptmod.toys.dsl_macro_expand
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from conceptmod import dsl
from conceptmod.toys.dsl_macro_expand import (
    DOCUMENTED_LAMBDA,
    FORBIDDEN_GEOMETRIC,
    GEOMETRIC_VERDICT,
    PHRASE,
    SOURCE,
    TARGET,
    TOY_ID,
    ExpandedRule,
    HonestyError,
    accept_game_row,
    claim_expand_pass,
    demo_arms,
    documented_text,
    documented_triple,
    failure_reasons,
    game_wiring_row,
    make_arm,
    run_arm,
    run_board,
)
from conceptmod.toys.leaderboard_honesty import HonestyError as BoardHonestyError


def _reports():
    return {row.name: row for row in (run_arm(arm, order=i) for i, arm in enumerate(demo_arms(), start=1))}


def test_honesty_error_is_the_leaderboard_type():
    assert HonestyError is BoardHonestyError


def test_documented_triple_is_the_written_formula():
    """b++:2λ | a=b:4λ | b%a:−λ at λ=0.1, with red/blue roles."""
    rules = documented_triple()
    assert DOCUMENTED_LAMBDA == 0.1
    assert [rule.alpha for rule in rules] == [
        2 * DOCUMENTED_LAMBDA,
        4 * DOCUMENTED_LAMBDA,
        -DOCUMENTED_LAMBDA,
    ]
    assert [rule.alpha for rule in rules] == [0.2, 0.4, -0.1]
    assert [(rule.op, rule.a, rule.b) for rule in rules] == [
        (dsl.EXAGGERATE, TARGET, ""),
        (dsl.WRITE, SOURCE, TARGET),
        (dsl.ORTHOGONAL, TARGET, SOURCE),
    ]
    assert rules[0].a == "blue"
    assert (rules[1].a, rules[1].b) == ("red", "blue")
    assert (rules[2].a, rules[2].b) == ("blue", "red")
    assert rules[2].alpha < 0
    assert all(rule.options == () for rule in rules)
    assert documented_text() == "blue++:0.2 | red=blue:0.4 | blue%red:-0.1"
    assert "replace" not in {
        dsl.EXAGGERATE, dsl.ERASE, dsl.WRITE, dsl.FREEZE,
        dsl.ORTHOGONAL, dsl.PIXEL, dsl.REWARD,
    }


def test_parser_matches_the_documented_triple():
    parsed = dsl.parse_phrase(PHRASE)
    want = documented_triple()
    assert [rule.op for rule in parsed] == [rule.op for rule in want]
    assert [rule.alpha for rule in parsed] == [rule.alpha for rule in want]
    assert [(rule.a, rule.b) for rule in parsed] == [(rule.a, rule.b) for rule in want]
    assert all(rule.options == {} for rule in parsed)
    spelled = dsl.parse_phrase(documented_text())
    assert [(rule.op, rule.a, rule.b, rule.alpha) for rule in spelled] == [
        (rule.op, rule.a, rule.b, rule.alpha) for rule in parsed
    ]


def test_locked_arm_passes_and_named_wrongs_fail():
    rows = _reports()
    locked = rows["locked_expand"]
    assert locked.won is True
    assert locked.reasons == ()
    assert locked.geometric_verdict == GEOMETRIC_VERDICT
    assert [rule.op for rule in locked.rules] == ["exaggerate", "write", "orthogonal"]
    assert [rule.alpha for rule in locked.rules] == [0.2, 0.4, -0.1]
    assert locked.text_rules == locked.rules

    expected = {
        "wrong_order": ("op_order",),
        "swapped_roles": ("swapped_roles",),
        "wrong_alphas": ("alphas",),
        "dropped_percent": ("dropped_percent",),
        "single_write": ("single_write",),
        "geometric_right": ("geometric_right",),
        "fourth_loss": ("fourth_loss",),
    }
    for name, reasons in expected.items():
        row = rows[name]
        assert row.won is False, name
        assert row.reasons == reasons, name
        assert row.role == "negative"

    geometric = rows["geometric_right"]
    assert geometric.rules == documented_triple()
    assert geometric.geometric_verdict == FORBIDDEN_GEOMETRIC
    assert failure_reasons(geometric.rules, "recipe") == ()

    fourth = rows["fourth_loss"]
    assert fourth.rules == (ExpandedRule("replace", "red", "blue", 0.1),)
    assert fourth.rules[0].op not in {
        dsl.EXAGGERATE, dsl.ERASE, dsl.WRITE, dsl.FREEZE, dsl.ORTHOGONAL, dsl.PIXEL,
    }

    dropped = rows["dropped_percent"]
    assert all(rule.op != dsl.ORTHOGONAL for rule in dropped.rules)
    single = rows["single_write"]
    assert [(rule.op, rule.a, rule.b) for rule in single.rules] == [("write", "red", "blue")]
    swapped = rows["swapped_roles"]
    assert (swapped.rules[0].a, swapped.rules[1].a, swapped.rules[1].b) == ("red", "blue", "red")
    ordered = rows["wrong_order"]
    assert [rule.op for rule in ordered.rules] == ["write", "exaggerate", "orthogonal"]
    assert [rule.alpha for rule in rows["wrong_alphas"].rules] == [1.0, 1.0, 1.0]


def test_board_pass_is_expand_honesty_not_geometry_right():
    board = run_board(include_game=False)
    verdict = board["verdict"]
    assert verdict["verdict"] == "PASS"
    assert verdict["winner"] == "locked_expand"
    assert verdict["ops"] == ["exaggerate", "write", "orthogonal"]
    assert verdict["alphas"] == [0.2, 0.4, -0.1]
    assert verdict["roles"] == [
        {"op": "exaggerate", "a": "blue", "b": ""},
        {"op": "write", "a": "red", "b": "blue"},
        {"op": "orthogonal", "a": "blue", "b": "red"},
    ]
    assert verdict["geometric_verdict"] == "recipe"
    assert verdict["geometry_right"] is False
    assert verdict["cpu_toy"] is True
    assert verdict["music_gpu_transfer"] is False
    assert verdict["anima_gpu_transfer"] is False
    assert set(verdict["negatives"]) == {
        "wrong_order",
        "swapped_roles",
        "wrong_alphas",
        "dropped_percent",
        "single_write",
        "geometric_right",
        "fourth_loss",
    }
    assert board["game"] is None


def test_claim_without_a_negative_errors():
    locked = _reports()["locked_expand"]
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_expand_pass(winner=locked, negatives=[])
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_expand_pass(winner=locked, negatives=())
    with pytest.raises(HonestyError, match="without a declared negative"):
        run_board((demo_arms()[0],), include_game=False)


def test_claim_rejects_geometric_right_on_the_winner():
    locked = _reports()["locked_expand"]
    lied = replace(
        locked,
        geometric_verdict="right",
        won=False,
        reasons=("geometric_right",),
    )
    negative = _reports()["wrong_order"]
    with pytest.raises(HonestyError, match="geometric right"):
        claim_expand_pass(winner=lied, negatives=[negative])


def test_claim_rejects_a_fourth_loss_on_the_winner():
    locked = _reports()["locked_expand"]
    invented = (ExpandedRule("replace", "red", "blue", 0.1),)
    lied = replace(
        locked,
        rules=invented,
        text_rules=invented,
        won=False,
        reasons=("fourth_loss",),
    )
    with pytest.raises(HonestyError, match="fourth loss"):
        claim_expand_pass(winner=lied, negatives=[_reports()["wrong_order"]])


def test_claim_rejects_a_bad_arm_that_matches():
    locked = _reports()["locked_expand"]
    fake = replace(
        _reports()["wrong_order"],
        rules=documented_triple(),
        geometric_verdict="recipe",
        won=True,
        reasons=(),
    )
    with pytest.raises(HonestyError, match="did not fail"):
        claim_expand_pass(winner=locked, negatives=[fake])


def test_dishonest_won_flag_is_refused():
    locked = _reports()["locked_expand"]
    lied = replace(locked, won=False)
    with pytest.raises(HonestyError, match="dishonest board"):
        claim_expand_pass(winner=lied, negatives=[_reports()["single_write"]])
    flipped = replace(_reports()["single_write"], won=True)
    with pytest.raises(HonestyError, match="dishonest board"):
        claim_expand_pass(winner=locked, negatives=[flipped])


def test_claim_rejects_a_negative_from_another_toy():
    with pytest.raises(HonestyError, match="same toy"):
        claim_expand_pass(
            winner=_reports()["locked_expand"],
            negatives=[replace(_reports()["wrong_order"], toy="two_pole_cloud")],
        )


def test_claim_rejects_a_winner_that_is_not_the_parser():
    other = replace(_reports()["wrong_order"], role="locked_shared", name="locked_expand")
    with pytest.raises(HonestyError, match="did not read the parser"):
        claim_expand_pass(winner=other, negatives=[_reports()["single_write"]])


def test_first_cell_must_be_the_documented_expand():
    arms = demo_arms()
    with pytest.raises(HonestyError, match="FIRST cell"):
        run_board(arms[1:], include_game=False)
    with pytest.raises(HonestyError, match="documented parser"):
        make_arm("wrong_order", "locked_shared", "wrong_order")
    with pytest.raises(HonestyError, match="cannot use the documented"):
        make_arm("locked_expand", "negative", "documented")


def test_two_pole_claim_pass_stays_closed():
    """This toy does not extend claim_pass. The two-pole gate still refuses it."""
    from conceptmod.toys.leaderboard_honesty import CellResult, claim_pass

    winner = CellResult(
        name="locked_shared",
        role="locked_shared",
        toy=TOY_ID,
        won=True,
        mean_abs=0.5,
        grad_med=0.4,
        nearest=0.5,
        cover_score=0.75,
        order=1,
        steps=8,
        seed=0,
    )
    negative = CellResult(
        name="wrong_order",
        role="negative",
        toy=TOY_ID,
        won=False,
        mean_abs=0.0,
        grad_med=0.0,
        nearest=1.0,
        cover_score=0.0,
        order=2,
        steps=8,
        seed=0,
    )
    with pytest.raises(HonestyError, match="two_pole_cloud"):
        claim_pass(winner=winner, negatives=[negative])
    with pytest.raises(HonestyError, match="without a declared negative"):
        claim_pass(winner=winner, negatives=[])


def test_game_wiring_pass_does_not_claim_geometric_right():
    row = game_wiring_row()
    accepted = accept_game_row(row)
    assert accepted["verdict"] == "PASS"
    assert accepted["pairing"] == "locked_shared"
    assert accepted["claim"] == "locked"
    assert accepted["game_verdict"] == "PASS"
    assert accepted["mismatches"] == []
    assert accepted["ops"] == ["exaggerate", "write", "orthogonal"]
    assert accepted["alphas"] == [0.2, 0.4, -0.1]
    assert accepted["geometric_verdict"] == "recipe"
    assert accepted["geometry_right"] is False
    assert accepted["cpu_toy"] is True
    assert accepted["music_gpu_transfer"] is False
    assert accepted["anima_gpu_transfer"] is False
    assert "geometric verdict" in accepted["note"]
    board = run_board()
    assert board["verdict"]["verdict"] == "PASS"
    assert board["verdict"]["geometry_right"] is False
    assert board["verdict"]["game_wiring"] == "PASS"
    assert board["game"]["geometry_right"] is False


def test_game_row_that_claims_right_is_refused():
    row = game_wiring_row()
    forged = {**row, "geometric_verdict": "right", "geometry_right": True}
    with pytest.raises(HonestyError, match="geometric right"):
        accept_game_row(forged)
    with pytest.raises(HonestyError, match="recipe"):
        accept_game_row({**row, "geometric_verdict": "needs help"})
    with pytest.raises(HonestyError, match="Music or Anima"):
        accept_game_row({**row, "music_gpu_transfer": True})
