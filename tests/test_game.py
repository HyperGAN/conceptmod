"""Phrase → GAN game → one CPU step under locked_shared.

A PASS here is formulation wiring on the 2-D fixture. It is not a
geometric verdict from docs/dsl.md and not a Music or Anima transfer.
"""

from __future__ import annotations

import pytest

from conceptmod import dsl, ops, toys
from conceptmod.game import (
    NOT_A_TRANSFER,
    GameError,
    LockedClaimError,
    feature_match,
    game,
    loss_game,
    parse_game,
)
from conceptmod.toys.locked_shared_floor import feature_match as floor_feature_match
from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer


def test_feature_match_is_the_floor_helper():
    assert feature_match is floor_feature_match


def test_parse_bare_phrase_and_stanza_with_freeze():
    assert parse_game("red=blue") == {"phrase": "red=blue"}
    spec = parse_game(
        """
        # keep the pattern axis
        phrase: red--|stripe#stripe
        pairing locked_shared
        claim locked
        """
    )
    assert spec["phrase"] == "red--|stripe#stripe"
    assert spec["pairing"] == "locked_shared"
    assert spec["claim"] == "locked"


def test_unknown_stanza_key_is_an_error():
    with pytest.raises(GameError, match="player"):
        parse_game("phrase red=blue\nplayer critic\n")


def test_loss_game_one_step_passes_locked_shared():
    match = loss_game("red=blue")
    assert isinstance(match.gan, GANLoss)
    assert match.gan.loss_type == "logistic"
    assert match.gan.mode == "rp"
    assert type(match.regularizer) is GradRegularizer
    assert match.regularizer.arm == "b_cap"
    assert match.regularizer.kappa == 1.0
    assert match.stamp == toys.locked_adv_defaults()
    assert "n_particles=12" in str(match)
    assert "does not sample a cloud" in str(match)
    names = [player.name for player in match.players()]
    assert names == ["student", "critic"]

    row = match.score()
    assert row["verdict"] == "PASS"
    assert row["mismatches"] == []
    assert row["claim"] == "locked"
    assert row["pairing"] == "locked_shared"
    assert row["adv"] == toys.locked_adv_defaults()
    assert row["rules"] == [
        {"op": "write", "a": "red", "b": "blue", "alpha": 1.0, "options": {}},
    ]
    assert row["step"] == 1
    assert match.steps_run == 1
    assert row["note"] == NOT_A_TRANSFER
    assert "Music" in row["note"]
    for key in ("phrase_loss", "d_loss", "g_loss", "penalty"):
        assert row[key] == row[key]  # not NaN
        assert abs(row[key]) < 1e6
    assert row["phrase_loss"] > 0.0
    assert row["fm_term"] == 0.0
    assert match.backend.lora_B.detach().abs().sum().item() > 0.0
    again = match.score()
    assert again["step"] == 1
    assert match.steps_run == 1


def test_student_objective_calls_rule_loss(monkeypatch):
    seen = []
    real = ops.rule_loss

    def wrapped(rule, ctx):
        seen.append((rule.op, rule.a, rule.b))
        return real(rule, ctx)

    monkeypatch.setattr(ops, "rule_loss", wrapped)
    match = loss_game("red=blue|stripe#stripe")
    match.step()
    assert seen == [("write", "red", "blue"), ("freeze", "stripe", "stripe")]


def test_stamp_is_read_from_toys(monkeypatch):
    def boom():
        raise RuntimeError("stamp source")

    monkeypatch.setattr(toys, "locked_adv_defaults", boom)
    with pytest.raises(RuntimeError, match="stamp source"):
        loss_game("red=blue")


def test_game_string_matches_loss_game():
    source = """
    phrase red++
    pairing locked_shared
    claim locked
    """
    from_string = game(source)
    from_call = loss_game("red++")
    assert [rule.op for rule in from_string.rules] == [dsl.EXAGGERATE]
    assert from_string.stamp == from_call.stamp
    row = from_string.score()
    assert row["verdict"] == "PASS"
    assert row["rules"][0]["op"] == "exaggerate"


def test_locked_claim_refuses_stranger_fm_and_thinned_kappa():
    with pytest.raises(LockedClaimError, match="stranger pairing refused"):
        loss_game("red=blue", pairing="stranger")
    with pytest.raises(LockedClaimError, match="FM-on under b_cap refused"):
        loss_game("red=blue", fm_weight=0.1)
    with pytest.raises(LockedClaimError, match="thinned b_cap refused"):
        game("phrase red=blue\nkappa 0.2\nclaim locked\n")
    with pytest.raises(LockedClaimError, match="only adv='locked_shared'"):
        loss_game("red=blue", adv="vanilla")


def test_drift_arms_score_fail():
    stranger = loss_game("red=blue", pairing="stranger", claim="drift")
    row = stranger.score()
    assert row["verdict"] == "FAIL"
    assert "stranger_pairing" in row["mismatches"]
    assert row["claim"] == "drift"
    assert "PASS" not in row["mismatches"]

    fm = loss_game("red--", fm_weight=0.1, claim="drift")
    fm.step()  # field still matches the frozen probes, so the FM value can be 0
    fm.step()
    fm_row = fm.score()
    assert fm_row["verdict"] == "FAIL"
    assert fm_row["step"] == 2
    assert "fm_on" in fm_row["mismatches"]
    assert fm.fm_weight == 0.1
    assert fm_row["fm_term"] != 0.0

    thin = loss_game("red++", kappa=0.2, claim="drift")
    thin_row = thin.score()
    assert thin_row["verdict"] == "FAIL"
    assert any(item.startswith("thinned_kappa") for item in thin_row["mismatches"])
    assert thin.regularizer.kappa == 0.2
    assert type(thin.regularizer) is GradRegularizer


def test_drift_claim_on_a_locked_shape_is_refused():
    with pytest.raises(GameError, match="named drift"):
        loss_game("red=blue", claim="drift")


def test_pixel_and_reward_do_not_run_on_the_cpu_game():
    with pytest.raises(GameError, match="render"):
        loss_game("a painting of a house^a photo of a house")
    with pytest.raises(GameError, match="not implemented"):
        game("phrase red++|;a bright sunset")


def test_replace_macro_expands_before_the_step():
    match = loss_game("red~blue")
    assert [rule.op for rule in match.rules] == [
        dsl.EXAGGERATE, dsl.WRITE, dsl.ORTHOGONAL,
    ]
    row = match.score()
    assert row["verdict"] == "PASS"
    assert row["note"] == NOT_A_TRANSFER


def test_random_prompt_materializes_or_refuses():
    with pytest.raises(dsl.DSLError, match="random_prompt"):
        loss_game("red%{random_prompt}")
    match = loss_game("red%{random_prompt}", random_prompt="stripe")
    assert match.rules[0].b == "stripe"
    assert match.score()["verdict"] == "PASS"


def test_deprecated_at_is_stripped():
    match = loss_game("@red++")
    assert match.rules[0].op == dsl.EXAGGERATE
    assert match.rules[0].a == "red"
