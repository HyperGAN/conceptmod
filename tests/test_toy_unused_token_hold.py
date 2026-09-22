"""Unused-token UNI hold: locked_shared keeps the pin; dropping the hold trashes it.

CPU only. Seed 0. Not an Anima, Supra, or Music GPU transfer.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from particlegan import GradientPenalty
from conceptmod.toys import unused_token_hold as hold


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_module_does_not_claim_a_gpu_transfer():
    source = inspect.getsource(hold)
    assert "Anima" in source and "GPU transfer" in source
    assert "huggingface.co" not in source
    assert "from_pretrained" not in source


def test_matched_pairs_skip_the_concept_slot():
    """Concept words are not held. A stranger partner uses the concept slot."""
    matched = hold.hold_pairs("matched")
    assert matched == [(hold.UNUSED, hold.UNUSED)]
    assert hold.CONCEPT not in [i for i, _ in matched]
    stranger = hold.hold_pairs("stranger")
    assert stranger == [(hold.UNUSED, hold.CONCEPT)]
    empty = hold.unused_hold_loss(torch.zeros(2, 2), torch.zeros(2, 2), [])
    assert float(empty) == 0.0


def test_locked_shared_passes_hold_and_concept_move():
    row = hold.train(hold.locked_recipe())
    assert row["passed"]
    assert row["reasons"] == ()
    assert row["verdict"] == "PASS"
    assert row["unused_hold"] >= hold.UNUSED_HOLD_MIN
    assert row["concept_move"] >= hold.CONCEPT_MOVE_MIN
    assert row["unused_hold"] == pytest.approx(0.9914, abs=1e-3)
    assert row["concept_move"] == pytest.approx(0.9677, abs=1e-3)
    assert row["scale0_err"] == pytest.approx(0.0, abs=1e-6)
    assert row["fm_weight"] == 0.0
    assert row["gan_mode"] == "rp"
    assert row["pairing"] == "matched"
    assert row["loss_type"] == "logistic"
    assert row["hold_weight"] == hold.HOLD_WEIGHT
    assert row["cover_weight"] == 1.5
    assert row["bcap_applied"] == hold.STEPS
    assert row["reg_faithful"]
    assert row["bcap_probe"] == pytest.approx(4.0, abs=1e-6)
    assert row["bcap_faithful"] == pytest.approx(4.0, abs=1e-6)


def test_no_hold_trashes_unused_and_keeps_the_concept_move():
    """Shared write splits the concept step in half onto the unused slot."""
    row = hold.train(hold.locked_recipe(name="no_hold", hold_weight=0.0))
    assert row["passed"] is False
    assert "unused_trashed" in row["reasons"]
    assert "no_hold" in row["reasons"]
    assert "concept_move" not in row["reasons"]
    assert row["concept_move"] >= hold.CONCEPT_MOVE_MIN
    assert row["unused_hold"] < hold.UNUSED_HOLD_MIN
    assert row["unused_hold"] == pytest.approx(0.4794, abs=1e-3)
    assert row["concept_move"] == pytest.approx(0.9585, abs=1e-3)
    assert row["unused_dist"] == pytest.approx(row["concept_delta_norm"] / 2.0, abs=1e-3)
    assert row["gan_mode"] == "rp"
    assert row["reg_faithful"]


def test_stranger_partner_trashes_the_unused_slot():
    row = hold.train(hold.locked_recipe(name="stranger_pairing", pairing="stranger"))
    assert row["passed"] is False
    assert "stranger_pairing" in row["reasons"]
    assert "unused_trashed" in row["reasons"]
    assert row["concept_move"] >= hold.CONCEPT_MOVE_MIN
    assert row["unused_hold"] < hold.UNUSED_HOLD_MIN
    assert row["unused_hold"] == pytest.approx(0.0243, abs=1e-3)
    assert row["gan_mode"] == "rp"
    assert row["pairing"] == "stranger"


def test_vanilla_fm_and_thin_bcap_fail_closed():
    vanilla = hold.train(hold.locked_recipe(name="stranger_vanilla", gan_mode="vanilla"))
    fm = hold.train(hold.locked_recipe(name="fm_on", fm_weight=1.0))
    thin = hold.train(hold.locked_recipe(name="thin_bcap"), hold.ThinnedBCap())
    assert vanilla["passed"] is False
    assert "stranger_pairing" in vanilla["reasons"]
    # Vanilla pairing can still pin the slot. The fail is the adv drift.
    assert vanilla["unused_hold"] >= hold.UNUSED_HOLD_MIN
    assert vanilla["concept_move"] >= hold.CONCEPT_MOVE_MIN
    assert "unused_trashed" not in vanilla["reasons"]

    assert fm["passed"] is False
    assert "fm_on" in fm["reasons"]
    assert fm["unused_hold"] >= hold.UNUSED_HOLD_MIN
    assert fm["concept_move"] >= hold.CONCEPT_MOVE_MIN

    assert thin["passed"] is False
    assert "thinned_bcap" in thin["reasons"]
    assert thin["unused_hold"] >= hold.UNUSED_HOLD_MIN
    assert thin["bcap_probe"] == pytest.approx(9.0, abs=1e-6)
    assert thin["bcap_faithful"] == pytest.approx(4.0, abs=1e-6)
    assert type(thin["recipe"]) is hold.UnusedHoldRecipe
    assert hold.ThinnedBCap.__bases__[0] is GradientPenalty


def test_family_passes_only_the_locked_arm(capsys):
    rows = hold.run_family()
    out = capsys.readouterr().out
    assert [row["name"] for row in rows] == [
        "locked_shared",
        "no_hold",
        "stranger_pairing",
        "stranger_vanilla",
        "fm_on",
        "thin_bcap",
    ]
    assert sum(row["passed"] for row in rows) == 1
    assert rows[0]["passed"] and rows[0]["name"] == "locked_shared"
    assert "unused-token arm=locked_shared verdict=PASS" in out
    assert "unused-token arm=no_hold verdict=FAIL" in out
    assert "reasons=unused_trashed" in out or "unused_trashed" in out
    assert "reasons=thinned_bcap" in out or "thinned_bcap" in out
    assert "unused-token family=uni_hold pass=1/6" in out
