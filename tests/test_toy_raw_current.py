"""Short execution regressions for raw-current research observation.

These checks prove weight provenance, not the historical scientific outcomes.
"""
import pytest
import torch

from conceptmod.toys import cover_leftover, mode_hold


def _record_updates(monkeypatch):
    current = {}
    original = torch.optim.Adam.step

    def record(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                current[id(parameter)] = parameter.detach().clone()
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", record)
    return current


def test_cover_scores_final_raw_residual_without_replacing_parameters(monkeypatch):
    raw = _record_updates(monkeypatch)
    original = cover_leftover.score_geometry
    seen = []

    def observe(residual, *args, **kwargs):
        for parameter in residual.parameters():
            assert torch.equal(parameter, raw[id(parameter)])
        seen.append(True)
        return original(residual, *args, **kwargs)

    monkeypatch.setattr(cover_leftover, "score_geometry", observe)
    row = cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=2, seed=41))
    assert seen
    assert row["protocol"] == "raw-current-no-weight-ema-v1"
    assert row["weight_source"] == "raw_current" and row["ema_decay"] is None
    assert cover_leftover.FORMULATION["ema_decay"] is None


def test_mode_hold_scores_raw_generator_and_particle_bank(monkeypatch):
    raw = _record_updates(monkeypatch)
    models = []
    generator_type = mode_hold.SimpleMLPGenerator
    prior_type = mode_hold.ParticlePrior
    original = mode_hold.diversity

    def generator(*args, **kwargs):
        model = generator_type(*args, **kwargs)
        models.append(model)
        return model

    def prior(*args, **kwargs):
        model = prior_type(*args, **kwargs)
        models.append(model)
        return model

    def observe(*args, **kwargs):
        for model in models:
            for parameter in model.parameters():
                assert torch.equal(parameter, raw[id(parameter)])
        return original(*args, **kwargs)

    monkeypatch.setattr(mode_hold, "SimpleMLPGenerator", generator)
    monkeypatch.setattr(mode_hold, "ParticlePrior", prior)
    monkeypatch.setattr(mode_hold, "diversity", observe)
    row = mode_hold.train_mode_hold(mode_hold.locked_recipe().replace(steps=2), seed=41, log=False)
    assert row["protocol"] == "raw-current-no-weight-ema-v1"
    assert row["weight_source"] == "raw_current" and row["ema_decay"] is None
    assert mode_hold.locked_recipe().ema_decay is None
    assert row["verdict"] == mode_hold.verdict(row)


def test_research_apis_refuse_weight_ema_enablement(monkeypatch):
    with pytest.raises(ValueError, match="weight EMA is forbidden"):
        mode_hold.locked_recipe().replace(ema_decay=.995)
    with pytest.raises(ValueError, match="weight EMA is forbidden"):
        cover_leftover.reject_unlocked({"ema_decay": .995})
    with pytest.raises(ValueError, match="weight EMA is forbidden"):
        cover_leftover.reject_unlocked({"ema": .995})
    monkeypatch.setitem(cover_leftover.FORMULATION, "ema_decay", .995)
    with pytest.raises(ValueError, match="weight EMA is forbidden"):
        cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=1))
