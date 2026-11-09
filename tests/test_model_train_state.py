"""Training commits must leave no partial updates or consumed autograd graphs."""

import pytest
import torch

from conceptmod import model_train, ops
from conceptmod.plan import compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession


class ScalarBackend:
    device = "cpu"

    def __init__(self):
        self.transformer = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
        self.weight = self.transformer[0].weight
        self.weight.data.fill_(0.2)
        self.partial_calls = 0
        self.predict_calls = []

    def training_modules(self):
        return {"transformer": self.transformer}

    def training_identity(self):
        return {"fixture": "scalar-source-target-v1"}

    def invalidate_caches(self):
        pass

    def trainable_parameters(self, method):
        self.transformer.train()
        return self.transformer.parameters()

    def partial_denoise(self, *args):
        self.partial_calls += 1
        return torch.zeros(1, 1), torch.tensor(1.0)

    def predict_v(self, prompt, z, timestep, frozen):
        self.predict_calls.append((prompt, frozen))
        base = torch.full_like(z, 0.3 if prompt == "target" else 0.1)
        return base if frozen else base + self.weight.square()


def expected_sgd_weight(updates, lr=0.1):
    # source#target minimizes (0.1 + weight**2 - 0.3)**2. All
    # microsteps within an accumulation group see identical weights.
    weight = 0.2
    for _ in range(updates):
        weight -= lr * 4 * weight * (weight * weight - 0.2)
    return weight


def train(backend, monkeypatch, **kwargs):
    # SGD makes an incorrect final-group gradient scale observable in weights;
    # Adam's normalization otherwise nearly hides this regression.
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.SGD)
    return model_train.train_model(
        backend, "source#target", lr=0.1, **kwargs)


def test_final_partial_group_is_committed_with_actual_group_mean(monkeypatch):
    backend = ScalarBackend()
    backend.weight.grad = torch.full_like(backend.weight, 9.0)

    history = train(backend, monkeypatch, iterations=3, accumulation_steps=2)

    assert len(history) == 2
    assert backend.weight.item() == pytest.approx(expected_sgd_weight(2))
    assert backend.weight.grad is None


def test_budget_smaller_than_accumulation_still_commits(monkeypatch):
    backend = ScalarBackend()

    train(backend, monkeypatch, iterations=1, accumulation_steps=4)

    assert backend.weight.item() == pytest.approx(expected_sgd_weight(1))
    assert backend.weight.grad is None


def test_early_stop_only_observes_committed_updates(monkeypatch):
    backend = ScalarBackend()
    seen = []
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.SGD)

    def observe(active, report):
        assert backend.weight.grad is None
        seen.append(backend.weight.item())
        assert report.microsteps == 3
        assert report.optimizer_steps == 1

    active = TrainingSession(backend, compile_phrase("source#target"),
                             TrainingConfig(iterations=7, accumulation_steps=3, lr=0.1),
                             observers=[observe])
    active.run(stop_when=lambda report: report.optimizer_steps == 1)

    assert active.completed_microsteps == 3
    assert seen == pytest.approx([expected_sgd_weight(1)])
    assert backend.weight.grad is None


def test_empty_budget_clears_stale_gradients_and_preserves_mixed_modes(monkeypatch):
    backend = ScalarBackend()
    backend.transformer.train()
    backend.transformer[0].eval()
    backend.weight.grad = torch.ones_like(backend.weight)

    history = train(backend, monkeypatch, iterations=0)

    assert history == []
    assert backend.weight.item() == pytest.approx(0.2)
    assert backend.weight.grad is None
    assert backend.transformer.training
    assert not backend.transformer[0].training


@pytest.mark.parametrize("failure", ["backward", "observer", "policy"])
def test_exceptions_clear_gradients_and_restore_modes(monkeypatch, failure):
    backend = ScalarBackend()
    backend.transformer.eval()
    backend.transformer[0].train()

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    kwargs = {"iterations": 2, "accumulation_steps": 2}
    if failure == "backward":
        original = model_train.ops.rule_loss
        calls = 0

        def loss(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                fail()
            return original(*args)

        monkeypatch.setattr(model_train.ops, "rule_loss", loss)
    elif failure == "observer":
        kwargs["observers"] = [fail]
    else:
        kwargs["stop_when"] = fail

    with pytest.raises(RuntimeError, match="injected failure"):
        train(backend, monkeypatch, **kwargs)

    assert backend.weight.grad is None
    assert not backend.transformer.training
    assert backend.transformer[0].training


@pytest.mark.parametrize("kwargs, name", [
    ({"iterations": -1}, "iterations"),
    ({"iterations": 1.5}, "iterations"),
    ({"accumulation_steps": 0}, "accumulation_steps"),
    ({"accumulation_steps": True}, "accumulation_steps"),
    ({"lr": float("nan")}, "lr"),
    ({"op_defaults": ops.OpDefaults(sample_steps=0)}, "sample_steps"),
])
def test_invalid_budgets_fail_before_backend_mutation(kwargs, name):
    backend = ScalarBackend()
    backend.transformer.eval()

    with pytest.raises(ValueError, match=name):
        model_train.train_model(backend, "source#target", **kwargs)

    assert not backend.transformer.training
    assert backend.partial_calls == 0


def test_context_releases_consumed_graphs_but_keeps_detached_predictions():
    backend = ScalarBackend()
    context = ops.StepContext(backend, stop_index=0, seed=0, cfg=ops.OpDefaults())
    first = context.v("source", "source", frozen=False, grad=True)
    reference = context.v("target", "source", frozen=True)
    (first - reference).square().mean().backward()

    context.clear_grad_cache()
    second = context.v("source", "source", frozen=False, grad=True)
    (second - reference).square().mean().backward()

    assert second is not first
    assert context.v("target", "source", frozen=True) is reference
    assert backend.predict_calls == [
        ("source", False), ("target", True), ("source", False)]
    assert backend.partial_calls == 1
    assert backend.weight.grad.item() == pytest.approx(2 * -0.128)
