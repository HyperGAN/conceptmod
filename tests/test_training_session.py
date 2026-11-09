import copy
import random
from types import SimpleNamespace

import pytest
import torch

from conceptmod import ops
from conceptmod.encoder_train import train_encoder
from conceptmod.model_train import train_model
from conceptmod.observers import HistoryObserver, LossLogger
from conceptmod.plan import compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession


class TinyBackend:
    device = "cpu"
    model_id = "tiny-fixture-v1"

    def __init__(self, *, stochastic=False):
        with torch.random.fork_rng(devices=[]):
            self.transformer = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.25))
            self.text_encoder = torch.nn.Linear(2, 2)
        self.transformer[0].weight.data.fill_(0.2)
        self.transformer[0].bias.data.fill_(0.1)
        self.text_encoder.weight.data.copy_(torch.eye(2))
        self.text_encoder.bias.data.zero_()
        self.frozen = copy.deepcopy(self.transformer).requires_grad_(False).eval()
        self.stochastic = stochastic
        self.invalidations = 0
        self.attachments = 0

    def training_modules(self):
        return {"model": self.transformer, "encoder": self.text_encoder, "original": self.frozen}

    def trainable_parameters(self, method):
        self.transformer.train()
        return self.transformer.parameters()

    def partial_denoise(self, prompt, stop_index, steps, guidance, generator):
        return torch.randn(1, 2, generator=generator), torch.tensor(0.5)

    def predict_v(self, prompt, z, timestep, frozen):
        features = z + len(prompt) / 20
        model = self.frozen if frozen else self.transformer
        if not self.stochastic:
            return model[0](features)
        return model(features) + 0.01 * torch.randn_like(features) + random.random() * 0.01

    def invalidate_caches(self):
        self.invalidations += 1

    def attach_encoder_lora(self, rank):
        self.attachments += 1
        return self.text_encoder.parameters()

    def _tokens(self, prompt):
        return torch.tensor([[[len(prompt) / 20, sum(map(ord, prompt)) % 19 / 20]]])

    def encode_text(self, prompt, frozen=False):
        # Reference encoding remains stable even as the trainable encoder changes.
        tokens = self._tokens(prompt)
        return SimpleNamespace(embeds=tokens if frozen else self.text_encoder(tokens), mask=None)

    def encode_text_grad(self, prompt):
        return self.encode_text(prompt)


def session(backend=None, *, config=None, stage="model", phrase="cat=dog", **kwargs):
    return TrainingSession(backend or TinyBackend(), compile_phrase(phrase, stage=stage),
                           config or TrainingConfig(iterations=7, accumulation_steps=3,
                                                    lr=0.01, seed=123), **kwargs)


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree_equal(left, right)
    else:
        assert first == second


def test_checkpoint_exact_stochastic_continuation_including_original_and_optimizer(tmp_path):
    continuous = session(TinyBackend(stochastic=True))
    continuous.run()
    split = session(TinyBackend(stochastic=True))
    split.step()
    split.step()
    checkpoint = tmp_path / "session.pt"
    split.save_checkpoint(checkpoint)
    resumed = session(TinyBackend(stochastic=True))
    resumed.backend.frozen[0].weight.data.fill_(99)
    resumed.load_checkpoint(checkpoint)
    resumed.run()

    assert_tree_equal(continuous.state_dict(), resumed.state_dict())
    assert resumed.completed_microsteps == 7
    assert resumed.completed_updates == 3
    assert resumed.reports[-1].group_microsteps == 1
    assert resumed.backend.invalidations == 1
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_sessions_have_independent_random_streams_and_do_not_perturb_globals():
    python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
    first = session(TinyBackend(stochastic=True))
    second = session(TinyBackend(stochastic=True))
    for _ in range(3):
        first.step()
        # unrelated caller work cannot change the next session's random stream
        external = torch.rand(5)
        expected = torch.get_rng_state().clone()
        second.step()
        assert torch.equal(torch.get_rng_state(), expected)
        assert external.shape == (5,)
    assert_tree_equal(first.state_dict(), second.state_dict())
    # Python random was never used by the intervening caller work.
    assert random.getstate() == python_before
    torch.set_rng_state(torch_before)


def test_session_construction_and_execution_preserve_all_entry_modes():
    backend = TinyBackend(stochastic=True)
    backend.transformer.train()
    backend.transformer[1].eval()
    backend.text_encoder.eval()
    before = [(module, module.training) for root in backend.training_modules().values()
              for module in root.modules()]
    active = session(backend)
    active.step()
    assert all(module.training == mode for module, mode in before)


def test_failed_update_is_not_committed_and_checkpoint_restores_recovery(monkeypatch):
    active = session(TinyBackend(stochastic=True))
    checkpoint = active.state_dict()
    original_step = active.optimizer.step

    def fail():
        original_step()
        raise RuntimeError("optimizer failed after touching weights")

    monkeypatch.setattr(active.optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        active.step()
    assert active.status == "failed"
    assert active.completed_updates == active.completed_microsteps == 0
    assert all(param.grad is None for param in active.params)
    with pytest.raises(RuntimeError, match="committed"):
        active.state_dict()
    with pytest.raises(RuntimeError, match="failed"):
        active.step()
    active.load_state_dict(checkpoint)
    monkeypatch.setattr(active.optimizer, "step", original_step)
    active.run()
    expected = session(TinyBackend(stochastic=True))
    expected.run()
    assert_tree_equal(active.state_dict(), expected.state_dict())


def test_observer_failure_keeps_a_committed_checkpoint():
    def observer(active, report):
        assert active.completed_updates == report.optimizer_steps == 1
        raise RuntimeError("observer failed")

    active = session(observers=[observer])
    with pytest.raises(RuntimeError, match="observer failed"):
        active.step()
    assert active.status == "ready"
    assert active.completed_microsteps == 3
    assert active.state_dict()["progress"]["optimizer_steps"] == 1


@pytest.mark.parametrize("tamper", ["config", "plan", "backend", "optimizer", "rng", "progress"])
def test_invalid_checkpoint_fails_before_model_mutation(tamper):
    active = session()
    active.step()
    original = active.state_dict()
    broken = copy.deepcopy(original)
    if tamper == "config":
        broken["config"]["seed"] += 1
    elif tamper == "plan":
        broken["plan"]["phrase"] = "a=b"
    elif tamper == "backend":
        broken["backend"]["original"]["0.weight"] = torch.zeros(8)
    elif tamper == "optimizer":
        first = next(iter(broken["optimizer"]["state"].values()))
        first["exp_avg"] = torch.zeros(8)
    elif tamper == "rng":
        broken["rng"]["torch"] = torch.zeros(3, dtype=torch.uint8)
    else:
        broken["progress"]["microsteps"] += 1
    with pytest.raises((ValueError, RuntimeError)):
        active.load_state_dict(broken)
    assert_tree_equal(active.state_dict(), original)


def test_explicit_prompt_source_and_wrapper_have_no_downloads(monkeypatch):
    backend = TinyBackend()
    with pytest.raises(ValueError, match="prompt_source"):
        train_model(backend, "cat++", iterations=1)
    history = train_model(backend, "cat++", iterations=3, accumulation_steps=2,
                          prompt_source=["a city", "a room"])
    assert len(history) == 2
    with pytest.raises(TypeError, match="observers/policy"):
        train_model(backend, "cat=dog", sample_dir="samples")


def test_encoder_preflight_never_silently_skips_and_invalidates_after_commits():
    backend = TinyBackend()
    with pytest.raises(ValueError):
        train_encoder(backend, "cat=dog | image^painting", iterations=1)
    assert backend.attachments == 0
    history = train_encoder(backend, "cat=dog", iterations=3, accumulation_steps=2)
    assert len(history) == 2
    assert backend.attachments == 1
    assert backend.invalidations == 2


def test_encoder_checkpoint_exact_continuation():
    config = TrainingConfig(iterations=5, accumulation_steps=2, lr=0.02, seed=9)
    continuous = session(config=config, stage="encoder")
    continuous.run()
    split = session(config=config, stage="encoder")
    split.step()
    resumed = session(config=config, stage="encoder")
    resumed.load_state_dict(split.state_dict())
    resumed.run()
    assert_tree_equal(continuous.state_dict(), resumed.state_dict())


def test_explicit_stopping_policy_and_observers_observe_only_commits():
    history, lines = HistoryObserver(), []
    active = session(observers=[history, LossLogger(lines.append)])
    reports = active.run(stop_when=lambda report: report.optimizer_steps >= 2)
    assert len(reports) == len(history.reports) == len(lines) == 2
    assert active.completed_microsteps == 6
    assert active.status == "ready"
    active.run()
    assert active.done
    active.close()
    with pytest.raises(RuntimeError, match="closed"):
        active.run()


@pytest.mark.parametrize("values", [
    {"iterations": -1}, {"accumulation_steps": 0}, {"accumulation_steps": True},
    {"iterations": 1.5}, {"lr": float("nan")}, {"encoder_rank": 0},
    {"curriculum_split": 0}, {"op_defaults": ops.OpDefaults(sample_steps=0)},
])
def test_invalid_budgets(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_checkpoint_unavailable_inside_update_and_bounded_explicit_groups(monkeypatch):
    active = session()
    original = active.backend.predict_v

    def predict(*args, **kwargs):
        with pytest.raises(RuntimeError, match="committed"):
            active.state_dict()
        return original(*args, **kwargs)

    monkeypatch.setattr(active.backend, "predict_v", predict)
    for count in (0, 4, True):
        with pytest.raises(ValueError):
            active.step(count)
    active.step(2)
    assert active.completed_microsteps == 2
    active.run()
    assert active.completed_microsteps == 7


def test_construction_and_step_do_not_advance_caller_rngs():
    backend = TinyBackend(stochastic=True)
    expected_python, expected_torch = random.getstate(), torch.get_rng_state().clone()
    active = session(backend)
    assert random.getstate() == expected_python
    assert torch.equal(torch.get_rng_state(), expected_torch)
    active.step()
    assert random.getstate() == expected_python
    assert torch.equal(torch.get_rng_state(), expected_torch)


def test_provider_state_load_failure_rolls_back_all_declared_weights():
    class HookBackend(TinyBackend):
        fail_load = False

        def training_state_dict(self):
            return {name: copy.deepcopy(module.state_dict())
                    for name, module in self.training_modules().items()}

        def load_training_state_dict(self, state):
            for name, module in self.training_modules().items():
                module.load_state_dict(state[name])
                if self.fail_load:
                    self.fail_load = False
                    raise ValueError("provider rejected state after partial load")

    active = session(HookBackend())
    before = active.state_dict()
    altered = copy.deepcopy(before)
    altered["backend"]["model"]["0.weight"].fill_(9)
    active.backend.fail_load = True
    with pytest.raises(ValueError, match="partial load"):
        active.load_state_dict(altered)
    assert_tree_equal(active.state_dict(), before)


def test_mutating_original_config_cannot_change_effective_session_policy():
    cfg = TrainingConfig(iterations=1)
    active = session(config=cfg)
    before = active.state_dict()["config"]
    cfg.op_defaults.sample_steps = 999
    active.step()
    assert active.state_dict()["config"] == before


def test_encoder_exaggeration_needs_no_external_prompt_dataset():
    active = session(stage="encoder", phrase="cat++", config=TrainingConfig(iterations=1))
    active.run()
    assert active.completed_updates == 1


def test_nonfinite_loss_never_commits_an_update(monkeypatch):
    active = session()
    before = active.backend.transformer[0].weight.detach().clone()
    monkeypatch.setattr(ops, "rule_loss", lambda rule, context: active.params[0].sum() * float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        active.step()
    assert active.status == "failed"
    assert active.completed_updates == active.completed_microsteps == 0
    assert torch.equal(active.backend.transformer[0].weight, before)
    assert all(param.grad is None for param in active.params)


def test_missing_capability_fails_before_any_parameter_selection():
    backend = TinyBackend()
    backend.render = None
    backend.trainable_parameters = lambda method: pytest.fail("must preflight before selection")
    with pytest.raises(TypeError, match="render"):
        session(backend, phrase="photo^painting")
