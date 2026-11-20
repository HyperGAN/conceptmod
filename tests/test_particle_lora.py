"""Full upstream-recipe smoke checks; no pretrained-model quality claims."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from particlegan import GANTrainer, ParticlePrior, Recipe, get_recipe
from particlegan.empf import EMPFCriticPenalty, EMPFPriorRegularizer
from conceptmod.particle_lora import ParticleLoRAGenerator


def setup(seed=4, recipe=None):
    torch.manual_seed(seed)
    host = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
    host.requires_grad_(False)
    original = deepcopy(host.state_dict())
    context = torch.tensor([[0.2, -0.5, 1.0]])
    # Keep the reference batch identical: BLAS may use different accumulation
    # kernels for a single row and a batch, with valid float32 roundoff.
    base = host(context.expand(4, -1)).detach()
    generator = ParticleLoRAGenerator(host, ("0", "2"), context, rank=8, z_dim=4)
    critic = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
    trainer = generator.make_trainer(critic, batch_size=8, total_steps=8, seed=12,
                                     recipe=recipe)
    return generator, trainer, original, base


def real():
    return torch.tensor([[-1., .3], [1., -.3]]).repeat(4, 1)


def test_official_recipe_updates_lora_and_real_prior_keeps_base_frozen():
    generator, trainer, original, base = setup()
    assert isinstance(trainer, GANTrainer)
    assert type(trainer.prior) is ParticlePrior
    assert trainer.prior.z.shape == (128, 4)
    assert trainer.recipe == get_recipe("empf", z_dim=4, num_particles=128,
                                       batch_size=8, total_steps=8,
                                       reg_every=1, ema_decay=0.0, serve_average=0.0)
    assert trainer.recipe.name == "empf"
    assert trainer.recipe.optimizer_family == "dualnorm"
    assert trainer.recipe.ema_decay == 0.0
    assert trainer.recipe.serve_average == 0.0
    assert getattr(trainer.opt_d, "ema_critic", None) is None
    assert isinstance(trainer.penalty, EMPFCriticPenalty)
    assert isinstance(trainer.prior_regularizer, EMPFPriorRegularizer)
    assert trainer.recipe.reg_every == trainer.penalty.regularizer.lazy_k == 1
    assert [g["lr"] for g in trainer.opt_g.param_groups] == pytest.approx([0.012, 0.03])
    assert [g["lr"] for g in trainer.opt_d.param_groups] == pytest.approx([0.018])
    before = trainer.prior.z.detach().clone()
    for index in range(1, 4):
        result = trainer.step(real(), collect_stats=True)
        assert result["step"] == trainer.completed_steps
        assert result["penalty_stats"]["applied"] is True
        assert trainer.penalty.regularizer.record.calls == index
    assert trainer.completed_steps == 3
    # Upstream retains zero-decay checkpoint copies; they must equal current
    # weights exactly rather than accumulate a moving-weight history.
    for current, mirror in ((trainer.G, trainer.ema_G), (trainer.prior, trainer.ema_prior)):
        _assert_state_equal(mirror.state_dict(), current.state_dict())
    assert not torch.equal(before, trainer.prior.z)
    assert any(torch.count_nonzero(layer.up.weight) for layer in generator.adapters)
    for name, layer in zip(generator.targets, generator.adapters):
        assert not any(p.requires_grad for p in layer.base.parameters())
        assert torch.equal(layer.base.weight, original[f"{name}.weight"])
        assert torch.equal(layer.base.bias, original[f"{name}.bias"])
    codes = trainer.prior.z[:4]
    assert torch.equal(generator(codes, strength=0), base)
    assert not torch.equal(generator(codes, strength=1), generator(codes, strength=0))
    # The same context genuinely depends on the sampled particle row.
    assert not torch.equal(generator(codes)[0], generator(codes)[1])


@pytest.mark.parametrize("preset", ["rhbi_optimized", "bispf_optimized", "empf"])
def test_checkpoint_replays_next_full_update(preset):
    recipe = get_recipe(preset, z_dim=4, num_particles=16, batch_size=8, total_steps=8,
                        ema_decay=0.0, serve_average=0.0)
    generator, trainer, _, _ = setup(recipe=recipe)
    trainer.step(real())
    saved = trainer.state_dict()
    result = trainer.step(real())
    expected = trainer.state_dict()
    restored_generator, restored, _, _ = setup(seed=99, recipe=Recipe.from_dict(saved["recipe"]))
    restored.load_state_dict(saved)
    actual = restored.step(real())
    assert result == actual
    _assert_state_equal(restored.state_dict(), expected)
    codes = trainer.prior.z[:4].detach()
    torch.testing.assert_close(generator(codes), restored_generator(codes), rtol=0, atol=0)


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_state_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("overrides", [
    {"ema_decay": .995}, {"serve_average": 4.0}, {"z_dim": 3},
])
def test_explicit_recipe_rejects_averaging_and_incompatible_latent_dimensions(overrides):
    recipe = get_recipe("gan", z_dim=4, ema_decay=0.0, serve_average=0.0).replace(**overrides)
    with pytest.raises(ValueError, match="weight EMA|z_dim"):
        setup(recipe=recipe)


def test_explicit_legacy_critic_anchor_is_rejected_even_with_generator_ema_disabled():
    recipe = get_recipe("ka2", z_dim=4, ema_decay=0.0, serve_average=0.0)
    with pytest.raises(ValueError, match="weight EMA"):
        setup(recipe=recipe)


@pytest.mark.parametrize("reg_every", [2, 4])
def test_explicit_recipe_cannot_skip_regularizer_steps(reg_every):
    recipe = get_recipe("empf", z_dim=4, reg_every=reg_every,
                        ema_decay=0.0, serve_average=0.0)
    with pytest.raises(ValueError, match=r"reg_every=1 \(lazy_k=1\)"):
        setup(recipe=recipe)


def test_checkpoint_from_another_formulation_is_not_silently_reinterpreted():
    recipe = get_recipe("bispf_optimized", z_dim=4, num_particles=128,
                        batch_size=8, total_steps=8, ema_decay=0.0, serve_average=0.0)
    _, historical, _, _ = setup(recipe=recipe)
    historical.step(real())
    _, current, _, _ = setup()
    with pytest.raises(ValueError, match="recipe does not match"):
        current.load_state_dict(historical.state_dict())


def test_rejects_trainable_or_unknown_targets_before_mutating_host():
    host = nn.Sequential(nn.Linear(3, 2))
    context = torch.ones(1, 3)
    with pytest.raises(ValueError, match="already be frozen"):
        ParticleLoRAGenerator(host, ("0",), context)
    host.requires_grad_(False)
    with pytest.raises(ValueError, match="must name a Linear"):
        ParticleLoRAGenerator(host, ("0", "missing"), context)
    assert isinstance(host[0], nn.Linear)
    with pytest.raises(ValueError, match="singleton"):
        ParticleLoRAGenerator(host, ("0",), context.repeat(2, 1))


def test_trainer_never_reactivates_frozen_dropout_or_running_statistics():
    torch.manual_seed(11)
    host = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.8), nn.Linear(4, 2))
    host.requires_grad_(False).eval()
    context = torch.ones(1, 3)
    original = deepcopy(host.state_dict())
    base = host(context.expand(16, -1)).detach()
    generator = ParticleLoRAGenerator(host, ("0", "3"), context, rank=4)
    critic = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    trainer = generator.make_trainer(critic, num_particles=16, batch_size=8, total_steps=4)
    for _ in range(2):
        trainer.step(real())
    assert not host.training and not host[1].training and not host[2].training
    for key in ("running_mean", "running_var", "num_batches_tracked"):
        assert torch.equal(host[1].state_dict()[key], original["1." + key])
    # Passing the Parameter itself must remain transient context, never become
    # a registered adapter parameter or leak into its checkpoint.
    assert torch.equal(generator(trainer.prior.z, strength=0), base)
    assert not any("_codes" in name for name in generator.state_dict())


class _CheckpointBlock(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
        self.calls = []

    def forward(self, value):
        self.calls.append(torch.is_grad_enabled())
        return super().forward(value)


class _CheckpointHost(nn.Module):
    def __init__(self, use_reentrant, use_closure=False):
        super().__init__()
        self.block = _CheckpointBlock()
        self.alias = self.block
        self.use_reentrant = use_reentrant
        self.use_closure = use_closure
        self.barrier = None
        self.fail = []
        self.nest = []

    def forward(self, value):
        assert self.block is self.alias
        assert not self.training and not self.block.training
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.fail:
            self.fail.pop()
            raise RuntimeError("intentional host failure")
        if self.nest:
            self.nest.pop()()
        if self.use_reentrant is None:
            return self.block(value)
        function = (lambda data: self.block(data)) if self.use_closure else self.block
        return checkpoint(function, value, use_reentrant=self.use_reentrant)


def checkpoint_setup(use_reentrant, use_closure=False):
    torch.manual_seed(23)
    host = _CheckpointHost(use_reentrant, use_closure).requires_grad_(False)
    generator = ParticleLoRAGenerator(host, ("block.0", "block.2"), torch.ones(1, 3), rank=4)
    for layer in generator.adapters:
        nn.init.normal_(layer.up.weight, std=.2)
    return generator


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("use_closure", [False, True])
def test_checkpoint_recomputation_preserves_routing_and_gradients(use_reentrant, use_closure):
    generator = checkpoint_setup(use_reentrant, use_closure)
    eager = checkpoint_setup(None)
    codes = torch.randn(4, 4, requires_grad=True)
    eager_codes = codes.detach().clone().requires_grad_(True)
    expected = eager(eager_codes, strength=.7)
    actual = generator(codes, strength=.7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(generator.host.block.calls) == 1
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert len(generator.host.block.calls) == 2  # real backward recomputation
    torch.testing.assert_close(codes.grad, eager_codes.grad, rtol=0, atol=0)
    for (name, parameter), (_, reference) in zip(generator.named_parameters(), eager.named_parameters()):
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None
    assert not generator.context.requires_grad and generator.context.grad is None
    assert not any("codes" in name for name in generator.state_dict())
    assert all(not hasattr(layer, "_codes") for layer in generator.adapters)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_pending_forwards_nested_calls_and_failures_have_independent_routing(use_reentrant):
    generator = checkpoint_setup(use_reentrant)
    eager = checkpoint_setup(None)
    first = torch.randn(2, 4, requires_grad=True)
    second = torch.randn(3, 4, requires_grad=True)
    # Invoke the same generator inside its host, with a different batch and
    # strength. The nested graph must not replace the outer invocation's codes.
    nested = []
    generator.host.nest.append(lambda: nested.append(generator(second, strength=-.4)))
    outer = generator(first, strength=.8)
    generator.host.fail.append(True)
    with pytest.raises(RuntimeError, match="intentional host failure"):
        generator(torch.zeros(1, 4))
    after = generator(second, strength=.3)
    torch.testing.assert_close(outer, eager(first, strength=.8), rtol=0, atol=0)
    torch.testing.assert_close(nested[0], eager(second, strength=-.4), rtol=0, atol=0)
    torch.testing.assert_close(after, eager(second, strength=.3), rtol=0, atol=0)
    (after.sum() + outer.sum() + nested[0].sum()).backward()
    assert first.grad is not None and second.grad is not None


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_concurrent_forward_contexts_are_independent(use_reentrant):
    generator = checkpoint_setup(use_reentrant)
    eager = checkpoint_setup(None)
    generator.host.barrier = Barrier(2)
    first, second = torch.randn(2, 4), torch.randn(3, 4)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_call = pool.submit(generator, first, strength=.8)
        second_call = pool.submit(generator, second, strength=-.4)
        first_output, second_output = first_call.result(), second_call.result()
    torch.testing.assert_close(first_output, eager(first, strength=.8), rtol=0, atol=0)
    torch.testing.assert_close(second_output, eager(second, strength=-.4), rtol=0, atol=0)
    (first_output.sum() + second_output.sum()).backward()
    assert all(layer.up.weight.grad is not None for layer in generator.adapters)
