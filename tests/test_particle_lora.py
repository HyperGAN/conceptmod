"""Full upstream-recipe smoke checks; no pretrained-model quality claims."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, ParticlePrior
from particlegan.bispf import BISPFCriticPenalty, SBNPPriorRegularizer
from conceptmod.particle_lora import ParticleLoRAGenerator


def setup(seed=4):
    torch.manual_seed(seed)
    host = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
    host.requires_grad_(False)
    original = deepcopy(host.state_dict())
    context = torch.tensor([[0.2, -0.5, 1.0]])
    base = host(context).detach()
    generator = ParticleLoRAGenerator(host, ("0", "2"), context, rank=8, z_dim=4)
    critic = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
    trainer = generator.make_trainer(critic, batch_size=8, total_steps=8, seed=12)
    return generator, trainer, original, base


def real():
    return torch.tensor([[-1., .3], [1., -.3]]).repeat(4, 1)


def test_official_recipe_updates_lora_and_real_prior_keeps_base_frozen():
    generator, trainer, original, base = setup()
    assert isinstance(trainer, GANTrainer)
    assert type(trainer.prior) is ParticlePrior
    assert trainer.prior.z.shape == (128, 4)
    assert trainer.recipe.name == "bispf_optimized"
    assert trainer.recipe.optimizer_family == "dualnorm"
    assert trainer.recipe.make_loss().loss == "non_saturating"
    assert isinstance(trainer.penalty, BISPFCriticPenalty)
    assert isinstance(trainer.prior_regularizer, SBNPPriorRegularizer)
    before = trainer.prior.z.detach().clone()
    for _ in range(3):
        result = trainer.step(real())
        assert result["step"] == trainer.completed_steps
    assert trainer.completed_steps == 3
    assert not torch.equal(before, trainer.prior.z)
    assert any(torch.count_nonzero(layer.up.weight) for layer in generator.adapters)
    for name, layer in zip(generator.targets, generator.adapters):
        assert not any(p.requires_grad for p in layer.base.parameters())
        assert torch.equal(layer.base.weight, original[f"{name}.weight"])
        assert torch.equal(layer.base.bias, original[f"{name}.bias"])
    codes = trainer.prior.z[:4]
    assert torch.equal(generator(codes, strength=0), base.expand(4, -1))
    assert not torch.equal(generator(codes, strength=1), generator(codes, strength=0))
    # The same context genuinely depends on the sampled particle row.
    assert not torch.equal(generator(codes)[0], generator(codes)[1])


def test_checkpoint_replays_next_full_update():
    generator, trainer, _, _ = setup()
    trainer.step(real())
    saved = trainer.state_dict()
    result = trainer.step(real())
    expected = trainer.state_dict()
    restored_generator, restored, _, _ = setup(seed=99)
    restored.load_state_dict(saved)
    actual = restored.step(real())
    assert result == actual
    for kind in ("G", "D", "prior"):
        for name, value in expected["models"][kind].items():
            torch.testing.assert_close(restored.state_dict()["models"][kind][name], value, rtol=0, atol=0)
    codes = trainer.prior.z[:4].detach()
    torch.testing.assert_close(generator(codes), restored_generator(codes), rtol=0, atol=0)


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
    base = host(context).detach()
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
    assert torch.equal(generator(trainer.prior.z, strength=0), base.expand(16, -1))
    assert not any("_codes" in name for name in generator.state_dict())
