"""Real CPU native/HG/HNDL/shared-game conformance; no pretrained qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

pytest.importorskip('particle_sliders.pg094')
from conceptmod.backends.cpu import CpuBackend
from conceptmod.modern import open_game, load_native, COHORT, BACKEND_STATES
from conceptmod.modern import session as module
from particle_sliders.pg09 import _hash, validate_pg09_domain
from particle_sliders.lifecycle import CheckpointError
from particle_sliders.field_critic import FieldCritic


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def seal(packet):
    packet['sha256'] = _hash({k: v for k, v in packet.items() if k != 'sha256'})
    return packet


def flags(runtime):
    roots = (runtime.transformer, runtime.text_encoder)
    return ([m.training for r in roots for m in r.modules()],
            [p.requires_grad for r in roots for p in r.parameters()])


def native_args():
    return dict(prompt='red', latent=torch.ones(1, 4, 8, 8), timestep=torch.tensor([500.]))


def test_native_peft_outputs_input_and_parameter_vjps_match_hndl():
    runtime, cfg = module._configuration()
    leaf = runtime.transformer.base_model.model.linear2
    nn.init.normal_(leaf.lora_B['default'].weight, std=.1)
    old = deepcopy(leaf)
    new = module.HNDLLoRA(leaf)
    assert type(new.graph).__module__ == 'hndl.torch'
    assert new.identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    assert isinstance(new.graph['down'], nn.Linear)
    assert 'product_source_sha256' in new.extra_repr()
    x = torch.randn(2, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = old(x), new(y)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    probe = torch.randn_like(actual)
    g0 = torch.autograd.grad((expected * probe).sum(),
        (x, old.lora_A['default'].weight, old.lora_B['default'].weight))
    g1 = torch.autograd.grad((actual * probe).sum(),
        (y, new.graph['down'].weight, new.graph['up'].weight))
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    with new.scaled(0):
        assert torch.equal(new(y), leaf.base_layer(y))
    full = new(y)
    with new.scaled(.5):
        torch.testing.assert_close(new(y), leaf.base_layer(y) + .5 * (full - leaf.base_layer(y)))
    assert new.strength == 1


def test_native_full_velocity_matches_existing_cpu_backend():
    runtime, cfg = module._configuration()
    reference = deepcopy(runtime)
    nn.init.normal_(reference.transformer.base_model.model.linear2.lora_B['default'].weight, std=.1)
    args = native_args()
    expected = reference.predict_v(args['prompt'], args['latent'], args['timestep'], frozen=False)
    frozen = runtime.predict_v(args['prompt'], args['latent'], args['timestep'], frozen=True)
    before, state = flags(runtime), deepcopy(runtime.transformer.state_dict())
    binding = module.CpuResponse(runtime, cfg)
    try:
        with torch.no_grad():
            binding.network.graph['up'].weight.copy_(
                reference.transformer.base_model.model.linear2.lora_B['default'].weight)
        torch.testing.assert_close(binding.response(**args), expected)
        assert torch.equal(binding.response(**args, strength=0), frozen)
    finally:
        binding.close()
    same(state, runtime.transformer.state_dict())
    assert flags(runtime) == before
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)


def test_critic_outputs_vjps_and_actual_public_penalty_second_derivatives():
    with open_game(steps=2) as session:
        graph = session.game.critic
        assert isinstance(graph.graph['field0'], nn.Conv2d)
        assert isinstance(graph.graph['field2'], nn.Conv2d)
        assert graph.graph['field0'].kernel_size == (1, 1)
        assert graph.graph['field2'].kernel_size == (3, 3)
        old = FieldCritic(4, 1, width=4)
        for name, layer in [('field0', old.field[0]), ('field2', old.field[2]),
                            ('condition', old.condition), ('score0', old.score[0]), ('score2', old.score[2])]:
            layer.load_state_dict(graph.graph[name].state_dict())
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        condition = torch.randn(2, 1)
        a, b = old(x, condition), graph(y, condition)
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
        ga = torch.autograd.grad(a.sum(), x, create_graph=True)[0]
        gb = torch.autograd.grad(b.sum(), y, create_graph=True)[0]
        torch.testing.assert_close(ga, gb, atol=2e-7, rtol=2e-5)
        # Use the actual public regularizer; do not restate its mathematics.
        real, fake = torch.randn_like(x), torch.randn_like(x)
        _, d0 = session.game.recipe.make_optimizers(session.game.network, old)
        _, d1 = session.game.recipe.make_optimizers(session.game.network, graph)
        p0 = session.game.recipe.make_critic_penalty(d0, collect_stats=True)
        p1 = session.game.recipe.make_critic_penalty(d1, collect_stats=True)
        a, b = p0(old, real, fake, condition), p1(graph, real, fake, condition)
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
        ga = torch.autograd.grad(a, old.field[0].weight)[0]
        gb = torch.autograd.grad(b, graph.graph['field0'].weight)[0]
        torch.testing.assert_close(ga, gb, atol=2e-7, rtol=3e-5)
        assert torch.isfinite(gb).all() and gb.abs().sum() > 0


def test_real_game_updates_exact_resume_ema_and_global_rng():
    before_rng = torch.get_rng_state().clone()
    with open_game(steps=4) as full:
        before = deepcopy(full.game.network.state_dict())
        full.train()
        expected = full.checkpoint()
        assert any(not torch.equal(v, full.game.network.state_dict()[k]) for k, v in before.items())
        assert full.ema.values is not None
        assert full.config.cohort == COHORT
        assert full.game._signature['response_shape'] == [4, 8, 8]
        assert full.game._signature['objective'] == 'g_adv+public_prior;d_adv+public_penalty'
        assert len({id(p) for p in full.game.network.parameters()}) == 2
    with open_game(steps=4) as split:
        split.train(stop_after=2)
        packet = split.checkpoint()
    with open_game(steps=4, resume=packet) as resumed:
        resumed.train()
        same(expected, resumed.checkpoint())
        zero, half, one = [resumed.evaluate(strength=v) for v in (0, .5, 1)]
        torch.testing.assert_close(half, zero + .5 * (one - zero), atol=1e-6, rtol=1e-5)
        assert not torch.equal(zero, one)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('selection', ['raw', 'ema'])
def test_native_inference_never_reconstructs_training(selection, monkeypatch, tmp_path):
    with open_game(steps=2) as session:
        session.train()
        path = tmp_path / 'native.pt'
        session.export_inference(path, selection=selection)
        args = native_args()
        values = session.game.network.state_dict() if selection == 'raw' else session.ema.values
        saved = deepcopy(session.game.network.state_dict())
        session.game.network.load_state_dict(values)
        expected = [session.game.backend.response(**args, strength=v) for v in (0, .5, 1)]
        session.game.network.load_state_dict(saved)
    monkeypatch.setattr(module, '_prepare', lambda *a, **k: pytest.fail('training factory used'))
    monkeypatch.setattr(module, 'ConditionalPG094', lambda *a, **k: pytest.fail('game constructed'))
    monkeypatch.setattr(module, 'HNDLFieldCritic', lambda *a, **k: pytest.fail('critic constructed'))
    with load_native(path) as loaded:
        assert loaded.selection == selection
        for strength, value in zip((0, .5, 1), expected):
            assert torch.equal(loaded.response(**args, strength=strength), value)
        assert not hasattr(loaded, 'game')
    # A clean interpreter has no session, prepared rows, critic or teacher cache.
    script = """
import sys, torch
from conceptmod.modern import load_native
from conceptmod.modern import session as m
m._prepare=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('training forbidden'))
with load_native(sys.argv[1]) as s:
 value=s.response(prompt='red',latent=torch.ones(1,4,8,8),timestep=torch.tensor([500.]))
 assert value.shape==(1,4,8,8) and torch.isfinite(value).all()
 assert not hasattr(s,'game')
"""
    result = subprocess.run([sys.executable, '-c', script, str(path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', [n for n in BACKEND_STATES if n != 'cpu'])
def test_all_seven_native_backends_refuse_before_model_loading(name, monkeypatch):
    monkeypatch.setattr(module, '_configuration', lambda *a, **k: pytest.fail('native construction'))
    with pytest.raises(ValueError, match='gated'):
        open_game(backend=name)


@pytest.mark.parametrize('phrase', ['blue=red', 'red++', 'red#blue', 'red=blue*2'])
def test_other_dsl_objectives_are_not_relabelled(phrase):
    with pytest.raises(ValueError):
        open_game(phrase)


def test_native_source_and_builder_tamper_reject_before_borrowed_mutation(monkeypatch):
    runtime, _ = module._configuration()
    state, modes = deepcopy(runtime.transformer.state_dict()), flags(runtime)
    original = module._sha
    monkeypatch.setattr(module, '_sha', lambda p: '0' * 64 if str(p).endswith('cpu.py') else original(p))
    with pytest.raises(ValueError, match='native source'):
        open_game(runtime=runtime)
    same(state, runtime.transformer.state_dict())
    assert flags(runtime) == modes
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)
    monkeypatch.undo()
    import particle_sliders.fibo_hndl as builder
    original = builder._sha
    monkeypatch.setattr(builder, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError, match='builder closure'):
        open_game(runtime=runtime)
    assert flags(runtime) == modes
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)


def test_model_and_shape_rejection_before_binding():
    runtime, _ = module._configuration()
    runtime.transformer.base_model.model.linear1.weight.data.add_(1)
    state, modes = deepcopy(runtime.transformer.state_dict()), flags(runtime)
    with pytest.raises(ValueError, match='base/rank'):
        open_game(runtime=runtime)
    same(state, runtime.transformer.state_dict())
    assert flags(runtime) == modes
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)
    with pytest.raises(ValueError, match='width >=170'):
        validate_pg09_domain((1, 256), response_domain='continuous_vector')


@pytest.mark.parametrize('target', ['cohort', 'source', 'shape', 'legacy'])
def test_checkpoint_mismatch_rejects_before_borrowed_hook(target):
    with open_game(steps=2) as session:
        packet = session.checkpoint()
    if target == 'cohort':
        packet['manifest']['config']['cohort'] = 'historical'
    elif target == 'source':
        packet['manifest']['game_signature']['implementation_sources']['particle_sliders.pg09'] = '0'*64
    elif target == 'shape':
        packet['manifest']['game_signature']['response_shape'] = [256]
    else:
        packet = {'legacy': torch.tensor(1)}
    if target != 'legacy':
        seal(packet)
    runtime, _ = module._configuration()
    modes, state = flags(runtime), deepcopy(runtime.transformer.state_dict())
    with pytest.raises(CheckpointError):
        open_game(steps=2, runtime=runtime, resume=packet)
    assert flags(runtime) == modes
    same(state, runtime.transformer.state_dict())
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)


def test_native_artifact_identity_rejected_before_attach(monkeypatch):
    with open_game(steps=2) as session:
        packet = session.export_inference()
    packet['specification']['identity']['model']['base_sha256'] = '0'*64
    seal(packet)
    calls = []
    monkeypatch.setattr(module.CpuResponse, 'attach', lambda *a: calls.append('native mutation'))
    with pytest.raises(CheckpointError, match='before binding'):
        load_native(packet)
    assert calls == []


@pytest.mark.parametrize('corruption', ['shape', 'nonfinite'])
def test_native_payload_rejects_before_any_hook_attachment(monkeypatch, corruption):
    with open_game(steps=2) as session:
        packet = session.export_inference()
    key = next(iter(packet['network']))
    if corruption == 'shape':
        packet['network'][key] = packet['network'][key].flatten()
    else:
        packet['network'][key].fill_(float('nan'))
    seal(packet)
    calls = []
    monkeypatch.setattr(module.CpuResponse, 'attach', lambda *a: calls.append('native mutation'))
    with pytest.raises(CheckpointError, match='payload differs before binding'):
        load_native(packet)
    assert calls == []


def test_borrowed_cleanup_on_prepare_step_observation_and_restore_failure(monkeypatch):
    runtime, _ = module._configuration()
    runtime.transformer.train()
    runtime.transformer.base_model.model.linear1.eval()
    before, modes = deepcopy(runtime.transformer.state_dict()), flags(runtime)
    original = module.HNDLFieldCritic
    monkeypatch.setattr(module, 'HNDLFieldCritic', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('critic failed')))
    with pytest.raises(RuntimeError, match='critic failed'):
        open_game(runtime=runtime)
    assert flags(runtime) == modes
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)
    monkeypatch.setattr(module, 'HNDLFieldCritic', original)
    with pytest.raises(RuntimeError, match='response failed'):
        with open_game(runtime=runtime, steps=2) as session:
            monkeypatch.setattr(session.game.backend, '_velocity',
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError('response failed')))
            session.step()
    assert flags(runtime) == modes
    same(before, runtime.transformer.state_dict())
    assert 'forward' not in vars(runtime.transformer.base_model.model.linear2)


def test_every_cleanup_runs_if_projection_owner_changes():
    runtime, _ = module._configuration()
    before = flags(runtime)
    session = open_game(runtime=runtime)
    leaf = runtime.transformer.base_model.model.linear2
    replacement = lambda x: x
    leaf.forward = replacement
    with pytest.raises(RuntimeError, match='ownership changed'):
        session.close()
    assert flags(runtime) == before
    assert leaf.forward is replacement
    del leaf.forward


def test_failed_observation_restores_graph_and_all_native_flags(monkeypatch):
    runtime, _ = module._configuration()
    before = flags(runtime)
    with open_game(runtime=runtime) as session:
        owned = flags(runtime)
        graph = deepcopy(session.game.network.state_dict())
        def fail(*args, **kwargs):
            runtime.transformer.train().requires_grad_(True)
            runtime.text_encoder.train().requires_grad_(True)
            runtime.transformer.gradient_checkpointing = True
            session.game.network.graph['up'].weight.data.add_(1)
            raise RuntimeError('observation failed')
        monkeypatch.setattr(session.game.backend, '_velocity', fail)
        with pytest.raises(RuntimeError, match='observation failed'):
            session.evaluate(strength=.5)
        assert flags(runtime) == owned
        assert not hasattr(runtime.transformer, 'gradient_checkpointing')
        assert session.game.network.strength == 1
        same(graph, session.game.network.state_dict())
    assert flags(runtime) == before


def test_late_shared_restore_failure_rolls_back_adapter_and_ema():
    with open_game(steps=3) as source, open_game(steps=3) as target:
        source.train(stop_after=2)
        target.train(stop_after=1)
        before, rng = target.checkpoint(), torch.get_rng_state().clone()
        calls = []
        def fail_once(module, incompatible):
            calls.append('copied')
            if len(calls) == 1:
                raise RuntimeError('native post-load failure')
        handle = target.game.network.register_load_state_dict_post_hook(fail_once)
        try:
            with pytest.raises(RuntimeError, match='post-load failure'):
                target.restore(source.checkpoint())
        finally:
            handle.remove()
        assert len(calls) == 2
        same(before, target.checkpoint())
        assert torch.equal(rng, torch.get_rng_state())


def test_changed_shared_core_source_rejects_before_native_construction(monkeypatch):
    original = module._sha
    monkeypatch.setattr(module, '_sha', lambda p: '0' * 64 if str(p).endswith('/pg09.py') else original(p))
    monkeypatch.setattr(module, '_configuration', lambda *a, **k: pytest.fail('native construction'))
    with pytest.raises(ValueError, match='core source closure'):
        open_game()


@pytest.mark.parametrize('strength', [-1., 1.1, float('nan')])
def test_native_strength_refusal_precedes_forward(strength):
    with open_game(steps=1) as session:
        packet = session.export_inference()
    with load_native(packet) as loaded:
        with pytest.raises(RuntimeError, match='strength'):
            loaded.response(**native_args(), strength=strength)


@pytest.mark.parametrize('field,value', [
    ('latent', torch.zeros(1, 256)),
    ('latent', torch.zeros(1, 4, 8, 8, dtype=torch.float64)),
    ('timestep', torch.tensor([1001.])),
])
def test_native_geometry_and_time_refusal(field, value):
    with open_game(steps=1) as session:
        packet = session.export_inference()
    with load_native(packet) as loaded:
        args = native_args()
        args[field] = value
        with pytest.raises(ValueError):
            loaded.response(**args)


def test_alpha_conditions_change_actual_teacher_student_and_vjp():
    from dataclasses import replace
    with open_game(steps=2) as session:
        rows = session.game.rows
        alphas = [prepared.row.guidance for prepared in rows]
        assert alphas == list(module.TRAINING_ALPHAS)
        assert torch.equal(session.game.conditions, torch.tensor(alphas).unsqueeze(1))
        assert torch.equal(rows[0].row.latent, rows[1].row.latent)
        assert torch.equal(rows[0].row.timestep, rows[1].row.timestep)
        backend = session.game.backend
        residuals = [backend.generated(row) - row.target for row in rows]
        assert not torch.equal(residuals[0], residuals[1])
        torch.testing.assert_close(residuals[0] / alphas[0], residuals[1] / alphas[1])
        gradients = [torch.autograd.grad(backend.generated(row).sum(),
            backend.network.graph['up'].weight)[0] for row in rows]
        assert gradients[1].abs().sum() > 0
        torch.testing.assert_close(gradients[0], alphas[0] * gradients[1])
        zero = backend.prepare(replace(rows[1].row, guidance=0.))
        assert torch.equal(backend.generated(zero), zero.target)
        torch.testing.assert_close(rows[0].target, rows[0].neutral + alphas[0] *
            (rows[1].target - rows[1].neutral))


@pytest.mark.parametrize('matrix', ['lora_A', 'lora_B'])
def test_borrowed_nonfresh_initialization_rejects_before_mutation(matrix):
    runtime, configuration = module._configuration()
    leaf = runtime.transformer.base_model.model.linear2
    getattr(leaf, matrix)['default'].weight.data.add_(.125)
    before, modes = deepcopy(runtime.transformer.state_dict()), flags(runtime)
    with pytest.raises(ValueError, match='fresh seed-matched A/B'):
        open_game(runtime=runtime)
    assert 'forward' not in vars(leaf)
    same(before, runtime.transformer.state_dict())
    assert flags(runtime) == modes
    assert module._lora_initialization_identity(leaf) != configuration['initialization_sha256']


def test_actual_shared_training_moves_student_toward_frozen_teacher():
    from dataclasses import replace
    import json
    with open_game(steps=128) as session:
        backend = session.game.backend
        rows = list(session.game.rows)
        # An unseen alpha tests the declared interpolation, not z/t generality.
        rows.append(backend.prepare(replace(rows[1].row, guidance=.6)))
        targets = [row.target.clone() for row in rows]
        def errors():
            with torch.no_grad():
                return [float((backend.generated(row) - row.target).square().mean()) for row in rows]
        before = errors()
        session.train()
        after = errors()
        assert all(end < .8 * start for start, end in zip(before, after))
        for row, target in zip(rows, targets):
            assert torch.equal(row.target, target)
        print('BEHAVIOR ' + json.dumps(dict(steps=128, alphas=[row.row.guidance for row in rows],
            before_teacher_mse=before, after_teacher_mse=after)))


def test_critic_initialization_is_seeded_independently_of_caller_rng():
    before = torch.get_rng_state().clone()
    with open_game(steps=2) as first:
        expected = deepcopy(first.game.critic.state_dict())
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(7)
    changed = torch.get_rng_state().clone()
    with open_game(steps=2) as second:
        same(expected, second.game.critic.state_dict())
    assert torch.equal(changed, torch.get_rng_state())
