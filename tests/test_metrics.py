import torch
import pytest

from conceptmod import metrics, ops
from tests.test_ops import MockBackend


def _report(backend, phrase, **kw):
    cfg = ops.OpDefaults(sample_steps=4, write_guidance=2.0)
    probes = metrics.capture_probes(backend, cfg, seeds=(0,), stop_fracs=(0.5,))
    return metrics.eval_phrase(backend, phrase, cfg, probes=probes, **kw)


def test_swap_ratio_is_one_before_any_edit():
    report = _report(MockBackend(), "cat=dog")
    assert report.swap_ratio == pytest.approx(1.0, abs=1e-5)
    assert report.done is False


def test_swap_ratio_hits_zero_when_write_target_is_met():
    backend = MockBackend()
    cfg = ops.OpDefaults(sample_steps=4, write_guidance=2.0)
    stop = 2
    g = torch.Generator(device="cpu").manual_seed(0 + 17 * stop)
    z, t = backend.partial_denoise("", stop, 4, 1.0, g)
    v0 = backend.predict_v("", z, t, frozen=True)
    vb = backend.predict_v("dog", z, t, frozen=True)
    vf = backend.predict_v("cat", z, t, frozen=True)
    target = v0 + 2.0 * (vb - v0)
    probes = metrics.ProbeSet(((z, t),))
    backend.delta.data = (target - vf)
    report = metrics.eval_phrase(backend, "cat=dog", cfg, probes=probes, swap_stop=0.05)
    assert report.swap_ratio == pytest.approx(0.0, abs=1e-5)
    assert report.done is True


def test_freeze_hold_is_zero_at_init():
    report = _report(MockBackend(), "#")
    assert report.hold == pytest.approx(0.0, abs=1e-6)
    assert report.done is False  # freeze-only phrase is not a stoppable edit


def test_composite_done_needs_write_and_hold():
    report = _report(MockBackend(), "#:0.4|human=robot:0.8|robot%human:-0.1")
    assert report.swap_ratio == pytest.approx(1.0, abs=1e-5)
    assert report.hold == pytest.approx(0.0, abs=1e-6)
    assert report.done is False
    assert "swap_ratio" in report.summary()


class _ModeBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2))
        self.transformer[0].eval()  # preserve this mixed child mode exactly
        self.text_encoder = torch.nn.Linear(2, 2).eval()
        self.seen_latents = []

    def assert_evaluating(self):
        assert not any(module.training for module in self.transformer.modules())
        assert not self.text_encoder.training

    def partial_denoise(self, *args):
        self.assert_evaluating()
        z, t = super().partial_denoise(*args)
        # Fixed seeds would still produce different probes after an update.
        return z + self.delta.detach(), t

    def predict_v(self, prompt, z, timestep, frozen):
        self.assert_evaluating()
        self.seen_latents.append(z.detach().clone())
        return super().predict_v(prompt, z, timestep, frozen)


def _modes(backend):
    return tuple(module.training for root in (backend.transformer, backend.text_encoder)
                 for module in root.modules())


def test_fixed_probes_are_explicit_owned_and_reused_after_model_updates(monkeypatch):
    backend = _ModeBackend()
    modes = _modes(backend)
    cfg = ops.OpDefaults(sample_steps=4)
    rng = torch.get_rng_state().clone()
    probes = metrics.capture_probes(backend, cfg, seeds=(0,), stop_fracs=(.5,))
    assert _modes(backend) == modes
    assert torch.equal(torch.get_rng_state(), rng)
    original = probes.samples[0][0].clone()
    backend.delta.data.add_(1)

    def no_sampling(*args, **kwargs):
        raise AssertionError("observation must not sample again")

    monkeypatch.setattr(backend, "partial_denoise", no_sampling)
    metrics.eval_phrase(backend, "cat=dog", cfg, probes=probes)
    metrics.eval_phrase(backend, "cat=dog", cfg, probes=probes)
    assert backend.partial_calls == 1
    assert all(torch.equal(z, original) for z in backend.seen_latents)
    assert _modes(backend) == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert backend.delta.grad is None


@pytest.mark.parametrize("operation", ["capture", "evaluate"])
def test_probe_modes_restore_after_failure(monkeypatch, operation):
    backend = _ModeBackend()
    modes = _modes(backend)
    probes = metrics.ProbeSet(((torch.zeros(1, *backend.latent_shape), torch.tensor(2.)),))

    def fail(*args, **kwargs):
        backend.assert_evaluating()
        raise RuntimeError("probe failed")

    method = "partial_denoise" if operation == "capture" else "predict_v"
    monkeypatch.setattr(backend, method, fail)
    with pytest.raises(RuntimeError, match="probe failed"):
        if operation == "capture":
            metrics.capture_probes(backend)
        else:
            metrics.eval_phrase(backend, "cat=dog", probes=probes)
    assert _modes(backend) == modes


def test_probe_set_owns_detached_tensors_and_rejects_empty_samples():
    z = torch.ones(1, 2, requires_grad=True)
    t = torch.tensor(3.)
    probes = metrics.ProbeSet(((z, t),))
    z.data.zero_()
    t.zero_()
    assert torch.equal(probes.samples[0][0], torch.ones(1, 2))
    assert probes.samples[0][1].item() == 3
    assert not probes.samples[0][0].requires_grad
    with pytest.raises(ValueError, match="at least one"):
        metrics.ProbeSet(())
    with pytest.raises(TypeError, match="explicitly captured"):
        metrics.eval_phrase(MockBackend(), "cat=dog", probes=[])
