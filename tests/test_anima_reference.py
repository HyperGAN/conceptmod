"""Anima's native pipeline must use a complete original-model view."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from conceptmod.backends.anima import AnimaBackend


class _Adapter:
    def __init__(self):
        self.enabled = True

    @contextmanager
    def disable_adapter(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


class _Pipeline:
    def __init__(self, fail):
        self.transformer = _Adapter()
        self.text_encoder = _Adapter()
        self.guider = SimpleNamespace(guidance_scale=None)
        self.fail = fail
        self.observed = None
        self.image = object()

    def __call__(self, **kwargs):
        self.observed = (self.transformer.enabled, self.text_encoder.enabled)
        if self.fail:
            raise RuntimeError("pipeline failed")
        return SimpleNamespace(images=[self.image])


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("encoder_lora", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_generation_reference_disables_all_edited_components(frozen, encoder_lora, fail):
    backend = object.__new__(AnimaBackend)
    backend.pipe = _Pipeline(fail)
    backend.transformer = backend.pipe.transformer
    backend.device = "cpu"
    backend.resolution = 16
    backend.generate_steps = 2
    backend.generate_guidance = 4.0
    backend.default_negative = ""
    backend.encoder_lora = encoder_lora
    if fail:
        with pytest.raises(RuntimeError, match="pipeline failed"):
            backend.generate("red", seed=1, frozen=frozen)
    else:
        assert backend.generate("red", seed=1, frozen=frozen) is backend.pipe.image
    assert backend.pipe.observed == (not frozen, not (frozen and encoder_lora))
    assert backend.transformer.enabled
    assert backend.pipe.text_encoder.enabled
