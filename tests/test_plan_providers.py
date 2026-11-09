from dataclasses import FrozenInstanceError, replace
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from conceptmod import dsl
from conceptmod.contracts import BackendCapabilities, model_views
from conceptmod.plan import EditPlan, compile_phrase
from conceptmod.providers import BackendProvider, BackendRegistry


def test_plan_roundtrip_and_macro_preserves_signed_alpha():
    plan = compile_phrase("cat~dog:0.35")
    assert [r.alpha for r in plan.rules] == [0.7, 1.4, -0.35]
    assert EditPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    with pytest.raises(FrozenInstanceError):
        plan.stage = "encoder"
    changed = plan.to_dict()
    changed["rules"][0]["alpha"] = 3
    with pytest.raises(dsl.DSLError, match="do not match"):
        EditPlan.from_dict(changed)


@pytest.mark.parametrize("phrase", ["red=blue:guidnace=99", "red=blue:nan", "red++:guidance=inf", ";red"])
def test_preflight_rejects_unknown_nonfinite_and_unimplemented(phrase):
    with pytest.raises(dsl.DSLError):
        compile_phrase(phrase)


def test_preflight_respects_stage_and_render_capabilities():
    with pytest.raises(dsl.DSLError, match="encoder"):
        compile_phrase("red^blue", stage="both")
    with pytest.raises(dsl.DSLError, match="encoder"):
        compile_phrase("red={random_prompt}", stage="encoder")
    with pytest.raises(dsl.DSLError, match="rendering"):
        compile_phrase("red^blue", capabilities=BackendCapabilities(differentiable_render=False))
    plan = compile_phrase("red={random_prompt}")
    assert plan.needs_prompt_source
    assert plan.materialize("hello|world")[0].b == "hello world"


def test_explicit_registry_is_lazy_and_rejects_ambiguous_providers(monkeypatch):
    descriptor = BackendProvider("fixture", "conceptmod_test_factory:create", "fixture-conceptmod")
    registry = BackendRegistry([descriptor], discover=False)
    assert registry.resolve("fixture") is descriptor
    assert "conceptmod_test_factory" not in sys.modules
    module = ModuleType("conceptmod_test_factory")
    module.create = lambda **options: SimpleNamespace(**options)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    loaded = registry.load("fixture", device="cpu")
    assert loaded.device == "cpu"
    assert loaded.capabilities == descriptor.capabilities
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(descriptor)
    with pytest.raises(LookupError, match="not installed"):
        registry.load("missing")


def test_provider_caps_follow_runtime_to_session_preflight(monkeypatch):
    from conceptmod.contracts import VELOCITY_OPERATORS
    descriptor = BackendProvider("fixture", "caps_test:create", "fixture-conceptmod",
                                 BackendCapabilities(operators=VELOCITY_OPERATORS,
                                                     differentiable_render=False))
    module = ModuleType("caps_test")
    module.create = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = descriptor.create()
    with pytest.raises(dsl.DSLError, match="not supported"):
        compile_phrase("red^blue", capabilities=backend.capabilities)


def test_views_apply_the_same_role_to_all_operations():
    class Backend:
        def encode_text(self, prompt, **options): return options["frozen"]
        def predict_v(self, *args, **options): return options["frozen"]
        def render(self, *args, **options): return options["frozen"]
        def generate(self, *args, **options): return options["frozen"]
    for view, expected in zip(model_views(Backend()), (True, False)):
        assert view.encode_text("a") is expected
        assert view.predict_v("a", None, None) is expected
        assert view.render("a", None, 1, 1) is expected
        assert view.generate("a", 1) is expected
