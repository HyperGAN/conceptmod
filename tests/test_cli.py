import json

import pytest

from conceptmod.cli import main
from conceptmod.providers import BackendRegistry
from conceptmod.testing import fixture_registry


def test_list_and_validate_without_a_model(capsys):
    assert main(["backends"], registry=BackendRegistry(discover=False)) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["validate", "--phrase", "red=blue"], registry=BackendRegistry(discover=False)) == 0
    assert json.loads(capsys.readouterr().out)["rules"][0]["op"] == "write"


def test_bad_plan_does_not_call_the_model_factory(monkeypatch):
    registry = fixture_registry()
    called = []
    monkeypatch.setattr(type(registry.resolve("cpu")), "create", lambda *a, **kw: called.append(True))
    with pytest.raises(SystemExit):
        main(["train", "--backend", "cpu", "--phrase", "red=blue:guidnace=4", "--out", "unused"], registry=registry)
    assert called == []


def test_cli_executes_and_exports_a_real_cpu_edit(tmp_path, capsys):
    pytest.importorskip("peft", reason="the source-only CPU adapter fixture requires .[test]")
    pytest.importorskip("safetensors", reason="the fixture exports safetensors edit components")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": {"device": "cpu", "lora_rank": 2},
                                  "training": {"iterations": 3, "accumulation_steps": 2, "lr": 0.01}}))
    destination = tmp_path / "run"
    assert main(["train", "--backend", "cpu", "--phrase", "red=blue", "--config", str(config),
                 "--out", str(destination)], registry=fixture_registry()) == 0
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert reports[-1]["optimizer_steps"] == 2
    assert reports[-1]["microsteps"] == 3
    assert (destination / "checkpoint.pt").is_file()
    assert (destination / "edit" / "edit_bundle.json").is_file()
    assert main(["inspect", str(destination / "edit")]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["metadata"]["weight_provenance"] == "raw-current"
    resumed = tmp_path / "resumed"
    assert main(["train", "--backend", "cpu", "--phrase", "red=blue", "--config", str(config),
                 "--resume", str(destination / "checkpoint.pt"), "--out", str(resumed)],
                registry=fixture_registry()) == 0
    import torch
    original = torch.load(destination / "checkpoint.pt", weights_only=True)
    restored = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert original["progress"] == restored["progress"]
    for name, value in original["backend"]["transformer"].items():
        assert torch.equal(value, restored["backend"]["transformer"][name])
