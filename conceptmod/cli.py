"""Small provider-aware CLI. Model options belong to explicit product config."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from conceptmod.plan import compile_phrase
from conceptmod.providers import BackendRegistry


def _object(path: str | None) -> dict:
    value = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = value.keys() - {"backend", "training", "prompts"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    for key in ("backend", "training"):
        if key in value and not isinstance(value[key], dict):
            raise ValueError(f"configuration {key} must be an object")
    return value


def main(argv=None, *, registry: BackendRegistry | None = None):
    parser = argparse.ArgumentParser(prog="conceptmod", description="Model-independent concept editing")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backends", help="list installed lightweight provider metadata")
    validate = sub.add_parser("validate", help="compile an edit without loading a model")
    validate.add_argument("--phrase", required=True)
    validate.add_argument("--stage", choices=("model", "encoder", "both"), default="model")
    validate.add_argument("--backend")
    train = sub.add_parser("train", help="execute one explicit model or encoder session")
    train.add_argument("--phrase", required=True)
    train.add_argument("--stage", choices=("model", "encoder"), default="model")
    train.add_argument("--backend", required=True)
    train.add_argument("--config", help="JSON with backend options, training options and prompt sequence")
    train.add_argument("--out", required=True, help="new output directory")
    train.add_argument("--resume", help="exact committed session checkpoint")
    inspect = sub.add_parser("inspect", help="read a complete edit bundle manifest")
    inspect.add_argument("bundle")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            from conceptmod.artifacts import read_edit_bundle

            print(json.dumps(read_edit_bundle(args.bundle), indent=2))
            return 0
        registry = registry if registry is not None else BackendRegistry()
        if args.command == "backends":
            print(json.dumps([{"name": name, "distribution": registry.resolve(name).distribution,
                               "stages": sorted(registry.resolve(name).capabilities.stages),
                               "operators": sorted(registry.resolve(name).capabilities.operators)}
                              for name in registry.names()], indent=2))
            return 0
        descriptor = registry.resolve(args.backend) if args.backend else None
        plan = compile_phrase(args.phrase, stage=args.stage,
                              capabilities=descriptor.capabilities if descriptor else None)
        if args.command == "validate":
            print(json.dumps(plan.to_dict(), indent=2))
            return 0

        from conceptmod.ops import OpDefaults
        from conceptmod.training import TrainingConfig, TrainingSession

        request = _object(args.config)
        training = dict(request.get("training", {}))
        overrides = training.pop("op_defaults", {})
        if not isinstance(overrides, dict):
            raise ValueError("op_defaults must be an object")
        defaults = {**dict(descriptor.training_defaults), **overrides}
        config = TrainingConfig(**training, op_defaults=OpDefaults(**defaults))
        prompts = request.get("prompts")
        if prompts is not None and (not isinstance(prompts, list) or any(
                not isinstance(p, str) or not p.strip() for p in prompts)):
            raise ValueError("prompts must be a list of nonempty strings")
        if plan.needs_prompt_source and not prompts:
            raise ValueError("this plan requires explicit prompts in its configuration")
        destination = Path(args.out)
        if destination.exists():
            raise FileExistsError(f"output already exists: {destination}")
        backend = descriptor.create(**request.get("backend", {}))
        # Dynamic product defaults (e.g. distilled variants) belong to the
        # instantiated product; explicit user options always win.
        runtime_defaults = backend.training_defaults()
        config = TrainingConfig(**training, op_defaults=OpDefaults(
            **{**runtime_defaults, **dict(descriptor.training_defaults), **overrides}))
        try:
            with TrainingSession(backend, plan, config, prompt_source=prompts) as session:
                if args.resume:
                    session.load_checkpoint(args.resume)
                destination.mkdir(parents=True)
                (destination / "request.json").write_text(json.dumps({
                    "provider": descriptor.name, "distribution": descriptor.distribution,
                    "plan": plan.to_dict(), "training": config.to_dict(),
                    "backend": request.get("backend", {}), "prompts": prompts,
                }, indent=2) + "\n")
                while not session.done:
                    report = session.step()
                    session.save_checkpoint(destination / "checkpoint.pt")
                    print(json.dumps(asdict(report)), flush=True)
                if not (destination / "checkpoint.pt").exists():
                    session.save_checkpoint(destination / "checkpoint.pt")
                backend.save_trained(str(destination / "edit"))
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        return 0
    except (ValueError, TypeError, LookupError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
