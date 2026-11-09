"""Committed training updates, isolated random streams and resumable state.

Providers own models, placements and serialization. Sessions own optimization and
never download prompts, render samples, select candidates or write diagnostics.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import random
import tempfile
from typing import Callable, Iterator, Sequence

import torch

from conceptmod import dsl, ops
from conceptmod.plan import EditPlan


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 500
    accumulation_steps: int = 1
    lr: float = 1e-5
    train_method: str = "xattn"
    seed: int = 0
    max_grad_norm: float = 1.0
    op_defaults: ops.OpDefaults = field(default_factory=ops.OpDefaults)
    encoder_rank: int = 8
    encoder_strength: float = 1.0
    curriculum_split: int = 20

    def __post_init__(self):
        for name, minimum in (("iterations", 0), ("accumulation_steps", 1),
                              ("seed", 0), ("encoder_rank", 1),
                              ("curriculum_split", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, minimum in (("lr", 0), ("max_grad_norm", 0), ("encoder_strength", 0)):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be finite and >= {minimum}")
        if not isinstance(self.op_defaults, ops.OpDefaults):
            raise TypeError("op_defaults must be OpDefaults")
        cfg = self.op_defaults
        for name in ("sample_steps", "pixel_render_steps", "pixel_grad_steps"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(cfg).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= cfg.probe_p <= 1:
            raise ValueError("probe_p must lie in [0, 1]")

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, value: dict) -> TrainingConfig:
        values = copy.deepcopy(value)
        values["op_defaults"] = ops.OpDefaults(**values["op_defaults"])
        return cls(**values)


@dataclass(frozen=True)
class StepReport:
    microsteps: int
    optimizer_steps: int
    group_microsteps: int
    loss: float
    rule_losses: tuple[tuple[str, float], ...]


def _modules(backend) -> dict[str, torch.nn.Module]:
    """Prefer the provider contract; retain 2.x attribute compatibility here."""
    if callable(getattr(backend, "training_modules", None)):
        modules = dict(backend.training_modules())
    else:
        modules = {name: getattr(backend, name) for name in
                   ("transformer", "text_encoder", "frozen")
                   if isinstance(getattr(backend, name, None), torch.nn.Module)}
    if not modules or not all(isinstance(k, str) and isinstance(v, torch.nn.Module)
                              for k, v in modules.items()):
        raise TypeError("provider must expose a nonempty training_modules() mapping")
    return modules


def _identity(backend) -> dict:
    if callable(getattr(backend, "training_identity", None)):
        return copy.deepcopy(backend.training_identity())
    metadata = (backend.artifact_metadata() if callable(getattr(backend, "artifact_metadata", None))
                else {"model_id": getattr(backend, "model_id", None)})
    return {"class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "metadata": copy.deepcopy(metadata)}


def _clear_caches(backend) -> None:
    invalidate = getattr(backend, "invalidate_caches", None)
    if callable(invalidate):
        invalidate()


@contextmanager
def _preserve_modes(backend, *, train: bool = False):
    roots = _modules(backend)
    modes = [(module, module.training) for root in roots.values() for module in root.modules()]
    try:
        if train:
            for root in roots.values():
                if any(param.requires_grad for param in root.parameters()):
                    root.train()
        yield
    finally:
        # Attachment may replace a root. Preserve its caller-visible root mode,
        # then restore every original submodule (including mixed-mode trees).
        for name, root in _modules(backend).items():
            if name in roots and root is not roots[name]:
                root.train(next(mode for module, mode in modes if module is roots[name]))
        for module, mode in modes:
            module.training = mode


class TrainingSession:
    """One explicit model or encoder plan and a declared microstep budget.

    ``step`` is one optimizer commit, with at most ``accumulation_steps``
    microsteps. Failed optimization invalidates the session until a committed
    checkpoint is loaded. Observer failures happen after commit and preserve it.
    Provider state hooks must include current weights, separate original copies,
    and mutable buffers; the fallback snapshots all declared training modules.
    """

    def __init__(self, backend, plan: EditPlan, config: TrainingConfig, *,
                 prompt_source: Sequence[str] | None = None, observers=()):
        if not isinstance(plan, EditPlan) or plan.stage not in {"model", "encoder"}:
            raise ValueError("a session needs one explicit model or encoder EditPlan")
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be TrainingConfig")
        # Roundtrip validates caller-constructed or tampered plans before model mutation.
        plan = EditPlan.from_dict(plan.to_dict(), capabilities=getattr(backend, "capabilities", None))
        config = TrainingConfig.from_dict(config.to_dict())
        if isinstance(prompt_source, (str, bytes)):
            raise TypeError("prompt_source must be a sequence of prompt strings")
        prompts = tuple(prompt_source) if prompt_source is not None else ()
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("prompt_source entries must be nonempty strings")
        if plan.stage == "model" and plan.needs_prompt_source and not prompts:
            raise ValueError("this plan requires an explicit nonempty prompt_source")
        required = ({"trainable_parameters", "predict_v", "partial_denoise"}
                    if plan.stage == "model" else
                    {"attach_encoder_lora", "encode_text", "encode_text_grad", "invalidate_caches"})
        if any(rule.op == dsl.PIXEL for rule in plan.rules):
            required.add("render")
        missing = sorted(name for name in required if not callable(getattr(backend, name, None)))
        if missing:
            raise TypeError(f"provider is missing required training methods: {missing}")
        state_hooks = [callable(getattr(backend, name, None)) for name in
                       ("training_state_dict", "load_training_state_dict")]
        if any(state_hooks) and not all(state_hooks):
            raise TypeError("providers must implement both training state hooks")
        modules = _modules(backend)
        self.backend, self.plan, self.config = backend, plan, config
        self._configuration = config.to_dict()
        self._op_cfg = copy.deepcopy(config.op_defaults)
        self._prompts = prompts
        self._observers = tuple(observers)
        if not all(callable(observer) for observer in self._observers):
            raise TypeError("observers must be callable")
        self.rng = random.Random(config.seed)
        self._python_rng = random.Random(config.seed + 1).getstate()
        self._cuda_devices = sorted({tensor.device.index for root in modules.values()
                                     for tensor in list(root.parameters()) + list(root.buffers())
                                     if tensor.device.type == "cuda"})
        self._torch_rng = torch.Generator(device="cpu").manual_seed(config.seed).get_state()
        self._cuda_rng = {index: torch.Generator(device=f"cuda:{index}").manual_seed(config.seed).get_state()
                          for index in self._cuda_devices}
        self.completed_microsteps = 0
        self.completed_updates = 0
        self.reports: list[StepReport] = []
        self.status = "ready"
        self._custom_state = all(state_hooks)
        with self._isolated_rng(), _preserve_modes(backend):
            if plan.stage == "encoder":
                from conceptmod.encoder_train import EncoderObjective
                self.params = list(backend.attach_encoder_lora(config.encoder_rank))
                self._objective = EncoderObjective(backend, plan, config)
            else:
                self.params = list(backend.trainable_parameters(config.train_method))
                self._objective = None
            if not self.params or not all(isinstance(p, torch.nn.Parameter) for p in self.params):
                raise ValueError("provider returned no trainable parameters")
            if len({id(param) for param in self.params}) != len(self.params):
                raise ValueError("provider returned duplicate trainable parameters")
            self.optimizer = torch.optim.AdamW(self.params, lr=config.lr)
            self.optimizer.zero_grad(set_to_none=True)
        self._backend_identity = _identity(backend)

    @contextmanager
    def _isolated_rng(self):
        external_python = random.getstate()
        with torch.random.fork_rng(devices=self._cuda_devices):
            random.setstate(self._python_rng)
            torch.set_rng_state(self._torch_rng)
            for index, state in self._cuda_rng.items():
                torch.cuda.set_rng_state(state, index)
            try:
                yield
            finally:
                self._python_rng = random.getstate()
                self._torch_rng = torch.get_rng_state().clone()
                self._cuda_rng = {index: torch.cuda.get_rng_state(index).clone()
                                  for index in self._cuda_devices}
                random.setstate(external_python)

    @property
    def done(self) -> bool:
        return self.completed_microsteps == self.config.iterations

    def _model_microstep(self) -> Iterator[tuple[str, torch.Tensor]]:
        cfg = self._op_cfg
        random_prompt = self.rng.choice(self._prompts) if self._prompts else None
        rules = self.plan.materialize(random_prompt)
        stop_index = self.rng.randrange(cfg.sample_steps)
        seed = self.rng.randrange(2**31)
        probe = None
        if any(rule.op == dsl.EXAGGERATE for rule in rules) and self.rng.random() < cfg.probe_p:
            probe = dsl.sanitize_prompt(self.rng.choice(self._prompts))[:200]
        context = ops.StepContext(self.backend, stop_index, seed, cfg, probe=probe)
        try:
            for rule in rules:
                yield rule.raw, rule.alpha * ops.rule_loss(rule, context)
                context.clear_grad_cache()
        finally:
            context.clear_grad_cache()

    def step(self, microsteps: int | None = None) -> StepReport:
        if self.status not in {"ready", "complete"}:
            raise RuntimeError(f"cannot step a {self.status} session")
        if self.done:
            raise StopIteration("the declared microstep budget is complete")
        remaining = self.config.iterations - self.completed_microsteps
        count = min(self.config.accumulation_steps, remaining) if microsteps is None else microsteps
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1
                or count > min(self.config.accumulation_steps, remaining)):
            raise ValueError("microsteps must fit the remaining accumulation group")
        self.status = "running"
        losses: dict[str, float] = {}
        total_loss = 0.0
        try:
            with self._isolated_rng(), _preserve_modes(self.backend, train=True):
                self.optimizer.zero_grad(set_to_none=True)
                for _ in range(count):
                    values = (self._model_microstep() if self._objective is None
                              else self._objective.losses(self.rng))
                    try:
                        for raw, loss in values:
                            value = float(loss.detach())
                            if not math.isfinite(value):
                                raise FloatingPointError(f"non-finite loss for {raw!r}")
                            total_loss += value / count
                            losses[raw] = losses.get(raw, 0.0) + value / count
                            (loss / count).backward()
                    finally:
                        values.close()
                torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm,
                                               error_if_nonfinite=True)
                self.optimizer.step()
                if self.plan.stage == "encoder":
                    _clear_caches(self.backend)
        except BaseException:
            self.status = "failed"
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        # Nothing that can fail inside optimization remains before these counts.
        self.completed_microsteps += count
        self.completed_updates += 1
        self.status = "complete" if self.done else "ready"
        report = StepReport(self.completed_microsteps, self.completed_updates,
                            count, total_loss, tuple(losses.items()))
        self.reports.append(report)
        for observer in self._observers:
            observer(self, report)
        return report

    def run(self, *, stop_when: Callable[[StepReport], bool] | None = None) -> list[StepReport]:
        if self.status not in {"ready", "complete"}:
            raise RuntimeError(f"cannot run a {self.status} session")
        produced = []
        while not self.done:
            report = self.step()
            produced.append(report)
            if stop_when is not None and stop_when(report):
                break
        return produced

    def _backend_state(self):
        if self._custom_state:
            return copy.deepcopy(self.backend.training_state_dict())
        return {name: copy.deepcopy(module.state_dict()) for name, module in _modules(self.backend).items()}

    def _load_backend_state(self, state):
        if self._custom_state:
            self.backend.load_training_state_dict(copy.deepcopy(state))
        else:
            for name, module in _modules(self.backend).items():
                module.load_state_dict(state[name], strict=True)
        _clear_caches(self.backend)

    def state_dict(self) -> dict:
        if self.status not in {"ready", "complete"}:
            raise RuntimeError("checkpoints require a valid committed update boundary")
        return {"schema_version": 1, "weight_provenance": "raw-current", "ema_decay": None,
                "plan": self.plan.to_dict(),
                "config": copy.deepcopy(self._configuration),
                "backend_identity": copy.deepcopy(self._backend_identity),
                "prompt_source": list(self._prompts), "backend": self._backend_state(),
                "optimizer_type": f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "progress": {"microsteps": self.completed_microsteps,
                             "optimizer_steps": self.completed_updates},
                "rng": {"python": self.rng.getstate(), "python_global": self._python_rng,
                        "torch": self._torch_rng.clone(),
                        "cuda": {k: v.clone() for k, v in self._cuda_rng.items()}},
                "reports": [asdict(report) for report in self.reports]}

    def _validate_checkpoint(self, state):
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported training checkpoint schema")
        for key, expected in (("weight_provenance", "raw-current"), ("ema_decay", None),
                              ("plan", self.plan.to_dict()), ("config", self._configuration),
                              ("backend_identity", self._backend_identity),
                              ("prompt_source", list(self._prompts)),
                              ("optimizer_type", f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}")):
            if key not in state or state[key] != expected:
                raise ValueError(f"checkpoint {key} does not match this session")
        progress = state["progress"]
        microsteps, updates = progress["microsteps"], progress["optimizer_steps"]
        if (type(microsteps) is not int or type(updates) is not int
                or not 0 <= microsteps <= self.config.iterations
                or not 0 <= updates <= microsteps or (microsteps > 0 and updates == 0)):
            raise ValueError("invalid checkpoint progress")
        reports = [StepReport(**report) for report in state["reports"]]
        if len(reports) != updates or sum(report.group_microsteps for report in reports) != microsteps:
            raise ValueError("checkpoint reports do not match committed progress")
        previous = 0
        for index, report in enumerate(reports, 1):
            if (type(report.group_microsteps) is not int or type(report.microsteps) is not int
                    or type(report.optimizer_steps) is not int
                    or not 1 <= report.group_microsteps <= self.config.accumulation_steps
                    or report.microsteps != previous + report.group_microsteps
                    or report.optimizer_steps != index or not math.isfinite(report.loss)):
                raise ValueError("invalid checkpoint update report")
            previous = report.microsteps
        if not self._custom_state:
            modules = _modules(self.backend)
            if set(state["backend"]) != set(modules):
                raise ValueError("checkpoint module names do not match")
            for name, module in modules.items():
                current = module.state_dict()
                saved = state["backend"][name]
                if set(saved) != set(current):
                    raise ValueError(f"checkpoint keys do not match {name}")
                for key, tensor in current.items():
                    value = saved[key]
                    if not isinstance(value, torch.Tensor) or tensor.shape != value.shape or tensor.dtype != value.dtype:
                        raise ValueError(f"checkpoint tensor does not match {name}.{key}")
        elif callable(getattr(self.backend, "validate_training_state_dict", None)):
            self.backend.validate_training_state_dict(state["backend"])
        groups = state["optimizer"]["param_groups"]
        if (len(groups) != len(self.optimizer.param_groups)
                or any(len(saved["params"]) != len(current["params"])
                       for saved, current in zip(groups, self.optimizer.param_groups))):
            raise ValueError("checkpoint optimizer parameter groups do not match")
        for saved, current in zip(groups, self.optimizer.param_groups):
            if {key: value for key, value in saved.items() if key != "params"} != {
                    key: value for key, value in current.items() if key != "params"}:
                raise ValueError("checkpoint optimizer configuration does not match")
            for param_id, param in zip(saved["params"], current["params"]):
                for value in state["optimizer"]["state"].get(param_id, {}).values():
                    if isinstance(value, torch.Tensor) and value.numel() != 1 and value.shape != param.shape:
                        raise ValueError("checkpoint optimizer tensor shape does not match")
        rng = state["rng"]
        random.Random().setstate(rng["python"])
        random.Random().setstate(rng["python_global"])
        torch.Generator(device="cpu").set_state(rng["torch"])
        if set(rng["cuda"]) != set(self._cuda_devices):
            raise ValueError("checkpoint CUDA random streams do not match")
        for index, value in rng["cuda"].items():
            torch.Generator(device=f"cuda:{index}").set_state(value)
        return reports

    def load_state_dict(self, state: dict) -> None:
        if self.status in {"running", "closed"}:
            raise RuntimeError(f"cannot restore a {self.status} session")
        reports = self._validate_checkpoint(state)
        previous_backend = self._backend_state()
        previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
        try:
            self._load_backend_state(state["backend"])
            self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
            if self._objective is not None:
                with self._isolated_rng(), _preserve_modes(self.backend):
                    self._objective.refresh()
        except BaseException:
            self._load_backend_state(previous_backend)
            self.optimizer.load_state_dict(previous_optimizer)
            raise
        self.rng.setstate(state["rng"]["python"])
        self._python_rng = state["rng"]["python_global"]
        self._torch_rng = state["rng"]["torch"].clone()
        self._cuda_rng = {key: value.clone() for key, value in state["rng"]["cuda"].items()}
        self.completed_microsteps = state["progress"]["microsteps"]
        self.completed_updates = state["progress"]["optimizer_steps"]
        self.reports = reports
        self.optimizer.zero_grad(set_to_none=True)
        self.status = "complete" if self.done else "ready"

    def save_checkpoint(self, path) -> None:
        """Atomically persist a committed snapshot; the caller chooses the path."""
        snapshot = self.state_dict()
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(snapshot, stream)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def close(self) -> None:
        if self.status == "running":
            raise RuntimeError("cannot close during an optimizer update")
        self.optimizer.zero_grad(set_to_none=True)
        self.status = "closed"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
