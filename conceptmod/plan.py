"""Pure phrase compilation. No model construction, downloads or device access."""
from __future__ import annotations

from dataclasses import dataclass
import math

from conceptmod import dsl
from conceptmod.contracts import BackendCapabilities, VELOCITY_OPERATORS


_OPTIONS = {
    dsl.WRITE: frozenset({"guidance"}),
    dsl.ERASE: frozenset({"guidance"}),
    dsl.EXAGGERATE: frozenset({"guidance"}),
    dsl.FREEZE: frozenset(), dsl.ORTHOGONAL: frozenset(), dsl.PIXEL: frozenset(),
}


@dataclass(frozen=True)
class PlannedRule:
    op: str
    a: str
    b: str
    alpha: float
    options: tuple[tuple[str, float], ...]
    raw: str

    def as_rule(self) -> dsl.Rule:
        return dsl.Rule(self.op, self.a, self.b, self.alpha, dict(self.options), self.raw)

    @property
    def needs_random_prompt(self) -> bool:
        return dsl.RANDOM_PROMPT in self.a or dsl.RANDOM_PROMPT in self.b


@dataclass(frozen=True)
class EditPlan:
    phrase: str
    stage: str
    rules: tuple[PlannedRule, ...]
    schema_version: int = 1

    @property
    def needs_prompt_source(self) -> bool:
        return any(r.needs_random_prompt or r.op == dsl.EXAGGERATE for r in self.rules)

    def materialize(self, random_prompt: str | None = None) -> list[dsl.Rule]:
        return dsl.materialize([r.as_rule() for r in self.rules], random_prompt)

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "phrase": self.phrase,
                "stage": self.stage, "rules": [
                    {"op": r.op, "a": r.a, "b": r.b, "alpha": r.alpha,
                     "options": dict(r.options), "raw": r.raw} for r in self.rules]}

    @classmethod
    def from_dict(cls, value: dict, *, capabilities: BackendCapabilities | None = None):
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise dsl.DSLError("unsupported edit-plan schema")
        plan = compile_phrase(value["phrase"], stage=value["stage"], capabilities=capabilities)
        if plan.to_dict() != value:
            raise dsl.DSLError("edit-plan rules do not match its phrase")
        return plan


def compile_phrase(phrase: str, *, stage: str = "model",
                   capabilities: BackendCapabilities | None = None) -> EditPlan:
    """Validate the complete requested edit before loading a backend.

    Encoder and model objectives differ. An explicit ``both`` request accepts
    only rules supported in both stages; use separate plans for unequal jobs.
    Legacy parsing remains available for examining archived unsupported phrases.
    """
    if stage not in {"model", "encoder", "both"}:
        raise dsl.DSLError(f"unknown training stage {stage!r}")
    caps = capabilities or BackendCapabilities()
    stages = {"model", "encoder"} if stage == "both" else {stage}
    if not stages <= caps.stages:
        raise dsl.DSLError(f"backend does not support requested stages {sorted(stages)}")
    planned = []
    try:
        rules = dsl.parse_phrase(phrase)
    except (ValueError, OverflowError) as error:
        raise dsl.DSLError(str(error)) from error
    for rule in rules:
        if rule.op not in _OPTIONS or rule.op not in caps.operators:
            raise dsl.DSLError(f"operator {rule.op!r} is not supported by this backend")
        unknown = rule.options.keys() - _OPTIONS[rule.op]
        if unknown:
            raise dsl.DSLError(f"unknown options {sorted(unknown)} in {rule.raw!r}")
        if not all(math.isfinite(v) for v in [rule.alpha, *rule.options.values()]):
            raise dsl.DSLError(f"non-finite value in {rule.raw!r}")
        if rule.op == dsl.PIXEL and not caps.differentiable_render:
            raise dsl.DSLError("pixel operator requires differentiable rendering")
        if "encoder" in stages:
            if rule.op not in VELOCITY_OPERATORS or rule.needs_random_prompt:
                raise dsl.DSLError(f"rule {rule.raw!r} is not supported by the encoder stage")
            if rule.options:
                raise dsl.DSLError(f"encoder stage does not implement options in {rule.raw!r}")
        planned.append(PlannedRule(rule.op, rule.a, rule.b, rule.alpha,
                                   tuple(sorted(rule.options.items())), rule.raw))
    return EditPlan(phrase, stage, tuple(planned))
