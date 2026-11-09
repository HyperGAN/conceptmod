"""Convenience execution of one explicit model TrainingSession."""
from __future__ import annotations

from conceptmod import ops
from conceptmod.plan import EditPlan, compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession


def train_model(backend, phrase: str | EditPlan, iterations: int = 500,
                lr: float = 1e-5, train_method: str = "xattn",
                accumulation_steps: int = 1, seed: int = 0,
                op_defaults: ops.OpDefaults | None = None, *,
                prompt_source=None, observers=(), stop_when=None):
    """Run a model plan and return one mean loss per committed update.

    Random-prompt plans require an injected prompt_source. Samples, diagnostic
    files and stopping metrics moved to explicit observers and stop_when policy;
    Use TrainingSession directly for progress counters and resumable checkpoints.
    """
    plan = phrase if isinstance(phrase, EditPlan) else compile_phrase(
        phrase, stage="model", capabilities=getattr(backend, "capabilities", None))
    if plan.stage != "model":
        raise ValueError("train_model requires a model EditPlan")
    config = TrainingConfig(iterations=iterations, lr=lr, train_method=train_method,
                            accumulation_steps=accumulation_steps, seed=seed,
                            op_defaults=op_defaults or ops.OpDefaults())
    with TrainingSession(backend, plan, config, prompt_source=prompt_source,
                         observers=observers) as session:
        session.run(stop_when=stop_when)
        return [report.loss for report in session.reports]
