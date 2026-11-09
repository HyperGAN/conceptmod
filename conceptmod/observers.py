"""Explicit post-commit observers, separate from optimization and selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from conceptmod.training import StepReport


@dataclass
class HistoryObserver:
    reports: list[StepReport] = field(default_factory=list)

    def __call__(self, session, report: StepReport):
        self.reports.append(report)


@dataclass
class LossLogger:
    """Send a report to a caller-selected sink at committed-update intervals."""
    write: Callable[[str], None]
    every: int = 1

    def __post_init__(self):
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError("every must be a positive integer")

    def __call__(self, session, report: StepReport):
        if report.optimizer_steps % self.every == 0:
            self.write(f"update {report.optimizer_steps} microstep {report.microsteps} loss {report.loss:.6g}")
