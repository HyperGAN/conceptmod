"""Explicit shared-game CPU cohort; historical DSL trainers keep their objectives."""
from .session import open_game, load_native, COHORT, BACKEND_STATES

__all__ = ["open_game", "load_native", "COHORT", "BACKEND_STATES"]
