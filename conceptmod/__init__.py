"""Model-independent concept editing: compile first, load products explicitly."""
from conceptmod.contracts import BackendCapabilities, ModelView, model_views
from conceptmod.plan import EditPlan, compile_phrase
from conceptmod.providers import BackendProvider, BackendRegistry

__version__ = "3.0.0"
__all__ = ["BackendCapabilities", "ModelView", "model_views", "EditPlan",
           "compile_phrase", "BackendProvider", "BackendRegistry"]
