"""Model-independent backend interface and installed product discovery."""
from conceptmod.backends.base import Backend, TextEmbeds
from conceptmod.providers import BackendProvider, BackendRegistry


def installed_backends(*, registry: BackendRegistry | None = None) -> tuple[str, ...]:
    return (registry if registry is not None else BackendRegistry()).names()


def load_backend(name: str, device: str | None = None, *,
                 registry: BackendRegistry | None = None, **options) -> Backend:
    """Load a separately installed product after explicit caller preflight.

    No default model, built-in catalogue or fallback loader is provided.
    Tests may supply conceptmod.testing.fixture_registry() explicitly.
    """
    if device is not None:
        options["device"] = device
    registry = registry if registry is not None else BackendRegistry()
    return registry.load(name, **{k: v for k, v in options.items() if v is not None})


__all__ = ["Backend", "TextEmbeds", "BackendProvider", "BackendRegistry",
           "installed_backends", "load_backend"]
