"""Explicit CPU fixtures for integration/research; never discovered as products."""
from conceptmod.contracts import BackendCapabilities
from conceptmod.providers import BackendProvider, BackendRegistry


def fixture_registry() -> BackendRegistry:
    """PEFT is optional; only constructing the fixture imports it."""
    return BackendRegistry([
        BackendProvider(name, "conceptmod.testing.peft_backend:CpuBackend",
                        "conceptmod[test]", BackendCapabilities())
        for name in ("cpu", "dummy")
    ], discover=False)
