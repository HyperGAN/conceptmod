"""Lazy installed model providers. Core contains no pretrained model catalogue."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
import math
import re

from conceptmod.contracts import BackendCapabilities


ENTRY_POINT_GROUP = "conceptmod.backends"


@dataclass(frozen=True)
class BackendProvider:
    name: str
    factory: str
    distribution: str
    capabilities: BackendCapabilities = BackendCapabilities()
    training_defaults: tuple[tuple[str, float | int], ...] = ()

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise ValueError("provider name must be a lowercase identifier")
        if ":" not in self.factory or not self.distribution:
            raise ValueError("provider needs module:factory and distribution identity")
        if not isinstance(self.capabilities, BackendCapabilities):
            raise TypeError("provider capabilities must be BackendCapabilities")
        defaults = tuple(tuple(item) for item in self.training_defaults)
        if (any(len(item) != 2 or not isinstance(item[0], str)
                or not isinstance(item[1], (int, float)) or not math.isfinite(item[1])
                for item in defaults)
                or len(dict(defaults)) != len(defaults)):
            raise ValueError("provider defaults must be unique finite scalar overrides")
        object.__setattr__(self, "training_defaults", defaults)

    def create(self, **options):
        module, name = self.factory.split(":", 1)
        factory = getattr(import_module(module), name)
        backend = factory(**options)
        advertised = getattr(backend, "capabilities", self.capabilities)
        if advertised != self.capabilities:
            raise ValueError(f"backend {self.name!r} runtime capabilities differ from its descriptor")
        try:
            backend.capabilities = self.capabilities
        except (AttributeError, TypeError) as error:
            raise TypeError("provider runtimes must expose their declared capabilities") from error
        return backend


class BackendRegistry:
    """Explicit registry of descriptors; instances and model weights stay local."""
    def __init__(self, providers=(), *, discover: bool = True):
        self._providers = {}
        for provider in providers:
            self.register(provider)
        if discover:
            for point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda p: p.name):
                descriptor = point.load()
                if not isinstance(descriptor, BackendProvider) or descriptor.name != point.name:
                    raise ValueError(f"{point.value} must export BackendProvider(name={point.name!r})")
                self.register(descriptor)

    def register(self, provider: BackendProvider):
        if not isinstance(provider, BackendProvider):
            raise TypeError("register a BackendProvider descriptor")
        if provider.name in self._providers:
            raise ValueError(f"duplicate backend provider {provider.name!r}")
        self._providers[provider.name] = provider

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def resolve(self, name: str) -> BackendProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise LookupError(
                f"backend {name!r} is not installed; install its conceptmod product "
                "package or pass an explicit BackendRegistry. Use `conceptmod backends` "
                "to list installed providers.") from None

    def load(self, name: str, **options):
        return self.resolve(name).create(**options)
