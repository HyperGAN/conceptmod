"""Curated distribution boundary, following particle-sliders-core."""
from pathlib import Path

from setuptools.command.build_py import build_py


CORE_MODULES = {
    "conceptmod": frozenset({
        "__init__", "artifacts", "cli", "contracts", "dsl", "encoder_train",
        "geometry", "metrics", "model_train", "observers", "ops", "particle_api",
        "particle_lora", "plan", "providers", "training", "verify",
    }),
    "conceptmod.backends": frozenset({"__init__", "base"}),
}


class CoreBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        allowed = CORE_MODULES[package]
        modules = super().find_package_modules(package, package_dir)
        selected = [entry for entry in modules if entry[1] in allowed]
        missing = allowed - {entry[1] for entry in selected}
        if missing:
            raise ValueError(f"Missing reviewed core modules in {package}: {sorted(missing)}")
        return selected

    def run(self):
        super().run()
        outputs = {Path(name).absolute() for name in self.get_outputs(include_bytecode=False)}
        root = Path(self.build_lib)
        if root.exists():
            for path in root.rglob("*"):
                if (path.is_file() or path.is_symlink()) and path.absolute() not in outputs:
                    path.unlink()
