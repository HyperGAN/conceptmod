"""conceptmod owns the formulation toys. particlegan stays primitives.

The published package may supply RpGAN and GradRegularizer. Toy modules that
were opened on ParticleGAN (orbit hold, leaderboard honesty, shared
trajectory, and the rest) must not be imported back from it.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOY_DIR = ROOT / "conceptmod" / "toys"
TEST_DIR = ROOT / "tests"

# Submodules and names that are core primitives, not gates.
ALLOWED_MODULES = frozenset({
    "particlegan",
    "particlegan.gan_loss",
    "particlegan.grad_regularizers",
    "particlegan.particle_prior",
    "particlegan.recipes",
    "particlegan.vicreg_loss",
    "particlegan.autoencoder",
})
ALLOWED_NAMES = frozenset({
    "GANLoss",
    "GradientPenalty",
    "GradRegularizer",
    "ParticlePrior",
    "ParticleRegularizer",
    "get_recipe",
    "Recipe",
})


def _python_files() -> list[Path]:
    toys = sorted(TOY_DIR.glob("*.py"))
    tests = sorted(TEST_DIR.glob("test_toy_*.py"))
    assert toys and tests
    return toys + tests


def _particlegan_imports(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "particlegan" or alias.name.startswith("particlegan."):
                    found.append((alias.name, ()))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "particlegan" or node.module.startswith("particlegan."):
                names = tuple(alias.name for alias in node.names)
                found.append((node.module, names))
    return found


def test_toys_import_only_published_primitives():
    seen = []
    for path in _python_files():
        for module, names in _particlegan_imports(path):
            seen.append((path.name, module, names))
            assert module in ALLOWED_MODULES, f"{path.name} imports toy module {module}"
            if names:
                bad = [name for name in names if name not in ALLOWED_NAMES]
                assert not bad, f"{path.name} imports {bad} from {module}"
    assert seen, "expected the toys to depend on particlegan primitives"


def test_docs_do_not_tell_you_to_run_toys_from_particlegan():
    blobs = [path.read_text() for path in _python_files() if path.name != "test_toy_home.py"]
    blobs.append((ROOT / "docs" / "formulation-toys.md").read_text())
    blobs.append((ROOT / "pyproject.toml").read_text())
    text = "\n".join(blobs)
    assert "python -m particlegan" not in text
    assert "particlegan @ git+" not in text
    pyproject = (ROOT / "pyproject.toml").read_text()
    requirements = (ROOT / "requirements.txt").read_text()
    assert "particlegan>=" in pyproject
    dep = next(line for line in requirements.splitlines() if line.startswith("particlegan"))
    assert dep.startswith("particlegan>=")
    assert "git+" not in dep
    for path in (ROOT / "conceptmod").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in {
                "GradRegularizer", "GANLoss", "GradientPenalty",
            }:
                raise AssertionError(f"{path} copies {node.name}; use the particlegan package")
