"""Build real artifacts to prevent source additions and stale outputs leaking."""
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_wheel_excludes_unreviewed_source_and_stale_build_outputs(tmp_path):
    pytest.importorskip("build")
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "setup.py", "_build_backend.py", "_core_build.py",
                 "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(source / name, project / name)
    shutil.copytree(source / "conceptmod", project / "conceptmod",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (project / "conceptmod" / "unreviewed_product.py").write_text("raise RuntimeError('not core')\n")
    stale = project / "build" / "lib" / "conceptmod" / "research"
    stale.mkdir(parents=True)
    (stale / "leaked.py").write_text("raise RuntimeError('stale build')\n")
    result = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation"],
                            cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    with zipfile.ZipFile(next((project / "dist").glob("*.whl"))) as wheel:
        modules = {name for name in wheel.namelist() if name.startswith("conceptmod/")}
        assert len(modules) == 19, sorted(modules)
        assert "conceptmod/training.py" in modules
        assert not any(name.startswith(("conceptmod/research/", "conceptmod/testing/")) for name in modules)
        assert "conceptmod/unreviewed_product.py" not in modules
