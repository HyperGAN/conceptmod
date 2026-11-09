"""The editing runtime does not import models, converters or research suites."""
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def test_core_imports_with_model_libraries_blocked():
    code = '''
import importlib.abc, sys
blocked = {'diffusers', 'transformers', 'peft', 'datasets', 'torchvision',
           'sentencepiece', 'accelerate', 'matplotlib', 'PIL'}
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('core imported optional model/presentation library: ' + fullname)
sys.meta_path.insert(0, NoModels())
import conceptmod
import conceptmod.model_train, conceptmod.training, conceptmod.geometry, conceptmod.providers
assert conceptmod.compile_phrase('red=blue').rules[0].op == 'write'
assert not any(name.startswith('conceptmod.toys') for name in sys.modules)
assert not any(name.startswith('conceptmod_converters') for name in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_core_distribution_does_not_depend_on_models():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["version"].startswith("3.")
    assert {dep.split('>=')[0].split('==')[0] for dep in project["dependencies"]} == {"torch", "particlegan"}
    assert {path.name for path in (root / "conceptmod" / "backends").glob("*.py")} == {"__init__.py", "base.py"}
