"""Check a wheel and the installed core outside the source checkout."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


INSTALLED_CHECK = r'''
import importlib.util
import json
from pathlib import Path
import sys
import conceptmod
from conceptmod import compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession

assert conceptmod.__version__ == '3.0.0'
assert compile_phrase('red=blue').rules[0].op == 'write'
for module in ('research', 'testing', 'toys', 'game', 'convert',
               'backends.sana', 'backends.anima', 'backends.supra'):
    assert importlib.util.find_spec('conceptmod.' + module) is None, module
assert not any(name in sys.modules for name in ('diffusers', 'transformers', 'peft'))
print(json.dumps({'version': conceptmod.__version__, 'origin': str(Path(conceptmod.__file__).resolve())}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if not args.wheel and not args.installed:
        parser.error("select --wheel and/or --installed")
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as archive:
            modules = {name for name in archive.namelist() if name.startswith("conceptmod/")}
            assert len(modules) == 19, sorted(modules)
            assert all(name.endswith(".py") for name in modules), sorted(modules)
            assert {name for name in modules if name.startswith("conceptmod/backends/")} == {
                "conceptmod/backends/__init__.py", "conceptmod/backends/base.py",
            }
            assert not any(name.startswith(("conceptmod/research/", "conceptmod/testing/",
                                            "conceptmod/convert")) for name in modules)
        print(f"{args.wheel.name}: 19 reviewed core modules; no products, research or fixtures")
    if args.installed:
        with tempfile.TemporaryDirectory(prefix="conceptmod-installed-check-") as directory:
            subprocess.run([sys.executable, "-I", "-c", INSTALLED_CHECK], cwd=directory, check=True)
            subprocess.run([sys.executable, "-I", "-m", "conceptmod.cli", "validate",
                            "--phrase", "red=blue"], cwd=directory, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
