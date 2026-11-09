"""Core geometry does not depend on executing the research suite."""
import subprocess
import sys


def test_runtime_erase_import_does_not_import_research_modules():
    subprocess.run([sys.executable, "-c", """
import sys
from conceptmod import ops_erase
from conceptmod.geometry import hold_dir
assert ops_erase.hold_dir is hold_dir
assert not any(name == 'conceptmod.toys' or name.startswith('conceptmod.toys.')
               for name in sys.modules)
"""], check=True)


def test_geometry_toy_aliases_are_lazy_and_identical():
    subprocess.run([sys.executable, "-c", """
import sys
import conceptmod.toys as toys
import conceptmod.geometry as geometry
assert not any(name.startswith('conceptmod.toys.') for name in sys.modules)
assert toys.hold_dir is geometry.hold_dir
assert toys.LeftoverField is geometry.LeftoverField
assert not any(name.startswith('conceptmod.toys.') for name in sys.modules)
from conceptmod.toys.cover_leftover import hold_dir, faithful_guard_e, LeftoverField
assert hold_dir is geometry.hold_dir
assert faithful_guard_e is geometry.faithful_guard_e
assert LeftoverField is geometry.LeftoverField
assert 'conceptmod.toys.mode_hold' not in sys.modules
"""], check=True)


def test_all_documented_research_exports_remain_available():
    import conceptmod.toys as toys

    for name in toys.__all__:
        assert getattr(toys, name) is not None, name
