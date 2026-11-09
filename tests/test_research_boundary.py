"""Production imports and the explicit research namespace stay independent."""
import subprocess
import sys


def test_core_runtime_imports_do_not_load_research():
    subprocess.run([sys.executable, "-c", """
import sys
import conceptmod
from conceptmod import artifacts, dsl, geometry, metrics, ops, plan, providers
from conceptmod import encoder_train, model_train, observers, particle_api, particle_lora, training
assert not any(name == 'conceptmod.research' or name.startswith('conceptmod.research.')
               for name in sys.modules)
"""], check=True)


def test_research_namespace_is_lightweight_and_old_paths_are_absent():
    subprocess.run([sys.executable, "-c", """
import importlib.util
import sys
import conceptmod.research
assert not any(name.startswith('conceptmod.research.') for name in sys.modules)
for name in ('analysis_2d', 'analysis_dsl', 'game', 'ops_erase', 'toys'):
    assert importlib.util.find_spec('conceptmod.' + name) is None
"""], check=True)


def test_research_erase_uses_core_geometry_without_loading_toys():
    subprocess.run([sys.executable, "-c", """
import sys
from conceptmod.research import ops_erase
from conceptmod.geometry import hold_dir
assert ops_erase.hold_dir is hold_dir
assert not any(name == 'conceptmod.research.toys' or name.startswith('conceptmod.research.toys.')
               for name in sys.modules)
"""], check=True)


def test_geometry_toy_exports_are_lazy_and_identical():
    subprocess.run([sys.executable, "-c", """
import sys
import conceptmod.research.toys as toys
import conceptmod.geometry as geometry
assert not any(name.startswith('conceptmod.research.toys.') for name in sys.modules)
assert toys.hold_dir is geometry.hold_dir
assert toys.LeftoverField is geometry.LeftoverField
assert not any(name.startswith('conceptmod.research.toys.') for name in sys.modules)
from conceptmod.research.toys.cover_leftover import hold_dir, faithful_guard_e, LeftoverField
assert hold_dir is geometry.hold_dir
assert faithful_guard_e is geometry.faithful_guard_e
assert LeftoverField is geometry.LeftoverField
assert 'conceptmod.research.toys.mode_hold' not in sys.modules
"""], check=True)


def test_all_documented_research_exports_remain_available():
    import conceptmod.research.toys as toys

    for name in toys.__all__:
        assert getattr(toys, name) is not None, name
