# Core package and particle-sliders template

Conceptmod follows the model-independent distribution boundary in
[particle-sliders-core 0.14.1](https://github.com/HyperGAN/particle-sliders/blob/ec5566323d46a04145e9ecfa02bf9ad55843be5e/packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md).

| Boundary | particle-sliders template | Conceptmod |
| --- | --- | --- |
| Core ownership | Shared lifecycle, state, adapters and observation | Editing language, model views, sessions, state, adapters and observation |
| Model ownership | Separate products inject native factories | Seven separate products register lightweight providers |
| Distribution | Reviewed module list; no native implementations or research runners | 19 reviewed modules; no native implementations, research or PEFT test fixture |
| Editable installs | Strict PEP 660, same boundary as wheels | Same strict installation policy |
| Checkpoints and weights | Explicit exact continuation; raw current weights | Same ownership guarantees; complete inference bundles are separate |
| Numerical implementation | Conditional ParticleGAN response lifecycle, PG 0.9.5 | Concept-editing DSL/session API, tested PG 0.9.4 |

These are corresponding ownership boundaries, not interchangeable numerical
APIs. Packaging does not upgrade an optimizer, change a formulation or qualify
pretrained media quality. Model weights use no EMA. Conceptmod's intended-effect
selection rule remains as documented in [plan.md](plan.md).

The package is built directly from this core repository; the template keeps its
package under `packages/particle-sliders-core` in a larger repository. That
directory difference does not expose additional installed modules.

## Install a versioned package

```sh
python -m pip install https://github.com/HyperGAN/conceptmod/releases/download/v3.0.0/conceptmod-3.0.0-py3-none-any.whl
conceptmod validate --phrase 'red=blue'
```

The release includes a wheel, source distribution and SHA-256 checksums. A pinned
VCS install is also supported:

```sh
python -m pip install 'git+https://github.com/HyperGAN/conceptmod.git@v3.0.0'
```

Install a [model product](providers.md) separately when execution needs one.

## Build and check

```sh
python -m pip install -e '.[test]' build twine
python scripts/check_distribution.py --installed
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/check_distribution.py --wheel dist/conceptmod-3.0.0-py3-none-any.whl
python -m pip install --force-reinstall --no-deps dist/conceptmod-3.0.0-py3-none-any.whl
python scripts/check_distribution.py --installed
```

`_core_build.py` lists every public runtime module. New source files do not ship
automatically, and stale files from older builds are removed from build outputs.
The packaging regression injects an unreviewed source file and a stale research
output into a temporary project, then checks an actual wheel excludes both.

`_build_backend.py` defaults editable installation to strict mode and rejects
loose editable modes that would expose the entire source tree. The installed
check runs in isolated Python outside the checkout, verifies that excluded
namespaces cannot import, and exercises the plan and CLI APIs.

Research, source-only fixtures, their tests and historical evidence remain in Git
and the source archive. Run the full source suite from the repository root.
Installed-core tests run separately outside the checkout, and product suites
verify their own native dependencies and complete model/encoder continuation.
