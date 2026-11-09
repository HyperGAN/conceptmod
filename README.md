# conceptmod

Compile concept-editing phrases into explicit training plans. Core owns the
language, loss composition, training lifecycle, checkpoints and complete edit
bundles. Pretrained models live in [separate product repositories](docs/providers.md).

```python
from conceptmod import compile_phrase

plan = compile_phrase("#:0.4|human=robot:0.8|robot%human:-0.1")
print(plan.to_dict())
```

No model is loaded to compile a phrase. Unknown options, unsupported operators
and non-finite values fail before training. Model capabilities determine which
stages and rendering operations are available.

## Install and choose a product

```bash
pip install https://github.com/HyperGAN/conceptmod/releases/download/v3.0.0/conceptmod-3.0.0-py3-none-any.whl
pip install 'git+https://github.com/HyperGAN/anima-conceptmod.git'
conceptmod backends
conceptmod validate --backend anima --phrase 'human=robot'
```

[Available products](docs/providers.md): SANA, Z-Image, Anima, Krea 2,
Qwen-Image, FLUX.2 Klein and Supra. Core requires Torch and ParticleGAN; model
libraries and model weights belong to the products. Core never downloads a
prompt dataset, picks a GPU or changes process-wide numerical settings.

## Execute a plan

```python
from conceptmod import BackendRegistry, compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession

provider = BackendRegistry().resolve("anima")
plan = compile_phrase("human=robot", capabilities=provider.capabilities)
backend = provider.create(device="cuda:0", lora_rank=16)
config = TrainingConfig(iterations=500, lr=1e-5)
with TrainingSession(backend, plan, config) as session:
    session.step()                       # one committed optimizer update
    session.save_checkpoint("step1.pt")
    session.run()
    backend.save_trained("my-edit")      # transformer + edited encoder
```

A session executes exactly one model or encoder stage. Compose separate sessions
when both are needed. Supply `prompt_source=[...]` for model-stage exaggeration
or random-prompt rules. Use explicit observer callbacks and caller stop policies;
training does not generate presentation files or select a scientific objective.
[API details](docs/api.md), [complete edit bundles](docs/edit-bundles.md).

The CLI accepts product options as data, with no model-specific flags:

```json
{"backend": {"device": "cuda:0", "lora_rank": 16},
 "training": {"iterations": 500, "lr": 0.00001}}
```

```bash
conceptmod train --backend anima --phrase 'human=robot' --config run.json --out my-run
conceptmod inspect my-run/edit
```

Each committed update atomically saves `my-run/checkpoint.pt`. Resume with the
same plan and configuration using `--resume my-run/checkpoint.pt` and a new
output directory. Training checkpoints include optimizer, random streams,
progress, current and original model state. Inference bundles do not simulate
exact continuation. Weights are raw current weights; weight EMA is disabled.

## Research and tools

- [DSL catalogue](docs/dsl/README.md) and [2-D analysis](docs/2d-analysis.md).
- [Formulation toys](docs/formulation-toys.md) under `conceptmod.research.toys`,
  available in the source checkout and excluded from the installed core.
- [Particle LoRA tensor integration](docs/particle-lora.md), using the official
  ParticleGAN trainer and optimizer formulas.
- [Model-format converters](https://github.com/HyperGAN/conceptmod-converters).
- [Historical 2.x model proofs](https://github.com/HyperGAN/conceptmod/tree/70f0cb2632190388eb4028ef80066558259494fc/outputs),
  retained as historical evidence rather than new-model qualification.
- [Implementation plan and verification record](docs/plan.md).
- [Distribution boundary and particle-sliders comparison](docs/distribution.md).

For intended-effect comparisons, rank the equal-case mean of matched finished
output intensity ON minus OFF. Surrogate velocity error, preservation and toy
formulation verdicts remain diagnostics.

## Development

```bash
pip install -e '.[test]'
python -m pytest -q
```

The command above runs the full source suite, including research. Tests use CPU
fixtures, not pretrained models or model downloads. Product tests live with
their implementations. Research and `conceptmod.testing` are source-only and
are excluded from both wheels and strict editable installs. See
[distribution checks](docs/distribution.md) for installed-package validation.
