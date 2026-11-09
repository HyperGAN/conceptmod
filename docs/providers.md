# Model products and providers

Conceptmod core contains no pretrained architectures, model identifiers or model
loaders. Each product is an independent package/repository. Install the
[versioned core package](https://github.com/HyperGAN/conceptmod/releases/tag/v3.0.0)
first, then install the product you want. A pinned source install from
`git+https://github.com/HyperGAN/conceptmod.git@v3.0.0` is also supported.

| Provider ID | Product repository / distribution |
| --- | --- |
| `sana` | [sana-conceptmod](https://github.com/HyperGAN/sana-conceptmod) |
| `zimage` | [zimage-conceptmod](https://github.com/HyperGAN/zimage-conceptmod) |
| `anima` | [anima-conceptmod](https://github.com/HyperGAN/anima-conceptmod) |
| `krea` | [krea2-conceptmod](https://github.com/HyperGAN/krea2-conceptmod) |
| `qwen` | [qwen-image-conceptmod](https://github.com/HyperGAN/qwen-image-conceptmod) |
| `klein` | [flux2-klein-conceptmod](https://github.com/HyperGAN/flux2-klein-conceptmod) |
| `supra` | [supra-conceptmod](https://github.com/HyperGAN/supra-conceptmod) |

Model-format conversion is independently available in
[conceptmod-converters](https://github.com/HyperGAN/conceptmod-converters).
It owns ComfyUI formats and Music/YuE mappings; core does not import it.

## Provider contract

A product's `__init__.py` and `provider.py` are lightweight metadata. They must
not import model runtimes, construct modules, download files, touch CUDA or
change numerical settings. Declare the descriptor as an entry point:

```toml
[project.entry-points."conceptmod.backends"]
example = "example_conceptmod.provider:provider"
```

```python
# example_conceptmod/provider.py (no backend import)
from conceptmod import BackendCapabilities, BackendProvider

provider = BackendProvider(
    name="example",
    factory="example_conceptmod.backend:ExampleBackend",
    distribution="example-conceptmod",
    capabilities=BackendCapabilities(),
)
```

`BackendRegistry` discovers descriptors. `resolve(name)` returns metadata;
`load(name, **options)` imports the factory and constructs the model. Duplicate
provider IDs fail rather than choosing an arbitrary installed package. Embedders
can pass `BackendRegistry([provider], discover=False)` for deterministic local
registration without installing an entry point.

Products implement the narrow protocols in `conceptmod.contracts` and can inherit
`conceptmod.backends.base.Backend` for generic artifacts, model views, cache
invalidation and declared training modules. `training_modules()` must include
all current and separate original modules needed for exact continuation. A
product's `original_view` must use both original encoder and original velocity
model, consistently for generation, rendering and direct predictions.

Products own native tensor layouts, scheduler conventions, adapter attachment,
device residency and safe base-weight loading. Core never discovers models by
walking source directories or imports a product to find an observer. Runtime
construction has an explicit owner; independent sessions need independent model
instances.

The source-only PEFT fixture is under `conceptmod.testing.peft_backend` and uses
the checkout's `.[test]` dependencies. Its explicit `fixture_registry()` is for
repository tests; neither the fixture nor research is installed with core.
