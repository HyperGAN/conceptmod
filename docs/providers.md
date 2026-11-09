# Model products and providers

Conceptmod core contains no pretrained architectures, model identifiers or model
loaders. Each product is an independent package/repository. Install core from
this repository and then install the product you want. Until the 3.x package is
published, install the core Git source first, rather than resolving a 2.x PyPI
release for a 3.x product.

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

The PEFT CPU fixture is under `conceptmod.testing.peft_backend` and requires
`conceptmod[peft-test]`. It is available only through explicit
`conceptmod.testing.fixture_registry()` and is not installed as a model product.
