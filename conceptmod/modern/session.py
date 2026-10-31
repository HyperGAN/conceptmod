"""Source-bound native CPU response training through the shared PG lifecycle.

This is a new frozen-positive-velocity GAN cohort, not a DSL rule_loss rewrite.
Only the exact CpuBackend class path has a qualified binding. No model loader
for another backend is called by this module.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from particle_sliders.fibo_hndl import verified_builder, HNDLFieldCritic
from particle_sliders.pg094 import ConditionalPG094, validate_pg09_domain
from particle_sliders.pg09 import _validate_module
from particle_sliders.pg09_backends import FrozenResponseAdapter, ResponseRow
from particle_sliders.lifecycle import (
    Backend, Binding, Capabilities, SessionConfig, CheckpointError, TRAINING,
    native_cleanup, read_packet, _check_packet, _restore_all,
)
from particle_sliders.inference import (
    InferenceBackend, InferenceBinding, native_identity, open_inference,
    read_inference, _validate_packet,
)

ROOT = Path(__file__).parent
NAME = 'conceptmod-dsl-cpu-hg-v1'
COHORT = 'conceptmod-cpu-native-velocity-pg094-hg-hndl010-v1'
PROTOCOL = 'conceptmod-cpu-peft-native-velocity-hg-v1'
BACKEND_STATES = {
    'cpu': 'CPU software fixture qualified; native 4x8x8 velocity, ordinary rank4 LoRA',
    'sana': 'gated: native field, timestep_scale and learned-variance split need own conformance',
    'zimage': 'gated: list inputs, reversed model time and negated velocity need own conformance',
    'anima': 'gated: preserve native singleton-time 5D field; critic squeeze parity required',
    'krea': 'gated: native packed token field; no admitted packed-token critic/domain binding',
    'qwen': 'gated: native packed token field and img_shapes; edit inputs separately unqualified',
    'klein': 'gated: native packed token field, position IDs and distilled/base identity required',
    'supra': 'gated: repository-specific transformer/PEFT targets and increasing flow time need conformance',
}
CAPABILITIES = Capabilities(
    generator_arm='adapter_gan', render=False, base_precisions=('float32',),
    render_outputs=('latent',), operations=('train', 'resume', 'evaluate', 'export', 'response'),
    qualification='Tiny CpuBackend software fixture only; no pretrained model/GPU qualification',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    """Reject an unreviewed installed core/native implementation before binding."""
    import particle_sliders
    from importlib.metadata import version
    lock = json.loads((ROOT / 'sources.json').read_text())
    if version('particle-sliders-core') != '0.12.0':
        raise ValueError('modern cohort requires the locked core0.12 source, not a version alias')
    root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(root)): _sha(p) for p in root.rglob('*')
              if p.is_file() and p.suffix in ('.py', '.json', '.hndl')}
    if actual != lock['core_sources']:
        raise ValueError('shared core source closure differs from the modern cohort')
    for name, digest in lock['native_sources'].items():
        if _sha(ROOT.parent / name) != digest:
            raise ValueError('ConceptMod native source differs: ' + name)
    verified_builder()
    validate_pg09_domain((1, 4, 8, 8), response_domain='continuous_field')
    return lock


class HNDLLoRA(nn.Module):
    """Exact CPU PEFT linear2 A/B computation; all trainables execute in HNDL."""
    def __init__(self, leaf):
        super().__init__()
        builder, identity = verified_builder()
        if leaf.active_adapters != ['default'] or leaf.merged or leaf.disable_adapters:
            raise ValueError('CPU binding requires one active unmerged default PEFT adapter')
        dropout = leaf.lora_dropout['default']
        if (leaf.use_dora.get('default', False)
                or not (isinstance(dropout, nn.Identity) or isinstance(dropout, nn.Dropout) and dropout.p == 0)):
            raise ValueError('DoRA/dropout do not belong to this CPU cohort')
        a, b = leaf.lora_A['default'], leaf.lora_B['default']
        if a.bias is not None or b.bias is not None:
            raise ValueError('CPU binding requires bias-free A/B matrices')
        source = (ROOT / 'lora.hndl').read_text()
        self.graph = builder(source,
            input_shape={'value': ['B', leaf.in_features]}, output_shape=['B', leaf.out_features],
            parameters={'rank': a.out_features, 'output_width': leaf.out_features})
        with torch.no_grad():
            self.graph['down'].weight.copy_(a.weight)
            self.graph['up'].weight.copy_(b.weight)
        self.factor = float(leaf.scaling['default'])
        self.names = ('transformer.linear2.lora_A.default.weight', 'transformer.linear2.lora_B.default.weight')
        self.strength = 1.
        self.identity = dict(identity, product_source_sha256=_sha(ROOT / 'lora.hndl'),
            input_shape=['B', leaf.in_features], output_shape=['B', leaf.out_features],
            rank=a.out_features, factor=self.factor,
            graph_plan=str(self.graph))
        object.__setattr__(self, '_base', leaf.base_layer)

    def extra_repr(self):
        return json.dumps(self.identity, sort_keys=True)

    @contextmanager
    def scaled(self, value):
        value = float(value)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError('finite strength required')
        previous, self.strength = self.strength, value
        try:
            yield self
        finally:
            self.strength = previous

    def forward(self, value):
        base = self._base(value)
        if self.strength == 0:
            return base
        return base + self.graph(value=value) * (self.factor * self.strength)


def _base_identity(runtime):
    values = {}
    for prefix, model in [('transformer', runtime.transformer), ('text_encoder', runtime.text_encoder)]:
        for name, value in model.state_dict().items():
            if '.lora_' not in name:
                tensor = value.detach().cpu().contiguous()
                values[prefix + '.' + name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _validate_runtime(runtime, configuration):
    from conceptmod.backends.cpu import CpuBackend
    if type(runtime) is not CpuBackend or runtime.encoder_lora or runtime.frozen is not None:
        raise ValueError('only exact CpuBackend without encoder LoRA/frozen copy is admitted')
    if runtime.latent_shape != (4, 8, 8) or runtime.device != 'cpu':
        raise ValueError('modern CPU cohort requires native [1,4,8,8] on CPU')
    if runtime.lora_rank != configuration['rank'] or _base_identity(runtime) != configuration['base_sha256']:
        raise ValueError('native CPU base/rank identity differs before binding')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32
           for model in (runtime.transformer, runtime.text_encoder) for p in model.parameters()):
        raise ValueError('CPU native parameters require float32 CPU')
    leaf = runtime.transformer.base_model.model.linear2
    if leaf.r != {'default': 4} or leaf.lora_alpha != {'default': 4} or leaf.scaling != {'default': 1.}:
        raise ValueError('native CPU PEFT rank/alpha/scale differs before binding')
    if 'forward' in vars(leaf):
        raise ValueError('CPU projection already has a forward owner')
    return leaf


class CpuResponse(FrozenResponseAdapter):
    family = 'conceptmod_cpu'
    time_coordinates = 'CpuBackend raw timestep = 1000*sigma'

    def __init__(self, runtime, configuration, *, attach=True):
        self.backend = runtime
        leaf = _validate_runtime(runtime, configuration)
        network = HNDLLoRA(leaf)  # builder/source failures precede native mutation
        self.configuration = deepcopy(configuration)
        self._closed = False
        self._restores = []
        self._leaf, self._forward = leaf, network.forward
        self._attached = False
        super().__init__(network, model_identity=dict(configuration,
            native_shape=[1, 4, 8, 8], qualification='CPU fixture only'))
        if attach:
            self.attach()

    def attach(self):
        if self._closed or self._attached or 'forward' in vars(self._leaf):
            raise RuntimeError('CPU binding already owned or closed')
        runtime = self.backend
        self._restores = [native_cleanup(runtime.transformer), native_cleanup(runtime.text_encoder)]
        try:
            runtime.transformer.eval().requires_grad_(False)
            runtime.text_encoder.eval().requires_grad_(False)
            self._leaf.forward = self._forward
            self._attached = True
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        def unhook():
            if not self._attached:
                return
            if vars(self._leaf).get('forward') is not self._forward:
                raise RuntimeError('CPU projection forward ownership changed')
            del self._leaf.forward
        _restore_all([unhook, *self._restores])

    @contextmanager
    def observation(self):
        restores = [native_cleanup(self.backend.transformer), native_cleanup(self.backend.text_encoder)]
        try:
            yield
        finally:
            _restore_all(restores)

    def _velocity(self, condition, latent, timestep):
        _validate_request(condition, latent, timestep)
        text = self.backend._encode_raw(condition)
        return self.backend._forward(self.backend.transformer, latent, timestep, text)

    def response(self, *, prompt, latent, timestep, strength=1., seed=0):
        del seed
        _validate_request(prompt, latent, timestep)
        with torch.no_grad(), self.network.scaled(strength):
            return self._velocity(prompt, latent, timestep).detach().clone()


def _validate_request(prompt, latent, timestep):
    if not isinstance(prompt, str):
        raise ValueError('prompt must be text')
    if (not isinstance(latent, torch.Tensor) or latent.shape != (1, 4, 8, 8)
            or latent.device.type != 'cpu' or latent.dtype != torch.float32
            or not torch.isfinite(latent).all()):
        raise ValueError('native CPU latent must be finite float32 [1,4,8,8]')
    if (not isinstance(timestep, torch.Tensor) or timestep.numel() != 1
            or timestep.device.type != 'cpu' or timestep.dtype != torch.float32
            or not torch.isfinite(timestep).all() or not 0 <= timestep.item() <= 1000):
        raise ValueError('CPU raw timestep requires a finite float32 scalar in [0,1000]')


def _configuration(seed=0):
    from conceptmod.backends.cpu import CpuBackend
    with torch.random.fork_rng(devices=[]):
        runtime = CpuBackend(device='cpu', lora_rank=4, seed=seed)
    return runtime, dict(model_seed=seed, rank=4, base_sha256=_base_identity(runtime))


def _native_validate(options, spec):
    if options:
        raise ValueError('native CPU inference accepts no source/model overrides')
    verify_sources()
    cfg = spec.configuration
    if (set(cfg) != {'model_seed', 'rank', 'base_sha256'} or cfg['rank'] != 4
            or type(cfg['model_seed']) is not int or not 0 <= cfg['model_seed'] < 2**31):
        raise ValueError('unknown native CPU configuration')
    if spec.identity['protocol'] != PROTOCOL:
        raise ValueError('native CPU protocol differs')


def _native_prepare(options, spec):
    runtime, configuration = _configuration(spec.configuration['model_seed'])
    if configuration != spec.configuration:
        raise ValueError('native CPU base identity differs before binding')
    native = CpuResponse(runtime, configuration, attach=False)
    try:
        identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
        if identity != spec.identity:
            raise CheckpointError('native CPU model/source/architecture differs before binding')
        native.attach()
        return InferenceBinding(native.network,
            identity,
            native.close, response=native.response, observation=native.observation)
    except BaseException:
        native.close()
        raise


NATIVE = InferenceBackend(NAME, CAPABILITIES, _native_prepare, _native_validate)


def _validate_options(options):
    allowed = {'phrase', 'steps', 'batch', 'seed', 'model_seed', 'ema_decay'}
    if set(options) - allowed:
        raise ValueError('unknown modern CPU option')
    verify_sources()
    _write_rule(options['phrase'])
    if type(options['model_seed']) is not int or not 0 <= options['model_seed'] < 2**31:
        raise ValueError('invalid CPU model_seed')


def _write_rule(phrase):
    from conceptmod import dsl
    rules = dsl.parse_phrase(phrase)
    if (len(rules) != 1 or rules[0].op != dsl.WRITE or rules[0].alpha != 1
            or rules[0].a != 'red' or rules[0].b != 'blue'):
        raise ValueError('modern CPU fixture admits only red=blue at alpha1; other DSL objectives stay historical')
    return rules[0]


def _prepare(options, runtime=None):
    expected, configuration = _configuration(options['model_seed'])
    runtime = expected if runtime is None else runtime
    native = CpuResponse(runtime, configuration)
    try:
        rule = _write_rule(options['phrase'])
        generator = torch.Generator().manual_seed(options['seed'])
        rows = [native.prepare(ResponseRow(rule.a, rule.b, '',
            torch.randn((1, 4, 8, 8), generator=generator), torch.tensor([time])))
            for time in (250., 750.)]
        critic = HNDLFieldCritic(4, 1, (8, 8), width=4)
        game = ConditionalPG094(native, rows, critic, conditions=torch.tensor([[.25], [.75]]),
            total_steps=options['steps'], batch_size=options['batch'], seed=options['seed'])
        specification = NATIVE.specification(native, model_identity=native.model_identity,
            protocol=PROTOCOL, configuration=configuration)
        return Binding(game, native.close, inference=specification, observation=native.observation)
    except BaseException:
        native.close()
        raise


FACTORY = Backend(NAME, CAPABILITIES, _prepare, _validate_options)


def open_game(phrase='red=blue', *, backend='cpu', steps=4, batch=2, seed=7,
              model_seed=0, ema_decay=.9, runtime=None, resume=None):
    """Open an actual shared session using the new GAN objective.

    A borrowed CpuBackend is restored on close/failure; HNDL weights are exported
    via the shared session. Historical PEFT artifacts are never overwritten.
    """
    if backend != 'cpu':
        raise ValueError(BACKEND_STATES.get(backend, 'unknown DSL backend'))
    config = SessionConfig(NAME, COHORT, phrase=phrase, steps=steps, batch=batch, seed=seed,
                           model_seed=model_seed, ema_decay=ema_decay)
    if resume is not None:
        packet = read_packet(resume) if isinstance(resume, (str, Path)) else deepcopy(resume)
        _check_packet(packet)
        if packet['schema'] != TRAINING or packet['manifest']['config'] != config.as_dict():
            raise CheckpointError('modern CPU config/cohort differs before native binding')
        # Validate the complete incoming state on an owned tiny CPU instance
        # before installing anything into a borrowed native model.
        staged = FACTORY.open(config)
        try:
            staged.restore(packet)
            if runtime is None:
                return staged
        except BaseException:
            staged.close()
            raise
        staged.close()
        resume = packet
    session = FACTORY.open(config, runtime=runtime)
    try:
        if resume is not None:
            session.restore(resume)
        return session
    except BaseException:
        session.close()
        raise


def load_native(packet, *, strength=1.):
    """Load data-independent raw/EMA native inference without a training factory."""
    packet = read_inference(packet) if isinstance(packet, (str, Path)) else deepcopy(packet)
    spec = _validate_packet(packet)
    _native_validate({}, spec)
    # The shared inference loader validates tensors before copying them. Check
    # that same schema on an unattached native candidate before installing any
    # product hook, including for a well-checksummed malformed payload.
    with torch.random.fork_rng(devices=[]):
        runtime, configuration = _configuration(spec.configuration['model_seed'])
        if configuration != spec.configuration:
            raise CheckpointError('native CPU base identity differs before binding')
        candidate = CpuResponse(runtime, configuration, attach=False)
        try:
            _validate_module(candidate.network, packet['network'])
        except (ValueError, TypeError, AttributeError) as exc:
            raise CheckpointError('native CPU payload differs before binding') from exc
        finally:
            candidate.close()
    return open_inference(packet, backend=NATIVE, operation='response', strength=strength)
