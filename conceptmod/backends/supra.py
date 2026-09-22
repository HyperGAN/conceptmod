"""Supra2-IMG backend (tiny DiT, frozen Flan-T5-Base, SD-VAE-FT-MSE).

Hub contract, pinned to ``SupraLabs/Supra2-IMG`` ``inference.py``:

- DiT ~104.1M (104,094,736): D_MODEL 576, DEPTH 14, N_HEADS 9, HEAD_DIM 64,
  MLP_RATIO 4, D_CTX 768
- Image 256, latent 32, patch 2, VAE scale 0.18215
- Text: frozen ``google/flan-t5-base``, ctx_len 128
- Euler flow ``t = i/K``, ``z <- z + dt * v``, CFG 3.0, 50 steps

Weights are not vendored. Pass a local ``model_final_ema.pt``, point
``SUPRA_CKPT`` at one, or use the Hugging Face cache. ``SUPRA_DOWNLOAD=1``
fetches the checkpoint. ``model_id="dummy"`` builds a tiny DiT on CPU with
no Hub access so pytest can exercise LoRA and ``predict_v``.

Cross-attention sees ``d_model``, not 768: ``ctx_proj`` maps Flan-T5 into
the DiT width first. Hub's ``DiTBlock`` takes a ``ctx_dim`` argument and
then ignores it (``Attention(..., ctx_dim=dim)``). That quirk is kept so
``cross_attn.kv`` stays ``Linear(576, 1152)``.

LoRA targets are qualified suffixes (``self_attn.proj``, ``cross_attn.proj``,
``cross_attn.q``, …). A bare ``proj`` matches both attention projections,
and a bare ``q`` matches only cross-attention ``q`` — easy to get wrong.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptmod.backends.base import Backend, TextEmbeds, require_cuda

HF_REPO = "SupraLabs/Supra2-IMG"
DEFAULT_MODEL = HF_REPO
CKPT_FILENAME = "model_final_ema.pt"

IMG_SIZE = 256
LATENT_SIZE = 32
LATENT_CH = 4
PATCH = 2
NUM_TOKENS = (LATENT_SIZE // PATCH) ** 2

D_MODEL = 576
DEPTH = 14
N_HEADS = 9
HEAD_DIM = 64
MLP_RATIO = 4.0
D_CTX = 768
MAX_CTX_LEN = 128
T5_NAME = "google/flan-t5-base"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
VAE_SCALE = 0.18215

DEFAULT_STEPS = 50
DEFAULT_CFG = 3.0
# Cheaper than the 50-step generate budget, same CFG the sampler uses.
TRAIN_STEPS = 8
TRAIN_CFG = 3.0
DUMMY_TRAIN_STEPS = 4

EXPECTED_DIT_PARAMS = 104_094_736

if D_MODEL // N_HEADS != HEAD_DIM:
    raise RuntimeError(f"HEAD_DIM {HEAD_DIM} != D_MODEL/N_HEADS")

# Path suffixes. PEFT matches ``key.endswith("." + target)``.
# Bare "proj" would hit self_attn.proj and cross_attn.proj together;
# bare "q" would hit cross_attn.q and not self_attn.qkv.
_LORA_TARGETS = [
    "self_attn.qkv",
    "self_attn.proj",
    "cross_attn.q",
    "cross_attn.kv",
    "cross_attn.proj",
]
# Flan-T5 attention linears are named q/k/v/o (not q_proj).
_TEXT_LORA_TARGETS = ["q", "k", "v", "o"]


@dataclass(frozen=True)
class SupraSpec:
    """Spatial / width knobs. Defaults are the Hub DiT."""

    latent_ch: int = LATENT_CH
    latent_size: int = LATENT_SIZE
    patch: int = PATCH
    d_model: int = D_MODEL
    depth: int = DEPTH
    n_heads: int = N_HEADS
    ctx_dim: int = D_CTX
    mlp_ratio: float = MLP_RATIO
    ctx_len: int = MAX_CTX_LEN

    def __post_init__(self) -> None:
        if self.latent_size % self.patch != 0:
            raise ValueError(
                f"latent_size {self.latent_size} not divisible by patch {self.patch}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
            )

    @property
    def num_tokens(self) -> int:
        side = self.latent_size // self.patch
        return side * side

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_ch, self.latent_size, self.latent_size)

    @property
    def image_size(self) -> int:
        return self.latent_size * (IMG_SIZE // LATENT_SIZE)


FULL_SPEC = SupraSpec()
DUMMY_SPEC = SupraSpec(
    d_model=64,
    depth=2,
    n_heads=4,
    ctx_dim=32,
    latent_size=8,
    ctx_len=16,
)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def euler_schedule(num_steps: int, device) -> tuple[torch.Tensor, float]:
    """Hub sampling grid: ``t_i = i/K``, ``dt = 1/K`` for ``i = 0..K-1``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    dt = 1.0 / float(num_steps)
    timesteps = torch.arange(num_steps, device=device, dtype=torch.float32) * dt
    return timesteps, dt


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _cached_hub_file(repo_id: str, filename: str) -> str | None:
    """Return a cached Hub file path. Does not download."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str) and os.path.isfile(cached):
        return cached
    return None


def resolve_checkpoint(
    path: str | None = None,
    *,
    download: bool = False,
    repo_id: str = HF_REPO,
) -> str | None:
    """Local file, ``SUPRA_CKPT``, ``./model_final_ema.pt``, or HF cache.

    ``download=True`` fetches ``model_final_ema.pt`` from ``repo_id``.
    ``SUPRA_DOWNLOAD`` is applied by ``SupraBackend``, not here, so a smoke
    script can pass ``download=False`` and stay offline. A miss returns None.
    """
    candidates: list[str] = []
    if path:
        candidates.append(path)
    env = os.environ.get("SUPRA_CKPT")
    if env and env not in candidates:
        candidates.append(env)
    default_local = os.path.join(".", CKPT_FILENAME)
    if default_local not in candidates:
        candidates.append(default_local)
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    cached = _cached_hub_file(repo_id, CKPT_FILENAME)
    if cached:
        return cached
    if download:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=CKPT_FILENAME)
    return None


def _split_checkpoint(state) -> tuple[dict, dict]:
    """Hub unwrap: ``ema`` if present, else ``model``, else the dict itself."""
    if not isinstance(state, dict):
        raise ValueError("Supra checkpoint is not a state dict")
    raw_cfg = state.get("config", {})
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    patch = cfg.get("patch", PATCH)
    if patch != PATCH:
        raise ValueError(f"Checkpoint PATCH={patch} != script PATCH={PATCH}")
    if "ema" in state or "model" in state:
        weights = state["ema"] if "ema" in state else state["model"]
    else:
        weights = state
    if not isinstance(weights, dict):
        raise ValueError("Supra checkpoint weights are not a state dict")
    return weights, cfg


def _load_weights(model: nn.Module, weights: dict) -> None:
    try:
        model.load_state_dict(weights, strict=True)
        return
    except RuntimeError as first:
        stripped = {}
        changed = False
        for key, value in weights.items():
            renamed = key.removeprefix("module.")
            changed = changed or renamed != key
            stripped[renamed] = value
        if not changed:
            raise first
        model.load_state_dict(stripped, strict=True)


def load_supra_checkpoint(model: nn.Module, path: str) -> dict:
    """Load Hub ``model_final_ema.pt`` (or a raw state dict) into ``model``.

    ``weights_only=False`` matches Hub ``inference.py``: the file stores a
    config dict alongside the EMA tensors.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights, cfg = _split_checkpoint(state)
    _load_weights(model, weights)
    return cfg


def _uncond_from_config(cfg: dict, device: str) -> TextEmbeds | None:
    """Stored empty-prompt embeds, if the checkpoint saved them."""
    if "uncond_text" not in cfg or "uncond_mask" not in cfg:
        return None
    text = cfg["uncond_text"]
    mask = cfg["uncond_mask"]
    if not torch.is_tensor(text) or not torch.is_tensor(mask):
        return None
    text = text.detach().float()
    mask = mask.detach().float()
    if text.ndim == 2:
        text = text.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return TextEmbeds(text.to(device), mask.to(device))


def _resolve_device(device: str, *, cuda_required: bool) -> str:
    if cuda_required:
        return str(require_cuda(device))
    dev = torch.device(device)
    if dev.type == "cuda":
        if torch.cuda.is_available():
            return str(dev)
        print(f"supra dummy: {device!r} requested but CUDA is unavailable; using cpu")
        return "cpu"
    return "cpu"


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: ``x * (1 + scale) + shift``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class Attention(nn.Module):
    """Multi-head self- or cross-attention."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int | None = None) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_self = ctx_dim is None
        if self.is_self:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        if self.is_self:
            qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        else:
            m = ctx.shape[1]
            q = self.q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(ctx).view(b, m, 2, self.n_heads, self.head_dim)
            k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask.bool()[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class DiTBlock(nn.Module):
    """DiT block: AdaLN-Zero self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Hub projects context to ``dim`` before the block, then passes
        # ``ctx_dim`` here and still builds cross-attn at ``dim``. Keep that.
        del ctx_dim
        self.cross_attn = Attention(dim, n_heads, ctx_dim=dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim, bias=True),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln(c).chunk(6, dim=1)
        )
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa)
        )
        x = x + self.cross_attn(self.norm_ca(x), ctx=ctx, ctx_mask=ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch, bias=True)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class SupraDiT(nn.Module):
    """Rectified-flow text-to-image DiT. Defaults match the Hub checkpoint."""

    def __init__(
        self,
        latent_ch: int = LATENT_CH,
        d_model: int = D_MODEL,
        depth: int = DEPTH,
        n_heads: int = N_HEADS,
        ctx_dim: int = D_CTX,
        mlp_ratio: float = MLP_RATIO,
        num_tokens: int = NUM_TOKENS,
        patch: int = PATCH,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.patch = patch
        self.x_embed = nn.Linear(latent_ch * patch * patch, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.t_embed = TimestepEmbedder(d_model)
        self.ctx_proj = nn.Linear(ctx_dim, d_model, bias=True)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, d_model, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(d_model, latent_ch * patch * patch)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = z.shape
        p = self.patch
        gh, gw = h // p, w // p
        if gh * gw != self.num_tokens:
            raise ValueError(
                f"latent {h}x{w} patch {p} -> {gh * gw} tokens, "
                f"pos_embed has {self.num_tokens}"
            )
        x = (
            z.view(b, c, gh, p, gw, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, gh * gw, c * p * p)
        )
        x = self.x_embed(x) + self.pos_embed
        cond = self.t_embed(t)
        ctx = self.ctx_proj(ctx)
        for blk in self.blocks:
            x = blk(x, cond, ctx, ctx_mask)
        x = self.final(x, cond)
        return (
            x.view(b, gh, gw, c, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, c, h, w)
        )


def build_dit(spec: SupraSpec | None = None) -> SupraDiT:
    spec = spec or FULL_SPEC
    return SupraDiT(
        latent_ch=spec.latent_ch,
        d_model=spec.d_model,
        depth=spec.depth,
        n_heads=spec.n_heads,
        ctx_dim=spec.ctx_dim,
        mlp_ratio=spec.mlp_ratio,
        num_tokens=spec.num_tokens,
        patch=spec.patch,
    )


class _DummySelfAttention(nn.Module):
    """q/k/v/o linears so encoder LoRA targets match Flan-T5 module names."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o(self.v(self.k(self.q(x))))


class DummyTextEncoder(nn.Module):
    """Byte embedding stand-in. No tokenizer download."""

    def __init__(self, dim: int, vocab: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.SelfAttention = _DummySelfAttention(dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.SelfAttention(self.embed(input_ids))


def _missing_ckpt_message() -> str:
    return (
        "Supra2-IMG weights not found (model_final_ema.pt). "
        "Pass ckpt=..., set SUPRA_CKPT, or place the file in the HF cache "
        f"for {HF_REPO}. SUPRA_DOWNLOAD=1 fetches it. "
        "model_id='dummy' is the CPU testbed and does not download."
    )


class SupraBackend(Backend):
    """LoRA-only Supra2-IMG. Frozen reference is the DiT with the adapter off.

    The 104M DiT stays fp32 (same idea as SANA: bf16 master weights plus Adam
    is a bad train dtype at this size). Hub inference autocasts to bf16; the
    Euler / CFG math is unchanged.
    """

    def __init__(
        self,
        device: str,
        model_id: str = DEFAULT_MODEL,
        resolution: int | None = None,
        lora_rank: int | None = None,
        ckpt: str | None = None,
        download: bool = False,
        dummy: bool = False,
        generate_steps: int | None = None,
        generate_guidance: float | None = None,
    ):
        if model_id is not None and str(model_id).lower() == "dummy":
            dummy = True
        self.dummy = dummy
        self.spec = DUMMY_SPEC if dummy else FULL_SPEC
        if resolution is not None and resolution != self.spec.image_size:
            kind = "dummy" if dummy else "Supra2-IMG"
            raise ValueError(
                f"supra {kind} resolution is {self.spec.image_size}, got {resolution}"
            )
        self.device = _resolve_device(device, cuda_required=not dummy)
        self.model_id = "dummy" if dummy else (model_id or DEFAULT_MODEL)
        self.resolution = self.spec.image_size
        self.latent_shape = self.spec.latent_shape
        self.ctx_len = self.spec.ctx_len
        self.compute_dtype = torch.float32
        self.generate_steps = (
            generate_steps if generate_steps is not None
            else (DUMMY_TRAIN_STEPS if dummy else DEFAULT_STEPS)
        )
        self.generate_guidance = (
            DEFAULT_CFG if generate_guidance is None else generate_guidance
        )
        self._stored_uncond: TextEmbeds | None = None
        self._text_cache: dict[tuple[str, bool], TextEmbeds] = {}
        self.encoder_lora = False
        self.ckpt_path: str | None = None
        self.tokenizer = None
        self.vae = None

        dit = build_dit(self.spec)
        if not dummy:
            self.ckpt_path = self._resolve_real_ckpt(ckpt, download)
            cfg = load_supra_checkpoint(dit, self.ckpt_path)
            if "ctx_len" in cfg:
                self.ctx_len = int(cfg["ctx_len"])
            self._stored_uncond = _uncond_from_config(cfg, self.device)

        dit.to(device=self.device, dtype=self.compute_dtype)
        if lora_rank is None:
            lora_rank = 16
            print("supra backend is LoRA-only; defaulting to rank 16")
        self.lora_rank = lora_rank
        self.transformer = self._attach_dit_lora(dit, lora_rank)
        self.frozen = None

        if dummy:
            self.text_encoder = DummyTextEncoder(self.spec.ctx_dim).to(self.device)
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        else:
            self._load_frozen_encoders()

        n = count_parameters(self.transformer.get_base_model())
        print(
            f"supra {'dummy ' if dummy else ''}DiT on {self.device}: "
            f"{n / 1e6:.1f}M params, latent {self.latent_shape}, "
            f"steps {self.generate_steps}, cfg {self.generate_guidance}"
        )

    def _resolve_real_ckpt(self, ckpt: str | None, download: bool) -> str:
        repo_id = HF_REPO
        explicit = ckpt
        if (
            self.model_id
            and self.model_id not in (DEFAULT_MODEL, "dummy")
            and os.path.isfile(self.model_id)
        ):
            explicit = self.model_id
        elif self.model_id and self.model_id not in (DEFAULT_MODEL, "dummy"):
            repo_id = self.model_id
        found = resolve_checkpoint(
            explicit,
            download=download or _env_flag("SUPRA_DOWNLOAD"),
            repo_id=repo_id,
        )
        if found is None:
            raise FileNotFoundError(_missing_ckpt_message())
        print(f"supra checkpoint: {found}")
        return found

    def _attach_dit_lora(self, module: nn.Module, rank: int) -> nn.Module:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=list(_LORA_TARGETS),
        )
        wrapped = get_peft_model(module, config)
        wrapped.to(self.device)
        for param in wrapped.parameters():
            if param.requires_grad:
                param.data = param.data.float().to(self.device)
        names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        self_hit = any(".self_attn." in n for n in names)
        cross_hit = any(".cross_attn." in n for n in names)
        outside = [
            n for n in names if ".self_attn." not in n and ".cross_attn." not in n
        ]
        if not names or not self_hit or not cross_hit or outside:
            raise RuntimeError(
                "supra LoRA must hit self_attn and cross_attn only, "
                f"got {names[:12]}"
            )
        if not any(".self_attn.qkv." in n for n in names):
            raise RuntimeError("supra LoRA missed self_attn.qkv")
        if not any(".cross_attn.q." in n for n in names):
            raise RuntimeError("supra LoRA missed cross_attn.q")
        if not any(".cross_attn.kv." in n for n in names):
            raise RuntimeError("supra LoRA missed cross_attn.kv")
        if not any(".self_attn.proj." in n for n in names):
            raise RuntimeError("supra LoRA missed self_attn.proj")
        if not any(".cross_attn.proj." in n for n in names):
            raise RuntimeError("supra LoRA missed cross_attn.proj")
        wrapped.eval()
        return wrapped

    def _load_frozen_encoders(self) -> None:
        from diffusers import AutoencoderKL
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
        self.text_encoder = T5EncoderModel.from_pretrained(T5_NAME)
        self.text_encoder.to(self.device).eval()
        self.text_encoder.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(VAE_NAME)
        self.vae.to(self.device).eval()
        self.vae.requires_grad_(False)

    def training_defaults(self) -> dict:
        if self.dummy:
            return {"sample_steps": DUMMY_TRAIN_STEPS, "sample_guidance": TRAIN_CFG}
        return {"sample_steps": TRAIN_STEPS, "sample_guidance": TRAIN_CFG}

    # ---------------- text ----------------

    def _dummy_ids(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        raw = [min(ord(ch), 255) for ch in prompt][: self.ctx_len]
        if not raw:
            # At least one valid token so SDPA is not handed an all-pad mask.
            raw = [0]
        mask = [1] * len(raw)
        while len(raw) < self.ctx_len:
            raw.append(0)
            mask.append(0)
        ids = torch.tensor([raw], device=self.device, dtype=torch.long)
        attn = torch.tensor([mask], device=self.device, dtype=torch.float32)
        return ids, attn

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        if self.dummy:
            ids, mask = self._dummy_ids(prompt)
            hidden = self.text_encoder(ids)
            return TextEmbeds(hidden.float(), mask)
        tok = self.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=self.ctx_len,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        hidden = self.text_encoder(**tok).last_hidden_state
        return TextEmbeds(hidden.float(), tok["attention_mask"].float())

    def _use_stored_uncond(self, frozen: bool) -> bool:
        if self._stored_uncond is None:
            return False
        if self.encoder_lora and not frozen:
            return False
        return True

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        if not self.encoder_lora:
            frozen = False
        if prompt == "" and self._use_stored_uncond(frozen):
            return self._stored_uncond
        key = (prompt, frozen)
        if key not in self._text_cache:
            if self.encoder_lora and frozen:
                with self.text_encoder.disable_adapter():
                    self._text_cache[key] = self._encode_raw(prompt)
            else:
                self._text_cache[key] = self._encode_raw(prompt)
        return self._text_cache[key]

    def encode_text_grad(self, prompt: str) -> TextEmbeds:
        return self._encode_raw(prompt)

    def attach_encoder_lora(self, rank: int = 8):
        from peft import LoraConfig, get_peft_model

        assert not self.encoder_lora, "encoder LoRA already attached"
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=list(_TEXT_LORA_TARGETS),
        )
        self.text_encoder = get_peft_model(self.text_encoder, config)
        self.text_encoder.to(self.device)
        for _name, param in self.text_encoder.named_parameters():
            if param.requires_grad:
                param.data = param.data.float().to(self.device)
        self.encoder_lora = True
        self._text_cache.clear()
        params = [p for p in self.text_encoder.parameters() if p.requires_grad]
        assert params, "encoder LoRA matched no q/k/v/o modules"
        return params

    # ---------------- velocity ----------------

    def _forward(self, model, z, timestep, text: TextEmbeds):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.device)
        t = timestep.reshape(-1).to(device=self.device, dtype=torch.float32)
        if t.numel() == 1:
            t = t.expand(z.shape[0])
        ctx = text.embeds.to(device=self.device, dtype=self.compute_dtype)
        mask = None if text.mask is None else text.mask.to(self.device)
        if ctx.shape[0] == 1 and z.shape[0] > 1:
            ctx = ctx.expand(z.shape[0], -1, -1)
            if mask is not None:
                mask = mask.expand(z.shape[0], -1)
        hidden = z.to(device=self.device, dtype=self.compute_dtype)
        return model(hidden, t, ctx, mask).float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            with torch.no_grad(), self.transformer.disable_adapter():
                return self._forward(self.transformer, z, timestep, text)
        return self._forward(self.transformer, z, timestep, text)

    # ---------------- sampling ----------------

    def _cfg(self, prompt, z, t, guidance, frozen):
        v = self.predict_v(prompt, z, t, frozen=frozen)
        # Hub: CFG when scale > 1. Empty prompt is already unconditional.
        if guidance is not None and guidance > 1.0 and prompt != "":
            v_u = self.predict_v("", z, t, frozen=frozen)
            v = v_u + guidance * (v - v_u)
        return v

    def _noise(self, generator):
        return torch.randn(
            (1, *self.latent_shape),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        timesteps, dt = euler_schedule(num_steps, self.device)
        z = self._noise(generator)
        for i, t in enumerate(timesteps):
            if i >= stop_index:
                return z, t
            v = self._cfg(prompt, z, t, guidance, frozen=False)
            z = z + dt * v
        return z, timesteps[-1]

    def render(self, prompt, generator, num_steps, guidance, grad_steps=0,
               frozen=False):
        timesteps, dt = euler_schedule(num_steps, self.device)
        z = self._noise(generator)
        n = len(timesteps)
        for i, t in enumerate(timesteps):
            with torch.set_grad_enabled(not frozen and i >= n - grad_steps):
                v = self._cfg(prompt, z, t, guidance, frozen=frozen)
                z = z + dt * v
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        with torch.set_grad_enabled(grad):
            if self.dummy or self.vae is None:
                rgb = torch.tanh(z[:, :3])
                img = F.interpolate(
                    rgb, scale_factor=IMG_SIZE // LATENT_SIZE, mode="nearest",
                )
            else:
                latents = z.to(device=self.device, dtype=self.vae.dtype)
                img = self.vae.decode(latents / VAE_SCALE).sample
            return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        from PIL import Image

        num_steps = self.generate_steps if num_steps is None else num_steps
        guidance = self.generate_guidance if guidance is None else guidance
        g = torch.Generator(device=self.device).manual_seed(seed)
        img = self.render(prompt, g, num_steps, guidance, grad_steps=0,
                          frozen=frozen)
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        assert params, "peft returned no trainable params"
        if train_method not in (None, "lora"):
            print(f"supra backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str) -> None:
        self.transformer.save_pretrained(path)
