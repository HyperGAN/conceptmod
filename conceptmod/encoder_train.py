"""Stage 1: train a LoRA on the *text encoder only*, in embedding space.

This is the modernized "notrigger" method from ntc-ai/sliders-conceptmod:
no diffusion sampling in the loop, so it is orders of magnitude faster than
model training and can be verified with images before stage 2 ever runs.

The mechanism differs from the CLIP-era original because Sana/Z-Image use
LLM encoders with variable-length valid tokens: instead of matching full
fixed-length CLIP sequences, we steer with *pooled concept directions*.
For a concept c, its direction is

    d_c = pool(E_f(c)) - pool(E_f(''))

(pool = attention-masked mean over valid token positions of the frozen
encoder). Ops then move every valid token embedding of random *probe
prompts*:

    c++   E_t(p) -> E_f(p) + s * d_c          (concept added everywhere)
    c--   E_t(p) -> E_f(p) - s * d_c          (concept pushed out everywhere)
    a=b   E_t(p_a) -> E_f(p_a) + (pool(E_f(p_b)) - pool(E_f(p_a)))
          where p_a / p_b are the same probe template filled with a / b
          ('=b' with empty a behaves like b++)
    a#b   E_t(template(a)) pinned to E_f(template(b)); bare '#' pins the
          neutral probe templates themselves (drift anchor)
    a%b   decorrelate pooled trained-b direction from frozen-a direction
          (negative alpha aligns instead - blend)

Kept from notrigger: the fixed-distance curriculum (move a fixed fraction
of the initial distance per step instead of jumping to the target, which
kept training from collapsing) and gradient clipping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from conceptmod import dsl
from conceptmod.contracts import ModelView

PROBE_TEMPLATES = [
    "a photo of {}",
    "a painting of {}",
    "{}",
    "a detailed picture of {}, high quality",
    "{} in a landscape",
    "a portrait of {}",
]

NEUTRAL_PROBES = [
    "a photo of a person",
    "a city street",
    "a landscape with trees",
    "an animal",
    "a room interior",
    "a portrait of a woman",
    "a bowl of fruit on a table",
]


def _token_features(embeds: torch.Tensor) -> torch.Tensor:
    """(B, L, D) or (B, L, layers, D) -> (B, L, D'). Krea stacks layer taps."""
    e = embeds.float()
    if e.ndim == 4:
        return e.flatten(-2)
    return e


def pool(text_embeds) -> torch.Tensor:
    """Attention-masked mean over valid token positions -> (B, D).
    Handles both padded (tensor + mask) and unpadded (list of tensors)
    embedding formats."""
    e, m = text_embeds.embeds, text_embeds.mask
    if isinstance(e, list):
        return torch.stack([_token_features(t.unsqueeze(0)).squeeze(0).mean(dim=0)
                            for t in e])
    e = _token_features(e)
    if m is None:
        return e.mean(dim=1)
    m = m.to(e.dtype).unsqueeze(-1)
    return (e * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def as_padded(text_embeds) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize either embedding format to (B, L, D) + boolean mask."""
    e, m = text_embeds.embeds, text_embeds.mask
    if isinstance(e, list):
        assert len(e) == 1, "encoder stage expects batch size 1"
        t = _token_features(e[0].unsqueeze(0))
        return t, torch.ones(t.shape[:2], device=t.device, dtype=torch.bool)
    e = _token_features(e)
    if m is None:
        m = torch.ones(e.shape[:2], device=e.device, dtype=torch.bool)
    return e, m


def masked_mse(a, b, mask) -> torch.Tensor:
    m = mask.to(a.dtype).unsqueeze(-1)
    return (((a.float() - b.float()) ** 2) * m).sum() / (m.sum() * a.shape[-1])


def fixed_distance_target(current, target, step_dist):
    """Notrigger's curriculum: a target at most ``step_dist`` away from
    ``current`` along the direction to ``target``."""
    diff = target - current
    dist = diff.norm(dim=-1, keepdim=True)
    direction = diff / (dist + 1e-8)
    move = torch.minimum(dist, torch.full_like(dist, float(step_dist)))
    return current + direction * move


class EncoderObjective:
    """Embedding-space objective with a checkpointed immutable reference."""

    def __init__(self, backend, plan, config):
        self.backend, self.plan, self.config = backend, plan, config
        self.original = ModelView(backend, True)
        self.rules = [rule.as_rule() for rule in plan.rules]
        self.refresh()

    def refresh(self):
        # Build a replacement reference atomically: a failed provider encoding
        # during checkpoint restore cannot leave a partially rebuilt objective.
        with torch.no_grad():
            pool_uncond = pool(self.original.encode_text("")).detach()
            directions = {}
            for rule in self.rules:
                for concept in {rule.a, rule.b} - {""}:
                    if concept not in directions:
                        directions[concept] = (
                            pool(self.original.encode_text(concept))
                            - pool_uncond).detach()
        step_dist = {}
        for rule in self.rules:
            concept = rule.b or rule.a
            if concept in directions:
                step_dist[rule.raw] = directions[concept].norm().item() / self.config.curriculum_split
        self.pool_uncond, self.directions, self.step_dist = pool_uncond, directions, step_dist

    def state_dict(self):
        return {"pool_uncond": self.pool_uncond.detach().clone(),
                "directions": {key: value.detach().clone() for key, value in self.directions.items()},
                "step_dist": dict(self.step_dist)}

    def validate_state_dict(self, state):
        import math

        if not isinstance(state, dict) or set(state) != {"pool_uncond", "directions", "step_dist"}:
            raise ValueError("invalid encoder objective reference state")
        if (not isinstance(state["directions"], dict) or set(state["directions"]) != set(self.directions)
                or not isinstance(state["step_dist"], dict) or set(state["step_dist"]) != set(self.step_dist)):
            raise ValueError("encoder objective concepts do not match")
        pairs = [(self.pool_uncond, state["pool_uncond"])]
        pairs += [(value, state["directions"][key]) for key, value in self.directions.items()]
        for current, saved in pairs:
            if (not isinstance(saved, torch.Tensor) or saved.shape != current.shape
                    or saved.dtype != current.dtype or not bool(torch.isfinite(saved).all())):
                raise ValueError("encoder objective reference tensor does not match")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in state["step_dist"].values()):
            raise ValueError("invalid encoder curriculum distance")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        pool_uncond = state["pool_uncond"].detach().to(self.pool_uncond.device).clone()
        directions = {key: state["directions"][key].detach().to(value.device).clone()
                      for key, value in self.directions.items()}
        self.pool_uncond, self.directions, self.step_dist = pool_uncond, directions, dict(state["step_dist"])

    def losses(self, rng):
        backend = self.backend
        strength = self.config.encoder_strength
        for rule in self.rules:
            template = rng.choice(PROBE_TEMPLATES)
            neutral = rng.choice(NEUTRAL_PROBES)
            if rule.op in (dsl.EXAGGERATE, dsl.ERASE) or (
                    rule.op == dsl.WRITE and rule.a == ""):
                concept = rule.b if rule.op == dsl.WRITE else rule.a
                sign = -1.0 if rule.op == dsl.ERASE else 1.0
                cur, mask = as_padded(backend.encode_text_grad(neutral))
                with torch.no_grad():
                    fro, _ = as_padded(self.original.encode_text(neutral))
                    target = fro.float() + sign * strength * self.directions[concept].unsqueeze(1)
                    capped = fixed_distance_target(cur.float().detach(), target,
                                                   self.step_dist[rule.raw])
                loss = masked_mse(cur, capped, mask)
            elif rule.op == dsl.WRITE:
                p_a, p_b = template.format(rule.a), template.format(rule.b)
                cur, mask = as_padded(backend.encode_text_grad(p_a))
                with torch.no_grad():
                    frozen_a = self.original.encode_text(p_a)
                    fro, _ = as_padded(frozen_a)
                    direction = (pool(self.original.encode_text(p_b))
                                 - pool(frozen_a)).unsqueeze(1)
                    target = fro.float() + strength * direction
                    capped = fixed_distance_target(cur.float().detach(), target,
                                                   self.step_dist[rule.raw])
                loss = masked_mse(cur, capped, mask)
            elif rule.op == dsl.FREEZE:
                p_a = template.format(rule.a) if rule.a else neutral
                p_b = template.format(rule.b) if rule.b else neutral
                cur, mask = as_padded(backend.encode_text_grad(p_a))
                with torch.no_grad():
                    fro, _ = as_padded(self.original.encode_text(p_b))
                length = min(cur.shape[1], fro.shape[1])
                loss = masked_mse(cur[:, :length], fro[:, :length], mask[:, :length])
            elif rule.op == dsl.ORTHOGONAL:
                d_a = self.directions[rule.a]
                d_b = pool(backend.encode_text_grad(rule.b)) - self.pool_uncond
                loss = F.cosine_similarity(d_a, d_b, dim=-1, eps=1e-6).abs().mean()
            else:
                raise ValueError(f"unsupported encoder rule {rule.raw!r}")
            yield rule.raw, rule.alpha * loss


def train_encoder(backend, phrase, iterations=400, lr=1e-4, rank=8, seed=0,
                  strength=1.0, curriculum_split=20, *,
                  accumulation_steps=1, observers=(), stop_when=None):
    """Execute a strict encoder plan using the session API.

    Logging and evaluation belong to explicitly supplied observers. Unsupported
    rules fail during compilation, before an encoder adapter is attached.
    """
    from conceptmod.plan import EditPlan, compile_phrase
    from conceptmod.training import TrainingConfig, TrainingSession

    plan = phrase if isinstance(phrase, EditPlan) else compile_phrase(
        phrase, stage="encoder", capabilities=getattr(backend, "capabilities", None))
    if plan.stage != "encoder":
        raise ValueError("train_encoder requires an encoder EditPlan")
    config = TrainingConfig(iterations=iterations, lr=lr, seed=seed,
                            accumulation_steps=accumulation_steps, encoder_rank=rank,
                            encoder_strength=strength, curriculum_split=curriculum_split)
    with TrainingSession(backend, plan, config, observers=observers) as session:
        session.run(stop_when=stop_when)
        return [report.loss for report in session.reports]
