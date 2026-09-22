"""Circle orbit hold: closed-loop feedforward vs radius drift.

Explicit Euler of a constant-speed tangential feedforward keeps heading and
linear speed on target while the radius walks outward. A perpendicular step
of length ``dt * omega * R`` obeys Pythagoras:

    r_{k+1}^2 = r_k^2 + (dt * omega * R)^2

so direction and speed can look fine on an ablated step. The closed-loop
residual head adds the inward radial correction that lands each step back on
the target circle. The gate requires that radius hold. Heading and speed are
necessary and not sufficient.

Adv shape is the particle-sliders locked_shared / #94 recipe, checked with
the real ``GradRegularizer`` (not a thinned kappa stub):

  RpGAN logistic (matched relativistic pair)
  b_cap coeff=1, kappa=1, norm=l2, lazy=1, anneal=none
  fm_weight=0
  cover_weight=1.5   # demo / locked_shared cover, not Music 1.0
  particle_l2=0.02, n_particles=12

``vicreg_weight`` (0.05 on the Music prior) is not applied: these particles
are the orbit state, not a latent particle prior. That omission is reported
here and is not a second adv recipe. No Music or Anima GPU transfer is claimed
from a pass on this CPU toy.

Tail the family with ``python -m conceptmod.toys.orbit_hold``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer

# Behavioral gates. The residual's extra speed is O((omega * dt)^2), so at the
# pinned step the held orbit stays above the heading floor. One ablated Euler
# step already inflates radius by sqrt(1+(omega*dt)^2)-1 ≈ 0.0308, above the
# radius cap; a pure heading/speed score would still pass that step.
DIRECTION_COS_MIN = 0.98
SPEED_REL_MAX = 0.05
RADIUS_REL_MAX = 0.02

# Demo cover from locked_shared. Music's 1.0 cover is a different posture.
DEMO_COVER = 1.5

_LOCKED_ADV = {
    "loss_type": "logistic",
    "gan_mode": "rp",
    "pairing": "matched",
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_kappa": 1.0,
    "reg_norm": "l2",
    "fm_weight": 0.0,
    "cover_weight": DEMO_COVER,
    "particle_l2": 0.02,
    "n_particles": 12,
}

# Kinematics are part of the toy protocol. Loosening them can hide Euler drift.
_PROTOCOL = {
    "radius": 1.0,
    "omega": 1.0,
    "dt": 0.25,
    "steps": 32,
}

_DRIFT_REASONS = {
    "loss_type": "loss_not_logistic",
    "gan_mode": "stranger_pairing",
    "pairing": "stranger_pairing",
    "reg_arm": "grad_arm",
    "reg_coeff": "bcap_coeff",
    "reg_kappa": "bcap_kappa",
    "reg_norm": "bcap_norm",
    "fm_weight": "fm_on",
    "cover_weight": "cover_weight",
    "particle_l2": "particle_l2",
    "n_particles": "n_particles",
    "radius": "toy_protocol",
    "omega": "toy_protocol",
    "dt": "toy_protocol",
    "steps": "toy_protocol",
}


@dataclass(frozen=True)
class OrbitRecipe:
    """Locked-shared orbit controller plus the residual-head switch.

    ``residual=False`` is the intentional ablation (zero radial head). It is
    not an adv-recipe change. Every other knob is pinned to locked_shared or
    to the toy protocol.
    """

    name: str = "locked_shared"
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    pairing: str = "matched"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    fm_weight: float = 0.0
    cover_weight: float = DEMO_COVER
    particle_l2: float = 0.02
    n_particles: int = 12
    residual: bool = True
    radius: float = 1.0
    omega: float = 1.0
    dt: float = 0.25
    steps: int = 32

    def __post_init__(self) -> None:
        if self.pairing not in ("matched", "stranger"):
            raise ValueError("pairing must be 'matched' or 'stranger'")
        if self.steps < 1 or self.n_particles < 3:
            raise ValueError("steps >= 1 and n_particles >= 3 are required")
        if min(self.radius, self.omega, self.dt) <= 0:
            raise ValueError("radius, omega, and dt must be positive")
        if self.omega * self.dt >= 1:
            raise ValueError("omega * dt must be < 1 so the radial correction is real")


@dataclass(frozen=True)
class OrbitReport:
    name: str
    passed: bool
    reasons: tuple[str, ...]
    radius_rel: float
    direction_cos: float
    speed_rel: float
    radius_pass: bool
    direction_pass: bool
    speed_pass: bool
    d_loss: float
    bcap: float
    cover: float
    particle_l2_term: float
    fm_weight: float
    fm_term: float
    critic_frozen: bool

    def line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        why = ",".join(self.reasons) if self.reasons else "-"
        return (
            f"ORBIT arm={self.name} verdict={verdict} "
            f"radius_rel={self.radius_rel:.6e} direction_cos={self.direction_cos:.6f} "
            f"speed_rel={self.speed_rel:.6e} "
            f"gates=r{int(self.radius_pass)}d{int(self.direction_pass)}s{int(self.speed_pass)} "
            f"d_loss={self.d_loss:.6f} bcap={self.bcap:.6f} "
            f"cover={self.cover:.6e} particle_l2={self.particle_l2_term:.6e} "
            f"fm_weight={self.fm_weight:g} fm_term={self.fm_term:.6e} "
            f"critic_frozen={int(self.critic_frozen)} reasons={why}"
        )


class ClosedLoopRadialHead:
    """Inward radial velocity that cancels one Euler step back onto radius R."""

    def __call__(self, radius_now: torch.Tensor, recipe: OrbitRecipe) -> torch.Tensor:
        shrink = math.sqrt(1.0 - (recipe.omega * recipe.dt) ** 2)
        return (recipe.radius * shrink - radius_now) / recipe.dt


class ZeroResidualHead:
    """Ablation: the radial head is present and contributes nothing."""

    def __call__(self, radius_now: torch.Tensor, recipe: OrbitRecipe) -> torch.Tensor:
        return torch.zeros_like(radius_now)


class HardcodedKappaBCap(GradRegularizer):
    """Negative control. Attributes claim locked b_cap; the cap center is 0.

    A thinned stub that ignores ``kappa`` and penalizes ``relu(n)^2``.
    """

    def __init__(self) -> None:
        super().__init__("b_cap", coeff=1.0, kappa=1.0, norm="l2")

    def center(self, step: int) -> float:
        return 0.0


def locked_recipe(**overrides) -> OrbitRecipe:
    return OrbitRecipe(**overrides)


def formulation_drift(recipe: OrbitRecipe) -> tuple[str, ...]:
    """Adv and protocol mismatches vs locked_shared. Residual is not a mismatch."""
    reasons = []
    pinned = {**_LOCKED_ADV, **_PROTOCOL}
    for key, code in _DRIFT_REASONS.items():
        if getattr(recipe, key) != pinned[key] and code not in reasons:
            reasons.append(code)
    return tuple(reasons)


def _cloud(recipe: OrbitRecipe) -> torch.Tensor:
    angles = torch.arange(recipe.n_particles, dtype=torch.float64) * (2.0 * math.pi / recipe.n_particles)
    return torch.stack((recipe.radius * torch.cos(angles), recipe.radius * torch.sin(angles)), dim=-1)


def _step(points: torch.Tensor, recipe: OrbitRecipe, head) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radius_now = torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    radial = points / radius_now
    tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=-1)
    speed = recipe.omega * recipe.radius
    velocity = speed * tangent + head(radius_now, recipe) * radial
    return points + recipe.dt * velocity, velocity, tangent


def _rollout(recipe: OrbitRecipe, head) -> tuple[torch.Tensor, float, float, float]:
    points = _cloud(recipe)
    cos_acc = []
    speed_acc = []
    radius_acc = []
    target_speed = recipe.omega * recipe.radius
    for _ in range(recipe.steps):
        points, velocity, tangent = _step(points, recipe, head)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        cos_acc.append((velocity * tangent).sum(-1) / speed)
        speed_acc.append((speed / target_speed - 1.0).abs())
        radius_acc.append((torch.linalg.vector_norm(points, dim=-1) / recipe.radius - 1.0).abs())
    direction = torch.cat(cos_acc).mean().item()
    speed_rel = torch.cat(speed_acc).mean().item()
    radius_rel = torch.cat(radius_acc).mean().item()
    return points, direction, speed_rel, radius_rel


def _cover(points: torch.Tensor) -> float:
    angles = torch.sort(torch.atan2(points[:, 1], points[:, 0])).values
    wrapped = torch.cat((angles[1:], angles[:1] + 2.0 * math.pi))
    gaps = wrapped - angles
    ideal = 2.0 * math.pi / points.shape[0]
    return (gaps - ideal).pow(2).mean().item()


def _frozen_critic() -> nn.Linear:
    """Fixed linear critic. Weights are not stepped; the penalty graph still sees them."""
    critic = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 0.0]], dtype=torch.float64))
    return critic


def _audit_bcap(regularizer, critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> tuple[float, tuple[str, ...]]:
    """Run the real b_cap and refuse a stub whose value, center, or graph disagrees."""
    reference = GradRegularizer("b_cap", coeff=1.0, kappa=1.0, norm="l2", lazy_k=1, target_anneal="none")
    ref_pen, ref_stats = reference.penalty(critic, real, fake, step=1)
    reasons = []
    try:
        pen, stats = regularizer.penalty(critic, real, fake, step=1)
    except Exception:
        return float("nan"), ("thinned_bcap",)
    if not isinstance(regularizer, GradRegularizer):
        reasons.append("thinned_bcap")
    if getattr(regularizer, "arm", None) != "b_cap":
        reasons.append("thinned_bcap")
    if getattr(regularizer, "kappa", None) != 1.0 or getattr(regularizer, "coeff", None) != 1.0:
        reasons.append("thinned_bcap")
    if getattr(regularizer, "norm", None) != "l2":
        reasons.append("thinned_bcap")
    center = stats.get("center", None) if isinstance(stats, dict) else None
    if center != ref_stats["center"] or not stats.get("applied", False):
        reasons.append("thinned_bcap")
    if not torch.is_tensor(pen) or not pen.requires_grad or not torch.allclose(pen, ref_pen):
        reasons.append("thinned_bcap")
    value = float(pen.detach()) if torch.is_tensor(pen) else float("nan")
    # de-dupe while keeping order
    deduped = tuple(dict.fromkeys(reasons))
    return value, deduped


def _paired_logits(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, pairing: str):
    real_logits = critic(real).squeeze(-1)
    fake_logits = critic(fake).squeeze(-1)
    if pairing == "stranger":
        fake_logits = fake_logits.roll(1, 0)
    return real_logits, fake_logits


def evaluate(recipe: OrbitRecipe, regularizer=None) -> OrbitReport:
    """Score one arm. A pass needs locked_shared and a held radius."""
    reasons = list(formulation_drift(recipe))
    head = ClosedLoopRadialHead() if recipe.residual else ZeroResidualHead()
    final, direction, speed_rel, radius_rel = _rollout(recipe, head)
    radius_pass = radius_rel <= RADIUS_REL_MAX
    direction_pass = direction >= DIRECTION_COS_MIN
    speed_pass = speed_rel <= SPEED_REL_MAX

    critic = _frozen_critic()
    weight_before = critic.weight.detach().clone()
    start = _cloud(recipe)
    held, _, _ = _step(start, locked_recipe(), ClosedLoopRadialHead())
    fake, _, _ = _step(start, recipe, head)
    # The manifold target is the feedforward step pulled back onto the circle.
    # The residual head matches it, so matched Rp logistic is softplus(0).
    # An ablated step leaves the circle; on this frozen linear critic the
    # logistic value barely moves, which is why radius is its own gate.
    real = held.detach()
    fake = fake.detach()
    loss = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    real_logits, fake_logits = _paired_logits(critic, real, fake, recipe.pairing)
    d_loss = float(loss.d_loss(real_logits, fake_logits).detach())
    fm_raw = (fake_logits - real_logits).pow(2).mean()
    fm_term = float((recipe.fm_weight * fm_raw).detach())

    reg = regularizer
    if reg is None:
        reg = GradRegularizer(
            recipe.reg_arm,
            coeff=recipe.reg_coeff,
            kappa=recipe.reg_kappa,
            norm=recipe.reg_norm,
            lazy_k=1,
            target_anneal="none",
        )
    bcap, thin_reasons = _audit_bcap(reg, critic, real, fake)
    for code in thin_reasons:
        if code not in reasons:
            reasons.append(code)
    critic_frozen = torch.equal(critic.weight, weight_before)

    cover = _cover(final)
    final_r = torch.linalg.vector_norm(final, dim=-1)
    particle_l2_term = recipe.particle_l2 * (final_r - recipe.radius).pow(2).mean().item()

    if not radius_pass:
        reasons.append("radius_drift")
    if not direction_pass:
        reasons.append("direction")
    if not speed_pass:
        reasons.append("speed")
    if not critic_frozen:
        reasons.append("critic_moved")

    passed = not reasons
    return OrbitReport(
        name=recipe.name,
        passed=passed,
        reasons=tuple(reasons),
        radius_rel=radius_rel,
        direction_cos=direction,
        speed_rel=speed_rel,
        radius_pass=radius_pass,
        direction_pass=direction_pass,
        speed_pass=speed_pass,
        d_loss=d_loss,
        bcap=bcap,
        cover=cover,
        particle_l2_term=particle_l2_term,
        fm_weight=recipe.fm_weight,
        fm_term=fm_term,
        critic_frozen=critic_frozen,
    )


def family_arms() -> list[tuple[OrbitRecipe, object | None]]:
    """One family: locked pass, residual ablation, and locked-shape refusals."""
    return [
        (locked_recipe(name="locked_shared"), None),
        (locked_recipe(name="ablate_residual", residual=False), None),
        (locked_recipe(name="stranger_vanilla", gan_mode="vanilla"), None),
        (locked_recipe(name="stranger_shuffle", pairing="stranger"), None),
        (locked_recipe(name="fm_on", fm_weight=1.0), None),
        (locked_recipe(name="hinge_drift", loss_type="hinge"), None),
        (locked_recipe(name="thin_bcap"), HardcodedKappaBCap()),
    ]


def run_family() -> list[OrbitReport]:
    reports = []
    for recipe, regularizer in family_arms():
        report = evaluate(recipe, regularizer)
        print(report.line(), flush=True)
        reports.append(report)
    return reports


def _main() -> None:
    reports = run_family()
    print("ORBIT family=radius_hold pass=%d/%d" % (sum(r.passed for r in reports), len(reports)), flush=True)


if __name__ == "__main__":
    _main()
