"""2D sheet recipe lifted onto a tilted Field3D plane (CPU).

Particle-sliders ``analysis/slider2d/field3d.py`` is the R³ leftover field
the 2D locked recipe is supposed to survive: ``b_cap``, demo cover,
``faithful_guard_e``, particles versus residual. That note says the
transfer has no 3D-only hacks. This toy is that transfer, closed form.

The 2D recipe is the guarded odd residual of
:class:`conceptmod.toys.cover_leftover.LeftoverField` (slider along û,
content along ĉ, unused ê stripped by ``faithful_guard_e``). On the
coordinate plane that vector is already the ambient paste
``(slider, content, 0, 0)``, and the cover gates pass.

The lift tilts û toward the Field3D leftover axis (ê, ambient z) by
``π/4``. The correction is the tangent pushforward

    δ = slider · û(θ) + content · ĉ

which puts the missing z component on the residual and keeps δ · ê = 0.
The same keep / leak / same-dir gates then match the plane. A naive copy
writes the 2D numbers into x, y and leaves z at 0, so û undershoots and
the normal component lands on ê.

Locked knobs are the demo stamp: RpGAN logistic, ``GradientPenalty``
``b_cap`` coeff=1 κ=1 l2, FM off, cover 1.5, n=12, ``particle_l2=0.02``.
κ stays 1 on the lift (no dimension rescaling). Stranger pairing, FM-on,
and a κ-hardcoded cap fail while the lifted geometry still holds.

One scored step. No train budget. A PASS is a CPU toy. It is not a Music
or Anima GPU transfer.

Tail the board with ``python -m conceptmod.toys.field_lift``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from particlegan import GANLoss, GradientPenalty

from conceptmod.toys.cover_leftover import (
    CONTENT_KEPT_MIN,
    LEAK_RATIO_MAX,
    LeftoverField,
    POLE_REL_ERR_MAX,
    SAME_DIR_MAX,
    U_KEPT_MIN,
    faithful_guard_e,
    leftover_bipolar,
)
from conceptmod.toys.locked_shared_floor import (
    COVER_WEIGHT,
    FM_WEIGHT,
    N_PARTICLES,
    PARTICLE_L2,
    ThinnedKappaCap,
    feature_match,
)


# Field3D leftover amplitudes (the cover toy's one-row sheet).
_SHEET = LeftoverField()
SLIDER = float(_SHEET.slider)
CONTENT = float(_SHEET.content)
LEAK = float(_SHEET.leak)
LYRIC = float(_SHEET.lyric)

# Tilt of û out of the coordinate plane, toward leftover ê.
LIFT_ANGLE = math.pi / 4

# Sheet offsets for the n=12 cloud. Symmetric, so the centroid stays on the pole.
_OFFSETS = torch.linspace(-0.05, 0.05, N_PARTICLES, dtype=torch.float64)


@dataclass(frozen=True)
class Frame:
    """Orthonormal (û, ĉ, ê, lyric) in R^4. ê is the lift normal."""

    name: str
    angle: float
    u: torch.Tensor
    content: torch.Tensor
    e: torch.Tensor
    lyric: torch.Tensor

    def neu(self) -> torch.Tensor:
        return LYRIC * self.lyric


def concept_frame(angle: float, name: str) -> Frame:
    """Tilt û in the (x, z) plane. θ=0 is the 2D coordinate sheet."""
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    dtype = torch.float64
    u = torch.tensor([c, 0.0, s, 0.0], dtype=dtype)
    content = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=dtype)
    # û × ĉ, so ê is the outward lift axis.
    e = torch.tensor([-s, 0.0, c, 0.0], dtype=dtype)
    lyric = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    frame = Frame(name=name, angle=float(angle), u=u, content=content, e=e, lyric=lyric)
    _check_frame(frame)
    return frame


def plane_frame() -> Frame:
    return concept_frame(0.0, "plane")


def lifted_frame() -> Frame:
    return concept_frame(LIFT_ANGLE, "tilted")


def _check_frame(frame: Frame) -> None:
    axes = (frame.u, frame.content, frame.e, frame.lyric)
    for axis in axes:
        if abs(float(axis.norm()) - 1.0) > 1e-8:
            raise RuntimeError("lift frame axis is not a unit vector")
    for i, left in enumerate(axes):
        for right in axes[i + 1 :]:
            if abs(float(left @ right)) > 1e-8:
                raise RuntimeError("lift frame axes are not orthogonal")


def sheet_coefficients() -> tuple[float, float]:
    """2D recipe: guarded slider and content. Leak is not a sheet coordinate."""
    return SLIDER, CONTENT


def pushforward(frame: Frame, slider: float = SLIDER, content: float = CONTENT) -> torch.Tensor:
    """Tangent lift of the 2D coefficient vector."""
    return float(slider) * frame.u + float(content) * frame.content


def naive_copy(slider: float = SLIDER, content: float = CONTENT) -> torch.Tensor:
    """Paste the 2D numbers into ambient x, y. The lift axis stays 0."""
    return torch.tensor([float(slider), float(content), 0.0, 0.0], dtype=torch.float64)


def lift_correction(frame: Frame, slider: float = SLIDER, content: float = CONTENT) -> torch.Tensor:
    """What the naive paste is missing on this frame. Zero on the plane."""
    return pushforward(frame, slider, content) - naive_copy(slider, content)


def guarded_odd(frame: Frame) -> torch.Tensor:
    """``faithful_guard_e`` odd part. Unused ê comes off; û and content stay."""
    neu = frame.neu()
    raw = pushforward(frame) + LEAK * frame.e
    plus, _minus = faithful_guard_e(neu + raw, neu - raw, neu, frame.e, frame.u)
    return plus - neu


def _cloud(frame: Frame, delta: torch.Tensor) -> torch.Tensor:
    return frame.neu() + delta + _OFFSETS.unsqueeze(1) * frame.u


class _SliderCritic(nn.Module):
    """Frozen linear critic along û. Slope is 1, so locked b_cap is quiet."""

    def __init__(self, direction: torch.Tensor) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lin.weight.copy_(direction.reshape(1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


def _kappa_probe(reg_cls: type) -> float:
    """Slope 0.4 is under κ=1 and over κ=0.2. A center hardcoded at 1 scores 0."""
    critic = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[0.4, 0.0]]))
    batch = torch.zeros(4, 2)
    probed = reg_cls(
        arm="b_cap", coeff=1.0, kappa=0.2, norm="l2",
        lazy_k=1, target_anneal="none",
    )
    got = probed.penalty(critic, batch, batch)[0]
    faithful = GradientPenalty(
        arm="b_cap", coeff=1.0, kappa=0.2, norm="l2",
    ).penalty(critic, batch, batch)[0]
    return abs(float(got.detach()) - float(faithful.detach()))


def _make_regularizer(kind: str) -> GradientPenalty:
    if kind == "faithful":
        cls = GradientPenalty
    elif kind == "thinned":
        cls = ThinnedKappaCap
    else:
        raise ValueError(f"unknown regularizer kind {kind!r}")
    return cls(
        arm="b_cap", coeff=1.0, kappa=1.0, norm="l2",
        lazy_k=1, target_anneal="none",
    )


@dataclass(frozen=True)
class LiftArm:
    """One board row. ``placement`` is the only geometric switch."""

    name: str
    placement: str
    frame: str = "tilted"
    gan_mode: str = "rp"
    pairing: str = "pair"
    fm_weight: float = FM_WEIGHT
    regularizer: str = "faithful"

    def __post_init__(self) -> None:
        if self.placement not in ("lift", "naive"):
            raise ValueError(f"placement must be 'lift' or 'naive', got {self.placement!r}")
        if self.frame not in ("plane", "tilted"):
            raise ValueError(f"frame must be 'plane' or 'tilted', got {self.frame!r}")
        if self.pairing not in ("pair", "stranger"):
            raise ValueError(f"pairing must be 'pair' or 'stranger', got {self.pairing!r}")
        if self.regularizer not in ("faithful", "thinned"):
            raise ValueError(f"unknown regularizer {self.regularizer!r}")


def _frame_of(arm: LiftArm) -> Frame:
    if arm.frame == "plane":
        return plane_frame()
    return lifted_frame()


def _place(arm: LiftArm, frame: Frame) -> torch.Tensor:
    if arm.placement == "naive":
        return naive_copy()
    return pushforward(frame)


def _geometry(delta: torch.Tensor, frame: Frame, teacher: torch.Tensor) -> dict[str, float]:
    on_u = float(delta @ frame.u)
    on_c = float(delta @ frame.content)
    on_e = float(delta @ frame.e)
    u_kept = on_u / SLIDER
    content_kept = on_c / CONTENT
    leak_ratio = abs(on_e) / (abs(on_u) + 1e-8)
    neu = frame.neu()
    target = neu + teacher
    pole_rel = float((neu + delta - target).norm() / target.norm().clamp_min(1e-8))
    bipolar = leftover_bipolar(delta, -delta)
    return {
        "u_kept": u_kept,
        "content_kept": content_kept,
        "leak_ratio": leak_ratio,
        "on_u": on_u,
        "on_content": on_c,
        "on_e": on_e,
        "pole_rel_err": pole_rel,
        "same_dir": float(bipolar["same_dir"]),
        "leak_frac": float(bipolar["leak_frac"]),
        "lyric_abs": abs(float(delta @ frame.lyric)),
    }


def _geometry_reasons(metrics: dict[str, float]) -> list[str]:
    reasons = []
    leaked = metrics["leak_ratio"] > LEAK_RATIO_MAX or metrics["pole_rel_err"] > POLE_REL_ERR_MAX
    if leaked:
        reasons.append("naive_lift")
    if metrics["u_kept"] < U_KEPT_MIN:
        reasons.append("undershoot")
    if metrics["content_kept"] < CONTENT_KEPT_MIN:
        reasons.append("content")
    if metrics["same_dir"] > SAME_DIR_MAX:
        reasons.append("even_leftover")
    if metrics["lyric_abs"] > 1e-6:
        reasons.append("lyric_leak")
    return reasons


def evaluate(arm: LiftArm) -> dict:
    """Score one arm. Locked lift matches the plane gates; the paste does not."""
    frame = _frame_of(arm)
    teacher = guarded_odd(frame)
    delta = _place(arm, frame)
    metrics = _geometry(delta, frame, teacher)

    real = _cloud(frame, teacher)
    fake = _cloud(frame, delta)
    critic = _SliderCritic(frame.u)
    weight_before = critic.lin.weight.detach().clone()
    real_logits = critic(real)
    fake_logits = critic(fake)
    paired_fake = fake_logits.flip(0) if arm.pairing == "stranger" else fake_logits

    locked_loss = GANLoss("logistic", "rp")
    arm_loss = GANLoss("logistic", arm.gan_mode)
    ref_d = locked_loss.d_loss(real_logits, fake_logits)
    arm_d = arm_loss.d_loss(real_logits, paired_fake)
    ref_g = locked_loss.g_loss(fake_logits, real_logits)
    arm_g = arm_loss.g_loss(paired_fake, real_logits)

    naive_cloud = _cloud(frame, naive_copy())
    fm_raw = feature_match(fake, naive_cloud)
    fm_term = float(arm.fm_weight) * fm_raw
    # Both poles. The teacher residual has cover 0; a pasted residual does not.
    cover = (delta - teacher).pow(2).mean() + ((-delta) - (-teacher)).pow(2).mean()
    particle_term = PARTICLE_L2 * _OFFSETS.pow(2).mean()
    ref_locked = ref_g + particle_term
    arm_total = arm_g + particle_term + COVER_WEIGHT * cover + fm_term

    reg = _make_regularizer(arm.regularizer)
    penalty, _stats = reg.penalty(critic, real, fake, step=1)
    ref_pen = GradientPenalty(
        arm="b_cap", coeff=1.0, kappa=1.0, norm="l2",
    ).penalty(critic, real, fake, step=1)[0]
    probe = _kappa_probe(type(reg))

    adv_err = abs(float(arm_d.detach()) - float(ref_d.detach()))
    # g is the locked objective on this arm's matched pair (FM off, cover of
    # the teacher residual). A paste, a stranger pair, or an FM term moves it.
    g_err = abs(float(arm_total.detach()) - float(ref_locked.detach()))
    cap_err = abs(float(penalty.detach()) - float(ref_pen.detach()))
    critic_frozen = bool(torch.equal(critic.lin.weight, weight_before))

    reasons = _geometry_reasons(metrics)
    if arm.pairing != "pair" or arm.gan_mode != "rp":
        reasons.append("stranger_pairing")
    if float(arm.fm_weight) != FM_WEIGHT:
        reasons.append("fm_on")
    if arm.regularizer != "faithful" or probe > 1e-5:
        reasons.append("thinned_kappa")
    if not critic_frozen:
        reasons.append("critic_moved")

    passed = not reasons
    row = {
        "arm": arm.name,
        "placement": arm.placement,
        "frame": frame.name,
        "angle": frame.angle,
        "pass": passed,
        "verdict": "PASS" if passed else "FAIL",
        "reasons": tuple(reasons),
        "fail_reasons": ",".join(reasons),
        "gan_mode": arm.gan_mode,
        "pairing": arm.pairing,
        "fm_weight": float(arm.fm_weight),
        "cover_weight": COVER_WEIGHT,
        "n_particles": N_PARTICLES,
        "particle_l2": PARTICLE_L2,
        "reg_class": "GradientPenalty" if type(reg) is GradientPenalty else type(reg).__name__,
        "d_loss": float(arm_d.detach()),
        "adv_abs_err": adv_err,
        "g_abs_err": g_err,
        "cap_abs_err": cap_err,
        "kappa_probe_abs_err": probe,
        "fm_term": float(fm_term.detach()),
        "cover_mse": float(cover.detach()),
        "particle_l2_term": float(particle_term.detach()),
        "critic_frozen": critic_frozen,
        "correction_norm": float(lift_correction(frame).norm()),
        **metrics,
    }
    return row


def board_arms() -> list[LiftArm]:
    """Plane pass, locked lift pass, then the four named failures."""
    return [
        LiftArm("locked_2d", "lift", frame="plane"),
        LiftArm("locked_lift", "lift", frame="tilted"),
        LiftArm("naive_copy", "naive", frame="tilted"),
        LiftArm("stranger_pairing", "lift", frame="tilted", pairing="stranger"),
        LiftArm("fm_on", "lift", frame="tilted", fm_weight=0.1),
        LiftArm("thinned_kappa", "lift", frame="tilted", regularizer="thinned"),
    ]


def run_board() -> list[dict]:
    rows = [evaluate(arm) for arm in board_arms()]
    for row in rows:
        print(_line(row), flush=True)
    return rows


def _line(row: dict) -> str:
    why = row["fail_reasons"] or "-"
    return (
        "field_lift arm=%s verdict=%s frame=%s u_kept=%.6f content=%.6f "
        "leak=%.6f pole=%.6f same_dir=%.6f adv_err=%.6e g_err=%.6e "
        "cap_err=%.6e kappa_probe=%.6e fm_term=%.6e reasons=%s"
        % (
            row["arm"],
            row["verdict"],
            row["frame"],
            row["u_kept"],
            row["content_kept"],
            row["leak_ratio"],
            row["pole_rel_err"],
            row["same_dir"],
            row["adv_abs_err"],
            row["g_abs_err"],
            row["cap_abs_err"],
            row["kappa_probe_abs_err"],
            row["fm_term"],
            why,
        )
    )


def format_board(rows: list[dict]) -> str:
    header = (
        "| arm | frame | u_kept | content | leak | pole_err | adv err | κ probe | fm term | gate |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        why = row["fail_reasons"] if row["verdict"] == "FAIL" else "2D recipe holds"
        lines.append(
            "| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3e | %.3f | %.3e | %s %s |"
            % (
                row["arm"],
                row["frame"],
                row["u_kept"],
                row["content_kept"],
                row["leak_ratio"],
                row["pole_rel_err"],
                row["adv_abs_err"],
                row["kappa_probe_abs_err"],
                row["fm_term"],
                row["verdict"],
                why,
            )
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    rows = run_board()
    print(format_board(rows), end="")
    n_pass = sum(1 for row in rows if row["pass"])
    print("field_lift family=2d_to_3d pass=%d/%d" % (n_pass, len(rows)), flush=True)


if __name__ == "__main__":
    _main()
