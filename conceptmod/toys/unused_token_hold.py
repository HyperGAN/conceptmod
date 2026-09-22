"""CPU unused-token / UNI hold for the image-slider posture.

An unused embed slot stays on ``encode(neu)`` while the concept slot moves.
That is the image analog of Music 3 lyric hold, reviewed against
HyperGAN/particle-sliders before this gate existed:

- ``docs/anima-slider.md`` — UNI on the neu/infer caption. Unused yaml
  attributes are pins for unused-token hold, not caption prefixes.
  Concept words are not held. ``--lm_target`` stays the Anima card
  (trajectory); Music 3 stays ``v9``. This toy does not train either.
- ``conceptmod/textsliders/anima_slider.py`` — ``anima_unused_hold_loss``
  and ``align_unused_positions``. Matched pairs pin the unused index to
  the neu unused index. A stranger partner points that index at the
  concept slot.
- ``docs/zimage-slider.md`` — frozen-embed unused-token hold is off on
  ZiT because those encoder embeds have no grad into LoRA. This toy is
  the posture where the slot *does* take a hold loss.
- ``analysis/slider2d/locked_baseline_defaults.py`` — locked_shared / #94.

The student writes one shared residual into every slot, plus a per-slot
correction. Concept RpGAN sees only the concept slot, so the shared write
and that slot's correction get the same gradient. Without a hold loss,
half of the concept step lands on the unused slot and the pin is trashed.
The hold loss (Anima ``hold_weight=1``) cancels that write on the unused
index only. Concept words stay out of the mask.

Locked shape that must PASS
---------------------------
* RpGAN logistic relativistic pair (``particlegan.GANLoss``).
* ParticleGAN ``GradientPenalty`` (``GradRegularizer``) ``b_cap``,
  coeff=1, kappa=1, norm=l2, lazy=1, anneal=none.
* ``fm_weight=0``.
* Adam betas ``(0, 0.99)``, shared LR ``5e-3``. Critic hidden 64.

Recorded, not a second train loss (same honesty as mode-hold's cover pin):

* demo cover ``1.5`` (Music 1.0 is a different posture)
* ``n_particles=12``, ``particle_l2=0.02`` — no particle cloud. This is
  an embed-slot student, not the Music cloud and not Hub 128.

A CPU PASS is not an Anima, Supra, or Music GPU transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import GANLoss, GradientPenalty


# Image-slider gates. Same 0.85 floor as the unipolar neu_hold / cover board.
UNUSED_HOLD_MIN = 0.85
CONCEPT_MOVE_MIN = 0.85

DIM = 2
N_SLOTS = 2
UNUSED = 0
CONCEPT = 1
N_ROWS = 8
LR = 5e-3
BETAS = (0.0, 0.99)
CRITIC_HIDDEN = 64
STEPS = 200
# Anima image-slider default (``DEFAULT_HOLD_WEIGHT``). Not Field3D cover.
HOLD_WEIGHT = 1.0
DEMO_COVER = 1.5

# Slot 0 is the subject/composition pin. Slot 1 is the concept direction.
NEU = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
CONCEPT_DIR = torch.tensor([0.0, 1.0])


@dataclass(frozen=True)
class UnusedHoldRecipe:
    """locked_shared card plus the unused-token hold switch.

    ``steps`` and ``seed`` are budget. ``hold_weight=0`` and
    ``pairing='stranger'`` are named drifts, not silent aliases.
    """

    name: str = "locked_shared"
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    pairing: str = "matched"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    reg_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cover_weight: float = DEMO_COVER
    n_particles: int = 12
    particle_l2: float = 0.02
    hold_weight: float = HOLD_WEIGHT
    steps: int = STEPS
    seed: int = 0

    def replace(self, **overrides) -> "UnusedHoldRecipe":
        return replace(self, **overrides)


def locked_recipe(**overrides) -> UnusedHoldRecipe:
    return UnusedHoldRecipe(**overrides)


class ThinnedBCap(GradientPenalty):
    """Advertises locked b_cap. The cap center is hardcoded at 0.

    At kappa=1 a faithful cap centers on 1. This stub does not.
    """

    def __init__(self) -> None:
        super().__init__(
            arm="b_cap",
            coeff=1.0,
            kappa=1.0,
            norm="l2",
            lazy_k=1,
            target_anneal="none",
        )

    def center(self, step: int = 0) -> float:
        return 0.0


def hold_pairs(pairing: str) -> list[tuple[int, int]]:
    """Unused-token partners. Concept index is never the prediction slot.

    ``matched`` pins unused → encode(neu) at the unused index.
    ``stranger`` pins unused → the concept slot (the wrong neu partner).
    """
    if pairing == "matched":
        return [(UNUSED, UNUSED)]
    if pairing == "stranger":
        return [(UNUSED, CONCEPT)]
    raise ValueError(f"pairing must be 'matched' or 'stranger', got {pairing!r}")


def unused_hold_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    """Masked MSE of unused positions onto the partner embed.

    ``pred`` / ``tgt`` are ``(T, D)``. Empty pairs contribute 0 — there is
    nothing to pin, which is the Anima fail-closed empty alignment.
    """
    if not pairs:
        return pred.reshape(-1)[:1].sum() * 0.0
    pred_idx = [i for i, _ in pairs]
    tgt_idx = [j for _, j in pairs]
    return F.mse_loss(pred[pred_idx], tgt[tgt_idx])


class SharedSlotStudent(nn.Module):
    """One residual added to every slot, plus a per-slot correction.

    Concept loss does not see the unused slot. The shared vector still
    moves it, unless the hold loss trains the unused correction to cancel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(DIM))
        self.slot = nn.Parameter(torch.zeros(N_SLOTS, DIM))
        self.register_buffer("neu", NEU.clone())

    def embeds(self, scale: float) -> torch.Tensor:
        # Scale 0 is the adapter-off identity (Anima UNI scale 0).
        return self.neu + float(scale) * (self.shared + self.slot)


class SlotCritic(nn.Module):
    """Two-layer LeakyReLU MLP on the concept slot. Hidden 64, locked card."""

    def __init__(self, hidden: int = CRITIC_HIDDEN) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(DIM, hidden), nn.LeakyReLU(0.2))
        self.out = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(z)).squeeze(-1)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        return self.hidden(z)


def _batch(vector: torch.Tensor, rows: int = N_ROWS) -> torch.Tensor:
    return vector.detach().unsqueeze(0).expand(rows, -1)


def score_student(student: SharedSlotStudent) -> dict[str, float]:
    """Unused hold and concept move at scale +1. Scale 0 is identity."""
    embeds = student.embeds(1.0).detach()
    unused = embeds[UNUSED]
    concept = embeds[CONCEPT]
    pin = student.neu[UNUSED]
    origin = student.neu[CONCEPT]
    dist = float((unused - pin).norm())
    target_norm = float(CONCEPT_DIR.norm())
    hold = 1.0 - min(1.0, dist / target_norm)
    delta = concept - origin
    delta_norm = float(delta.norm())
    if delta_norm <= 1e-8:
        cos = 0.0
    else:
        cos = float(F.cosine_similarity(delta.unsqueeze(0), CONCEPT_DIR.unsqueeze(0)))
    mag = max(0.0, 1.0 - abs(delta_norm / target_norm - 1.0))
    move = max(0.0, cos) * mag
    scale0 = student.embeds(0.0).detach()
    return {
        "unused_hold": hold,
        "concept_move": move,
        "unused_dist": dist,
        "concept_cos": cos,
        "concept_delta_norm": delta_norm,
        "scale0_err": float((scale0 - student.neu).norm()),
    }


def bcap_probe(reg: GradientPenalty) -> tuple[float, float, bool]:
    """Faithful b_cap on ||grad|| = 3 is 4. A center-0 stub is 9."""
    critic = nn.Linear(DIM, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 0.0]]))
    batch = torch.zeros(4, DIM)
    champion = GradientPenalty(
        arm="b_cap",
        coeff=1.0,
        kappa=1.0,
        norm="l2",
        lazy_k=1,
        target_anneal="none",
    )
    ref, ref_stats = champion.penalty(critic, batch, batch, step=1)
    try:
        got, stats = reg.penalty(critic, batch, batch, step=1)
    except Exception:
        return float("nan"), float(ref.detach()), False
    same = (
        type(reg) is GradientPenalty
        and torch.is_tensor(got)
        and torch.allclose(got, ref)
        and stats.get("center") == ref_stats.get("center")
        and bool(stats.get("applied", False))
    )
    return float(got.detach()), float(ref.detach()), bool(same)


def formulation_reasons(recipe: UnusedHoldRecipe, *, reg_faithful: bool) -> list[str]:
    """Drift from the locked card. Hold weight 0 is the named no-hold arm."""
    reasons: list[str] = []
    if recipe.loss_type != "logistic":
        reasons.append("loss_not_logistic")
    if recipe.gan_mode != "rp" or recipe.pairing != "matched":
        reasons.append("stranger_pairing")
    if float(recipe.fm_weight) != 0.0:
        reasons.append("fm_on")
    reg_locked = (
        recipe.reg_arm == "b_cap"
        and float(recipe.reg_coeff) == 1.0
        and float(recipe.reg_kappa) == 1.0
        and recipe.reg_norm == "l2"
        and int(recipe.reg_lazy) == 1
        and recipe.target_anneal == "none"
    )
    if not reg_locked or not reg_faithful:
        reasons.append("thinned_bcap")
    if float(recipe.cover_weight) != DEMO_COVER:
        reasons.append("cover_weight")
    if int(recipe.n_particles) != 12 or float(recipe.particle_l2) != 0.02:
        reasons.append("particle_cloud")
    if float(recipe.hold_weight) != HOLD_WEIGHT:
        reasons.append("no_hold")
    return reasons


def gate_reasons(
    recipe: UnusedHoldRecipe,
    metrics: dict[str, float],
    *,
    reg_faithful: bool,
) -> list[str]:
    reasons = formulation_reasons(recipe, reg_faithful=reg_faithful)
    if metrics["unused_hold"] < UNUSED_HOLD_MIN:
        reasons.append("unused_trashed")
    if metrics["concept_move"] < CONCEPT_MOVE_MIN:
        reasons.append("concept_move")
    if metrics["scale0_err"] > 1e-6:
        reasons.append("scale0_identity")
    return list(dict.fromkeys(reasons))


def _log(name: str, step: int, metrics: dict[str, float], verdict: str | None = None) -> None:
    tail = f" verdict={verdict}" if verdict else ""
    print(
        f"unused-token arm={name} step={step} "
        f"hold={metrics['unused_hold']:.4f} concept={metrics['concept_move']:.4f} "
        f"unused_dist={metrics['unused_dist']:.4f}{tail}",
        flush=True,
    )


def _make_regularizer(recipe: UnusedHoldRecipe) -> GradientPenalty:
    return GradientPenalty(
        arm=recipe.reg_arm,
        coeff=recipe.reg_coeff,
        kappa=recipe.reg_kappa,
        norm=recipe.reg_norm,
        lazy_k=recipe.reg_lazy,
        target_anneal=recipe.target_anneal,
    )


def train(recipe: UnusedHoldRecipe, regularizer: GradientPenalty | None = None) -> dict:
    """Fit one arm. Prints a tailable line at the checkpoints."""
    torch.manual_seed(int(recipe.seed))
    student = SharedSlotStudent()
    critic = SlotCritic()
    gan = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    reg = regularizer if regularizer is not None else _make_regularizer(recipe)
    opt_g = torch.optim.Adam(student.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    real = _batch(CONCEPT_DIR)
    pairs = hold_pairs(recipe.pairing)
    bcap_applied = 0
    for step in range(int(recipe.steps)):
        fake = student.embeds(1.0)[CONCEPT].unsqueeze(0).expand(N_ROWS, -1).detach()
        opt_d.zero_grad(set_to_none=True)
        penalty, stats = reg.penalty(critic, real, fake, step=step + 1)
        if stats.get("applied"):
            bcap_applied += 1
        d_loss = gan.d_loss(critic(real), critic(fake)) + penalty
        d_loss.backward()
        opt_d.step()

        critic.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        fake_g = student.embeds(1.0)[CONCEPT].unsqueeze(0).expand(N_ROWS, -1)
        g_loss = gan.g_loss(critic(fake_g), critic(real).detach())
        # Demo cover and the n=12 cloud are recorded on the card and are
        # not added here. The train pin is the unused-token hold.
        if float(recipe.fm_weight) != 0.0:
            real_feat = critic.features(real).detach().mean(0)
            fake_feat = critic.features(fake_g).mean(0)
            g_loss = g_loss + float(recipe.fm_weight) * (real_feat - fake_feat).pow(2).mean()
        loss = g_loss
        if float(recipe.hold_weight) != 0.0:
            embeds = student.embeds(1.0)
            loss = loss + float(recipe.hold_weight) * unused_hold_loss(embeds, student.neu, pairs)
        loss.backward()
        critic.requires_grad_(True)
        opt_g.step()

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(recipe.steps):
            _log(recipe.name, step + 1, score_student(student))

    metrics = score_student(student)
    probe, faithful, reg_ok = bcap_probe(reg)
    reasons = gate_reasons(recipe, metrics, reg_faithful=reg_ok)
    if bcap_applied != int(recipe.steps):
        reasons.append("bcap_skipped")
        reasons = list(dict.fromkeys(reasons))
    passed = not reasons
    row = {
        "name": recipe.name,
        "recipe": recipe,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "reasons": tuple(reasons),
        "bcap_applied": bcap_applied,
        "bcap_probe": probe,
        "bcap_faithful": faithful,
        "reg_faithful": reg_ok,
        "hold_weight": float(recipe.hold_weight),
        "fm_weight": float(recipe.fm_weight),
        "gan_mode": recipe.gan_mode,
        "pairing": recipe.pairing,
        "loss_type": recipe.loss_type,
        "cover_weight": float(recipe.cover_weight),
        "n_particles": int(recipe.n_particles),
        "particle_l2": float(recipe.particle_l2),
        **metrics,
    }
    why = ",".join(reasons) if reasons else "-"
    print(
        f"unused-token arm={recipe.name} verdict={row['verdict']} "
        f"hold={metrics['unused_hold']:.4f} concept={metrics['concept_move']:.4f} "
        f"reasons={why}",
        flush=True,
    )
    return row


def family_arms() -> list[tuple[UnusedHoldRecipe, GradientPenalty | None]]:
    """Locked pass, the hold ablation, and the locked-shape refusals."""
    return [
        (locked_recipe(name="locked_shared"), None),
        (locked_recipe(name="no_hold", hold_weight=0.0), None),
        (locked_recipe(name="stranger_pairing", pairing="stranger"), None),
        (locked_recipe(name="stranger_vanilla", gan_mode="vanilla"), None),
        (locked_recipe(name="fm_on", fm_weight=1.0), None),
        (locked_recipe(name="thin_bcap"), ThinnedBCap()),
    ]


def run_family() -> list[dict]:
    rows = []
    for recipe, regularizer in family_arms():
        rows.append(train(recipe, regularizer))
    n_pass = sum(row["passed"] for row in rows)
    print(
        f"unused-token family=uni_hold pass={n_pass}/{len(rows)}",
        flush=True,
    )
    return rows


def main() -> None:
    torch.set_num_threads(1)
    print("unused-token family=uni_hold device=cpu", flush=True)
    run_family()


if __name__ == "__main__":
    main()
