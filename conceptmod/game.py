"""Compile a loss phrase into a two-player GAN game on the 2-D CPU fixture.

The phrase is the student's concept objective (:func:`conceptmod.ops.rule_loss`).
The critic is the locked_shared adversary: ``particlegan`` ``GANLoss`` in
RpGAN logistic mode, plus ``GradRegularizer`` ``b_cap``. The stamp is
:func:`conceptmod.toys.locked_adv_defaults`. This module does not restate
that recipe and does not sample the toys' particle cloud.

A formulation ``PASS`` means the step used that wiring. It is not a 2-D
geometric verdict (those stay in ``docs/dsl.md``) and it is not a Music
or Anima GPU transfer.

Negative controls (stranger pairing, FM-on, thinned kappa) are refused
when the game claims ``locked``. Built with ``claim="drift"`` they run
one step and score ``FAIL``.

Not a multi-player language. The stanza keys that work are listed in
``docs/dsl/SPEC.md``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer

from conceptmod import dsl, ops
from conceptmod.analysis_2d import (
    DEFAULT_LR,
    DEFAULT_SEED,
    PROBE_PROMPTS,
    TwoAxisBackend,
    _cfg,
    _pin_bare_templates,
    _probe_zt,
    snapshot,
)
from conceptmod import toys
from conceptmod.toys.locked_shared_floor import LOCKED, feature_match


NOT_A_TRANSFER = (
    "CPU toy on the 2-D fixture. Formulation PASS is locked_shared wiring, "
    "not a 2-D geometric verdict and not a Music or Anima GPU transfer."
)

_STANZA_KEYS = frozenset({
    "phrase", "pairing", "claim", "fm", "kappa", "adv", "seed",
})
_PAIRINGS = frozenset({"locked_shared", "stranger"})
_CLAIMS = frozenset({"locked", "drift"})


class GameError(ValueError):
    """The stanza or phrase cannot be built into a CPU game."""


class LockedClaimError(GameError):
    """A drifted knob was claimed as locked_shared."""


@dataclass(frozen=True)
class Player:
    """One side of the game. ``objective`` is what ``step`` minimizes."""

    name: str
    role: str
    objective: str


@dataclass
class GameStep:
    """Tensors from one critic update and one student update."""

    step: int
    phrase_loss: float
    d_loss: float
    g_loss: float
    penalty: float
    fm_term: float
    d_real: torch.Tensor
    d_fake: torch.Tensor
    g_real: torch.Tensor
    g_fake: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor
    critic_snap: nn.Module
    probe: object


class PlaneCritic(nn.Module):
    """Host critic on the 2-D CFG plane. Not a formulation recipe."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2))
        self.out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(x)).squeeze(-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


def _content_lines(source: str) -> list[str]:
    """Drop blank lines and comments.

    ``#`` is also the freeze operator (``stripe#stripe``). A comment is a
    line that starts with ``#``, or the tail of a line after `` #``
    (hash preceded by whitespace). Write freeze without spaces around
    ``#`` inside a stanza.
    """
    lines = []
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        comment = stripped.find(" #")
        if comment != -1:
            stripped = stripped[:comment].strip()
        if stripped:
            lines.append(stripped)
    return lines


def _split_key(line: str) -> tuple[str | None, str]:
    parts = line.split(None, 1)
    head = parts[0].lower().rstrip(":")
    if head not in _STANZA_KEYS:
        return None, line
    rest = parts[1].strip() if len(parts) > 1 else ""
    if rest.startswith(":"):
        rest = rest[1:].strip()
    return head, rest


def parse_game(source: str) -> dict:
    """Parse a one-line phrase or a small game stanza. Does not build modules.

    A stanza line is ``key value`` or ``key: value``. A line that starts
    with ``#``, and the tail of a line after `` #``, are comments. Freeze
    inside a phrase is written without spaces (``stripe#stripe``) so it
    is not read as a comment. Keys: ``phrase``, ``pairing``, ``claim``,
    ``fm``, ``kappa``, ``adv``, ``seed``. Anything else is an error —
    including keys this module does not implement.
    """
    lines = _content_lines(source)
    if not lines:
        raise GameError("empty game")
    keyed = [_split_key(line)[0] for line in lines]
    if all(key is None for key in keyed):
        if len(lines) != 1:
            raise GameError(
                "a bare phrase is one line; put it after 'phrase' in a stanza"
            )
        return {"phrase": lines[0]}
    spec: dict = {}
    for line in lines:
        key, rest = _split_key(line)
        if key is None:
            raise GameError(
                f"stanza line {line!r} is not a key "
                f"(expected one of {sorted(_STANZA_KEYS)})"
            )
        if key in spec:
            raise GameError(f"duplicate game key {key!r}")
        if not rest:
            raise GameError(f"game key {key!r} needs a value")
        spec[key] = rest
    if "phrase" not in spec:
        raise GameError("game stanza needs a phrase line")
    return _coerce_spec(spec)


def _coerce_spec(spec: dict) -> dict:
    out: dict = {"phrase": spec["phrase"]}
    if "pairing" in spec:
        out["pairing"] = spec["pairing"]
    if "claim" in spec:
        out["claim"] = spec["claim"]
    if "adv" in spec:
        out["adv"] = spec["adv"]
    if "fm" in spec:
        out["fm_weight"] = _parse_float(spec["fm"], "fm")
    if "kappa" in spec:
        out["kappa"] = _parse_float(spec["kappa"], "kappa")
    if "seed" in spec:
        try:
            out["seed"] = int(spec["seed"])
        except ValueError as exc:
            raise GameError(f"bad seed {spec['seed']!r}") from exc
    return out


def _parse_float(text: str, name: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise GameError(f"bad {name} {text!r}") from exc


def _refuse_locked_drift(*, pairing: str, fm_weight: float, kappa: float, stamp: dict) -> None:
    """Refuse stranger / FM-on / thinned kappa when the caller claims locked.

    FM and kappa go through :func:`conceptmod.toys.reject_unlocked` so the
    wording stays the formulation toy's. Stranger pairing here is the
    flipped relativistic batch (``GANLoss`` stays ``mode='rp'``), which
    that helper expresses as a ``gan_mode`` drift — so the pairing check
    is local and names the flip.
    """
    if pairing != "locked_shared":
        raise LockedClaimError(
            "stranger pairing refused: locked_shared pairs each frozen "
            f"probe with the trained probe in the same order, got pairing={pairing!r}"
        )
    if fm_weight != stamp["fm_weight"]:
        try:
            toys.reject_unlocked({"fm_weight": fm_weight})
        except ValueError as exc:
            raise LockedClaimError(str(exc)) from exc
    if kappa != stamp["reg_kappa"]:
        try:
            toys.reject_unlocked({"reg_kappa": kappa})
        except ValueError as exc:
            raise LockedClaimError(str(exc)) from exc


def loss_game(
    phrase: str,
    *,
    adv: str = "locked_shared",
    pairing: str = "locked_shared",
    fm_weight: float = 0.0,
    kappa: float | None = None,
    claim: str = "locked",
    seed: int = DEFAULT_SEED,
    random_prompt: str | None = None,
) -> "Game":
    """Build a game whose student objective is ``phrase``.

    ``adv`` must be ``"locked_shared"``. There is no second recipe to
    select. ``claim="locked"`` (the default) refuses stranger pairing,
    a non-zero ``fm_weight``, and a ``kappa`` other than the stamp.
    ``claim="drift"`` runs that negative control and scores ``FAIL``.
    """
    if adv != "locked_shared":
        raise LockedClaimError(
            f"only adv='locked_shared' is defined (from locked_adv_defaults), got {adv!r}"
        )
    if pairing not in _PAIRINGS:
        raise GameError(
            f"unknown pairing {pairing!r} (expected one of {sorted(_PAIRINGS)})"
        )
    if claim not in _CLAIMS:
        raise GameError(f"unknown claim {claim!r} (expected one of {sorted(_CLAIMS)})")
    stamp = toys.locked_adv_defaults()
    applied_kappa = float(stamp["reg_kappa"] if kappa is None else kappa)
    fm_weight = float(fm_weight)
    drifted = (
        pairing != "locked_shared"
        or fm_weight != float(stamp["fm_weight"])
        or applied_kappa != float(stamp["reg_kappa"])
    )
    if claim == "locked":
        _refuse_locked_drift(
            pairing=pairing,
            fm_weight=fm_weight,
            kappa=applied_kappa,
            stamp=stamp,
        )
    elif not drifted:
        raise GameError(
            "claim drift needs a named drift: pairing stranger, "
            "fm_weight != 0, or kappa != locked reg_kappa"
        )
    rules = dsl.parse_phrase(phrase)
    if any(rule.needs_random_prompt for rule in rules):
        if random_prompt is None:
            raise dsl.DSLError(
                "phrase needs {random_prompt}; pass random_prompt= to loss_game"
            )
        rules = dsl.materialize(rules, random_prompt)
    for rule in rules:
        if rule.op == dsl.PIXEL:
            raise GameError(
                "pixel '^' has no render on TwoAxisBackend; "
                "the CPU game does not run it (see docs/dsl/operators/pixel.md)"
            )
        if rule.op == dsl.REWARD:
            raise GameError(
                "';' ImageReward is parsed and not implemented "
                "(see docs/dsl/operators/reward.md)"
            )
    return Game(
        phrase=phrase,
        rules=rules,
        stamp=stamp,
        pairing=pairing,
        fm_weight=fm_weight,
        kappa=applied_kappa,
        claimed_locked=claim == "locked",
        seed=seed,
    )


def game(source: str, **overrides) -> "Game":
    """Build a game from a phrase or a stanza. Keyword arguments override the stanza.

    .. code-block:: python

        game("red=blue")
        game('''
        phrase red=blue
        pairing locked_shared
        claim locked
        ''')
    """
    spec = parse_game(source)
    for key, value in overrides.items():
        if value is not None:
            spec[key] = value
    phrase = spec.pop("phrase")
    return loss_game(phrase, **spec)


class Game:
    """Student (phrase rules) versus critic (RpGAN + ``b_cap``) on ``TwoAxisBackend``."""

    def __init__(
        self,
        *,
        phrase: str,
        rules: list[dsl.Rule],
        stamp: dict,
        pairing: str,
        fm_weight: float,
        kappa: float,
        claimed_locked: bool,
        seed: int,
    ) -> None:
        self.phrase = phrase
        self.rules = list(rules)
        self.stamp = dict(stamp)
        self.pairing = pairing
        self.fm_weight = float(fm_weight)
        self.kappa = float(kappa)
        self.claimed_locked = claimed_locked
        self.seed = int(seed)
        self.steps_run = 0
        self.history: list[GameStep] = []

        torch.manual_seed(self.seed)
        self.backend = TwoAxisBackend(seed=self.seed)
        self._z, self._t = _probe_zt(self.backend, self.seed)
        self.critic = PlaneCritic()
        self.gan = GANLoss(
            loss_type=self.stamp["loss_type"],
            mode=self.stamp["gan_mode"],
        )
        self.regularizer = GradRegularizer(
            arm=self.stamp["reg_arm"],
            coeff=float(self.stamp["reg_coeff"]),
            kappa=self.kappa,
            norm=self.stamp["reg_norm"],
            lazy_k=int(self.stamp["reg_lazy"]),
            target_anneal=self.stamp["target_anneal"],
            total_steps=0,
        )
        self.opt_student = torch.optim.Adam(
            self.backend.trainable_parameters(), lr=DEFAULT_LR,
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LOCKED.lr)
        # Fixture guidance (write 1, erase 0, exaggerate 3). Not a new recipe.
        self._op_cfg = _cfg()

    def players(self) -> tuple[Player, Player]:
        student_obj = "phrase rule_loss + RpGAN g_loss"
        if self.fm_weight != 0.0:
            student_obj += " + fm_weight * feature_match"
        return (
            Player("student", "generator", student_obj),
            Player("critic", "critic", "RpGAN d_loss + GradRegularizer b_cap"),
        )

    def __str__(self) -> str:
        claim = "locked" if self.claimed_locked else "drift"
        stamp = self.stamp
        lines = [
            f"Game  phrase={self.phrase!r}  claim={claim}  pairing={self.pairing}",
            "fixture  TwoAxisBackend  color=red/blue  pattern=stripe/dot",
            (
                "stamp  "
                f"loss={stamp['loss_type']} mode={stamp['gan_mode']} "
                f"arm={stamp['reg_arm']} coeff={float(stamp['reg_coeff']):g} "
                f"kappa={float(stamp['reg_kappa']):g} "
                f"fm={float(stamp['fm_weight']):g}  source=locked_adv_defaults"
            ),
            (
                "recorded  "
                f"n_particles={stamp['n_particles']} "
                f"cover={float(stamp['cover_weight']):g} "
                f"posture={stamp['cover_posture']}  "
                "(toys own these; this step does not sample a cloud)"
            ),
            (
                f"applied  pairing={self.pairing} "
                f"kappa={float(self.regularizer.kappa):g} fm={self.fm_weight:g}"
            ),
            "player   role       objective",
        ]
        for player in self.players():
            lines.append(f"{player.name:<8} {player.role:<10} {player.objective}")
        lines.append("index  op           a                 b                 alpha")
        for index, rule in enumerate(self.rules):
            lines.append(
                f"{index:<5}  {rule.op:<12} {(rule.a or '∅'):<17} "
                f"{(rule.b or '∅'):<17} {rule.alpha:g}"
            )
        if not self.claimed_locked:
            lines.append(
                "score    FAIL — this build is a named drift, not locked_shared"
            )
        return "\n".join(lines)

    def _align(self, real_logits: torch.Tensor, fake_logits: torch.Tensor):
        if self.pairing == "locked_shared":
            return real_logits, fake_logits
        if self.pairing == "stranger":
            return real_logits, fake_logits.flip(0)
        raise GameError(f"unknown pairing {self.pairing!r}")

    def _plane(self, frozen: bool) -> torch.Tensor:
        rows = []
        color_axis = self.backend.d_color.reshape(-1)
        pattern_axis = self.backend.d_pattern.reshape(-1)
        for prompt in PROBE_PROMPTS:
            velocity = self.backend.predict_v(prompt, self._z, self._t, frozen=frozen)
            uncond = self.backend.predict_v("", self._z, self._t, frozen=True)
            delta = (velocity - uncond).reshape(-1)
            rows.append(torch.stack([
                torch.dot(delta, color_axis),
                torch.dot(delta, pattern_axis),
            ]))
        return torch.stack(rows)

    def step(self) -> GameStep:
        """One critic update, then one student update. The student loss is the phrase plus RpGAN ``g_loss``."""
        restore = _pin_bare_templates()
        try:
            return self._step_pinned()
        finally:
            restore()

    def _step_pinned(self) -> GameStep:
        self.steps_run += 1
        step_index = self.steps_run
        real = self._plane(frozen=True).detach()
        fake_det = self._plane(frozen=False).detach()
        critic_snap = copy.deepcopy(self.critic)

        d_real = self.critic(real)
        d_fake = self.critic(fake_det)
        paired_real, paired_fake = self._align(d_real, d_fake)
        d_adv = self.gan.d_loss(paired_real, paired_fake)
        penalty, _stats = self.regularizer.penalty(
            self.critic, real, fake_det, step=step_index,
        )
        self.opt_critic.zero_grad(set_to_none=True)
        (d_adv + penalty).backward()
        self.opt_critic.step()

        self.opt_student.zero_grad(set_to_none=True)
        ctx = ops.StepContext(
            self.backend,
            stop_index=2,
            seed=self.seed + step_index - 1,
            cfg=self._op_cfg,
        )
        phrase_loss = 0.0
        for rule in self.rules:
            loss = rule.alpha * ops.rule_loss(rule, ctx)
            phrase_loss += float(loss.detach())
            loss.backward()
            ctx._v = {key: value for key, value in ctx._v.items() if not key[3]}

        fake = self._plane(frozen=False)
        g_real = self.critic(real)
        g_fake = self.critic(fake)
        g_paired_real, g_paired_fake = self._align(g_real, g_fake)
        g_adv = self.gan.g_loss(g_paired_fake, g_paired_real)
        student = g_adv
        fm_term = 0.0
        if self.fm_weight != 0.0:
            matched = feature_match(self.critic.features(real), self.critic.features(fake))
            fm_term = float(matched.detach())
            student = student + self.fm_weight * matched
        student.backward()
        torch.nn.utils.clip_grad_norm_(self.backend.trainable_parameters(), 1.0)
        self.opt_student.step()
        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_student.zero_grad(set_to_none=True)

        taken = GameStep(
            step=step_index,
            phrase_loss=phrase_loss,
            d_loss=float(d_adv.detach()),
            g_loss=float(g_adv.detach()),
            penalty=float(penalty.detach()),
            fm_term=fm_term,
            d_real=d_real.detach().clone(),
            d_fake=d_fake.detach().clone(),
            g_real=g_real.detach().clone(),
            g_fake=g_fake.detach().clone(),
            real=real.detach().clone(),
            fake=fake_det.detach().clone(),
            critic_snap=critic_snap,
            probe=snapshot(self.backend, self._z, self._t, step_index),
        )
        self.history.append(taken)
        return taken

    def score(self) -> dict:
        """Formulation row for the latest step. Runs one step if none has."""
        if not self.history:
            self.step()
        taken = self.history[-1]
        mismatches = self._mismatches(taken)
        verdict = "PASS" if self.claimed_locked and not mismatches else "FAIL"
        if not self.claimed_locked and not mismatches:
            mismatches.append("drift build produced no named mismatch")
            verdict = "FAIL"
        probe = taken.probe
        return {
            "phrase": self.phrase,
            "pairing": self.pairing,
            "claim": "locked" if self.claimed_locked else "drift",
            "verdict": verdict,
            "mismatches": mismatches,
            "phrase_loss": taken.phrase_loss,
            "d_loss": taken.d_loss,
            "g_loss": taken.g_loss,
            "penalty": taken.penalty,
            "fm_term": taken.fm_term,
            "adv": toys.locked_adv_defaults(),
            "rules": [
                {
                    "op": rule.op,
                    "a": rule.a,
                    "b": rule.b,
                    "alpha": rule.alpha,
                    "options": dict(rule.options),
                }
                for rule in self.rules
            ],
            "players": [
                {"name": player.name, "role": player.role, "objective": player.objective}
                for player in self.players()
            ],
            "probe": {
                "step": probe.step,
                "color_on_red": probe.color_on_red,
                "pattern_on_stripe": probe.pattern_on_stripe,
                "color_on_stripe": probe.color_on_stripe,
                "pattern_on_red": probe.pattern_on_red,
                "write_cosine": probe.write_cosine,
                "stripe_hold": probe.stripe_hold,
            },
            "step": taken.step,
            "cloud": (
                "not sampled; n_particles and cover_weight stay on "
                "locked_adv_defaults and are not a second cloud"
            ),
            "note": NOT_A_TRANSFER,
        }

    def _mismatches(self, taken: GameStep) -> list[str]:
        bad: list[str] = []
        live = toys.locked_adv_defaults()
        if self.stamp != live:
            bad.append("stamp diverged from locked_adv_defaults()")
        if type(self.gan) is not GANLoss:
            bad.append(f"adversary: {type(self.gan).__name__} is not particlegan.GANLoss")
        elif self.gan.loss_type != live["loss_type"] or self.gan.mode != live["gan_mode"]:
            bad.append(
                f"adversary: GANLoss({self.gan.loss_type!r}, {self.gan.mode!r}) "
                f"!= locked ({live['loss_type']!r}, {live['gan_mode']!r})"
            )
        if type(self.regularizer) is not GradRegularizer:
            bad.append(
                f"regularizer: {type(self.regularizer).__name__} is not GradRegularizer"
            )
        else:
            if self.regularizer.arm != live["reg_arm"]:
                bad.append(
                    f"thinned_kappa: arm {self.regularizer.arm!r} != {live['reg_arm']!r}"
                )
            if float(self.regularizer.kappa) != float(live["reg_kappa"]):
                bad.append(
                    "thinned_kappa: "
                    f"kappa {float(self.regularizer.kappa):g} != locked {float(live['reg_kappa']):g}"
                )
            if float(self.regularizer.coeff) != float(live["reg_coeff"]):
                bad.append(
                    f"thinned_kappa: coeff {self.regularizer.coeff} != {live['reg_coeff']}"
                )
            if self.regularizer.norm != live["reg_norm"]:
                bad.append(
                    f"thinned_kappa: norm {self.regularizer.norm!r} != {live['reg_norm']!r}"
                )
        if self.pairing != "locked_shared":
            bad.append("stranger_pairing")
        fm_locked = float(live["fm_weight"])
        if self.fm_weight != fm_locked or (self.fm_weight == 0.0 and taken.fm_term != 0.0):
            bad.append("fm_on")
        if not math.isfinite(taken.phrase_loss):
            bad.append("phrase loss is not finite")
        if not math.isfinite(taken.d_loss) or not math.isfinite(taken.g_loss):
            bad.append("GAN loss is not finite")
        if not math.isfinite(taken.penalty):
            bad.append("b_cap penalty is not finite")

        reference = GANLoss(loss_type=live["loss_type"], mode=live["gan_mode"])
        want_real, want_fake = self._align(taken.d_real, taken.d_fake)
        d_want = reference.d_loss(want_real, want_fake)
        d_err = abs(float(d_want.detach()) - taken.d_loss)
        if d_err > 1e-5:
            bad.append(f"adv: abs err {d_err:.3e} vs RpGAN logistic on the saved pair")
        g_real, g_fake = self._align(taken.g_real, taken.g_fake)
        g_want = reference.g_loss(g_fake, g_real)
        g_err = abs(float(g_want.detach()) - taken.g_loss)
        if g_err > 1e-5:
            bad.append(f"g: abs err {g_err:.3e} vs RpGAN logistic g_loss")

        reg = GradRegularizer(
            arm=self.regularizer.arm,
            coeff=float(self.regularizer.coeff),
            kappa=float(self.regularizer.kappa),
            norm=self.regularizer.norm,
            lazy_k=int(self.regularizer.lazy_k),
            target_anneal=self.regularizer.target_anneal,
            total_steps=0,
        )
        cap_want, _stats = reg.penalty(
            taken.critic_snap, taken.real, taken.fake, step=taken.step,
        )
        cap_err = abs(float(cap_want.detach()) - taken.penalty)
        if cap_err > 1e-5:
            bad.append(
                f"b_cap: abs err {cap_err:.3e} vs GradRegularizer on the pre-step critic"
            )
        return bad
