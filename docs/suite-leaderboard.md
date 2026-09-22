# Suite leaderboard (cross-toy)

**Module:** `conceptmod/toys/suite_leaderboard.py`

This board answers: **which adv config(s) PASS every applicable CPU toy?**

It is **not** `leaderboard_honesty` (that module is the single-toy two-pole
cloud + `claim_pass` gate). Do not confuse the two.

## Candidates (rows)

| config | meaning |
|---|---|
| `locked_shared` | Demo LOCKED stamp (`locked_adv_defaults` / `LOCKED`): RpGAN logistic, `b_cap` coeff=1 κ=1, FM off, cover **1.5**, n=12, `particle_l2=0.02` |
| `music_cover_1_0` | Cover **1.0** + music posture, with the required drift report |
| `stranger_pair` | Stranger pairing drift |
| `fm_on` | `fm_weight=0.1` |
| `thinned_kappa` | Thinned / κ-hardcoded `b_cap` |
| `vanilla_logistic` | `gan_mode=vanilla` |
| `hub128` | `n_particles=128` |

## Toy columns and applicability

| kind | meaning |
|---|---|
| **stamp** | Locked-shared / demo stamp toys. A **stamp sweep** winner must PASS every applicable stamp cell (N/A excluded). |
| **dsl** | Phrase expand / phrase jobs / game-geometry. Scored once under `locked_shared`, or N/A for cover drifts. No second phrase recipe per config. |

Cover posture is **not** a column on this board. Demo cover 1.5 and Music cover 1.0 are named claims in `conceptmod/toys/cover_posture_fork.py`. That toy still PASSes demo claim + demo pin, and music claim + music pin, and still FAILs a silent swap or the wrong pin. Scoring every suite candidate under both claims would report config identity (`locked_shared` is not the music config) as a formulation FAIL. Run that gate on its own:

```bash
python -m conceptmod.toys.cover_posture_fork
```

**PASS all stamp toys ≠ Music / Anima / Supra GPU transfer.** A suite PASS is CPU formulation only.

N/A appears when a family does not fork that knob (for example LoRA path targets, late-collapse selection rules, or DSL cover drifts). FAIL means the family's locked arm / refuse path / named drift arm rejects the config.

## Run

```bash
python -m conceptmod.toys.suite_leaderboard
python -m conceptmod.toys.suite_leaderboard --stamp-only --quiet

pytest tests/test_toy_suite_leaderboard.py -q
```

Expected stamp sweep winner: **`locked_shared`** (demo). `music_cover_1_0` still FAILs stamp toys that claim the demo floor. It does not appear as a suite FAIL for the demo claim, and `locked_shared` does not appear as a suite FAIL for the music claim.

Honesty: claiming a stamp sweep with an empty negative list raises `HonestyError`.
