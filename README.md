# Possession submission for Barcelona vs Athletic: B2 plus models learned from open data

**Main candidate: `submissions/barca_hybrid_pmdx_d1_C_ctx.csv`, submitted as `submissions/subject47.csv` (identical bytes).**

It is B2 with two learned models added:

- **Possession model.** It decides which team has the ball on every live frame. It is trained on DFL (Bundesliga) and Metrica open data, and it looks at a few seconds of context around each frame.
- **Stoppage model.** It may only lengthen B2's stoppages.

Every submission rebuilds with one command, and gives identical bytes each time.

## Scores

The public window is frames 11185–18684: 5 minutes, 17 possessions and 12 changes of team.

| File | What it is | spellF1 (ranking) | macroF1 | Wrong frames | Possessions, full match |
|---|---|---|---|---|---|
| `barca_hybrid_bc.csv` | B2, the earlier best | 0.560 | 0.905 | 739 | 237 |
| `barca_hybrid_dmx.csv` | B2 + stoppage model | 0.560 | 0.903 | 747 | 229 |
| `barca_hybrid_pm.csv` | possession model (DFL), decoder D0 | 0.627 | 0.898 | 787 | 207 |
| `barca_hybrid_pmdx.csv` | + stoppage model, D0 | 0.627 | 0.897 | 795 | 193 |
| `barca_hybrid_pmdx_d1.csv` | + decoder D1 (model trained on DFL) | 0.653 | 0.917 | 621 | 228 |
| `barca_hybrid_pmdx_d1_C.csv` | + model trained on DFL and Metrica | 0.653 | 0.917 | 622 | 216 |
| **`barca_hybrid_pmdx_d1_C_ctx.csv`** | **+ context features (47 instead of 25)** | 0.643 | **0.926** | **550** | 225 |

**How the main candidate was chosen.** It does best on every frame-level measure:

- **Wrong frames:** 550, against 622 for `pmdx_d1_C`.
- **Wrong team:** 321 frames, against 393.
- **macroF1:** 0.926, the highest of any file.

Its public spellF1 is 0.010 lower than `pmdx_d1_C`. That comes from one short possession merged in the 5-minute window, a quarter of the noise band. It was chosen on the external tests instead: it beat the 25-feature model on all 10 held-out DFL and Metrica matches, by +0.036 spellF1 on average. The public window is treated only as a check that nothing broke.

Notes on these numbers:

- **Noise.** The organisers give a noise band of about 0.04 spellF1. The public window holds only 17 possessions.
- **The main candidate against the DFL-only version.** On the public window they are equal: they differ by 5 s inside it and 159 s over the whole match. C was chosen because it was better on both external test sets (see below).
- **The stoppage model barely shows in the public window.** Only 0.3 s of its 217 s of changes falls inside it, so its support comes from DFL.

**SHA-256** (a rebuild must match):

```
c41c5ccb9db471d836f56ff2766d0bc1106cda987bbc90a7890bc1268ded224e  barca_hybrid_bc.csv
993908a05a5c5d4a13e0bb5781442f306f5e7162d4bbec041dce80b629bf6383  barca_hybrid_dmx.csv
639120c1d4fb23598090285ecf76a4d77a3929993e0b06bf82a2c39dac6d2dc9  barca_hybrid_pm.csv
3142a138460db59a2dc1f315afda92e90d351a62e5360d7fb8dc3567d3e682fa  barca_hybrid_pmdx.csv
ed0264d76b600818e86ad9839033966f349f45d10e539c03085b98837d6e25ee  barca_hybrid_pmdx_d1.csv
92ae311dabc499d412916656a3fd4b028c7525fca4bd0f7f64312be139fb943a  barca_hybrid_pmdx_d1_C.csv
802177e4de6e044824935af750d90e8fed7e657a717eed3b7d7a102712d8b993  barca_hybrid_pmdx_d1_C_ctx.csv  (= subject47.csv)
```

## What changed from B2

B2 is described in `b2_share/README.md`. In short:

- **Dead ball:** rules built from Smart Tagging set pieces, cues for untagged stoppages, and the moments the broadcast cuts away from play.
- **Team:** a hand-tuned decoder (Track A v5).

### External data

The competition rules allow external data about other games.

- **DFL / IDSSE open data:** 7 Bundesliga matches. Every frame has a dead/alive flag and a possession flag.
- **Metrica Sports sample data:** 3 matches. Labels are built from Metrica's events (`src/barca/metrica.py`).
  - **Possession:** the team of the latest set piece, recovery, pass or shot.
  - **Dead ball:** from a ball out, a foul, a shot going out or a goal, until the next set piece.
  - These labels look like our public ground truth: about 5 possessions per live minute, typically 6–7 s long, and a third shorter than 4 s.

### Preparing the external data so it transfers

- **DFL is degraded to look like our tracking** (`src/barca/degrade.py`). The ball is smoothed by 6 frames and players by 2. Ball gaps are cut in following our gap pattern, and players drop out now and then. This was calibrated against our tracking alone, with no labels.
- **The DFL ball during stoppages is simulated.** DFL freezes the ball for the whole of every stoppage, which would leak the answer. It is replaced by an ATD-like ball: missing about 65% of the time, otherwise carried towards the restart spot.
- **DFL possession is approximated as "controlled".** DFL's flag means "last touch", so a short spell (under 3 s) that hands the ball straight back is merged. Results were also checked with 2 s and 5 s.
- **Metrica is smoothed the same way.** Its ball gaps are already like ours, so no gaps are added.

### Possession model (`--poss-model`, `src/barca/poss_model.py`)

- **Inputs, per frame (25 tracking features):** each team's distance to the ball; a learned touch detector (`src/barca/touch_model.py`, trained on DFL); the last and next touch and control events; the ball and each team relative to the goals; team spread.
- **Model:** gradient-boosted trees give P(home has the ball).
- **Decoder D1** turns those probabilities into possessions in two steps:
  1. a 2-state Viterbi with a switch cost of 8;
  2. any spell shorter than 3 s that hands the ball straight back is merged.
- **Earlier decoder D0** used a switch cost of 64 and merged too many short possessions.

**Tests, all run before any public check:**

| Test | What was compared | Result |
|---|---|---|
| DFL, leave one match out | DFL-trained model against our tracking-based rules | spellF1 0.671 vs 0.526 (3 s cut). Better at 2 s and 5 s and on every match |
| Metrica (never used in training) | D1 against D0 | +0.035 and +0.040 spellF1, so the old decoder's low count did cost |
| Training sets, held-out Metrica games | DFL + Metrica (C) against DFL only | 0.707 vs 0.689 |
| Training sets, DFL leave one out | DFL + Metrica (C) against DFL only | 0.692 vs 0.681 |

- **C (DFL + Metrica) against A (DFL only).** C missed our pre-set bar of +0.02 on Metrica by 0.0014. The team chose it anyway, on the condition that it at least tie on the public window, and it tied. A later re-run with corrected Metrica labels, and an ensemble test, both kept C ahead on average.
- **Metrica label fix (2026-11-15).** A stoppage now starts when the ball leaves the pitch, not at the kick before it. Goal kicks count as restarts. All Metrica results above were re-checked with these labels.
- **Ensembles were tested and not adopted.** Averaging bagged A and C models gained at most +0.004.

**Context features** (`features(..., context=True)`, `scripts/ctx_eval.py`) add 22 features to the 25. Some look at windows of ±0.5, 1, 2 and 4 s around each frame:

- each team's highest touch probability;
- the average gap between the two teams' nearest-player distances;
- the share of the window in which home is nearer the ball.

Others describe the frame itself:

- the ball's velocity along the attacking axis;
- each team's players ahead of the ball;
- the 1 s trend in each team's distance to the ball.

They let the model see short interceptions that a single frame misses. Results, trained on C with decoder D1:

| Held-out spellF1 | 25 features | 47 features |
|---|---|---|
| DFL | 0.689 | 0.724 |
| Metrica | 0.674 | 0.712 |

The 47-feature model was better on all 7 DFL and all 3 Metrica matches.

### Stoppage model (`--dead-model`, `src/barca/dead_model.py`)

- **Inputs:** players only. The ball is left out because of the frozen-ball issue.
- **Effect:** it may only lengthen B2's stoppages. It never removes, shortens or adds one.
- **DFL test with stand-in Smart Tagging tags:** alive/dead F1 rose from 0.695 to 0.819. False stoppages stayed at 7.
- **On our match:** it lengthens stoppages by 209 s. The broadcast was away from live play for most of those frames.

## Rebuild

**What you need:**

- numpy, pandas, scipy and scikit-learn. Tested with Python 3.14, scikit-learn 1.9.1 and scipy 1.18.1.
- The same scikit-learn version, so the saved models load.
- `kloppy` only for retraining with Metrica game 3.

Put the folders next to the competition kit:

```
<your folder>/
  scripts/  src/barca/  cache/dfl/      from this share
  more-than-a-hack-2027/                the competition kit, with data/tactical and data/broadcast
```

Then rebuild the main candidate:

```
python scripts/make_hybrid.py --final        # = --poss-model --dead-model --poss-decoder d1 --poss-model-file cache/dfl/poss_model_C_ctx.pkl -> submissions/subject47.csv
```

This writes the submission under the team's file name, `subject47.csv`, with the same bytes as `barca_hybrid_pmdx_d1_C_ctx.csv`.

Other variants:

```
python scripts/make_hybrid.py --poss-model --dead-model --poss-decoder d1 --poss-model-file cache/dfl/poss_model_C.pkl --out submissions/barca_hybrid_pmdx_d1_C.csv
python scripts/make_hybrid.py --poss-model --dead-model --poss-decoder d1      # -> barca_hybrid_pmdx_d1.csv (DFL-only model)
python scripts/make_hybrid.py --poss-model --dead-model                        # -> barca_hybrid_pmdx.csv
python scripts/make_hybrid.py --broadcast                                      # -> barca_hybrid_bc.csv (B2)
```

- Each command prints the public scores and full-match statistics.
- The first run builds `cache/`, so it takes longer.
- The rebuild uses the saved models in `cache/dfl/`. The open data is not needed for it.

## Retraining (needs the open data)

1. **DFL:** put the IDSSE data in `DFL/DFL_Data/` (figshare, doi 10.6084/m9.figshare.28196177).
2. **Metrica:** put the sample-data repo in `Metrica/sample-data-master/sample-data-master/`.
3. Run:

```
python scripts/dead_model_eval.py --fair --bar     # stoppage model + its DFL test -> cache/dfl/dead_players_fair_model.pkl
python scripts/poss_model_eval.py --bar            # DFL-only possession model + DFL test -> cache/dfl/poss_model.pkl
python scripts/poss_decode_study.py                # decoder study on DFL
python scripts/metrica_eval.py                     # external test on Metrica
python scripts/metrica_train_eval.py               # training-set comparison
python scripts/metrica_train_eval.py --save C      # -> cache/dfl/poss_model_C.pkl
python scripts/ctx_eval.py                         # context features test -> cache/dfl/poss_model_C_ctx.pkl
```

`scripts/ensemble_eval.py`, `scripts/decoder_tune.py` and `scripts/dead_metrica_eval.py` hold tests that were run and not adopted.

**Do not use `dead_BM_F_model.pkl`.** `dead_metrica_eval.py` can write it to `cache/dfl/`, and its high scores are an artefact: Metrica blanks the ball while it is out of play, and our tracking does not.

## Data credits (must stay in the technical document)

- **DFL open data:** © Deutsche Fußball Liga (DFL), CC BY 4.0. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
- **Metrica Sports sample data:** https://github.com/metrica-sports/sample-data. Its README asks for responsible use and acknowledgement of the source.

Both sources are other games, named and explained here as the rules require.

**Not in this repository:** the match data, the competition kit, `cache/` (except the four models above) and the submission files. `.gitignore` keeps them out. The match data is licensed for the competition only. Rebuild the submission with `--final` and check its SHA-256 against the one above.
