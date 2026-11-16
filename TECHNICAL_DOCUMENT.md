# Ball possession for Barcelona – Athletic: a rule-based dead-ball layer with learned possession and stoppage models

*More than a Hack 2027, Phase 1. Technical document.*

**Submission:** `subject47.csv`, 199,319 rows, valid under `evaluation/validate_submission.py`. It is written by `python scripts/make_hybrid.py --final` (section 8).
**SHA-256:** `802177e4de6e044824935af750d90e8fed7e657a717eed3b7d7a102712d8b993`

**What it is.** Two layers:

1. **A rule-based dead-ball layer.** It uses the tactical tracking, Smart Tagging set pieces and the broadcast feed. A learned stoppage model may only lengthen the stoppages it finds.
2. **A learned possession model.** It decides H or A on every live frame. It was trained only on public labelled data from other matches (DFL and Metrica).

**How choices were made.** We chose between versions on held-out external matches with acceptance rules fixed before each run. The public 5-minute window was kept as a correctness check.

**Results on held-out external matches** (spellF1, possession layer):

| Source | Our model | Tracking-based rules |
|---|---|---|
| DFL (7 matches) | 0.72 | 0.53 |
| Metrica (3 matches) | 0.71 | 0.54–0.64 |
| SkillCorner (20 matches, never trained on) | 0.72 | – |

**Public window:** spellF1 0.643 and macroF1 0.926, against 0.560 and 0.905 for our earlier rule-only pipeline.

---

## 1. Approach and why

The task has two different problems, and we treat them separately:

- **Is the ball in play?** The provided match has strong evidence for this: Smart Tagging set-piece tags, the moment the broadcast cuts away from play, players slowing down and walking, and the ball leaving the pitch. Hand-built rules use this evidence well.
- **Which team has controlled possession?** This needs judgement about loose balls, deflections and short touches. Hand-made rules plateaued here: our earlier pipeline's public spellF1 was 0.56. Its main errors were changes of team caught seconds late and short possessions it never saw.

So the final system keeps a rule-based dead-ball layer and replaces the team decision with a model learned from labelled data of other matches. Two facts shaped the design:

1. **The 5-minute public window can't be used to choose between versions.** It has 17 possessions, and one possession moves spellF1 by about 6 points. Every model choice was therefore made on held-out external matches, with an acceptance rule written down before each run. Several plausible improvements were rejected under those rules (section 3).
2. **External tracking does not look like ours.** Our tactical tracking smooths the ball heavily and has gaps. Live-play ball acceleration is 6.6 / 18.7 / 33.8 m/s² at the 50th / 90th / 99th percentiles, against 19.8 / 97.6 / 604 for raw DFL. The ball is missing on 13.6% of live frames. A touch detector trained on raw DFL fired at a third of the expected rate on our match. Every external dataset is therefore degraded to our tracking's statistics before training, as described in section 4.

### 1.1 Dead-ball layer (rules, plus a learned extension)

**Per-frame evidence** (positive means dead), from the tactical tracking:

- **Ball:** a slow ball near a line; the ball out of the pitch; a ball gap that starts near a line.
- **Players:** players outside the pitch; the onset of a drop in player speed (`src/barca/stoppages.py`), aimed at stoppages that have no tag.
- **Control events:** live-play evidence inside them (a player within 2 m of an uncontested, slow ball for at least 0.5 s).
- **Broadcast cut-aways:** the moment the broadcast cuts away from live play, +1.0 from 3 s before to 5 s after the cut. It stops at the first tracking sign that play has restarted (`src/barca/broadcast.py`). Only the onset of a cut is used, because a broadcast can show a replay after play has resumed.
- **Set pieces:** Smart Tagging `SET PIECES` tags. A restart is located at the first struck ball whose position fits the tag type. For line restarts, the stoppage begins where the ball left the pitch (`src/barca/set_pieces.py`).

**Decoding:** a two-state Viterbi (switch cost 25), then minimum durations: a stoppage lasts at least 4 s and a live gap at least 2 s.

**Learned extension** (`src/barca/dead_model.py`):

- A gradient-boosted classifier of P(dead) from player behaviour only: average and fastest player speed over several windows, the share of players walking, players off the pitch, players in the boxes, team spread and the distance between team centres.
- It is trained on DFL; why the ball is excluded is explained in section 4.
- Its log-odds are added to the evidence, and the layer is decoded again. Only stoppages that overlap one found by the rules are kept ("extend-only"), so it can lengthen a stoppage but never adds or removes one.

### 1.2 Possession layer (learned) — `src/barca/poss_model.py`

**Inputs: 47 tracking-only features per live frame.** None use Smart Tagging or the broadcast, which external data does not have.

- **Ball and players:** nearest home and away player to the ball, and the margin between them; the ball and both team centroids along the attacking direction; team spread; ball speed.
- **Touch detector:** a learned model (`src/barca/touch_model.py`) trained on DFL "last-touch" flips. It gives the probability that each team touches the ball now. Held-out F1 is 0.33 on degraded DFL, against 0.17 for a velocity-kink rule.
- **Event context:** the team and age of the last and next detected touch and control event.
- **Multi-scale context:** over ±0.5, 1, 2 and 4 s, each team's highest touch probability, the average distance margin and the share of time home is nearer the ball. Also the ball's velocity along the attacking axis, each team's players ahead of the ball, and each team's 1 s distance trend.

**Model:** gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, default settings with early stopping) give P(home in possession). The training set is DFL plus Metrica, with Metrica weighted to the same total as DFL.

**Decoder "D1"**, applied inside each live spell:

1. a 2-state Viterbi on the log-odds with a switch cost of 8;
2. any team spell shorter than 3 s that hands the ball straight back to the other team is merged into the surrounding possession.

## 2. Possession rules (where we drew the lines)

| Situation | How it is labelled |
|---|---|
| Ball out, foul, offside, goal | **D** from the moment the ball leaves the pitch or play stops, to the restart: the first struck ball at a position that fits the tag type. Stoppages without a tag come from player-speed onset and ball cues |
| Restart | Live from the restart. The restarting team is not taken from Smart Tagging's `Team` field, which proved unreliable. On the full match, in the frames a "tag team restarts" rule would relabel, the player at the ball belonged to the tag's team 22 times and to the other side 70 times. The team comes from the possession model's reading of the tracking |
| Loose ball, contested ball | No change. The model's probability sits near 0.5, and the switch cost keeps the last team, as the challenge text asks |
| Deflection, touch without control, aerial duel without follow-up | Not a change. A spell under 3 s that returns the ball to the previous team is merged (decoder D1). Our training labels apply the same 3 s rule to DFL's last-touch flags |
| Change of controlled possession | Dated at the frame where the model's evidence flips for good. On held-out DFL, changes are typically about 0.5–1 s from the true moment |

The **3 s** threshold is an approximation of "controlled possession". It was checked against alternatives (sections 3 and 6).

## 3. Assets: what we used and what we dropped

| Asset | Used for | Kept? |
|---|---|---|
| Tactical tracking (ATD) | everything | yes |
| Broadcast tracking (ATD) + sync map | moments the broadcast cuts away (dead-ball evidence) | yes |
| Smart Tagging `SET PIECES` | stoppage timing and restart location | yes |
| Smart Tagging phases | team evidence in the earlier rule-based team layer | replaced by the possession model |
| Homography, both videos | WASB video ball detector (HRNet) | dropped (below) |

**Tried and dropped,** all under acceptance rules fixed before the run:

| What we tried | Why it was dropped |
|---|---|
| **WASB video ball** to fill tactical-ball gaps (4 variants) | In a test where we hid the ball ourselves, it gained +2.3 points against a +5 bar. It could not confirm touches either: the video track kinks about once a second |
| **Velocity-kink touch rule** | Over-split possessions (297–344 vs 237) |
| **"Next receiver" team signal** | +0.2 points agreement with Metrica's tags, against a +1 bar |
| **Snapping change times to detected touches** | Worse timing on DFL (1.07 → 1.20 s) |
| **Ensembles / bagging** | +0.004 at best |
| **A second round of context features** (66 features) | −0.004 |
| **Context features for the touch detector** | −0.008 |
| **Self-training on the target match** | −0.002. The pseudo-labels were 95–99% accurate but covered only easy frames |
| **Model-capacity tuning, training on a 2 s cut** | Defaults were best |
| **Re-tuning the decoder** (cost 12–16, 0–2 s) | +0.05 on Metrica but −0.02 on DFL, so the sources disagree |
| **Stoppage models with ball features**, from DFL, Metrica and SkillCorner | Section 4 |
| **SkillCorner as extra possession training data** | +0.028 on SkillCorner but −0.011 on DFL and −0.017 on Metrica |
| **"Full" (not extend-only) and "trim-end" stoppage models** | More false stoppages, or no effect |

## 4. Missing and noisy tracking

- **Our tracking (ATD).**
  - The ball is missing on 13.6% of live frames, in 88 gaps that are typically 5 s long.
  - Players are tracked about 21.4 per frame.
  - The ball track is smoothed. All learned models use scikit-learn's native handling of missing values, so a frame with no ball still gets a prediction from the player features. The decoders bridge gaps by keeping the current state.
- **Making external data look like ours** (`src/barca/degrade.py`). These settings were calibrated against our tracking only, with no labels:
  - **Ball:** Gaussian smoothing with σ = 6 frames. Live acceleration percentiles become 5.8 / 19.2 / 37.7 m/s².
  - **Players:** σ = 2 frames.
  - **Ball gaps:** inserted at our gap rate (1.5 per live minute), with lengths drawn from our own gaps.
  - **Players:** dropped out at our rate.

  After this, the touch detector's firing rate on our match matched DFL's (5.95 vs 6.22 per live minute; before: 2.1).
- **Provider artefacts during stoppages.** These decided which models could be trusted:
  - **DFL** freezes the ball for the whole of every stoppage (all 103 in the match we checked first; 100% of dead frames). A stoppage model with ball features scored a fake alive/dead F1 of 0.99. So DFL's dead-time ball is replaced by a simulated one: missing about 65% of the time, otherwise carried towards the restart spot, as in our tracking.
  - **Metrica** blanks the ball almost exactly while it is dead: it vanishes within 2 frames of the stoppage start in 91–98% of stoppages. Our tracking does not do this; in two of the four public stoppages the ball is visible throughout.
  - **SkillCorner** has neither artefact. But a stoppage model learned from its broadcast-camera ball added frames on our match that look like live play: the broadcast was away from play only 23% of the time (live play: 10%, our stoppages: 67%).

  So the deployed stoppage model uses **players only**.
- **Simulated tags for fair tests.** DFL, Metrica and SkillCorner have no Smart Tagging. Their stoppage tests use stand-in set-piece tags built from each dataset's own restart events. Each restart type is kept at the rate Smart Tagging tags it in our match (for example, 26 of 44 throw-ins and 5 of about 25 free kicks), with ±1 s of timing error.

## 5. Training data (all from other matches)

**Nothing from the provided match was labelled, by hand or otherwise, and no human-produced data about it was used.** The public 5-minute ground truth was never used for training. It served to check that the pipeline was correct, and as a final check of variants named in advance.

**Use of the public window in decisions.** In two early decisions, the D1 decoder and the DFL + Metrica training set, a public-window condition was part of the acceptance rule. Both choices are independently supported by held-out external results:

- **D1:** better than the earlier decoder on all 3 Metrica games, by +0.014 to +0.057.
- **DFL + Metrica:** ahead of DFL-only on average on both benches.

Every later choice used external results only.

| Dataset | Source and licence | Labels: what, by whom | Amount | Used for |
|---|---|---|---|---|
| **DFL / IDSSE open data** | Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12:195; data © Deutsche Fußball Liga (DFL), CC BY 4.0 | Per-frame `BallStatus` (alive/dead) and `BallPossession` (team) in the TRACAB position data, supplied by the DFL. The release does not document how they are produced. We derive "controlled" possession by merging hand-backs under 3 s | 7 Bundesliga matches, about 660 min (369 live min, 2,351 last-touch changes) | Touch detector, possession model, stoppage model; tests |
| **Metrica Sports sample data** | github.com/metrica-sports/sample-data (use responsibly, acknowledge the source) | Event data coded by Metrica's annotators. We derive per-frame labels by rule (`src/barca/metrica.py`). **Team:** the team of the latest set piece, recovery, pass or shot. **Dead:** from the ball leaving the pitch, a foul, a shot going out or a goal, to the next set piece or goal kick | 3 matches, about 285 min | Possession model; tests |
| **SkillCorner open data** | github.com/SkillCorner/opendata, MIT licence (credit SkillCorner) | Phases of play and dynamic events from SkillCorner. We derive team possession from the phases, and stoppages from gaps between phases plus interruption events | 20 A-League 2024/25 matches | **Tests only** (training trials rejected) |

## 6. Validation

Every candidate was judged leave-one-match-out on held-out matches, under an acceptance rule written into the script before it ran. The rules typically required:

- a combined gain of at least 0.01–0.015 spellF1;
- no loss on any source beyond 0.005;
- wins on most held-out matches.

The deciding results:

| Step | Before → after (held-out spellF1) |
|---|---|
| Possession model vs tracking rules, DFL (3 s cut) | 0.526 → 0.671 (better at 2 s and 5 s, and on every match) |
| D1 decoder vs D0, Metrica | better on all 3 games |
| Context features (47 vs 25), DFL / Metrica | 0.689 / 0.674 → 0.724 / 0.712 (all 10 matches better) |
| Stoppage extension, DFL stand-in bench | alive/dead F1 0.695 → 0.819, false stoppages 7 → 7 |

**Final check on the public window:**

| | spellF1 | macroF1 | Wrong frames |
|---|---|---|---|
| Earlier rule-based pipeline | 0.560 | 0.905 | 739 |
| Submission | 0.643 | 0.926 | 550 |

## 7. Limitations, and what we would do with more time

- **Short possessions are under-detected.** The model predicts about 225 possessions for the full match. On the external data it produces fewer possessions per live minute than the truth (about 4.8 vs 5.3 on DFL, 3.8–4.3 vs 5.8–6.6 on Metrica).
- **Label style varies by provider, and our sources disagree.** Metrica's human event labels favour less merging (a lower switch cost and a shorter minimum spell). DFL favours D1. Adding SkillCorner helps on SkillCorner and hurts elsewhere. The best decoder for our ground truth depends on how many short possessions it contains, which we cannot see.
- **The dead-ball layer still accounts for about 40% of the public-window errors.** It depends on Smart Tagging and the broadcast. Stoppages with neither are found only from player behaviour, and ball cues could not be learned from any external source because of provider artefacts.
- **"Controlled possession" is approximated by a 3 s hand-back rule,** not modelled directly.

**With more time** we would:

- **Get more labelled data like ours:** the same camera setup and the same labelling convention as the ground truth. This is the one thing every experiment pointed to.
- **Decode dead ball and team jointly,** instead of fixing the dead-ball mask first.
- **Train with home and away swapped,** to remove any home/away bias.
- **Feed the broadcast ball into the possession features** where the tactical ball is missing.

## 8. Reproduction

**Requirements:**

- **Python 3.14** with the pinned `requirements.txt`: numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1. The saved models need exactly scikit-learn 1.9.1.
- **The competition kit as `more-than-a-hack-2027/`** next to `scripts/`, `src/` and `cache/` (`git clone https://github.com/metrica-sports/more-than-a-hack-2027.git` inside this repository). The kit is not in this repository.
- **The match data** unpacked into `data/` (`data/tactical/` and `data/broadcast/`, see `data/README.md`). The kit's own `data/` or `MTAH_DATA_DIR` also work. The data is not in this repository.

**Steps:**

```
pip install -r requirements.txt
python scripts/make_hybrid.py --final
python more-than-a-hack-2027/evaluation/validate_submission.py submissions/subject47.csv
```

- **What `--final` does:** it is shorthand for the submitted configuration, `--poss-model --dead-model --poss-decoder d1 --poss-model-file cache/dfl/poss_model_C_ctx.pkl`. That configuration always writes `submissions/subject47.csv` unless `--out` names another file. The command can be run from any folder.
- **Run time:** a few minutes. The first run also builds a tracking cache in `cache/`.
- **Output:** the script prints the public-window check (spellF1 0.6429, macroF1 0.9256, 550 wrong frames) and `wrote …/submissions/subject47.csv`.
- **Check:** the file's SHA-256 must be `802177e4de6e044824935af750d90e8fed7e657a717eed3b7d7a102712d8b993`. Use `sha256sum submissions/subject47.csv` on Linux or macOS, or `Get-FileHash submissions\subject47.csv -Algorithm SHA256` in PowerShell. A clean copy of this repository with only the kit added reproduces it byte for byte.
- **The models it uses:**
  - `cache/dfl/poss_model_C_ctx.pkl`: the touch and possession models;
  - `cache/dfl/dead_players_fair_model.pkl`: the stoppage model.

**Retraining** (needs the three open datasets and `kloppy` 3.19.1 for Metrica game 3):

| Script | What it does |
|---|---|
| `dead_model_eval.py --fair --bar` | trains and tests the stoppage model |
| `ctx_eval.py` | trains and tests the possession model with context features |
| `touch_model_eval.py --degrade` | tests the touch detector |

The other scripts in `scripts/` hold the rejected experiments. Each states its acceptance rule at the top.

## Credits

- **DFL open data:** © Deutsche Fußball Liga (DFL), CC BY 4.0. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
- **Metrica Sports sample data:** https://github.com/metrica-sports/sample-data
- **SkillCorner open data:** https://github.com/SkillCorner/opendata (MIT)
- **WASB-SBDT ball detector:** Tarashima et al., BMVC 2023 (tried, not used in the submission)
