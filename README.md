# More than a Hack 2027 — team subject47

Phase 1 submission: ball possession (`H` / `A` / `D` / `X`) for every frame of Barcelona – Athletic.

- **Dead ball:** rules built on the tracking, the Smart Tagging set pieces and the moments the broadcast cuts away from play. A stoppage model trained on open data may lengthen these stoppages.
- **Possession:** a gradient-boosted model, trained on open data from other matches (DFL / Bundesliga and Metrica), gives the probability that the home team has the ball. A decoder turns those probabilities into possessions.

The method, the possession rules, the training data, the validation and the limitations are in **[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)**.

The submission rebuilds with one command and gives the same bytes every time.

## Rebuild the submission

**What you need:** Python 3.14 and the match data bundle (from enrolment in More than a Hack 2027).

**1. Clone this repository and go into it:**

```
git clone https://github.com/Dishanthskumar/more-than-a-hack-2027-Subject47.git
cd more-than-a-hack-2027-Subject47
```

**2. Install the pinned packages:**

```
pip install -r requirements.txt
```

scikit-learn must be exactly 1.9.1, or the saved models will not load.

**3. Clone the competition kit inside this repository:**

```
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
```

**4. Unpack the match data into `data/`,** keeping the folder and file names from the bundle (see `data/README.md`).

When steps 1–4 are done, the folder looks like this:

```
more-than-a-hack-2027-Subject47/          <- this repository; run every command from here
├─ scripts/make_hybrid.py                 the rebuild script
├─ src/barca/                             the code it uses
├─ cache/dfl/                             the 4 saved models
├─ requirements.txt
├─ data/
│  ├─ README.md
│  ├─ tactical/                           <- from the data bundle (step 4)
│  │  ├─ BAR-ATH_tactical.mp4
│  │  ├─ BAR-ATH_tactical_atd.xml
│  │  ├─ BAR-ATH_tactical_atd.txt
│  │  ├─ BAR-ATH_tactical_homography.csv
│  │  └─ BAR-ATH_tactical_smart_tagging.xml
│  └─ broadcast/                          <- from the data bundle (step 4)
│     ├─ BAR-ATH_broadcast.mp4
│     ├─ BAR-ATH_broadcast_atd.xml
│     ├─ BAR-ATH_broadcast_atd.txt
│     └─ BAR-ATH_broadcast_homography.csv
└─ more-than-a-hack-2027/                 <- the competition kit (step 3)
```

The match data and the kit are ignored by git, so they cannot be committed by mistake.

**5. Rebuild the submission:**

```
python scripts/make_hybrid.py --final
```

- **Output:** `submissions/subject47.csv`. It is created by the run.
- **Run time:** a few minutes. The first run also builds a cache in `cache/`.
- **Printed check:** public spellF1 0.6429, macroF1 0.9256, 550 wrong frames.
- **SHA-256:** must be `802177e4de6e044824935af750d90e8fed7e657a717eed3b7d7a102712d8b993`. Check it with `sha256sum submissions/subject47.csv` (Linux, macOS) or `Get-FileHash submissions\subject47.csv -Algorithm SHA256` (PowerShell).

**6. Validate it with the competition's checker:**

```
python more-than-a-hack-2027/evaluation/validate_submission.py submissions/subject47.csv
```

The rebuild uses the saved models in `cache/dfl/`, so the open data is not needed for it.

## What is in this repository

| Path | What it is |
|---|---|
| `scripts/make_hybrid.py` | builds the submission (`--final`) |
| `src/barca/` | the pipeline: tracking loaders, dead-ball rules, possession and stoppage models, decoder |
| `cache/dfl/` | the saved models, trained on open data only (see below) |
| `scripts/*_eval.py`, `*_study.py`, `*_tune.py` | the tests behind each decision in the technical document |
| `data/` | empty; the match data goes here |
| `TECHNICAL_DOCUMENT.md` | the method and its validation |

**Not in this repository:** the match data, the competition kit, the generated cache and the submission file. `.gitignore` keeps them out, because the match data is licensed for the competition only.

## Retraining the models (needs the open data)

1. **DFL:** put the IDSSE data in `DFL/DFL_Data/` (figshare, doi 10.6084/m9.figshare.28196177).
2. **Metrica:** put the sample-data repository in `Metrica/sample-data-master/sample-data-master/`.
3. **Metrica game 3** also needs `pip install kloppy==3.19.1`.
4. Run:

```
python scripts/dead_model_eval.py --fair --bar     # stoppage model + its DFL test -> cache/dfl/dead_players_fair_model.pkl
python scripts/poss_model_eval.py --bar            # DFL-only possession model + DFL test -> cache/dfl/poss_model.pkl
python scripts/poss_decode_study.py                # decoder study on DFL
python scripts/metrica_eval.py                     # external test on Metrica
python scripts/metrica_train_eval.py               # DFL-only vs DFL + Metrica training
python scripts/metrica_train_eval.py --save C      # -> cache/dfl/poss_model_C.pkl
python scripts/ctx_eval.py                         # context features test -> cache/dfl/poss_model_C_ctx.pkl (the submitted model)
```

`scripts/ensemble_eval.py`, `scripts/decoder_tune.py` and `scripts/dead_metrica_eval.py` hold tests that were run and not adopted. Do not use the `dead_BM_F_model.pkl` that `dead_metrica_eval.py` can write: its scores are an artefact of Metrica blanking the ball while it is out of play.

## Data credits

- **DFL open data:** © Deutsche Fußball Liga (DFL), CC BY 4.0. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
- **Metrica Sports sample data:** https://github.com/metrica-sports/sample-data

Both are other matches. No labels from Barcelona – Athletic were used for training.
