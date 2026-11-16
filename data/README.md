# Unpack the match data here

This folder is empty on purpose. The match assets are gated: enrol in More than a Hack 2027,
accept the terms and get the download link as described in the competition kit's
[DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md).

Unpack the bundle here, keeping the names it arrives with:

```
data/
├─ tactical/                                  the evaluation reference
│  ├─ BAR-ATH_tactical.mp4                    row i of a submission is frame i of this
│  ├─ BAR-ATH_tactical_atd.xml                tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt                tracking, players and ball
│  ├─ BAR-ATH_tactical_homography.csv         one 3x3 matrix per frame
│  └─ BAR-ATH_tactical_smart_tagging.xml      638 tactical phases and set pieces
└─ broadcast/                                 NOT synchronised to tactical
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

The submission uses both folders: the tactical tracking and Smart Tagging for everything, and the
broadcast recording to find when the broadcast cuts away from live play.

## Where the code looks for the data

In this order (`src/barca/atd.py`):

1. `MTAH_DATA_DIR`, if it is set:
   ```bash
   export MTAH_DATA_DIR=/mnt/big-disk/mtah2027       # macOS / Linux
   $env:MTAH_DATA_DIR = "D:\mtah2027"                # Windows PowerShell
   ```
2. this folder, if `data/tactical/` exists;
3. the competition kit's own `more-than-a-hack-2027/data/`.

File names are built by the kit's `games.py`, so nothing in this repository spells one out.

## Nothing in here is committed

`.gitignore` keeps everything except this README out of git. The terms accepted at enrolment
license the match data for the competition only, and this repository is public: committing the
data, or anything derived from it frame by frame, would be redistribution.
