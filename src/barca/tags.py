"""Evidence from Metrica Smart Tagging (automatic, not human-produced).

Two signals, both calibrated against the tracking itself, never against labels:

* Phase tags (BUILD UP, PROGRESSION, ...): their Team is the team in possession. The DEFENDING
  tags are exact mirrors of the attacking ones (same intervals, Team = the defending side), so
  only the attacking side is used and nothing is double-counted. Each code's weight is the
  log-odds of its agreement with confident tracking ownership, so a tag that agrees with the
  tracking 89% of the time counts for more than one that agrees 57%.
* Set pieces: an interval is a fixed window that starts a few seconds before the restart (35 s
  windows, restart ~5 s in; 5 s kick-off windows, restart ~3.4 s in, which the two published
  kickoff frames confirm). Before the restart the ball is dead; just after it the restarting team
  has the ball, when the tag names one (throw-ins never do).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "more-than-a-hack-2027", "starter"))
import load_smart_tagging  # noqa: E402

HOME_TEAM_TAG = "Barcelona"      # DATA.md: H = Barcelona, the first team in the ATD metadata
AWAY_TEAM_TAG = "Atletic Club"

PHASE_CODES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL",
               "BALL IN FINAL THIRD", "BALL IN THE BOX", "CREATING CHANCES", "PLAYERS IN THE BOX")
SET_PIECE_CODE = "SET PIECES"
RESTART_DELAY_S = {"long": 5.0, "kickoff": 3.4}   # where the restart sits inside the tag window
RESTART_TEAM_S = 4.0                              # how long the restarting team is favoured


def _side(team):
    return {HOME_TEAM_TAG: 1.0, AWAY_TEAM_TAG: -1.0}.get(team, 0.0)


def evidence(n_frames, pseudo_home, pseudo_mask, fps=25, events=None):
    """dict of (n,) arrays.

    phase_team: signed log-odds, + for home, - for away (summed over overlapping phases).
    pre_restart: 1 on frames between a set-piece window start and its restart.
    restart_team: +1 / -1 for a few seconds after a restart whose team is named, 0 otherwise.
    pseudo_home: 1 where the tracking says home has the ball, 0 away; pseudo_mask: trusted frames.
    """
    events = events if events is not None else load_smart_tagging.load()
    phase = np.zeros(n_frames)
    pre = np.zeros(n_frames)
    restart = np.zeros(n_frames)
    restart_frames = []
    calibration = {}

    for code in PHASE_CODES:
        spans = [(e["start_frame"], e["end_frame"], _side(e["tags"].get("Team")))
                 for e in events if (e["code"] or "").strip().upper() == code]
        spans = [(max(a, 0), min(b, n_frames), s) for a, b, s in spans if s]
        agree = total = 0
        for a, b, s in spans:
            m = pseudo_mask[a:b]
            agree += int(np.sum((pseudo_home[a:b][m] == 1) == (s > 0)))
            total += int(m.sum())
        rate = (agree + 1.0) / (total + 2.0)            # Laplace, so an empty code is neutral
        weight = float(np.log(rate / (1.0 - rate)))
        calibration[code] = (round(rate, 3), total)
        if weight <= 0:
            continue                                     # no better than chance: ignore
        for a, b, s in spans:
            phase[a:b] += s * weight

    for e in events:
        if (e["code"] or "").strip().upper() != SET_PIECE_CODE:
            continue
        kind = "kickoff" if (e["tags"].get("Type") or "").lower().startswith("kick off") else "long"
        r = e["start_frame"] + int(round(RESTART_DELAY_S[kind] * fps))
        a = max(e["start_frame"], 0)
        pre[a:min(r, n_frames)] = 1.0
        if 0 <= r < n_frames:
            restart_frames.append(r)
        s = _side(e["tags"].get("Team"))
        if s:
            restart[r:min(r + int(RESTART_TEAM_S * fps), n_frames)] = s
    return {"phase_team": phase, "pre_restart": pre, "restart_team": restart,
            "restart_frames": np.array(sorted(restart_frames), dtype=np.int64),
            "calibration": calibration}
