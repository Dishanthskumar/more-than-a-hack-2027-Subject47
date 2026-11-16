"""Cues for stoppages that have no set-piece tag (fouls, offsides, injuries), and evidence from them.

The existing detector's speed cue is a 3 s mean of the average player speed, so a stoppage only
shows once everyone has slowed for a while; short stoppages (about 5 s) are missed and long ones
are found late. These cues aim at the onset instead.
"""

import warnings

import numpy as np
import pandas as pd

from barca import dead_ball as db
from barca.loaders import FPS

SLOW_MS = 1.0          # a player under 1 m/s is standing or walking
STILL_BALL_MS = 0.5    # a placed ball
LINE_M = 3.0           # "away from the lines" for a lost ball
WIN_S = 3.0            # before / after window for the speed drop


def _past_mean(x, frames):
    return pd.Series(x).rolling(frames, min_periods=frames // 3).mean().shift(1).to_numpy()


def _future_mean(x, frames):
    return pd.Series(x[::-1]).rolling(frames, min_periods=frames // 3).mean().to_numpy()[::-1]


def cues(c, tr):
    """dict of per-frame float arrays, NaN where undefined."""
    ps = tr["ps"].astype(np.float64)
    tracked = np.isfinite(ps)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        slow = np.where(tracked.any(1), (ps < SLOW_MS).sum(1) / tracked.sum(1), np.nan)
        p90 = np.nanpercentile(ps, 90, axis=1)
    sp = c["player_speed"]
    w = int(WIN_S * FPS)
    with np.errstate(invalid="ignore"):
        bmed = db.centred_median(c["ball_speed"], FPS)
        still = np.where(c["ball_seen"] > 0, (bmed < STILL_BALL_MS).astype(float), np.nan)
        lost_mid = ((c["ball_seen"] == 0) & (c["gap_start_line_dist_m"] > LINE_M)).astype(float)
    return {
        "speed_1s": db.centred_mean(sp, FPS),
        "slow_share_1s": db.centred_mean(slow, FPS),
        "p90_speed_1s": db.centred_mean(p90, FPS),
        "speed_drop_3s": _past_mean(sp, w) - _future_mean(sp, w),
        "speed_after_3s": _future_mean(sp, w),
        "ball_speed_1s": bmed,
        "ball_still_1s": db.centred_mean(still, FPS),
        "ball_lost_mid": lost_mid,
    }


# Evidence variants, constants fixed 2026-11-09 from the full-match cue study before scoring
# (scripts/stoppage_cues.py): next-3 s player speed AUC 0.105 at onsets vs 0.163 for the centred
# 1 s mean; slow-player share median 0.12 live, 0.49 tagged stoppages.
SLOW_PIVOT, SLOW_SCALE, SLOW_W = 0.30, 0.10, 0.5


def evidence(c, tr, level):
    """Speed evidence replacing dead_ball's speed part. level 1: onset-aware speed;
    level 2: + slow-player share. Same clip formula and constants as dead_ball.evidence."""
    with np.errstate(invalid="ignore"):
        centred = db.centred_mean(c["player_speed"], db.SPEED_SMOOTH_S * FPS)
        future = _future_mean(c["player_speed"], int(db.SPEED_SMOOTH_S * FPS))
        sp = np.fmin(centred, future)
        e = np.nan_to_num(np.clip((db.SPEED_PIVOT_MS - sp) / db.SPEED_SCALE_MS,
                                  -db.SPEED_CLIP, db.SPEED_CLIP), nan=0.0)
        if level >= 2:
            slow = cues(c, tr)["slow_share_1s"]
            e += SLOW_W * np.nan_to_num(np.clip((slow - SLOW_PIVOT) / SLOW_SCALE, -2, 2), nan=0.0)
    return e
