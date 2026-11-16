"""Fill tactical ball gaps from the broadcast ATD (Track B, step 3).

A tactical frame takes the broadcast ball only if: the tactical ball is missing there; the
broadcast frame that maps to it was verified (sync.verify, cache/sync_<GAME>.npz); the broadcast
ball is seen there; and the filled point is physically consistent with its neighbours (no step
over MAX_STEP_M per frame, the same 50 m/s limit as the clip-test linker).

The two ATD runs disagree by > 3 m on ~19% of synced frames, so filled frames are flagged and the
model trusts them less (decoder fill_weight; no touches, no off-pitch calls from them).
"""

import os

import numpy as np

from . import atd, sync

MAX_STEP_M = 2.0


def sync_map(game=atd.GAME):
    """Broadcast -> tactical frame map (-1 = not verified). Built and cached on first use."""
    path = os.path.join(atd.CACHE_DIR, "sync_{}.npz".format(game))
    if not os.path.isfile(path):
        v = sync.verify(sync.load_streams())
        os.makedirs(atd.CACHE_DIR, exist_ok=True)
        np.savez(path, half=v["half"].astype(np.int8), tactical_frame=v["tactical_frame"],
                 threshold_m=v["threshold_m"])
    with np.load(path) as z:
        return z["tactical_frame"]


def fused_ball(tac, game=atd.GAME):
    """(ball (n,3) normalised, filled (n,) bool) with broadcast fill in tactical gaps."""
    ball = tac["ball"].copy()
    bc = atd.load_broadcast(game)
    tmap = sync_map(game)
    n = len(ball)
    src = np.flatnonzero((tmap >= 0) & np.isfinite(bc["ball"][:, 0]))
    dst = tmap[src]
    keep = (dst < n) & ~np.isfinite(ball[dst, 0])
    src, dst = src[keep], dst[keep]
    ball[dst] = bc["ball"][src]
    filled = np.zeros(n, bool)
    filled[dst] = True

    pitch = tac["pitch"]
    xy = ball[:, :2].astype(np.float64) * pitch[None, :]
    step = np.full(n, np.nan)
    step[1:] = np.hypot(*(xy[1:] - xy[:-1]).T)
    bad_prev = np.zeros(n, bool)
    bad_prev[1:] = step[1:] > MAX_STEP_M
    bad_next = np.zeros(n, bool)
    bad_next[:-1] = step[1:] > MAX_STEP_M
    drop = filled & (bad_prev | bad_next)
    ball[drop] = np.nan
    filled &= ~drop
    return ball, filled, {"filled": int(filled.sum()), "dropped_by_speed_gate": int(drop.sum())}


def overlay(f, fused):
    """Copy ball-dependent features of the fused track onto f where the ball was filled.

    off_pitch stays native (False there): a 1-3 m error near a line must not stop play.
    """
    filled = fused["filled"]
    for k in ("seen", "d_home", "d_away", "d_min", "ball_speed"):
        f[k] = np.where(filled, fused[k], f[k])
    f["filled"] = filled
    return f
