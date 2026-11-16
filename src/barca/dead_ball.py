"""Dead-ball cues and detector: evidence -> 2-state Viterbi -> minimum durations -> restart snap.

Constants were fixed from the cue study (full match, no ground truth) before any public scoring.
"""

import warnings

import numpy as np
import pandas as pd

from barca.loaders import FPS, PITCH_M, run_lengths

L, W = PITCH_M

# detector constants
SWITCH_COST = 25.0
MIN_DEAD_S, MIN_LIVE_S = 4.0, 2.0
TAG_PRE_S, TAG_POST_S = 10.0, 2.0
RESTART_FROM_S, RESTART_SEARCH_S = 1.0, 15.0
RESTART_BALL_MS = 5.0
RESTART_DEFAULT_S = 4.0

# evidence constants
SPEED_SMOOTH_S = 3.0
SPEED_PIVOT_MS, SPEED_SCALE_MS, SPEED_CLIP = 1.3, 0.3, 2.0
BALL_SLOW_MS, BALL_NEAR_LINE_M, BALL_W = 2.0, 6.0, 1.5
BALL_OUT_W, PLAYERS_OUT_W = 3.0, 2.0
TAG_W = 2.0
GAP_MIN_S, GAP_LINE_M, GAP_W = 5.0, 3.0, 1.0

SET_PIECE = "SET PIECES"


def centred_mean(x, frames):
    return pd.Series(x).rolling(int(frames), center=True, min_periods=1).mean().to_numpy()


def centred_median(x, frames):
    return pd.Series(x).rolling(int(frames), center=True, min_periods=1).median().to_numpy()


def line_dist(x, y):
    """Distance to the nearest pitch line in metres, positive inside the pitch."""
    return np.minimum(np.minimum(x, L - x), np.minimum(y, W - y))


def set_piece_tags(tags):
    return sorted((t for t in tags if (t["code"] or "").strip().upper() == SET_PIECE),
                  key=lambda t: t["start_frame"])


def tag_starts(tags):
    return np.array([t["start_frame"] for t in set_piece_tags(tags)], dtype=np.int64)


def cues(tr, hom, tags):
    """dict of 14 float arrays over all frames, NaN where undefined."""
    n = len(tr["ball"])
    bx, by = tr["ball"][:, 0].astype(np.float64) * L, tr["ball"][:, 1].astype(np.float64) * W
    seen = np.isfinite(bx)
    c = {"ball_seen": seen.astype(float)}

    speed = np.full(n, np.nan)
    both = seen[1:] & seen[:-1]
    step = np.hypot(np.diff(bx), np.diff(by)) * FPS
    speed[1:][both] = step[both]
    c["ball_speed"] = speed

    bld = line_dist(bx, by)
    c["ball_line_dist_m"] = bld
    with np.errstate(invalid="ignore"):
        c["ball_out"] = (seen & (bld < -0.5)).astype(float)

    ps = tr["ps"].astype(np.float64)
    tracked = np.isfinite(tr["px"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows -> NaN
        c["player_speed"] = np.nanmean(ps, axis=1)
    c["player_speed_1s"] = centred_mean(c["player_speed"], FPS)
    c["n_players"] = tracked.sum(1).astype(float)
    pld = line_dist(tr["px"].astype(np.float64) * L, tr["py"].astype(np.float64) * W)
    with np.errstate(invalid="ignore"):
        c["players_outside"] = (pld < -1.0).sum(1).astype(float)

    cam = np.full(n, np.nan)
    cam[1:] = np.hypot(np.diff(hom["h02"]), np.diff(hom["h12"]))
    c["cam_motion_1s"] = centred_mean(cam, FPS)

    since = np.zeros(n)
    gap_len = np.zeros(n)
    gap_line = np.full(n, np.nan)
    for s, k in zip(*run_lengths(~seen)):
        since[s:s + k] = np.arange(1, k + 1) / FPS
        gap_len[s:s + k] = k / FPS
        if s > 0:
            gap_line[s:s + k] = bld[s - 1]
    c["since_ball_s"], c["gap_len_s"], c["gap_start_line_dist_m"] = since, gap_len, gap_line

    starts = tag_starts(tags)
    in_tag = np.zeros(n)
    for t in set_piece_tags(tags):
        in_tag[max(t["start_frame"], 0):min(t["end_frame"], n)] = 1
    c["in_set_piece_tag"] = in_tag
    idx = np.searchsorted(starts, np.arange(n))
    nxt = np.where(idx < len(starts), starts[np.minimum(idx, len(starts) - 1)], np.nan) \
        if len(starts) else np.full(n, np.nan)
    c["s_to_next_set_piece"] = (nxt - np.arange(n)) / FPS
    return c


def evidence(c, starts, parts=("speed", "ball", "confirm", "tags", "gap")):
    """Per-frame evidence, > 0 favours dead. `parts` selects which terms are summed."""
    n = len(c["player_speed"])
    e = np.zeros(n)
    with np.errstate(invalid="ignore"):
        if "speed" in parts:
            sp = centred_mean(c["player_speed"], SPEED_SMOOTH_S * FPS)
            v = np.clip((SPEED_PIVOT_MS - sp) / SPEED_SCALE_MS, -SPEED_CLIP, SPEED_CLIP)
            e += np.nan_to_num(v, nan=0.0)
        if "ball" in parts:
            slow = centred_median(c["ball_speed"], FPS) < BALL_SLOW_MS
            e += BALL_W * (slow & (c["ball_line_dist_m"] < BALL_NEAR_LINE_M))
        if "confirm" in parts:
            e += BALL_OUT_W * (c["ball_out"] > 0) + PLAYERS_OUT_W * (c["players_outside"] >= 1)
        if "tags" in parts:
            tag = np.zeros(n, bool)
            for s in starts:
                tag[max(s - int(TAG_PRE_S * FPS), 0):min(s + int(TAG_POST_S * FPS), n)] = True
            e += TAG_W * tag
        if "gap" in parts:
            e += GAP_W * ((c["gap_len_s"] > GAP_MIN_S) & (c["gap_start_line_dist_m"] < GAP_LINE_M))
    return e


def viterbi_dead(e, switch_cost=SWITCH_COST):
    """Two states: each frame costs +e/2 if live, -e/2 if dead; each switch costs switch_cost."""
    n = len(e)
    d2l, l2d = np.zeros(n, bool), np.zeros(n, bool)
    live, dead = e[0] / 2, -e[0] / 2
    for t, v in enumerate(e.tolist()[1:], 1):
        stay_l, sw_l = live, dead + switch_cost
        stay_d, sw_d = dead, live + switch_cost
        d2l[t], l2d[t] = sw_l < stay_l, sw_d < stay_d
        live, dead = min(stay_l, sw_l) + v / 2, min(stay_d, sw_d) - v / 2
    out = np.zeros(n, bool)
    state = dead < live
    for t in range(n - 1, -1, -1):
        out[t] = state
        if t and (l2d[t] if state else d2l[t]):
            state = not state
    return out


def min_durations(dead):
    dead = dead.copy()
    for s, k in zip(*run_lengths(~dead)):            # merge short live gaps between dead spells
        if k < MIN_LIVE_S * FPS and s > 0 and s + k < len(dead):
            dead[s:s + k] = True
    for s, k in zip(*run_lengths(dead)):             # then drop short dead spells
        if k < MIN_DEAD_S * FPS:
            dead[s:s + k] = False
    return dead


def snap_restarts(dead, c, starts):
    """Dead from 10 s before each set-piece tag to the first struck ball, live from there."""
    dead, n = dead.copy(), len(dead)
    with np.errstate(invalid="ignore"):
        fast = centred_median(c["ball_speed"], 3) > RESTART_BALL_MS
    for s in starts:
        lo, hi = min(s + int(RESTART_FROM_S * FPS), n), min(s + int(RESTART_SEARCH_S * FPS), n)
        hits = np.flatnonzero(fast[lo:hi])
        r = lo + int(hits[0]) if len(hits) else min(s + int(RESTART_DEFAULT_S * FPS), n)
        dead[max(s - int(TAG_PRE_S * FPS), 0):r] = True
        end = r
        while end < n and dead[end]:
            end += 1
        dead[r:end] = False
    return dead


def detect(c, starts, snap=True, parts=("speed", "ball", "confirm", "tags", "gap")):
    dead = min_durations(viterbi_dead(evidence(c, starts, parts)))
    return snap_restarts(dead, c, starts) if snap else dead
