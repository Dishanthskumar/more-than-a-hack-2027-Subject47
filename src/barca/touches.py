"""Ball touches as kinks in the ball's path: a sudden change of velocity (speed or direction).

A pass, header, block or deflection changes the ball's velocity within a frame or two; in free
flight or rolling it changes slowly. The ATD ball track is smooth (turn angle at > 3 m/s:
median 0.6 deg, 99th percentile 12.9 deg per frame), so the velocity change between the K frames
before and the K frames after a frame marks a touch candidate.

Constants fixed 2026-11-10 from the track's noise only, before any study or scoring:
  K = 4 frames (0.16 s) each side; a candidate needs every frame in the window seen and no
  per-frame step over MAX_STEP_M (the ball_fusion glitch gate); peaks >= MIN_DV_MS, at least SEP_S
  apart.
"""

import numpy as np

from barca.loaders import FPS, PITCH_M

K = 4
MIN_DV_MS = 3.0
SEP_S = 0.4
MAX_STEP_M = 2.0


def ball_xy_m(tr):
    b = tr["ball"].astype(np.float64)
    return b[:, :2] * np.asarray(PITCH_M)


def velocity_change(xy, k=K):
    """(n,) |mean velocity of the k steps after t - mean of the k steps before t| (m/s), NaN if unknown."""
    n = len(xy)
    step = np.full((n, 2), np.nan)
    step[1:] = xy[1:] - xy[:-1]                    # step[t] = move from t-1 to t
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(step).all(1) | (np.hypot(*step.T) > MAX_STEP_M)
    step[bad] = np.nan
    out = np.full(n, np.nan)
    for t in range(k, n - k):
        before = step[t - k + 1:t + 1]             # moves ending at t
        after = step[t + 1:t + k + 1]              # moves starting at t
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        out[t] = np.hypot(*(after.mean(0) - before.mean(0))) * FPS
    return out


def velocity_change_fast(xy, k=K):
    """Vectorised velocity_change (same result)."""
    n = len(xy)
    step = np.full((n, 2), np.nan)
    step[1:] = xy[1:] - xy[:-1]
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(step).all(1) | (np.hypot(*step.T) > MAX_STEP_M)
    step[bad] = np.nan
    good = ~np.isnan(step[:, 0])
    s = np.where(good[:, None], step, 0.0)
    cs = np.vstack([np.zeros((1, 2)), np.cumsum(s, 0)])          # cs[i] = sum step[:i]
    cg = np.concatenate([[0], np.cumsum(good)])
    out = np.full(n, np.nan)
    t = np.arange(k, n - k)
    before = (cs[t + 1] - cs[t - k + 1]) / k
    after = (cs[t + k + 1] - cs[t + 1]) / k
    ok = (cg[t + 1] - cg[t - k + 1] == k) & (cg[t + k + 1] - cg[t + 1] == k)
    out[t[ok]] = np.hypot(*(after[ok] - before[ok]).T) * FPS
    return out


def kinks(dv, min_dv=MIN_DV_MS, sep_s=SEP_S):
    """Frames of local maxima of dv >= min_dv, greedily kept strongest first, >= sep_s apart."""
    sep = int(sep_s * FPS)
    cand = np.flatnonzero(np.nan_to_num(dv, nan=0.0) >= min_dv)
    taken = np.zeros(len(dv), bool)
    out = []
    for t in cand[np.argsort(-dv[cand], kind="stable")]:
        if taken[max(t - sep, 0):t + sep + 1].any():
            continue
        taken[t] = True
        out.append(int(t))
    return np.array(sorted(out), dtype=np.int64)


# Touch rule (pre-registered 2026-11-10, before scoring; see scripts/touch_eval.py).
TOUCH_DIST_M = 3.0             # toucher's nearest-player distance
TOUCH_DV_MAX_MS = 8.0          # stronger kinks are mostly far from players (noise, bounces)
TOUCH_MARGIN_M = 1.0           # T2 only: the toucher at least 1 m closer than the other side
TOUCH_MIN_S = 0.5              # a new possession at least as long as a control event's dwell


def qualifies(t, team, dist, margin, dv, need_margin):
    with np.errstate(invalid="ignore"):
        ok = team[t] >= 0 and dist[t] < TOUCH_DIST_M and MIN_DV_MS <= dv[t] < TOUCH_DV_MAX_MS
        return bool(ok and (not need_margin or margin[t] >= TOUCH_MARGIN_M))


def apply(labels, live, kinks_, team, dist, margin, dv, events, need_margin, dead_code=2):
    """Each qualifying kink by the side NOT in the current label starts a possession for the
    toucher, lasting until the other side's next control event or qualifying kink, or the end of
    the live spell; kept only if >= TOUCH_MIN_S. Kinks inside a control event are ignored.
    Returns (labels, list of (start, stop, team) possessions added)."""
    out = labels.copy()
    n = len(out)
    ev = sorted(events)
    in_ctrl = np.zeros(n, bool)
    for s, e, _ in ev:
        in_ctrl[s:e] = True
    q = [int(t) for t in kinks_ if live[t] and not in_ctrl[t]
         and qualifies(t, team, dist, margin, dv, need_margin)]
    added = []
    for i, t in enumerate(q):
        tt = int(team[t])
        if out[t] == tt or out[t] == dead_code:
            continue
        end = t
        while end < n and live[end]:
            end += 1
        nxt_ev = next((s for s, e, et in ev if s > t and et != tt), n)
        nxt_k = next((u for u in q[i + 1:] if team[u] != tt), n)
        end = min(end, nxt_ev, nxt_k)
        if end - t < TOUCH_MIN_S * FPS:
            continue
        out[t:end] = tt
        added.append((t, end, tt))
    return out, added
