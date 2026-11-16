"""Coarse sync of the broadcast ATD to the tactical timeline (Track B, step 1).

Both ATD files give positions in the same normalised pitch frame, so at the same moment of play
the players and the ball sit at the same pitch spots whatever either camera was doing. Slot ids
differ (51 vs 46) and the broadcast sees only the players in shot, so the match is team- and
identity-agnostic and one-sided: every broadcast player should be near SOME tactical player.

Convention: tactical_frame = broadcast_frame + offset (both 0-based video frames).

Procedure (all constants fixed from the design, not tuned on any score):
1. Anchor windows of WINDOW_S seconds every ANCHOR_EVERY_S on the broadcast, kept only where its
   homography is valid throughout (live play, not replays or close-ups) and enough players show.
2. For each anchor, every offset on a COARSE_STEP grid over the whole tactical file is scored by
   the window-mean of capped nearest-neighbour distances (players, plus the ball counted
   BALL_WEIGHT times). Then refined at 1-frame steps around the best.
3. Sharpness = best cost of any offset more than EXCLUDE_S away / best cost. Flat curves (players
   standing still, too few in shot) are dropped.
4. Accepted anchors are clustered by offset; the live feed should give one offset per half.

Only aggregates leave this module.
"""

import numpy as np

from . import atd

FPS = 25
WINDOW_S = 20.0
ANCHOR_EVERY_S = 120.0
MIN_PLAYERS = 6
CAP_M = 5.0
BALL_WEIGHT = 3.0
COARSE_STEP = 10            # frames between candidate offsets on the coarse grid
COARSE_SAMPLE = 25          # frames between window samples on the coarse pass (1 s)
REFINE_RADIUS = 15          # frames around the coarse best
REFINE_STEP = 2             # sample spacing within the window when refining
EXCLUDE_S = 2.0
MIN_SHARPNESS = 1.3
CLUSTER_FRAMES = 50
CHUNK = 128


def _metres(xy, pitch):
    return xy * pitch[None, None, :] if xy.ndim == 3 else xy * pitch[None, :]


def _window_cost(bp, bb, tp, tb):
    """Mean capped NN cost for each offset.

    bp (S,P,2) broadcast players, bb (S,2) broadcast ball; tp (O,S,Q,2), tb (O,S,2) tactical at
    each candidate offset. NaN = not observed. Returns (O,) costs.
    """
    d = np.sqrt(((bp[None, :, :, None, :] - tp[:, :, None, :, :]) ** 2).sum(-1))  # O,S,P,Q
    d = np.where(np.isnan(d), np.inf, d)
    nn = np.minimum(d.min(-1), CAP_M)                                           # O,S,P
    pv = ~np.isnan(bp[..., 0])                                                   # S,P
    nn = np.where(pv[None], nn, 0.0)
    num = nn.sum((1, 2))
    den = np.full(len(tp), float(pv.sum()))
    db = np.sqrt(((bb[None] - tb) ** 2).sum(-1))                                 # O,S
    both = np.isfinite(db)
    num = num + BALL_WEIGHT * np.where(both, np.minimum(db, CAP_M), 0.0).sum(1)
    den = den + BALL_WEIGHT * both.sum(1)
    return num / np.maximum(den, 1.0)


def _scan(b_players, b_ball, t_players, t_ball, start, offsets, step):
    """Cost of the broadcast window [start, start + WINDOW) at each offset (samples every step)."""
    s = np.arange(start, start + int(WINDOW_S * FPS), step)
    bp, bb = b_players[s], b_ball[s]
    keep = ~np.all(np.isnan(bp[..., 0]), axis=0)          # drop slots never seen in the window
    bp = bp[:, keep]
    n_t = len(t_players)
    out = np.full(len(offsets), np.inf)
    for k in range(0, len(offsets), CHUNK):
        o = offsets[k:k + CHUNK]
        idx = s[None, :] + o[:, None]
        ok = (idx[:, 0] >= 0) & (idx[:, -1] < n_t)
        if not ok.any():
            continue
        idx = idx[ok]
        out[k:k + CHUNK][ok] = _window_cost(bp, bb, t_players[idx], t_ball[idx])
    return out


def _pack(players):
    """Move observed slots to the front and drop columns that are never used (speed only)."""
    order = np.argsort(np.isnan(players[..., 0]), axis=1, kind="stable")
    packed = np.take_along_axis(players, order[..., None], axis=1)
    width = int(np.isfinite(players[..., 0]).sum(1).max())
    return np.ascontiguousarray(packed[:, :width])


def load_streams():
    tac = atd.load()
    bc = atd.load_broadcast()
    pitch = tac["pitch"]
    import sys
    sys.path.insert(0, atd.KIT)
    from starter import load_homography
    _, _, b_valid = load_homography.load(source="broadcast")
    return (_pack(_metres(tac["players"][..., :2].astype(np.float64), pitch)),
            _metres(tac["ball"][:, :2].astype(np.float64), pitch),
            _metres(bc["players"][..., :2].astype(np.float64), pitch),
            _metres(bc["ball"][:, :2].astype(np.float64), pitch),
            b_valid)


def anchors(b_players, b_valid):
    w = int(WINDOW_S * FPS)
    starts = np.arange(0, len(b_players) - w, int(ANCHOR_EVERY_S * FPS))
    n_vis = np.isfinite(b_players[..., 0]).sum(1)
    good = [int(a) for a in starts
            if b_valid[a:a + w].all() and n_vis[a:a + w].mean() >= MIN_PLAYERS]
    return good, len(starts)


def fit_anchor(streams, start):
    tp, tb, bp, bb, _ = streams
    lo, hi = -start, len(tp) - start - int(WINDOW_S * FPS)
    coarse = np.arange(lo - lo % COARSE_STEP, hi, COARSE_STEP)
    cost = _scan(bp, bb, tp, tb, start, coarse, COARSE_SAMPLE)
    i = int(np.argmin(cost))
    best = coarse[i]
    far = np.abs(coarse - best) > EXCLUDE_S * FPS
    sharp = float(cost[far].min() / max(cost[i], 1e-6))
    fine = np.arange(best - REFINE_RADIUS, best + REFINE_RADIUS + 1)
    fcost = _scan(bp, bb, tp, tb, start, fine, REFINE_STEP)
    j = int(np.argmin(fcost))
    return {"start": start, "offset": int(fine[j]), "cost_m": float(fcost[j]),
            "sharpness": sharp}


def mirror_cost(streams, start, offset, pitch):
    tp, tb, bp, bb, _ = streams
    flip = np.array([pitch[0], pitch[1]])
    return float(_scan(flip - bp, flip - bb, tp, tb, start, np.array([offset]), REFINE_STEP)[0])


def cluster(fits):
    acc = sorted([f for f in fits if f["sharpness"] >= MIN_SHARPNESS], key=lambda f: f["offset"])
    groups = []
    for f in acc:
        if groups and f["offset"] - groups[-1][-1]["offset"] <= CLUSTER_FRAMES:
            groups[-1].append(f)
        else:
            groups.append([f])
    out = []
    for g in sorted(groups, key=len, reverse=True):
        off = np.array([f["offset"] for f in g], dtype=float)
        st = np.array([f["start"] for f in g], dtype=float)
        slope = float(np.polyfit(st, off, 1)[0]) * 10 * 60 * FPS if len(g) >= 3 else None
        out.append({"n": len(g), "offset_median": int(np.median(off)),
                    "offset_min": int(off.min()), "offset_max": int(off.max()),
                    "broadcast_from": int(st.min()), "broadcast_to": int(st.max()),
                    "median_cost_m": round(float(np.median([f["cost_m"] for f in g])), 2),
                    "drift_frames_per_10min": None if slope is None else round(slope, 2)})
    return out, len(acc)


def ball_agreement(streams, offset, b_from, b_to):
    """Share of broadcast frames in [b_from, b_to] (both balls seen) within 1 m / 3 m."""
    tp, tb, bp, bb, valid = streams
    b = np.arange(b_from, b_to + 1)
    t = b + offset
    ok = (t >= 0) & (t < len(tb))
    b, t = b[ok], t[ok]
    d = np.sqrt(((bb[b] - tb[t]) ** 2).sum(-1))
    m = np.isfinite(d) & valid[b]
    return {"n": int(m.sum()),
            "within_1m": round(float((d[m] <= 1).mean()), 3) if m.any() else None,
            "within_3m": round(float((d[m] <= 3).mean()), 3) if m.any() else None}


# ---- Step 2: per-frame verification ---------------------------------------------------------
#
# Each broadcast frame is checked at each half's offset with the PLAYERS ONLY (capped NN cost,
# smoothed over SMOOTH_S). The ball is kept out so it can validate the result independently.
# The acceptance threshold comes from the same cost at deliberately wrong offsets (WRONG_SHIFTS_S,
# i.e. what a replay looks like): a frame is verified if its cost is below the WRONG_QUANTILE of
# that "wrong" distribution. No labels involved.

OFFSETS = {1: 4014, 2: 26919}     # from step 1 (scripts/sync_report.py)
SMOOTH_S = 2.0
FRAME_MIN_PLAYERS = 4
WRONG_SHIFTS_S = (-30.0, -10.0, 10.0, 30.0)
WRONG_QUANTILE = 0.05
FRAME_CHUNK = 4096


def frame_costs(tp, bp, offset):
    """(m,) players-only capped NN cost of each broadcast frame at tactical = b + offset (NaN if
    the frame has too few players or maps outside the tactical file)."""
    m = len(bp)
    out = np.full(m, np.nan)
    for a in range(0, m, FRAME_CHUNK):
        b = np.arange(a, min(m, a + FRAME_CHUNK))
        t = b + offset
        ok = (t >= 0) & (t < len(tp))
        b, t = b[ok], t[ok]
        P, Q = bp[b], tp[t]                                        # (k,P,2), (k,Q,2)
        d = np.sqrt(((P[:, :, None, :] - Q[:, None, :, :]) ** 2).sum(-1))
        d = np.where(np.isnan(d), np.inf, d)
        nn = np.minimum(d.min(-1), CAP_M)
        pv = ~np.isnan(P[..., 0])
        n = pv.sum(1)
        c = np.where(pv, nn, 0.0).sum(1) / np.maximum(n, 1)
        out[b] = np.where(n >= FRAME_MIN_PLAYERS, c, np.nan)
    return out


def smooth(x, half):
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.clip(i - half, 0, len(x)), np.clip(i + half + 1, 0, len(x))
    n = cn[hi] - cn[lo]
    return np.where(ok & (n > 0), (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def verify(streams, offsets=OFFSETS):
    """Per broadcast frame: chosen half (0 = none), tactical frame (-1 = none), and diagnostics."""
    tp, tb, bp, bb, valid = streams
    half = int(SMOOTH_S * FPS / 2)
    cost = {h: smooth(frame_costs(tp, bp, o), half) for h, o in offsets.items()}
    wrong = []
    for h, o in offsets.items():
        for s in WRONG_SHIFTS_S:
            w = smooth(frame_costs(tp, bp, o + int(s * FPS)), half)
            wrong.append(w[valid & np.isfinite(w)])
    wrong = np.concatenate(wrong)
    threshold = float(np.quantile(wrong, WRONG_QUANTILE))
    m = len(bp)
    stack = np.stack([np.where(np.isfinite(cost[h]), cost[h], np.inf) for h in sorted(offsets)])
    best = np.argmin(stack, 0)
    best_cost = stack[best, np.arange(m)]
    ok = valid & (best_cost < threshold)
    chosen = np.where(ok, np.array(sorted(offsets))[best], 0)
    tac = np.full(m, -1, dtype=np.int64)
    for h, o in offsets.items():
        sel = chosen == h
        tac[sel] = np.arange(m)[sel] + o
    both_ok = valid & (stack < threshold).all(0)
    return {"half": chosen, "tactical_frame": tac, "threshold_m": threshold,
            "cost": cost, "wrong_median_m": float(np.median(wrong)), "both_halves_fit": both_ok}


def runs(mask):
    """Lengths (frames) of True runs."""
    x = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.flatnonzero(np.diff(x))
    return d[1::2] - d[::2]
