"""Learned detector of possession-changing touches, trained on DFL open data (src/barca/dfl.py).

Target (DFL): frames where the last-touch team flips H <-> A with the ball alive on both sides (a
flip through a dead spell is a restart, not a touch). DFL's BallPossession behaves like "last
touch", so its flips mark the moment the other team touched the ball.

Features are tracking-only and computable on the BAR-ATH ATD (ball height left out: missing on
~90% of ATD frames). One row per frame and candidate team X (O = the other team):
  dX, dO           nearest X / O player to the ball (m), and their minimum over +-4 frames
  relX             ball speed relative to X's nearest player (m/s)
  speed, dv4, dv2  ball speed; change of mean velocity over 4 / 2 frames each side (m/s)
  turn             change of direction over 4 frames each side (degrees)
  ctxX             share of frames t-50..t-6 where X was the nearer team (who had it before)
  ctx_dX, ctx_dO   mean nearest distances t-25..t-6
  nX5, nO5         players of each team within 5 m
  line             ball distance to the nearest pitch line (m, positive inside)
Label: 1 if a flip TO X lies within +-LABEL_W frames.

Events: per frame the better team's probability; local maxima >= threshold, at least SEP frames
apart; then alternation (a flip must go to the other team than the previous kept flip, within a
live spell).
"""

import numpy as np
import pandas as pd

from barca import touches

FPS = 25
LABEL_W = 2
SEP = 10
FEATURES = ["dX", "dO", "dXw", "dOw", "relX", "speed", "dv4", "dv2", "turn",
            "ctxX", "ctx_dX", "ctx_dO", "nX5", "nO5", "line"]
# 2026-11-16, touch context (features(..., context=True)), fixed before testing:
CTX_WINDOWS = (6, 12, 25)          # frames each side: 0.25, 0.5, 1 s
CTX_FEATURES = ["%s_%d" % (f, w) for w in CTX_WINDOWS for f in ("dv4_max", "turn_max", "dX_min", "dO_min", "relX_min")] + ["dv4_peak"]


def _roll(x, win, how, shift=0, center=False):
    s = pd.Series(x).rolling(win, min_periods=1, center=center)
    r = getattr(s, how)().to_numpy()
    return np.roll(r, shift) if shift else r


def _vel(xy):
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * FPS / 2.0
    return v


def _turn(xy, k=4):
    n = len(xy)
    out = np.full(n, np.nan)
    step = np.full((n, 2), np.nan)
    step[1:] = xy[1:] - xy[:-1]
    cs = np.vstack([np.zeros((1, 2)), np.nancumsum(step, 0)])
    t = np.arange(k, n - k)
    before = cs[t + 1] - cs[t - k + 1]
    after = cs[t + k + 1] - cs[t + 1]
    nb, na = np.hypot(*before.T), np.hypot(*after.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", before, after) / (nb * na)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang[(nb < 1e-3) | (na < 1e-3)] = 0.0
    out[t] = ang
    return out


def features(players, ball, slot_team, pitch, context=False):
    """{0: (n, F) for X = home, 1: (n, F) for X = away}. Inputs as atd.load() (normalised)."""
    L, W = float(pitch[0]), float(pitch[1])
    b = ball[:, :2].astype(np.float64) * [L, W]
    p = players[..., :2].astype(np.float64) * [L, W]
    with np.errstate(invalid="ignore"):
        d = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    vb, vp = _vel(b), _vel(p)
    speed = np.hypot(*vb.T)
    dv4 = touches.velocity_change_fast(b, 4)
    dv2 = touches.velocity_change_fast(b, 2)
    turn = _turn(b)
    line = np.minimum(np.minimum(b[:, 0], L - b[:, 0]), np.minimum(b[:, 1], W - b[:, 1]))
    seen = np.isfinite(b[:, 0])
    per = {}
    for team in (0, 1):
        m = slot_team == team
        dm = np.where(m[None], d, np.inf)
        j = dm.argmin(1)
        dist = dm[np.arange(len(d)), j]
        dist = np.where(np.isfinite(dist) & seen, dist, np.nan)
        rel = np.hypot(*(vb - vp[np.arange(len(d)), j]).T)
        n5 = np.where(seen, (dm < 5).sum(1), np.nan)
        per[team] = (dist, rel, n5)
    nearer = np.where(np.isfinite(per[0][0]) & np.isfinite(per[1][0]),
                      (per[0][0] <= per[1][0]).astype(float), np.nan)
    out = {}
    for X in (0, 1):
        O = 1 - X
        dX, relX, nX5 = per[X]
        dO, _, nO5 = per[O]
        isX = nearer if X == 0 else 1.0 - nearer
        cols = {
            "dX": dX, "dO": dO,
            "dXw": _roll(dX, 9, "min", center=True), "dOw": _roll(dO, 9, "min", center=True),
            "relX": relX, "speed": speed, "dv4": dv4, "dv2": dv2, "turn": turn,
            "ctxX": _roll(isX, 45, "mean", shift=6), "ctx_dX": _roll(dX, 20, "mean", shift=6),
            "ctx_dO": _roll(dO, 20, "mean", shift=6), "nX5": nX5, "nO5": nO5, "line": line}
        names = FEATURES
        if context:
            for w in CTX_WINDOWS:
                win = 2 * w + 1
                cols["dv4_max_%d" % w] = _roll(dv4, win, "max", center=True)
                cols["turn_max_%d" % w] = _roll(turn, win, "max", center=True)
                cols["dX_min_%d" % w] = _roll(dX, win, "min", center=True)
                cols["dO_min_%d" % w] = _roll(dO, win, "min", center=True)
                cols["relX_min_%d" % w] = _roll(relX, win, "min", center=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols["dv4_peak"] = dv4 / _roll(dv4, 51, "mean", center=True)
            names = FEATURES + CTX_FEATURES
        out[X] = np.column_stack([cols[f] for f in names]).astype(np.float32)
    return out


def flips(labels, blocks):
    """[(frame, new_team)]: live H <-> A changes inside blocks (labels 0 H, 1 A, 2 D)."""
    out = []
    for a, b in blocks:
        w = labels[a:b + 1]
        idx = np.flatnonzero((w[1:] != w[:-1]) & (w[1:] != 2) & (w[:-1] != 2)) + 1
        out += [(a + int(i), int(w[i])) for i in idx]
    return out


def targets(n, fl):
    y = {0: np.zeros(n, np.int8), 1: np.zeros(n, np.int8)}
    for f, t in fl:
        y[t][max(f - LABEL_W, 0):f + LABEL_W + 1] = 1
    return y


def events(p0, p1, live, threshold, sep=SEP):
    """[(frame, team)] from per-team probabilities: peaks, spacing, then alternation per live spell."""
    p = np.where(live, np.fmax(np.nan_to_num(p0), np.nan_to_num(p1)), 0.0)
    team = (np.nan_to_num(p1) > np.nan_to_num(p0)).astype(int)
    peak = (p >= threshold) & (p >= _roll(p, 2 * 5 + 1, "max", center=True))
    cand = np.flatnonzero(peak)
    taken, kept = np.zeros(len(p), bool), []
    for t in cand[np.argsort(-p[cand], kind="stable")]:
        if taken[max(t - sep, 0):t + sep + 1].any():
            continue
        taken[t] = True
        kept.append(int(t))
    return alternate(sorted(kept), team, live)


def alternate(frames, team, live):
    start = live & ~np.r_[False, live[:-1]]
    spell = np.cumsum(start)
    out, last, cur = [], None, None
    for t in frames:
        if spell[t] != cur:
            cur, last = spell[t], None
        if last is None or team[t] != last:
            out.append((t, int(team[t])))
            last = team[t]
    return out


SNAP_BEFORE_S = 5.0     # pre-registered 2026-11-14: a change may move up to 5 s earlier ...
SNAP_AFTER_S = 1.0      # ... or 1 s later


def live_changes(labels, live):
    return np.flatnonzero(live[1:] & live[:-1] & (labels[1:] != labels[:-1])
                          & (labels[1:] < 2) & (labels[:-1] < 2)) + 1


def snap_changes(labels, live, ev, before_s=SNAP_BEFORE_S, after_s=SNAP_AFTER_S):
    """Move each live H <-> A change to the nearest detected flip to the NEW team within
    [c - before, c + after], never crossing the neighbouring change or the spell start.
    Returns (labels, list of moves in frames: event - change)."""
    out = labels.copy()
    start = live & ~np.r_[False, live[:-1]]
    spell_start = np.maximum.accumulate(np.where(start, np.arange(len(live)), 0))
    ch = live_changes(labels, live)
    ev_f = {t: np.array(sorted(f for f, tt in ev if tt == t), dtype=np.int64) for t in (0, 1)}
    moves = []
    for i, c in enumerate(ch):
        new, old = int(labels[c]), int(labels[c - 1])
        prev = ch[i - 1] if i and ch[i - 1] >= spell_start[c] else spell_start[c]
        nxt = ch[i + 1] if i + 1 < len(ch) and spell_start[ch[i + 1]] == spell_start[c] else len(labels)
        lo, hi = max(c - int(before_s * FPS), prev + 1), min(c + int(after_s * FPS), nxt - 1)
        f = ev_f[new]
        cand = f[(f >= lo) & (f <= hi)]
        if not len(cand):
            continue
        g = int(cand[np.argmin(np.abs(cand - c))])
        if g < c:
            out[g:c] = new
        elif g > c:
            out[c:g] = old
        moves.append(g - c)
    return out, moves


def match_events(true, pred, tol):
    """(tp, fp, fn): greedy one-to-one by time, same team, |dt| <= tol."""
    used = set()
    tp = 0
    tf = np.array([f for f, _ in true])
    tt = np.array([t for _, t in true])
    for f, t in sorted(pred):
        if not len(tf):
            break
        c = np.flatnonzero((np.abs(tf - f) <= tol) & (tt == t))
        c = [i for i in c if i not in used]
        if c:
            used.add(min(c, key=lambda i: abs(tf[i] - f)))
            tp += 1
    return tp, len(pred) - tp, len(true) - tp
