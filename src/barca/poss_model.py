"""Learned team-in-possession model, trained on degraded DFL open data (team layer).

Pre-registered 2026-11-14 (see scripts/poss_model_eval.py for the bench and the go rule).
Per live frame, P(home in possession) from tracking-only features computed the same way on DFL
and on the BAR-ATH ATD, then a 2-state Viterbi inside each live spell (switch cost from the bench).

Features (home perspective; "xa" = x along home's attacking direction, 0 = home goal, 1 = away goal):
  dH, dA, dHw, dAw, margin  nearest home / away player to the ball (m), min over +-4 frames, dA - dH
  pTH, pTA                  learned touch detector: probability of a flip TO home / away here
  lastT, lastT_age, nextT, nextT_in   last / next detected touch flip: team (+1 H, -1 A, 0) and s
  lastC, lastC_age, nextC, nextC_in, inC   same for control events (possession.control_events)
  near_past2, near_fut2     share of the past / next 2 s with home nearer the ball
  ball_xa, ball_yc          ball position (xa; distance from the centre line across, 0..0.5)
  cH_xa, cA_xa, spreadH, spreadA   team centroids (xa) and spreads (m)
  bspeed                    ball speed (m/s)
"""

import numpy as np
import pandas as pd

from barca import touch_model as TM

FPS = 25
CAP_S = 30.0
FEATURES = ["dH", "dA", "dHw", "dAw", "margin", "pTH", "pTA", "lastT", "lastT_age", "nextT", "nextT_in",
            "lastC", "lastC_age", "nextC", "nextC_in", "inC", "near_past2", "near_fut2",
            "ball_xa", "ball_yc", "cH_xa", "cA_xa", "spreadH", "spreadA", "bspeed"]
# 2026-11-15, multi-scale context (features(..., context=True)), fixed before testing:
CTX_WINDOWS_S = (0.5, 1, 2, 4)
CTX_FEATURES = ["%s_%gs" % (f, w) for w in CTX_WINDOWS_S for f in ("pTH_max", "pTA_max", "margin_mean", "near_share")] +     ["ball_vxa", "ball_vxa_1s", "nH_ahead", "nA_ahead", "dH_trend", "dA_trend"]
# 2026-11-15, context round 2 (features(..., context=2)), fixed before testing:
CTX2_FEATURES = ["dH2", "dH3", "dA2", "dA3", "relH", "relA", "recvH", "recvA",
                 "pTH_max_8s", "pTA_max_8s", "margin_mean_8s", "near_share_8s",
                 "lastT2", "lastT3", "nTH_4s", "nTA_4s", "lastC2", "spell_age", "spell_start_near"]


def _roll(x, win, how, center=True):
    return getattr(pd.Series(x).rolling(int(win), center=center, min_periods=1), how)().to_numpy()


def _last_next(frames, teams, n, live):
    """Per frame: team / age (s) of the last event at or before t, team / wait (s) of the next after
    t, both within the same live spell (0 and CAP_S otherwise)."""
    start = live & ~np.r_[False, live[:-1]]
    spell = np.cumsum(start) * live
    lt, la, nt, ni = np.zeros(n), np.full(n, CAP_S), np.zeros(n), np.full(n, CAP_S)
    ev_team = np.zeros(n)
    has = np.zeros(n, bool)
    for f, t in zip(frames, teams):
        has[f], ev_team[f] = True, 1.0 if t == 0 else -1.0
    last_f, last_t = -1, 0.0
    for i in range(n):
        if has[i]:
            last_f, last_t = i, ev_team[i]
        if last_f >= 0 and live[i] and spell[last_f] == spell[i]:
            lt[i], la[i] = last_t, min((i - last_f) / FPS, CAP_S)
    nxt_f, nxt_t = -1, 0.0
    for i in range(n - 1, -1, -1):
        if nxt_f >= 0 and live[i] and spell[nxt_f] == spell[i]:
            nt[i], ni[i] = nxt_t, min((nxt_f - i) / FPS, CAP_S)
        if has[i]:
            nxt_f, nxt_t = i, ev_team[i]
    return lt, la, nt, ni


def attack_sign(players, slot_team, blocks, pitch):
    """Per frame +1 if home attacks towards x = L in this block, else -1 (from mean team x)."""
    L = float(pitch[0])
    x = players[..., 0].astype(np.float64) * L
    out = np.ones(len(x))
    st = np.asarray(slot_team)
    for a, b in blocks:
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            h = np.nanmean(x[a:b + 1][:, st == 0])
            aw = np.nanmean(x[a:b + 1][:, st == 1])
        out[a:b + 1] = 1.0 if h < aw else -1.0
    return out


def features(players, ball, slot_team, pitch, blocks, live, p_touch, touch_events, control_events, context=False):
    L, W = float(pitch[0]), float(pitch[1])
    n = len(ball)
    b = ball[:, :2].astype(np.float64) * [L, W]
    p = players[..., :2].astype(np.float64) * [L, W]
    st = np.asarray(slot_team)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        d = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1])
        dH = np.nanmin(np.where(st[None] == 0, d, np.nan), 1)
        dA = np.nanmin(np.where(st[None] == 1, d, np.nan), 1)
        cH = np.stack([np.nanmean(p[:, st == 0, 0], 1), np.nanmean(p[:, st == 0, 1], 1)], 1)
        cA = np.stack([np.nanmean(p[:, st == 1, 0], 1), np.nanmean(p[:, st == 1, 1], 1)], 1)
        sH = np.sqrt(np.nanvar(p[:, st == 0, 0], 1) + np.nanvar(p[:, st == 0, 1], 1))
        sA = np.sqrt(np.nanvar(p[:, st == 1, 0], 1) + np.nanvar(p[:, st == 1, 1], 1))
        nearer = np.where(np.isfinite(dH) & np.isfinite(dA), (dH <= dA).astype(float), np.nan)
    sign = attack_sign(players, slot_team, blocks, pitch)
    xa = lambda x: np.where(sign > 0, x / L, 1 - x / L)  # noqa: E731
    bs = np.full(n, np.nan)
    bs[1:-1] = np.hypot(*(b[2:] - b[:-2]).T) * FPS / 2
    tf = [f for f, _ in touch_events]
    tt = [t for _, t in touch_events]
    lT, lTa, nT, nTi = _last_next(tf, tt, n, live)
    cf = [s for s, _, _ in control_events]
    ct = [t for _, _, t in control_events]
    lC, lCa, nC, nCi = _last_next(cf, ct, n, live)
    inC = np.zeros(n)
    for s, e, t in control_events:
        inC[s:e] = 1.0 if t == 0 else -1.0
    cols = {"dH": dH, "dA": dA, "dHw": _roll(dH, 9, "min"), "dAw": _roll(dA, 9, "min"), "margin": dA - dH,
            "pTH": p_touch[0], "pTA": p_touch[1], "lastT": lT, "lastT_age": lTa, "nextT": nT, "nextT_in": nTi,
            "lastC": lC, "lastC_age": lCa, "nextC": nC, "nextC_in": nCi, "inC": inC,
            "near_past2": np.r_[np.nan, _roll(nearer, 2 * FPS, "mean", center=False)[:-1]],
            "near_fut2": np.r_[np.nan, _roll(nearer[::-1], 2 * FPS, "mean", center=False)[:-1]][::-1],
            "ball_xa": xa(b[:, 0]), "ball_yc": np.abs(b[:, 1] / W - 0.5),
            "cH_xa": xa(cH[:, 0]), "cA_xa": xa(cA[:, 0]), "spreadH": sH, "spreadA": sA, "bspeed": bs}
    names = FEATURES
    if context:
        for w in CTX_WINDOWS_S:
            win = max(int(round(2 * w * FPS)) + 1, 3)
            cols["pTH_max_%gs" % w] = _roll(p_touch[0], win, "max")
            cols["pTA_max_%gs" % w] = _roll(p_touch[1], win, "max")
            cols["margin_mean_%gs" % w] = _roll(dA - dH, win, "mean")
            cols["near_share_%gs" % w] = _roll(nearer, win, "mean")
        vx = np.full(n, np.nan)
        vx[1:-1] = (b[2:, 0] - b[:-2, 0]) * FPS / 2
        cols["ball_vxa"] = vx * sign
        cols["ball_vxa_1s"] = _roll(vx * sign, FPS + 1, "mean")
        bxa = xa(b[:, 0])
        pxa = np.where(sign[:, None] > 0, p[..., 0] / L, 1 - p[..., 0] / L)
        with np.errstate(invalid="ignore"):
            ahead = pxa > bxa[:, None]
        cols["nH_ahead"] = np.where(np.isfinite(bxa), (ahead & (st[None] == 0)).sum(1), np.nan)
        cols["nA_ahead"] = np.where(np.isfinite(bxa), (ahead & (st[None] == 1)).sum(1), np.nan)
        h = FPS // 2
        cols["dH_trend"] = np.r_[np.full(h, np.nan), dH[2 * h:] - dH[:-2 * h], np.full(h, np.nan)]
        cols["dA_trend"] = np.r_[np.full(h, np.nan), dA[2 * h:] - dA[:-2 * h], np.full(h, np.nan)]
        names = FEATURES + CTX_FEATURES
    if context and context >= 2:
        _context2(cols, p, b, st, d, dH, dA, nearer, p_touch, touch_events, control_events, live, n)
        names = names + CTX2_FEATURES
    return np.column_stack([cols[f] for f in names]).astype(np.float32)


def _history(frames, teams, n, live, k):
    """(n, k): teams (+1 H / -1 A) of the 1st..k-th most recent events at or before t, same live spell."""
    start = live & ~np.r_[False, live[:-1]]
    spell = np.cumsum(start) * live
    out = np.zeros((n, k))
    ev = np.zeros(n)
    has = np.zeros(n, bool)
    for f, t in zip(frames, teams):
        if 0 <= f < n:
            has[f], ev[f] = True, 1.0 if t == 0 else -1.0
    hist, cur = [], None
    for i in range(n):
        if spell[i] != cur:
            hist, cur = [], spell[i]
        if has[i] and live[i]:
            hist = [ev[i]] + hist[:k - 1]
        if live[i]:
            out[i, :len(hist)] = hist
    return out


def _context2(cols, p, b, st, d, dH, dA, nearer, p_touch, touch_events, control_events, live, n):
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        for team, tag in ((0, "H"), (1, "A")):
            dd = np.sort(np.where(st[None] == team, d, np.nan), 1)
            cols["d%s2" % tag], cols["d%s3" % tag] = dd[:, 1], dd[:, 2]
        vb = np.full_like(b, np.nan)
        vb[1:-1] = (b[2:] - b[:-2]) * FPS / 2
        vp = np.full_like(p, np.nan)
        vp[1:-1] = (p[2:] - p[:-2]) * FPS / 2
        dmask = np.where(np.isfinite(d), d, np.inf)
        for team, tag in ((0, "H"), (1, "A")):
            j = np.where(st[None] == team, dmask, np.inf).argmin(1)
            cols["rel%s" % tag] = np.hypot(*(vb - vp[np.arange(n), j]).T)
        # likely receiver: player distance to the ball's straight path over the next 1 s
        rel = p - b[:, None]
        v2 = (vb ** 2).sum(1)
        tproj = np.clip((rel * vb[:, None]).sum(2) / np.where(v2 > 1e-6, v2, np.nan)[:, None], 0, 1.0)
        point = b[:, None] + vb[:, None] * tproj[..., None]
        perp = np.hypot(*(p - point).transpose(2, 0, 1))
        for team, tag in ((0, "H"), (1, "A")):
            cols["recv%s" % tag] = np.nanmin(np.where(st[None] == team, perp, np.nan), 1)
    win = 2 * 8 * FPS + 1
    cols["pTH_max_8s"] = _roll(p_touch[0], win, "max")
    cols["pTA_max_8s"] = _roll(p_touch[1], win, "max")
    cols["margin_mean_8s"] = _roll(dA - dH, win, "mean")
    cols["near_share_8s"] = _roll(nearer, win, "mean")
    th = _history([f for f, _ in touch_events], [t for _, t in touch_events], n, live, 3)
    cols["lastT2"], cols["lastT3"] = th[:, 1], th[:, 2]
    for team, tag in ((0, "H"), (1, "A")):
        ind = np.zeros(n)
        for f, t in touch_events:
            if t == team and 0 <= f < n:
                ind[f] = 1
        cols["nT%s_4s" % tag] = np.r_[0, _roll(ind, 4 * FPS, "sum", center=False)[:-1]]
    ch = _history([s for s, _, _ in control_events], [t for _, _, t in control_events], n, live, 2)
    cols["lastC2"] = ch[:, 1]
    start = live & ~np.r_[False, live[:-1]]
    idx = np.where(start, np.arange(n), 0)
    s0 = np.maximum.accumulate(idx)
    cols["spell_age"] = np.where(live, np.minimum((np.arange(n) - s0) / FPS, CAP_S), 0.0)
    near0 = np.where(np.isfinite(nearer), nearer * 2 - 1, 0.0)
    cols["spell_start_near"] = np.where(live, np.array([near0[k] for k in s0]), 0.0)


def decode(p_home, live, switch_cost):
    """2-state Viterbi inside each live spell on logit(p_home); dead frames -> 2."""
    n = len(p_home)
    out = np.full(n, 2, np.int8)
    q = np.clip(p_home, 1e-4, 1 - 1e-4)
    e = np.log(q / (1 - q))
    start = live & ~np.r_[False, live[:-1]]
    for s in np.flatnonzero(start):
        t = s
        while t < n and live[t]:
            t += 1
        seg = (e[s:t] / 2).tolist()
        m, c = len(seg), float(switch_cost)
        bh, ba = bytearray(m), bytearray(m)           # back-pointers for home (0) / away (1)
        h, a = seg[0], -seg[0]                        # home scores +e/2, away -e/2
        for i in range(1, m):
            x = seg[i]
            if h >= a - c:
                nh = h
            else:
                nh, bh[i] = a - c, 1
            if a >= h - c:
                na, ba[i] = a, 1
            else:
                na = h - c
            h, a = nh + x, na - x
        path = np.empty(m, np.int8)
        state = 0 if h >= a else 1
        for i in range(m - 1, -1, -1):
            path[i] = state
            if i:
                state = bh[i] if state == 0 else ba[i]
        out[s:t] = path
    return out


def absorb_short(labels, min_s):
    """A team run shorter than min_s with the OTHER team on both sides (inside one live spell)
    goes to that team, repeated until stable: the 'controlled possession' rule applied to the
    prediction. min_s = 0 leaves labels unchanged."""
    if min_s <= 0:
        return labels
    out = labels.copy()
    lim = min_s * FPS
    while True:
        cut = np.flatnonzero(out[1:] != out[:-1]) + 1
        bd = np.r_[0, cut, len(out)]
        runs = [(s, e, int(out[s])) for s, e in zip(bd[:-1], bd[1:])]
        changed = False
        for i in range(1, len(runs) - 1):
            s, e, k = runs[i]
            pk, nk = runs[i - 1][2], runs[i + 1][2]
            if k < 2 and e - s < lim and pk == nk and pk < 2 and pk != k:
                out[s:e] = pk
                changed = True
        if not changed:
            return out


MODEL_FILE = "poss_model.pkl"      # in dfl.CACHE, written by scripts/poss_model_eval.py --bar


D1 = (8, 3)        # decoder D1: switch cost 8, then absorb_short 3 s (DFL study; better on Metrica)


def bar_features(S, labels, saved):
    """(F, live) on the BAR-ATH ATD for a saved model dict: graded live frames of `labels`."""
    from barca import atd, possession
    n = len(labels)
    graded = np.zeros(n, bool)
    for s, e in S["runs"]:
        graded[s:e] = True
    live = graded & (labels != 2)
    blocks = [(s, e - 1) for s, e in S["runs"]]
    data = atd.load()
    st = data["slot_team"].astype(int)
    Ft = TM.features(data["players"], data["ball"], st, data["pitch"], context=saved.get("touch_context", False))
    pt = touch_probs(saved["touch"], Ft, live)
    ctrl = possession.control_events(S["team"], S["dist"], S["margin"], S["cues"]["ball_speed"])
    ev = touch_event_list(pt, live, saved["touch_th"])
    F = features(data["players"], data["ball"], st, data["pitch"], blocks, live, pt, ev, ctrl,
                 context=saved.get("context", False))
    return F, live


def load_saved(model_path=None):
    import os
    import pickle
    from barca import dfl
    with open(model_path or os.path.join(dfl.CACHE, MODEL_FILE), "rb") as fh:
        return pickle.load(fh)


def apply_bar(S, labels, model_path=None, decoder="d0", d1=None):
    """Replace the team on every graded live frame of `labels` (dead = 2 kept) with the saved
    model's decoded team. S = possession_pipeline.setup(). Same steps as poss_model_eval.bar()."""
    saved = load_saved(model_path)
    F, live = bar_features(S, labels, saved)
    q = np.zeros(len(labels))
    models = saved["poss"] if isinstance(saved["poss"], list) else [saved["poss"]]   # ensembles: average
    q[live] = np.mean([mdl.predict_proba(F[live])[:, 1] for mdl in models], axis=0)
    c, m = d1 or D1
    team = decode(q, live, saved["cost"]) if decoder == "d0" else absorb_short(decode(q, live, c), m)
    return np.where(live, team, labels).astype(np.int8)


def touch_probs(touch_model, F_touch, live):
    """p of a flip to home / away from a fitted touch model on touch_model.features() output."""
    out = {}
    for t in (0, 1):
        q = np.zeros(len(live))
        q[live] = touch_model.predict_proba(F_touch[t][live])[:, 1]
        out[t] = q
    return out


def touch_event_list(p, live, threshold):
    return TM.events(p[0], p[1], live, threshold)
