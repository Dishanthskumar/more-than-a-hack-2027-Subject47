"""Learned dead-ball probability from tracking, trained on degraded DFL (dfl.py + degrade.py).

DFL BallStatus (alive / dead) means what our ground truth's D means: dead share and dead-spell
lengths agree with the ATD (see memory note). Features are tracking-only and computed the same
way on DFL and on the BAR-ATH ATD (speeds from positions, not a reported speed channel):
  ball      seen; speed and its centred 1 s / 3 s medians; past / future 3 s mean speed; signed
            distance to the nearest line (negative = outside) and its minimum over the past 2 s;
            seconds since last seen and length of the current gap; seconds the ball has been
            still (1 s median < 2 m/s) and seconds until it next moves (capped at 60)
  players   mean speed: now, centred 1 s / 3 s, past / future 3 s; share under 1 m/s; mean of the
            fastest player's speed over 1 s; players tracked; players outside the pitch;
            nearest player to the ball; players within 10 m of the ball
p_dead feeds B2's dead layer as extra evidence w * logit(p) (set_pieces.detect(extra=...)).
"""

import numpy as np
import pandas as pd

from barca.loaders import run_lengths

FPS = 25
STILL_MS = 2.0
CAP_S = 60.0
FEATURES = ["seen", "bspeed", "bspeed_m1", "bspeed_m3", "bspeed_past3", "bspeed_fut3", "line",
            "line_min_past2", "since_seen", "gap_len", "still_since", "still_until",
            "pspeed", "pspeed_c1", "pspeed_c3", "pspeed_past3", "pspeed_fut3", "slow_share",
            "pmax_1s", "n_players", "players_out", "near_d", "n_10"]
# 2026-11-14, after the DFL frozen-ball artefact (the ball is frozen through every DFL dead spell):
# option 1 uses players only, plus four formation features fixed before running.
PLAYER_FEATURES = ["pspeed", "pspeed_c1", "pspeed_c3", "pspeed_past3", "pspeed_fut3", "slow_share",
                   "pmax_1s", "n_players", "players_out", "spread_x", "spread_y", "n_in_box", "centroid_gap"]
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
# 2026-11-14, Metrica dead model: all ball features + all player features (incl. formation). For
# DFL rows the ball features are set to NaN (DFL's dead-time ball is frozen / simulated), so only
# Metrica, whose ball is really tracked through stoppages, teaches the ball part.
FULL_FEATURES = FEATURES + ["spread_x", "spread_y", "n_in_box", "centroid_gap"]
BALL_FEATURES = ["seen", "bspeed", "bspeed_m1", "bspeed_m3", "bspeed_past3", "bspeed_fut3", "line",
                 "line_min_past2", "since_seen", "gap_len", "still_since", "still_until", "near_d", "n_10"]


def _roll(x, win, how, center=True):
    return getattr(pd.Series(x).rolling(int(win), center=center, min_periods=1), how)().to_numpy()


def _past(x, win, how="mean"):
    return np.r_[np.nan, _roll(x, win, how, center=False)[:-1]]


def _future(x, win, how="mean"):
    return _past(x[::-1], win, how)[::-1]


def _line(x, y, L, W):
    return np.minimum(np.minimum(x, L - x), np.minimum(y, W - y))


def _run_clock(flag, forward=True):
    """Seconds since flag last became True (counting along a True run), 0 where False; capped."""
    out = np.zeros(len(flag))
    for s, k in zip(*run_lengths(flag)):
        out[s:s + k] = np.arange(1, k + 1) / FPS if forward else np.arange(k, 0, -1) / FPS
    return np.minimum(out, CAP_S)


def features(players, ball, pitch, slot_team=None, kind="all"):
    """kind "all" -> FEATURES; "players" -> PLAYER_FEATURES; "full" -> FULL_FEATURES (both need
    slot_team except "all")."""
    L, W = float(pitch[0]), float(pitch[1])
    b = ball[:, :2].astype(np.float64) * [L, W]
    p = players[..., :2].astype(np.float64) * [L, W]
    n = len(b)
    seen = np.isfinite(b[:, 0])
    bs = np.full(n, np.nan)
    bs[1:-1] = np.hypot(*(b[2:] - b[:-2]).T) * FPS / 2
    ps = np.full(p.shape[:2], np.nan)
    ps[1:-1] = np.hypot(p[2:, :, 0] - p[:-2, :, 0], p[2:, :, 1] - p[:-2, :, 1]) * FPS / 2
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        pspeed = np.nanmean(ps, 1)
        pmax = np.nanmax(ps, 1)
        slow = np.nanmean(np.where(np.isfinite(ps), ps < 1.0, np.nan), 1)
        d = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1])
        near = np.nanmin(np.where(np.isfinite(d), d, np.nan), 1)
        n10 = np.where(seen, (d < 10).sum(1), np.nan)
        pl_line = _line(p[..., 0], p[..., 1], L, W)
        out_n = (pl_line < -1.0).sum(1).astype(float)
    line = _line(b[:, 0], b[:, 1], L, W)
    m1 = _roll(bs, FPS, "median")
    moving = np.nan_to_num(m1, nan=0.0) > STILL_MS
    still = seen & ~moving
    since_seen = _run_clock(~seen)
    gap_len = np.zeros(n)
    for s, k in zip(*run_lengths(~seen)):
        gap_len[s:s + k] = k / FPS
    cols = {
        "seen": seen.astype(float), "bspeed": bs, "bspeed_m1": m1, "bspeed_m3": _roll(bs, 3 * FPS, "median"),
        "bspeed_past3": _past(bs, 3 * FPS), "bspeed_fut3": _future(bs, 3 * FPS), "line": line,
        "line_min_past2": _past(line, 2 * FPS, "min"), "since_seen": since_seen,
        "gap_len": np.minimum(gap_len, CAP_S), "still_since": _run_clock(still),
        "still_until": _run_clock(still, forward=False),
        "pspeed": pspeed, "pspeed_c1": _roll(pspeed, FPS, "mean"), "pspeed_c3": _roll(pspeed, 3 * FPS, "mean"),
        "pspeed_past3": _past(pspeed, 3 * FPS), "pspeed_fut3": _future(pspeed, 3 * FPS), "slow_share": slow,
        "pmax_1s": _roll(pmax, FPS, "mean"), "n_players": np.isfinite(p[..., 0]).sum(1).astype(float),
        "players_out": out_n, "near_d": near, "n_10": n10}
    if kind in ("players", "full"):
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            cols["spread_x"] = np.nanstd(p[..., 0], 1)
            cols["spread_y"] = np.nanstd(p[..., 1], 1)
            in_box = ((p[..., 0] < BOX_DEPTH_M) | (p[..., 0] > L - BOX_DEPTH_M)) \
                & (np.abs(p[..., 1] - W / 2) < BOX_HALF_WIDTH_M)
            cols["n_in_box"] = in_box.sum(1).astype(float)
            st = np.asarray(slot_team)
            c0 = np.stack([np.nanmean(p[:, st == 0, 0], 1), np.nanmean(p[:, st == 0, 1], 1)], 1)
            c1 = np.stack([np.nanmean(p[:, st == 1, 0], 1), np.nanmean(p[:, st == 1, 1], 1)], 1)
            cols["centroid_gap"] = np.hypot(*(c0 - c1).T)
        names = PLAYER_FEATURES if kind == "players" else FULL_FEATURES
        return np.column_stack([cols[f] for f in names]).astype(np.float32)
    return np.column_stack([cols[f] for f in FEATURES]).astype(np.float32)


MODEL_PATH = "dead_players_fair_model.pkl"     # in dfl.CACHE, written by scripts/dead_model_eval.py --fair --bar


def extend_only(with_model, rules):
    """Extend-only (2026-11-14): the rules' dead mask, plus the dead spells of the with-model mask
    that overlap one of the rules' spells. Never removes or shortens a stoppage the rules found,
    never adds one they did not."""
    out = rules.copy()
    for s, k in zip(*run_lengths(with_model)):
        if rules[s:s + k].any():
            out[s:s + k] = True
    return out


def bar_evidence(data, model_path=None):
    """w * logit(p_dead) on the BAR-ATH ATD from the saved players-only model (atd.load() dict)."""
    import os
    import pickle
    from barca import dfl
    with open(model_path or os.path.join(dfl.CACHE, MODEL_PATH), "rb") as fh:
        saved = pickle.load(fh)
    F = features(data["players"], data["ball"], data["pitch"], data["slot_team"].astype(int), saved["features"])
    return saved["wc"] * logit(saved["model"].predict_proba(F)[:, 1])


def saved_mode(model_path=None):
    """"extend" (default, extend_only) or "full" (use the with-model mask as is)."""
    import os
    import pickle
    from barca import dfl
    with open(model_path or os.path.join(dfl.CACHE, MODEL_PATH), "rb") as fh:
        return pickle.load(fh).get("mode", "extend")


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def tr_from(d):
    """A loaders.load_tracking()-shaped dict from a dfl.load() / degrade() dict."""
    pl = d["players"]
    return {"ball": d["ball"][:, :2], "px": pl[..., 0], "py": pl[..., 1], "ps": pl[..., 2],
            "slot_team": d["slot_team"].astype(np.int64), "n_slots": pl.shape[1]}
