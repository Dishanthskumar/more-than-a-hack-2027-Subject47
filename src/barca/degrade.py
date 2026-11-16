"""Make clean DFL tracking look like the BAR-ATH ATD, so models trained on DFL transfer.

Calibrated 2026-11-14 without labels (live = B2's live mask, used only to pick frames):
  ball smoothing    Gaussian, sigma 6 frames: live ball acceleration p50/p90/p99 becomes 5.8 / 19.2 /
                    37.1 m/s^2 against the ATD's 6.6 / 18.7 / 33.8 (raw DFL: 19.8 / 97.6 / 604)
  player smoothing  Gaussian, sigma 2 frames: 1.04 / 3.02 / 5.98 against 1.08 / 3.23 / 6.16
  ball gaps         the ATD's live gap pattern: gap starts at the ATD rate per live frame, lengths
                    drawn from the ATD's own live gap lengths (13.6% of live frames missing)
  player dropouts   runs of 2 s at a rate giving the ATD's 21.4 present per frame instead of 22
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from barca.loaders import run_lengths

BALL_SIGMA = 6
PLAYER_SIGMA = 2
PLAYER_MISSING = 1 - 21.4 / 22.0
PLAYER_DROP_FRAMES = 50


def atd_gap_profile(ball, live):
    """(gap starts per live frame, array of gap lengths in frames) from the ATD."""
    s, k = run_lengths(live & ~np.isfinite(ball[:, 0]))
    return len(s) / max(int(live.sum()), 1), np.asarray(k, np.int64)


def _smooth(x, sigma, blocks):
    """Gaussian along time inside each block; NaN runs are bridged for filtering and put back."""
    out = x.copy()
    for a, b in blocks:
        w = x[a:b + 1]
        nan = ~np.isfinite(w)
        if nan.all():
            continue
        filled = w.copy()
        flat = filled.reshape(len(w), -1)
        for j in range(flat.shape[1]):
            col = flat[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 2:
                continue
            idx = np.arange(len(col))
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        sm = gaussian_filter1d(filled, sigma, axis=0, mode="nearest")
        sm[nan] = np.nan
        out[a:b + 1] = sm
    return out


# Dead-time ball for DFL (fair dead-ball bench, 2026-11-14). DFL freezes the ball through every
# dead spell, which leaks the label; the ATD's dead-time ball is missing ~65% of the time, evenly
# through the spell (fifths: .65 .69 .69 .68 .59) and moves at ~4.7 m/s when seen (carried).
# Simulation: carried in a straight line from where play stopped to the restart spot, plus smooth
# noise (2 m, 1 s), and hidden in alternating runs (visible mean 2 s, hidden mean 3.7 s = 65%).
DEAD_MISSING = 0.65
DEAD_VIS_RUN_S = 2.0
DEAD_NOISE_M = 2.0


def simulate_dead_ball(d, seed=0):
    rng = np.random.default_rng(seed)
    out = dict(d)
    ball = d["ball"].astype(np.float64).copy()
    L, W = float(d["pitch"][0]), float(d["pitch"][1])
    vis_mean = DEAD_VIS_RUN_S * 25
    hid_mean = vis_mean * DEAD_MISSING / (1 - DEAD_MISSING)
    for a, b in d["blocks"]:
        dead = d["labels"][a:b + 1] == 2
        for s, k in zip(*run_lengths(dead)):
            s0, e0 = a + s, a + s + k                       # dead frames [s0, e0)
            p0 = ball[s0 - 1, :2] if s0 > a else None
            p1 = ball[e0, :2] if e0 <= b else None
            p0 = p1 if p0 is None else p0
            p1 = p0 if p1 is None else p1
            if p0 is None:
                ball[s0:e0, :2] = np.nan
                continue
            f = np.linspace(0, 1, k)[:, None]
            path = p0 + (p1 - p0) * f
            noise = gaussian_filter1d(rng.normal(0, 1, (k, 2)), 25, axis=0, mode="nearest")
            noise *= DEAD_NOISE_M / max(float(np.std(noise)), 1e-9)
            path = path + noise / [L, W]
            hide = np.zeros(k, bool)
            i, visible = 0, rng.random() >= DEAD_MISSING
            while i < k:
                run = max(1, int(rng.exponential(vis_mean if visible else hid_mean)))
                hide[i:i + run] = not visible
                i, visible = i + run, not visible
            path[hide] = np.nan
            ball[s0:e0, :2] = path
            ball[s0:e0, 2] = 0.0
    out["ball"] = ball.astype(np.float32)
    return out


def degrade(d, gap_rate, gap_lengths, seed=0):
    """Copy of a dfl.load() dict with smoothed, gappy ball and smoothed players with dropouts.
    Labels are unchanged."""
    rng = np.random.default_rng(seed)
    out = dict(d)
    ball = d["ball"].astype(np.float64)
    ball[:, :2] = _smooth(ball[:, :2], BALL_SIGMA, d["blocks"])
    live = d["labels"] != 2
    li = np.flatnonzero(live)
    n_gaps = rng.poisson(gap_rate * len(li))
    for s in rng.choice(li, size=min(n_gaps, len(li)), replace=False):
        ball[s:s + int(rng.choice(gap_lengths))] = np.nan
    out["ball"] = ball.astype(np.float32)
    pl = d["players"].astype(np.float64)
    pl[..., :2] = _smooth(pl[..., :2], PLAYER_SIGMA, d["blocks"])
    n, slots = pl.shape[:2]
    rate = PLAYER_MISSING / PLAYER_DROP_FRAMES
    for j in range(slots):
        for s in np.flatnonzero(rng.random(n) < rate):
            pl[s:s + PLAYER_DROP_FRAMES, j] = np.nan
    out["players"] = pl.astype(np.float32)
    return out
