"""Per-frame evidence from the tactical ATD. No labels, no frame-specific constants.

All distances in metres, speeds as reported by the ATD (m/s).
"""

import warnings

import numpy as np

OFF_PITCH_MARGIN_M = 0.5


def _rolling_mean(x, half):
    """Centred NaN-aware moving mean over +/- half frames."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def compute(data, fps=25):
    """dict of (n,) arrays: seen, off_pitch, d_home, d_away, d_min, player_speed, n_players."""
    L, W = data["pitch"]
    players, ball, team = data["players"], data["ball"], data["slot_team"]

    bx, by = ball[:, 0].astype(np.float64) * L, ball[:, 1].astype(np.float64) * W
    px, py = players[:, :, 0].astype(np.float64) * L, players[:, :, 1].astype(np.float64) * W
    seen = np.isfinite(bx) & np.isfinite(by)

    with np.errstate(invalid="ignore"):
        dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d_home = dist[:, team == 0].min(axis=1)
    d_away = dist[:, team == 1].min(axis=1)

    off = seen & ((bx < -OFF_PITCH_MARGIN_M) | (bx > L + OFF_PITCH_MARGIN_M)
                  | (by < -OFF_PITCH_MARGIN_M) | (by > W + OFF_PITCH_MARGIN_M))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames (halftime)
        speed = np.nanmean(players[:, :, 2].astype(np.float64), axis=1)
    n_players = np.isfinite(px).sum(axis=1)

    # Ball speed from positions (the reported channel is missing on ~91% of seen frames),
    # central difference then a +/- 2 frame mean to tame the noise.
    raw = np.full(len(bx), np.nan)
    raw[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * fps / 2.0
    ball_speed = _rolling_mean(raw, 2)

    return {
        "ball_speed": ball_speed,
        "seen": seen,
        "off_pitch": off,
        "d_home": d_home,
        "d_away": d_away,
        "d_min": np.minimum(d_home, d_away),
        "player_speed": _rolling_mean(speed, fps // 2),
        "n_players": n_players,
    }


def ball_rel_speed(data, fps=25):
    """(n,) speed of the ball relative to its nearest player (m/s), NaN where unknown.

    Low = the ball moves WITH that player (carried, controlled); high = it is going past them.
    Central differences on raw positions (the ATD ball track is already smooth).
    """
    L, W = data["pitch"]
    b = data["ball"][:, :2].astype(np.float64) * [L, W]
    p = data["players"][..., :2].astype(np.float64) * [L, W]
    n = len(b)
    v = np.full_like(b, np.nan)
    v[1:-1] = (b[2:] - b[:-2]) * fps / 2.0
    pv = np.full_like(p, np.nan)
    pv[1:-1] = (p[2:] - p[:-2]) * fps / 2.0
    with np.errstate(invalid="ignore"):
        dist = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    j = dist.argmin(1)
    return np.hypot(*(v - pv[np.arange(n), j]).T)


PSEUDO_REACH_M = 1.5
PSEUDO_MARGIN_M = 2.0


def pseudo_ownership(f):
    """(home 0/1 per frame, mask): the tracking's own confident answer, for self-supervision.

    Trusted only where the ball is on the pitch, within reach of a player, and clearly nearer
    one team than the other. Produced by a rule, never by a person.
    """
    with np.errstate(invalid="ignore"):
        margin = np.abs(f["d_away"] - f["d_home"])
        mask = (f["seen"] & ~f["off_pitch"] & (f["d_min"] < PSEUDO_REACH_M)
                & np.isfinite(margin) & (margin > PSEUDO_MARGIN_M))
    return (f["d_home"] < f["d_away"]).astype(np.int8), mask
