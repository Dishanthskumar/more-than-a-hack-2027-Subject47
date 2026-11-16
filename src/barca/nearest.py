"""Nearest tracked player to the ball, their team, distance and margin over the other side."""

import numpy as np

NOBODY = -1


def nearest_team(ball_m, px_m, py_m, slot_team):
    """Per frame: (slot, team int8 with -1 = nobody, distance m, margin m).

    margin = |nearest home - nearest away|; small = contested, inf when only one team is tracked.
    The other side's nearest distance is dist + margin.
    """
    d = np.hypot(px_m - ball_m[:, :1], py_m - ball_m[:, 1:])
    d = np.where(np.isfinite(d), d, np.inf)
    slot = d.argmin(1)
    dist = d[np.arange(len(d)), slot]
    found = np.isfinite(dist)
    home = slot_team == 0
    with np.errstate(invalid="ignore"):  # inf - inf on frames with no ball; masked below
        margin = np.abs(np.where(home, d, np.inf).min(1) - np.where(home, np.inf, d).min(1))
    return (np.where(found, slot, NOBODY),
            np.where(found, slot_team[slot], NOBODY).astype(np.int8),
            np.where(found, dist, np.nan), np.where(found, margin, np.nan))


def from_tracking(tr, pitch=(105.3, 68.0)):
    """nearest_team on a loaders.load_tracking() dict, converting to metres."""
    L, W = np.float32(pitch[0]), np.float32(pitch[1])
    ball = np.stack([tr["ball"][:, 0] * L, tr["ball"][:, 1] * W], 1)
    return nearest_team(ball, tr["px"] * L, tr["py"] * W, tr["slot_team"])
