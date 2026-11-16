"""Dead-ball evidence from the moment the broadcast cuts away from live play.

"Covered" = some verified broadcast frame maps to this tactical frame (sync.verify via
ball_fusion.sync_map: players-only match at the half's offset). "Away" = none does: replays,
close-ups, crowd shots, and also sync failures (few players in shot, broadcast ATD missing).

Only the MOMENT of a cut is used, never how long the broadcast stays away: at 22 of 59 tagged
restarts the broadcast was still away when play resumed (median 2.6 s more), and 37 of 158
cut-aways fell in play the detector calls live. So:

  1  a cut onset adds CUT_W of dead evidence from CUT_PRE_S before it to CUT_POST_S after it
     (the director cuts 2-4 s after the ball goes out: coverage 0.87 at the exit, 0.46 at +3 s);
  2  tracking decides the restart: the evidence stops at the first frame from the onset on where
     a player is in control (control event) or the ball is struck inside the pitch.

CUT_W = 1.0 (as dead_ball.GAP_W): clearly live play has speed evidence -2, so a cut alone cannot
make it dead; it decides borderline, slow play. All constants fixed 2026-11-10 from full-match
profiles with no ground truth, before any scoring.
"""

import numpy as np

from barca import ball_fusion
from barca import dead_ball as db
from barca.loaders import FPS, run_lengths

CUT_W = 1.0
CUT_PRE_S, CUT_POST_S = 3.0, 5.0
FLICKER_S = 1.0          # covered / away runs shorter than this are sync noise
MIN_BEFORE_S = 2.0       # covered this long before the cut
MIN_AWAY_S = 2.0         # away this long after it
LIVE_IN_M = 1.0          # as set_pieces.LIVE_IN_M


def coverage(n):
    tm = ball_fusion.sync_map()
    cov = np.zeros(n, bool)
    cov[tm[(tm >= 0) & (tm < n)]] = True
    return cov


def clean(cov, flicker_s=FLICKER_S):
    """Flip runs shorter than flicker_s (away gaps first, then covered blips)."""
    out = cov.copy()
    k_min = int(flicker_s * FPS)
    for val in (False, True):
        for s, k in zip(*run_lengths(out == val)):
            if k < k_min and s > 0 and s + k < len(out):
                out[s:s + k] = not val
    return out


def cut_onsets(cov, graded):
    """Frames where a cleaned covered run (>= MIN_BEFORE_S) gives way to an away run (>= MIN_AWAY_S)."""
    cc = clean(cov)
    starts, lens = run_lengths(~cc)
    out = []
    for s, k in zip(starts, lens):
        b = int(MIN_BEFORE_S * FPS)
        if s >= b and k >= MIN_AWAY_S * FPS and cc[s - b:s].all() and graded[s]:
            out.append(int(s))
    return out


def restart_cue(c, in_ctrl):
    """Per frame: tracking shows live play (player in control, or ball struck inside the pitch)."""
    with np.errstate(invalid="ignore"):
        struck = (db.centred_median(c["ball_speed"], 3) > db.RESTART_BALL_MS) \
            & (c["ball_line_dist_m"] > LIVE_IN_M)
    return in_ctrl | struck


def evidence(onsets, live_cue, n, w=CUT_W):
    """CUT_W from onset - CUT_PRE_S until onset + CUT_POST_S or the first live cue at/after the onset."""
    e = np.zeros(n)
    pre, post = int(CUT_PRE_S * FPS), int(CUT_POST_S * FPS)
    for s in onsets:
        hi = min(s + post, n)
        hit = np.flatnonzero(live_cue[s:hi])
        end = s + int(hit[0]) if len(hit) else hi
        e[max(s - pre, 0):end] += w
    return e
