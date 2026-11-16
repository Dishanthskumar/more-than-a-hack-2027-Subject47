"""Who receives the ball next? A per-frame team signal from the real future ball path.

Approach 1 (2026-11-14), pre-registered before any scoring. The decoder's ownership term today is
the nearest player's distance margin, which flips to a defender while a pass flies past him and
lags at interceptions until the new team satisfies the control rule. Offline we know where the
ball goes, so at every frame we look ahead to the next RECEPTION and use its team instead:
during a completed pass the receiver is a teammate (possession stays), and at an interception the
new team shows up as soon as the ball is on its way to them.

Reception at frame u (every constant reused, none chosen here):
  ball seen, on the pitch, native (not a broadcast/video fill), nearest player within
  boundaries.TOUCH_REACH_M (1.5 m), the other side at least possession.CONTROL_MARGIN_M (1 m)
  further (contested frames give no evidence), and either the ball moving with that player
  (relative speed < control.WITH_PLAYER_MPS, 2 m/s) or a kink (touches.py) within touches.K frames.
Signal at frame t: +1 / -1 for the team (H / A) of the first reception at or after t within
control.PASS_GAP_S (3 s) in the same block, else 0.

Variants for the decoder (decoder.emissions, f["receiver_team"], f["receiver_mode"]):
  R1 replace  the nearest-margin term is replaced by own_gain * signal (filled frames keep it)
  R2 add      own_gain * signal is added to the nearest-margin term
"""

import numpy as np

from barca import boundaries, control, possession, touches
from barca.loaders import FPS, PITCH_M

REACH_M = boundaries.TOUCH_REACH_M
MARGIN_M = possession.CONTROL_MARGIN_M
WITH_MPS = control.WITH_PLAYER_MPS
KINK_WIN = touches.K
HORIZON = int(control.PASS_GAP_S * FPS)


def receptions(f, rel_speed, kink_frames):
    """(n,) int8: +1 home reception, -1 away, 0 none."""
    n = len(f["seen"])
    near_kink = np.zeros(n, bool)
    for k in kink_frames:
        near_kink[max(k - KINK_WIN, 0):k + KINK_WIN + 1] = True
    with np.errstate(invalid="ignore"):
        on = f["seen"] & ~f["off_pitch"] & (f["d_min"] < REACH_M) \
            & (np.abs(f["d_home"] - f["d_away"]) >= MARGIN_M) & ((rel_speed < WITH_MPS) | near_kink)
    if "filled" in f:
        on &= ~f["filled"]
    out = np.zeros(n, np.int8)
    out[on & (f["d_home"] < f["d_away"])] = 1
    out[on & (f["d_home"] > f["d_away"])] = -1
    return out


def next_receiver(rec, blocks, horizon=HORIZON):
    """(n,) float: sign of the first reception at or after t within horizon frames, same block."""
    n = len(rec)
    out = np.zeros(n)
    for first, last in blocks:
        nxt_val, nxt_at = 0, None
        for t in range(last, first - 1, -1):
            if rec[t]:
                nxt_val, nxt_at = int(rec[t]), t
            if nxt_at is not None and nxt_at - t <= horizon:
                out[t] = nxt_val
    return out


def add_to_features(f, data, blocks, mode):
    """Post-hook for make_submission.build_features: adds receiver_team / receiver_mode to f."""
    tac = data["ball"][:, :2].astype(np.float64) * np.asarray(PITCH_M)
    kinks = touches.kinks(touches.velocity_change_fast(tac))
    rel = f["ball_rel_speed"] if "ball_rel_speed" in f else None
    if rel is None:
        from barca import features
        rel = features.ball_rel_speed(data)
    rec = receptions(f, rel, kinks)
    f["receptions"] = rec
    f["receiver_team"] = next_receiver(rec, blocks)
    f["receiver_mode"] = mode
    return f
