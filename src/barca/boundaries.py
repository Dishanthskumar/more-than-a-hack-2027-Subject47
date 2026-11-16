"""Move each decoded transition to the moment the possession rules point at.

The Viterbi pass decides WHAT happened (which segments exist and their states). It is late on
WHEN, because a switch has to be paid for with several seconds of evidence. This pass keeps every
segment and its state and only slides each boundary, inside a small window, to the event that
defines it (CHALLENGE.md, "Change of possession"):

* H <-> A: possession passes when the new team gains the ball, and a loose or in-flight ball
  stays with the last team that had it. So the boundary is the new team's first touch after the
  old team's last touch.
* live -> D: the ball goes dead when it leaves the pitch. Snapped to the first off-pitch frame;
  other stoppages (fouls, whistles) have no tracking signature, so they stay where they are.
* D -> live: play restarts on the tagged restart if there is one nearby, otherwise on the first
  frame the ball moves on the pitch.

Windows and thresholds are fixed from the decoder's design (it switches up to ~1.6 s late), not
tuned on any score. Deterministic.
"""

import numpy as np

H, A, D = "H", "A", "D"

BACK_FRAMES = 75        # 3 s: how far a boundary may move earlier
FWD_FRAMES = 25         # 1 s: how far it may move later
MIN_SEGMENT = 5         # never shrink a segment below this many frames
TOUCH_REACH_M = 1.5
BALL_MOVING_MPS = 3.0


def touch_team(f):
    """Per frame: 'H' / 'A' where a player is on the ball (ball on pitch, within reach), else ''."""
    on = f["seen"] & ~f["off_pitch"] & (f["d_min"] < TOUCH_REACH_M)
    if "filled" in f:
        on = on & ~f["filled"]      # broadcast fill is too coarse (1-3 m) to call a touch
    out = np.full(len(on), "", dtype="<U1")
    out[on & (f["d_home"] <= f["d_away"])] = H
    out[on & (f["d_home"] > f["d_away"])] = A
    return out


def _segments(w):
    cut = np.flatnonzero(w[1:] != w[:-1]) + 1
    bounds = np.concatenate(([0], cut, [len(w)]))
    return [[int(a), int(b), w[a]] for a, b in zip(bounds[:-1], bounds[1:])]


def _new_boundary(t, old, new, lo, hi, touch, off, moving, restarts, team_changes, snap_dead):
    a, b = max(lo, t - BACK_FRAMES), min(hi, t + FWD_FRAMES)
    if a > b:
        return t
    if old in (H, A) and new in (H, A) and team_changes is not None:
        frames, teams = team_changes
        near = frames[(frames >= a) & (frames <= b) & (teams == new)]
        return int(near[np.argmin(np.abs(near - t))]) if len(near) else t
    if (old == D or new == D) and not snap_dead:
        return t
    if old in (H, A) and new in (H, A):
        window = touch[a:b + 1]
        old_touch = np.flatnonzero(window == old)
        start = old_touch[-1] + 1 if len(old_touch) else 0
        new_touch = np.flatnonzero(window[start:] == new)
        return a + start + int(new_touch[0]) if len(new_touch) else t
    if new == D:
        hits = np.flatnonzero(off[a:b + 1])
        return a + int(hits[0]) if len(hits) else t
    # old == D: a restart
    near = restarts[(restarts >= a) & (restarts <= b)]
    if len(near):
        return int(near[np.argmin(np.abs(near - t))])
    hits = np.flatnonzero(moving[a:b + 1])
    return a + int(hits[0]) if len(hits) else t


def refine(labels, f, blocks, restart_frames=None, team_changes=None, snap_dead=True):
    """Copy of labels with each transition slid to its defining event. Segment states unchanged.

    team_changes: optional [(frame, team)] control changes (control.py). When given, an H <-> A
    boundary snaps to the nearest change to the new team instead of the first-touch rule.
    snap_dead=False leaves every boundary into or out of D where the decoder put it.
    """
    if team_changes is not None:
        team_changes = (np.array([c for c, _ in team_changes], dtype=np.int64),
                        np.array([t for _, t in team_changes], dtype="<U1"))
    out = labels.copy()
    touch = touch_team(f)
    off = f["off_pitch"]
    with np.errstate(invalid="ignore"):
        moving = f["seen"] & ~off & (f["ball_speed"] > BALL_MOVING_MPS)
    restarts = np.asarray(restart_frames if restart_frames is not None else [], dtype=np.int64)
    moved = []

    for first, last in blocks:
        w = out[first:last + 1]
        segs = _segments(w)
        for i in range(1, len(segs)):
            prev, cur = segs[i - 1], segs[i]
            t = cur[0]
            lo = prev[0] + MIN_SEGMENT
            hi = cur[1] - MIN_SEGMENT
            if lo > hi:
                continue
            g = _new_boundary(first + t, prev[2], cur[2], first + lo, first + hi,
                              touch, off, moving, restarts, team_changes, snap_dead) - first
            g = min(max(g, lo), hi)
            moved.append(g - t)
            prev[1] = cur[0] = g
        for a, b, s in segs:
            w[a:b] = s
    moved = np.array(moved) if moved else np.zeros(1)
    return out, {"n_transitions": int(len(moved)),
                 "share_moved": round(float(np.mean(moved != 0)), 3),
                 "median_shift_frames": float(np.median(moved)),
                 "share_earlier": round(float(np.mean(moved < 0)), 3)}
