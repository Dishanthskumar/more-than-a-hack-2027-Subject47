"""Controlled possession from touch sequences.

CHALLENGE.md: possession changes when the other team gains CONTROLLED possession; a touch without
control, a deflection, or an aerial duel without follow-up control is not a change. Its suggested
proxies: a player keeps the ball for a few seconds, or a team strings passes together. Encoded
here, with constants fixed from that wording (not tuned on any score):

* A touch event is a run of frames where a player is on the ball (boundaries.touch_team), with
  same-team runs merged across a short dropout.
* Team T gains control at a touch event if that event lasts >= DWELL_S, or T touches the ball
  again within PASS_GAP_S with no opposing touch in between (a completed pass).
* The change is dated to the START of that controlling touch. Uncontrolled opposing touches in
  between leave possession where it was.
"""

import numpy as np

from .boundaries import touch_team

MERGE_FRAMES = 5
DWELL_S = 1.0
PASS_GAP_S = 3.0
HOLD_AFTER_TOUCH_S = 5.0
WITH_PLAYER_MPS = 2.0   # v5: ball moving with the player (relative speed below this) = on the ball
# v6: an ambiguous change (brief touches, ball never moved with either player) is accepted if the
# formations swing towards the new team: shape-model log-odds 2-6 s after minus 2-6 s before.
# Threshold = 75th percentile of that shift at random frames with a random team (null), fixed
# before looking at any result.
SHAPE_BEFORE = (-150, -50)
SHAPE_AFTER = (50, 150)
SHAPE_CONFIRM = 0.74


def touch_events(f, blocks):
    """[(start, end_inclusive, team)] per block, in time order."""
    tt = touch_team(f)
    events = []
    for first, last in blocks:
        w = tt[first:last + 1]
        idx = np.flatnonzero(w != "")
        if not len(idx):
            continue
        s = e = int(idx[0])
        team = w[s]
        for i in idx[1:].tolist():
            if w[i] == team and i - e <= MERGE_FRAMES + 1:
                e = i
            else:
                events.append((first + s, first + e, team))
                s = e = i
                team = w[i]
        events.append((first + s, first + e, team))
    return events


def with_player(events, rel_speed):
    """Per event: did the ball move WITH the player on any frame of it (not just go past)?"""
    with np.errstate(invalid="ignore"):
        on = rel_speed < WITH_PLAYER_MPS
    return [bool(on[s:e + 1].any()) for s, e, _ in events]


def shape_shift(shape_logit, frame, team, clip=3.0):
    sl = shape_logit
    a, b = frame + SHAPE_BEFORE[0], frame + SHAPE_BEFORE[1]
    c, d = frame + SHAPE_AFTER[0], frame + SHAPE_AFTER[1]
    if a < 0 or d > len(sl):
        return 0.0
    before, after = np.clip(sl[a:b], -clip, clip), np.clip(sl[c:d], -clip, clip)
    if not (np.isfinite(before).any() and np.isfinite(after).any()):
        return 0.0
    return (1.0 if team == "H" else -1.0) * float(np.nanmean(after) - np.nanmean(before))


def control_changes(events, blocks, fps=25, carried=None, confirm=None):
    """[(frame, team)]: each time control passes to a different team (first per block included).

    carried (v5): per-event with_player flags. Then a brief touch confirmed by a teammate's next
    touch counts only if the ball moved with the player in at least one of the two; two pass-bys
    in a row (ball running past two defenders) do not take possession, a one-touch pass to a
    teammate who controls it does.
    """
    dwell, gap = int(DWELL_S * fps), int(PASS_GAP_S * fps)
    out = []
    for first, last in blocks:
        idx = [k for k, e in enumerate(events) if first <= e[0] <= last]
        ev = [events[k] for k in idx]
        cv = [carried[k] for k in idx] if carried is not None else None
        current = None
        for i, (s, e, team) in enumerate(ev):
            if team == current:
                continue
            controlled = (e - s + 1) >= dwell
            if not controlled and i + 1 < len(ev):
                ns, _, nteam = ev[i + 1]
                controlled = nteam == team and ns - e <= gap
                if controlled and cv is not None and not (cv[i] or cv[i + 1]):
                    controlled = confirm is not None and confirm(s, team)
            if controlled:
                out.append((s, team))
                current = team
    return out


def evidence(f, blocks, n_frames, fps=25):
    """(control_team (n,) +1 home / -1 away / 0 unknown, change list).

    From each control change, the controlling team is asserted until the next change, but only
    while that team has touched the ball within HOLD_AFTER_TOUCH_S; beyond that the decoder's
    other evidence decides.
    """
    events = touch_events(f, blocks)
    carried = with_player(events, f["ball_rel_speed"]) if "ball_rel_speed" in f else None
    confirm = None
    if f.get("shape_confirm") is not None and "shape_logit" in f:
        sl = f["shape_logit"]
        confirm = lambda frame, team: shape_shift(sl, frame, team) >= SHAPE_CONFIRM  # noqa: E731
    changes = control_changes(events, blocks, fps, carried, confirm)
    hold = int(HOLD_AFTER_TOUCH_S * fps)
    out = np.zeros(n_frames)
    for k, (c, team) in enumerate(changes):
        last_block = next(last for first, last in blocks if first <= c <= last)
        stop = changes[k + 1][0] if k + 1 < len(changes) and changes[k + 1][0] <= last_block \
            else last_block + 1
        sign = 1.0 if team == "H" else -1.0
        own = [e for e in events if c <= e[0] < stop and e[2] == team]
        # cover from the change to HOLD after each of this team's touches, never past `stop`
        for s, e, _ in own:
            out[s:min(e + 1 + hold, stop)] = sign
    return out, changes
