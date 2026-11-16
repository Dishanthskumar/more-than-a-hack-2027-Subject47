"""Possession from control events, following the challenge's definition.

While the ball is alive one team always has it; a contested or loose ball stays with the last team;
a touch without control is no change. Labels: HOME, AWAY, DEAD = 0, 1, 2.
"""

import numpy as np

from barca.dead_ball import centred_median
from barca.loaders import FPS, run_lengths

HOME, AWAY, DEAD = 0, 1, 2

CONTROL_DIST_M = 2.0       # ATD positions are smoothed body centres
CONTROL_MARGIN_M = 1.0     # the other side at least 1 m further
CONTROL_BALL_MAX_MS = 8.0  # faster = travelling (pass, shot, clearance)
CONTROL_DWELL_S = 0.5

CONFIRM_HOLD_S = 1.5       # rejected variant: a change needs a 1.5 s hold ...
CONFIRM_REPEAT_S = 3.0     # ... or a repeat control within 3 s with no opposing event between


def control_conditions(team, dist, margin, ball_speed, dist_m=CONTROL_DIST_M):
    speed = centred_median(ball_speed, 5)
    with np.errstate(invalid="ignore"):
        cond = {"ball_seen": team >= 0, "near": dist <= dist_m,
                "uncontested": margin >= CONTROL_MARGIN_M,
                "not_flying": ~(speed > CONTROL_BALL_MAX_MS)}
    cond["ok"] = cond["ball_seen"] & cond["near"] & cond["uncontested"] & cond["not_flying"]
    return cond


def control_events(team, dist, margin, ball_speed, dist_m=CONTROL_DIST_M,
                   dwell_s=CONTROL_DWELL_S):
    """[(start, stop, team)] sorted: runs >= dwell where one team is in control."""
    ok = control_conditions(team, dist, margin, ball_speed, dist_m)["ok"]
    events = []
    for t in (HOME, AWAY):
        for s, k in zip(*run_lengths(ok & (team == t))):
            if k >= dwell_s * FPS:
                events.append((int(s), int(s + k), t))
    return sorted(events)


def live_spells(dead):
    return [(int(s), int(s + k)) for s, k in zip(*run_lengths(~dead))]


def carry_labels(base, dead):
    """Fallback: baseline team in live spells, its D frames filled with the last team, head back-filled."""
    out = np.where(dead, DEAD, base).astype(np.int8)
    for a, b in live_spells(dead):
        w = out[a:b]
        known = np.flatnonzero(w != DEAD)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w != DEAD, np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[a:b] = w[carry]
    return out


def nearest_switch(team, pos, e_old, s, old, new):
    """Halfway between the old team's last nearest frame + 1 and the new team's first after it."""
    w = team[pos:s]
    hit = np.flatnonzero(w == old)
    f = pos + int(hit[-1]) if len(hit) else e_old - 1
    after = np.flatnonzero(team[f + 1:s + 1] == new)
    g = f + 1 + int(after[0]) if len(after) else s
    return (f + 1 + g) // 2


def labels_from_events(events, dead, fallback, switch_at="mid", team=None):
    """D on dead frames; inside each live spell the team of the last control event.

    switch_at: where a change of team lands between the old team's last control (e_old) and the
    new team's first (s): "start" = s, "handover" = e_old, "mid" = halfway (best),
    "nearest" = halfway across the nearest-team flip (needs `team`).
    """
    out = np.where(dead, DEAD, fallback).astype(np.int8)
    starts = np.array([s for s, _, _ in events], dtype=np.int64)
    for a, b in live_spells(dead):
        i, j = np.searchsorted(starts, a), np.searchsorted(starts, b)
        if i == j:
            continue                           # no control in this spell: keep the fallback
        cur, pos, last_end = events[i][2], a, events[i][1]
        for s, e, t in events[i:j]:
            if t != cur:
                e_old = min(last_end, s)
                if switch_at == "nearest":
                    cut = nearest_switch(team, pos, e_old, s, cur, t)
                else:
                    cut = {"start": s, "handover": e_old, "mid": (e_old + s) // 2}[switch_at]
                cut = max(pos, cut)
                out[pos:cut], cur, pos = cur, t, cut
            last_end = e
        out[pos:b] = cur
    return out


def confirmed(events, dead):
    """Rejected variant: drop changes of team that are neither held nor repeated."""
    starts = np.array([s for s, _, _ in events], dtype=np.int64)
    keep = []
    for a, b in live_spells(dead):
        i, j = np.searchsorted(starts, a), np.searchsorted(starts, b)
        spell = events[i:j]
        if not spell:
            continue
        cur = spell[0][2]
        keep.append(spell[0])
        for k in range(1, len(spell)):
            s, e, t = spell[k]
            if t == cur:
                keep.append(spell[k])
                continue
            ok = e - s >= CONFIRM_HOLD_S * FPS
            for s2, _, t2 in spell[k + 1:]:
                if ok or s2 > e + CONFIRM_REPEAT_S * FPS or t2 != t:
                    break                       # too late, or an opposing event came first
                ok = True
            if ok:
                keep.append(spell[k])
                cur = t
    return keep
