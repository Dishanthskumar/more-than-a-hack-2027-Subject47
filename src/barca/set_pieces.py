"""Set-piece-aware dead balls from the Smart Tagging, on top of dead_ball.py.

Three rules, each switchable; all off reproduces dead_ball.detect(snap=True) exactly.

  backtrack  For line restarts (throw-in, corner, goal kick) the dead spell starts where the ball
             left the pitch, not at a fixed 10 s before the tag: the first ball exit after the last
             live action in the 30 s before the tag. No exit found -> the old 10 s.
  locate     The restart is the first struck ball whose position fits the tag type (throw-in on a
             touchline, corner at a corner, goal kick in the goal area, kick-off on the centre
             spot). No fitting strike -> the old first struck ball.
  goals      Each GOALS tag: dead from the tag start to the kick-off restart that follows.
             REJECTED (2026-11-09), default off: a GOALS interval starts BEFORE the goal (a set
             piece at 3-10 s and attacking phases at 11-19 s inside it), so this marks the live
             attack dead (1.8-2.5 m/s, ball 5-9 m/s). The ball never shows entering the goal
             mouth either, so the goal moment cannot be located; the detector already calls
             73-90% of each interval dead.

restart_team() gives the tag's named team the ball from each restart until the other side's
first control event. REJECTED (2026-11-09): in the frames it flips, the player within 2 m of the
ball is from the old label's team 70 times and the tag's team 22, so the tag's Team is not
reliably the taker. Kept only for the record; not used in the submission.

All constants come from the laws of the game, pitch geometry and the old detector's constants,
fixed before any scoring (2026-11-09).
"""

import numpy as np

from barca import dead_ball as db
from barca.loaders import FPS, PITCH_M

L, W = PITCH_M
HOME, AWAY, DEAD = 0, 1, 2

EXIT_SEARCH_S = 30.0       # how far before the tag the ball may have gone out
EXIT_LINE_M = 3.0          # ball this close to a line on the frame before it goes out / is lost
OUT_M = 0.5                # out = more than 0.5 m beyond a line (as dead_ball.ball_out)
LIVE_IN_M = 1.0            # live action = fast ball at least 1 m inside the pitch
LOC_TOL_M = 3.0            # restart position tolerance
GOAL_AREA_X_M = 5.5 + 2.0  # goal area depth + tolerance
GOAL_AREA_HALF_Y_M = 9.16 + 2.0
GOAL_MOUTH_HALF_Y_M = 3.66 + 2.0
LOC_LOOKBACK = 3           # frames before the fast-ball onset where the ball still sits at the spot
TEAM_CODES = {"Barcelona": HOME, "Atletic Club": AWAY}
LINE_TYPES = ("throw-in", "corner kick", "goal kick")


def tag_list(tags):
    return [{"start": t["start_frame"], "end": t["end_frame"],
             "type": (t["tags"].get("Type") or "").strip().lower(),
             "team": TEAM_CODES.get(t["tags"].get("Team"))} for t in db.set_piece_tags(tags)]


def goal_tags(tags):
    return sorted(((t["start_frame"], t["end_frame"]) for t in tags
                   if (t["code"] or "").strip().upper() == "GOALS"))


def ball_m(tr):
    b = tr["ball"].astype(np.float64)
    return b[:, 0] * L, b[:, 1] * W


def loc_ok(kind, x, y):
    if not np.isfinite(x):
        return False
    if kind == "throw-in":
        return min(y, W - y) < LOC_TOL_M
    if kind == "corner kick":
        return min(x, L - x) < LOC_TOL_M and min(y, W - y) < LOC_TOL_M
    if kind == "goal kick":
        return min(x, L - x) < GOAL_AREA_X_M and abs(y - W / 2) < GOAL_AREA_HALF_Y_M
    if kind.startswith("kick off"):
        return abs(x - L / 2) < LOC_TOL_M and abs(y - W / 2) < LOC_TOL_M
    return True                                   # free kick: anywhere


def checkable(kind):
    return kind in LINE_TYPES or kind.startswith("kick off")


def exits(bx, by, goal_mouth=False):
    """Bool per frame: the ball, seen near a line on the previous frame, is now out or lost."""
    seen = np.isfinite(bx)
    ld = db.line_dist(bx, by)
    with np.errstate(invalid="ignore"):
        before = seen[:-1] & (ld[:-1] < EXIT_LINE_M) & (ld[:-1] > -OUT_M)
        if goal_mouth:
            before &= (np.minimum(bx[:-1], L - bx[:-1]) < EXIT_LINE_M) \
                & (np.abs(by[:-1] - W / 2) < GOAL_MOUTH_HALF_Y_M)
        after = ~seen[1:] | (ld[1:] < -OUT_M)
    out = np.zeros(len(bx), bool)
    out[1:] = before & after
    return out


def restarts(c, bx, by, tl, locate=True):
    """Per tag: (restart frame, location verified: True / False / None = not checkable)."""
    n = len(bx)
    with np.errstate(invalid="ignore"):
        fast = db.centred_median(c["ball_speed"], 3) > db.RESTART_BALL_MS
    out = []
    for t in tl:
        s = t["start"]
        lo, hi = min(s + int(db.RESTART_FROM_S * FPS), n), min(s + int(db.RESTART_SEARCH_S * FPS), n)
        hits = lo + np.flatnonzero(fast[lo:hi])
        default = int(hits[0]) if len(hits) else min(s + int(db.RESTART_DEFAULT_S * FPS), n)
        if not (locate and checkable(t["type"])):
            out.append((default, None))
            continue
        onsets = [h for h in hits if h == lo or not fast[h - 1]]
        found = None
        for h in onsets:
            if any(loc_ok(t["type"], bx[k], by[k]) for k in range(max(h - LOC_LOOKBACK, 0), h + 1)):
                found = int(h)
                break
        out.append((found, True) if found is not None else (default, False))
    return out


def dead_from(c, bx, by, tl, rs, backtrack=True):
    """Per tag: first dead frame before it, and whether it came from a ball exit."""
    n = len(bx)
    ex = exits(bx, by)
    with np.errstate(invalid="ignore"):
        live = (db.centred_median(c["ball_speed"], 3) > db.RESTART_BALL_MS) \
            & (c["ball_line_dist_m"] > LIVE_IN_M)
    out = []
    for i, t in enumerate(tl):
        s = t["start"]
        fallback = max(s - int(db.TAG_PRE_S * FPS), 0)
        if not (backtrack and t["type"] in LINE_TYPES):
            out.append((fallback, False))
            continue
        lo = max(s - int(EXIT_SEARCH_S * FPS), 0)
        if i:
            lo = max(lo, min(rs[i - 1][0], s))       # not before the previous restart
        hi = min(s, n)
        lv = np.flatnonzero(live[lo:hi])
        a = lo + int(lv[-1]) + 1 if len(lv) else lo  # after the last live action
        e = np.flatnonzero(ex[a:hi])
        out.append((a + int(e[0]), True) if len(e) else (fallback, False))
    return out


def detect(c, tr, tags, backtrack=True, locate=True, goals=False, onset=0, extra=None):
    """Dead mask plus details. All switches off == dead_ball.detect(c, starts, snap=True).

    onset: 0 = dead_ball's centred speed cue; 1, 2 = stoppages.evidence levels (untagged stoppages).
    extra: optional per-frame evidence added before decoding (> 0 favours dead).
    """
    bx, by = ball_m(tr)
    n = len(bx)
    tl = tag_list(tags)
    rs = restarts(c, bx, by, tl, locate)
    df = dead_from(c, bx, by, tl, rs, backtrack)

    if onset:
        from barca import stoppages
        e = db.evidence(c, [], parts=("ball", "confirm", "gap")) + stoppages.evidence(c, tr, onset)
    else:
        e = db.evidence(c, [], parts=("speed", "ball", "confirm", "gap"))
    if extra is not None:
        e = e + extra
    tagw = np.zeros(n, bool)
    for t, (d0, _) in zip(tl, df):
        tagw[d0:min(t["start"] + int(db.TAG_POST_S * FPS), n)] = True
    dead = db.min_durations(db.viterbi_dead(e + db.TAG_W * tagw))
    raw = dead.copy()

    for t, (r, _), (d0, _) in zip(tl, rs, df):          # snap, as dead_ball.snap_restarts
        dead[d0:r] = True
        end = r
        while end < n and dead[end]:
            end += 1
        dead[r:end] = False

    goal_info = []
    if goals:
        for gs, ge in goal_tags(tags):                  # rejected rule, see the docstring
            ko = [k for k, t in enumerate(tl) if t["type"] == "kick off (after goal)"
                  and gs <= t["start"] <= ge + 5 * FPS]
            if not ko:
                goal_info.append({"kickoff": False})
                continue
            rk = rs[ko[0]][0]
            added = int((~dead[gs:rk]).sum())
            dead[gs:rk] = True
            goal_info.append({"kickoff": True, "goal_to_restart_s": round((rk - gs) / FPS, 1),
                              "frames_added": added})

    return dead, {"tags": tl, "restarts": rs, "dead_from": df, "goals": goal_info, "raw": raw}


def restart_team(labels, dead, tl, rs, events):
    """From each restart whose tag names a team, that team until the other side's first control.

    The cut sits halfway between the named team's last control and the other side's first, as in
    possession.labels_from_events(switch_at="mid"). Returns (labels, frames changed, restarts used).
    """
    out = labels.copy()
    n = len(out)
    starts = np.array([s for s, _, _ in events], dtype=np.int64)
    used = 0
    for t, (r, _) in zip(tl, rs):
        team = t["team"]
        if team is None or r >= n or dead[r]:
            continue
        b = r
        while b < n and not dead[b]:
            b += 1
        cut, last_end = b, r
        for s, e, tt in events[int(np.searchsorted(starts, r)):]:
            if s >= b:
                break
            if tt != team:
                cut = max(r, (min(last_end, s) + s) // 2)
                break
            last_end = e
        out[r:cut] = team
        used += 1
    return out, int(np.sum(out != labels)), used
