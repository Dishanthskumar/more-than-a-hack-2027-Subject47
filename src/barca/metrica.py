"""Metrica Sports sample data (games 1 and 2 CSV, game 3 EPTS + JSON via kloppy) for testing and
training (other games only).

Source: Metrica Sports sample data, https://github.com/metrica-sports/sample-data. The README asks
for responsible use and acknowledgement. Tracking at 25 fps, 0-1 coordinates with (0, 0) top left
(flipped here to bottom left like the other loaders), pitch 105 x 68 m; anonymised.

There are no per-frame flags, so labels come from the events (fixed 2026-11-14 before any statistic):
  live team  the team of the most recent SET PIECE, RECOVERY, PASS or SHOT (start frame); BALL LOST
             and CHALLENGE alone change nothing (a contested ball stays with the last team)
  dead       from a stopping event to the start frame of the next restart in the period:
             BALL OUT (END frame: the ball is outside the pitch there in 94-98% of cases, at the
             start frame in none), FAULT RECEIVED (start), SHOT whose subtype ends OUT or GOAL
             (end frame), BALL LOST subtype OFFSIDE / HAND BALL / END HALF (start). A restart is
             a SET PIECE or any event whose subtype contains GOAL KICK (Metrica records goal kicks
             as PASS / BALL LOST, not SET PIECE).
  A restart with no stopping event since the previous restart (10-13 per game, mostly free kicks
  after a foul logged as a CHALLENGE / BALL LOST) is dead from the end frame of the last event
  before it.
  Amended 2026-11-14 (first version: BALL OUT start frame, restarts = SET PIECE only, which left
  live play after goal kicks labelled dead; dead share 48-53% -> see memory note).
load(game) returns a dict shaped like dfl.load(): players (n, slots, 3) with x, y normalised and
speed NaN, ball (n, 3), slot_team, pitch, labels (0 H, 1 A, 2 D), blocks (one per period, inclusive).
"""

import os

import numpy as np
import pandas as pd

from barca import atd

DATA = os.path.join(atd.PROJECT, "Metrica", "sample-data-master", "sample-data-master", "data")
GAMES = (1, 2, 3)
CACHE = os.path.join(atd.CACHE_DIR, "metrica")
PITCH = (105.0, 68.0)
TEAM_EVENTS = ("SET PIECE", "RECOVERY", "PASS", "SHOT")


def _tracking(game, side):
    path = os.path.join(DATA, "Sample_Game_%d" % game, "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (game, side))
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = []
    for c in players:
        i = cols.index(c)
        xy.append(raw.iloc[:, [i, i + 1]].to_numpy(np.float64))
    i = cols.index("Ball")
    ball = raw.iloc[:, [i, i + 1]].to_numpy(np.float64)
    return raw["Period"].to_numpy(), raw["Frame"].to_numpy(), np.stack(xy, 1), ball


def events(game):
    if game != 3:
        return pd.read_csv(os.path.join(DATA, "Sample_Game_%d" % game, "Sample_Game_%d_RawEventsData.csv" % game))
    import json
    raw = json.load(open(os.path.join(DATA, "Sample_Game_3", "Sample_Game_3_events.json"), encoding="utf-8"))["data"]
    rows = []
    for e in raw:
        st = e.get("subtypes")
        if isinstance(st, dict):
            sub = st.get("name")
        elif isinstance(st, list):
            sub = "-".join(x.get("name", "") for x in st)
        else:
            sub = None
        tid = (e.get("team") or {}).get("id")
        rows.append({"Team": {"FIFATMA": "Home", "FIFATMB": "Away"}.get(tid), "Type": e["type"]["name"],
                     "Subtype": sub, "Period": e.get("period"), "Start Frame": e["start"]["frame"],
                     "End Frame": e["end"]["frame"]})
    return pd.DataFrame(rows)


def _game3():
    """(period, frame, home (n, p, 2), away (n, q, 2), ball (n, 2)) from the EPTS files via kloppy, cached."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "game3.npz")
    if not os.path.isfile(path):
        from kloppy import metrica as km
        d3 = os.path.join(DATA, "Sample_Game_3")
        ds = km.load_tracking_epts(meta_data=os.path.join(d3, "Sample_Game_3_metadata.xml"),
                                   raw_data=os.path.join(d3, "Sample_Game_3_tracking.txt"), coordinates="metrica")
        df = ds.to_df()
        side = {}
        for tm in ds.metadata.teams:
            ids = [pl.player_id for pl in tm.players if "%s_x" % pl.player_id in df.columns]
            side[str(tm.ground).lower().split(".")[-1]] = np.stack(
                [df[["%s_x" % i, "%s_y" % i]].to_numpy(np.float64) for i in ids], 1)
        np.savez_compressed(path, period=df["period_id"].to_numpy(), frame=df["frame_id"].to_numpy(),
                            home=side["home"], away=side["away"], ball=df[["ball_x", "ball_y"]].to_numpy(np.float64))
    with np.load(path) as z:
        return z["period"], z["frame"], z["home"], z["away"], z["ball"]


def labels_from_events(ev, frame0, n, blocks):
    """(n,) int8 labels from the events (see module doc). Frames are 1-based in the files."""
    lab = np.full(n, 2, np.int8)
    team = {"Home": 0, "Away": 1}
    idx = lambda f: int(f) - frame0  # noqa: E731
    ev = ev.sort_values("Start Frame", kind="stable")
    for a, b in blocks:
        e = ev[(ev["Start Frame"] - frame0 >= a) & (ev["Start Frame"] - frame0 <= b)]
        cur = None
        last = a
        for _, r in e.iterrows():
            f = idx(r["Start Frame"])
            if r["Type"] in TEAM_EVENTS and r["Team"] in team:
                if cur is not None:
                    lab[last:f] = cur
                cur, last = team[r["Team"]], f
        if cur is not None:
            lab[last:b + 1] = cur
        sub = e["Subtype"].fillna("").astype(str).str.upper()
        is_restart = (e["Type"] == "SET PIECE") | sub.str.contains("GOAL KICK")
        restarts = np.sort(e.loc[is_restart, "Start Frame"].map(idx).to_numpy())
        stops = []
        for (_, r), s in zip(e.iterrows(), sub):
            if r["Type"] == "BALL OUT":
                stops.append(idx(r["End Frame"]))
            elif r["Type"] == "FAULT RECEIVED":
                stops.append(idx(r["Start Frame"]))
            elif r["Type"] == "SHOT" and (s.endswith("OUT") or s.endswith("GOAL")):
                stops.append(idx(r["End Frame"]))
            elif r["Type"] == "BALL LOST" and any(k in s for k in ("OFFSIDE", "HAND BALL", "END HALF")):
                stops.append(idx(r["Start Frame"]))
        starts_ = e["Start Frame"].map(idx).to_numpy()
        ends_ = e["End Frame"].map(idx).to_numpy()
        stops_arr = np.array(sorted(stops))
        prev = a - 1
        for r in restarts:
            if not ((stops_arr > prev) & (stops_arr <= r)).any():
                before = np.flatnonzero(starts_ < r)
                if len(before):
                    f = int(max(ends_[before[-1]], starts_[before[-1]]))
                    if prev < f < r:
                        stops.append(f)
            prev = r
        for s in stops:
            nxt = restarts[restarts > s]
            lab[s:(int(nxt[0]) if len(nxt) else b + 1)] = 2
        first = restarts[0] if len(restarts) else a
        lab[a:first] = 2                                # before the period's kick-off
    return lab


TAG_TYPES = {"THROW IN": "Throw-in", "CORNER KICK": "Corner Kick", "FREE KICK": "Free Kick",
             "KICK OFF": "Kick Off", "GOAL KICK": "Goal Kick"}


def set_piece_tags(game, n, blocks, seed=0):
    """Smart-Tagging-like set-piece tags from the Metrica restarts, kept per type at BAR-ATH's
    tagging rates and shaped like dfl.set_piece_tags (same constants)."""
    from barca import dfl
    rng = np.random.default_rng(seed)
    e = events(game).sort_values("Start Frame", kind="stable")
    frame0 = 1
    out = []
    for _, r in e.iterrows():
        sub = str(r["Subtype"] or "").upper() if isinstance(r["Subtype"], str) else ""
        kind = None
        if r["Type"] == "SET PIECE":
            kind = next((v for k, v in TAG_TYPES.items() if sub.startswith(k)), None)
        elif "GOAL KICK" in sub:
            kind = "Goal Kick"
        if kind is None or kind == "Free Kick" and "PENALTY" in sub or rng.random() >= dfl.TAG_KEEP[kind]:
            continue
        f = int(r["Start Frame"]) - frame0
        if not 0 <= f < n:
            continue
        k = next(i for i, (a, b) in enumerate(blocks) if a <= f <= b) if any(a <= f <= b for a, b in blocks) else None
        if k is None:
            continue
        if kind == "Kick Off":
            first = not any(o["tags"]["Type"].startswith("Kick Off") and blocks[k][0] <= o["restart"] <= blocks[k][1] for o in out)
            kind_out, (off, length) = ("Kick Off (Start)" if first else "Kick Off (After Goal)"), dfl.TAG_WINDOW_S["kickoff"]
        else:
            kind_out, (off, length) = kind, dfl.TAG_WINDOW_S["long"]
        start = f - int(round((off + rng.uniform(-dfl.TAG_JITTER_S, dfl.TAG_JITTER_S)) * 25))
        out.append({"code": "SET PIECES", "start_frame": start, "end_frame": start + int(length * 25),
                    "start_s": start / 25, "end_s": (start + int(length * 25)) / 25, "restart": f,
                    "tags": {"Team": "N/A", "Type": kind_out}})
    return out


def load(game):
    if game == 3:
        ph, fh, home, away, ball = _game3()
    else:
        ph, fh, home, ball = _tracking(game, "Home")
        pa, fa, away, _ = _tracking(game, "Away")
        if not (np.array_equal(fh, fa) and np.array_equal(ph, pa)):
            raise ValueError("home and away tracking frames differ")
    n = len(fh)
    frame0 = int(fh[0])
    pl = np.concatenate([home, away], 1)
    players = np.full((n, pl.shape[1], 3), np.nan, np.float32)
    players[..., 0] = pl[..., 0]
    players[..., 1] = 1 - pl[..., 1]                    # (0, 0) top left -> bottom left
    b = np.full((n, 3), np.nan, np.float32)
    b[:, 0], b[:, 1], b[:, 2] = ball[:, 0], 1 - ball[:, 1], 0.0
    slot_team = np.r_[np.zeros(home.shape[1], np.int8), np.ones(away.shape[1], np.int8)]
    blocks = []
    for p in np.unique(ph):
        w = np.flatnonzero(ph == p)
        blocks.append((int(w[0]), int(w[-1])))
    return {"players": players, "ball": b, "slot_team": slot_team, "pitch": np.array(PITCH),
            "labels": labels_from_events(events(game), frame0, n, blocks), "blocks": blocks}
