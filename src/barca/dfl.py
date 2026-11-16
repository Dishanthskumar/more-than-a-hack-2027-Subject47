"""DFL / IDSSE open data (Bassek et al. 2025, Scientific Data 12:195; CC BY 4.0, data (c) Deutsche
Fussball Liga) converted to this project's layout, as an external labelled set (other games only).

Position XML (TRACAB, 25 fps): one FrameSet per person and half; ball frames carry
BallPossession (1 home, 2 away) and BallStatus (1 alive, 0 dead). Labels follow the competition's
recipe: D where the ball is dead, otherwise the owning team (0 H, 1 A).

convert(match_id) parses once with a line regex (the files are 350-420 MB) and caches
cache/dfl/<match_id>.npz; load(match_id) returns a dict shaped like atd.load():
  players (n, slots, 3)  x, y normalised to 0-1 (origin bottom-left), speed m/s; NaN = absent
  ball (n, 3)            x, y normalised, z metres
  slot_team (slots,)     0 home, 1 away
  pitch (2,)             metres
  labels (n,) int8       0 H, 1 A, 2 D
  blocks [(first, last)] one per half, inclusive, in the concatenated frame index
"""

import glob
import os
import re
from array import array

import numpy as np

from barca import atd

DATA_DIR = os.path.join(atd.PROJECT, "DFL", "DFL_Data")
CACHE = os.path.join(atd.CACHE_DIR, "dfl")
SECTIONS = ("firstHalf", "secondHalf")

_SET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?[^>]*? S="([-\d.]+)"'
                    r'(?:[^>]*? BallPossession="(\d)" BallStatus="(\d)")?')


def match_ids():
    return sorted(re.search(r"(DFL-MAT-\w+)\.xml$", p).group(1)
                  for p in glob.glob(os.path.join(DATA_DIR, "*positions_raw*.xml")))


def _file(kind, match_id):
    hits = glob.glob(os.path.join(DATA_DIR, "*{}*{}.xml".format(kind, match_id)))
    if len(hits) != 1:
        raise FileNotFoundError("{} for {}: {}".format(kind, match_id, hits))
    return hits[0]


def teams(match_id):
    text = open(_file("matchinformation", match_id), encoding="utf-8").read()
    home = re.search(r'HomeTeamId="([^"]+)"', text).group(1)
    guest = re.search(r'GuestTeamId="([^"]+)"', text).group(1)
    size = re.search(r'PitchX="([\d.]+)"[^>]*PitchY="([\d.]+)"', text)
    return home, guest, size


class _Track:
    """Compact per-FrameSet storage: frame number, x, y, z, speed, possession, status."""

    def __init__(self):
        self.n, self.x, self.y = array("i"), array("f"), array("f")
        self.z, self.s, self.poss, self.stat = array("f"), array("f"), array("b"), array("b")

    def add(self, g):
        self.n.append(int(g[0]))
        self.x.append(float(g[1]))
        self.y.append(float(g[2]))
        self.z.append(float(g[3]) if g[3] else 0.0)
        self.s.append(float(g[4]))
        self.poss.append(int(g[5]) if g[5] else 0)
        self.stat.append(int(g[6]) if g[6] else 0)

    def arrays(self):
        return {k: np.frombuffer(getattr(self, k), dtype={"i": np.int32, "f": np.float32, "b": np.int8}[
            getattr(self, k).typecode]) for k in ("n", "x", "y", "z", "s", "poss", "stat")}


def parse(match_id):
    home, guest, _ = teams(match_id)
    sets = {}                       # (section, team, person) -> _Track
    pitch = (105.0, 68.0)
    cur = None
    with open(_file("positions_raw", match_id), encoding="utf-8") as fh:
        for line in fh:
            if "<Frame " in line:
                if cur is not None:
                    m = _FRAME.search(line)
                    if m:
                        cur.add(m.groups())
                continue
            m = _SET.search(line)
            if m:
                key = m.groups()
                cur = sets.setdefault(key, _Track()) if key[1] in (home, guest, "BALL") else None
                continue
            p = re.search(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"', line)
            if p:
                pitch = (float(p.group(1)), float(p.group(2)))
    return home, guest, pitch, {k: v.arrays() for k, v in sets.items()}


def convert(match_id, force=False):
    os.makedirs(CACHE, exist_ok=True)
    out_path = os.path.join(CACHE, match_id + ".npz")
    if os.path.isfile(out_path) and not force:
        return out_path
    home, guest, (L, W), sets = parse(match_id)
    persons = sorted({(t, p) for (_, t, p) in sets if t != "BALL"}, key=lambda tp: (tp[0] != home, tp[1]))
    slot_team = np.array([0 if t == home else 1 for t, _ in persons], np.int8)
    halves, blocks, start = [], [], 0
    for sec in SECTIONS:
        ball = next((v for (s, t, _), v in sets.items() if s == sec and t == "BALL"), None)
        if ball is None or not len(ball["n"]):
            continue
        n0 = int(ball["n"][0])
        n = int(ball["n"][-1]) - n0 + 1
        b = np.full((n, 3), np.nan, np.float32)
        lab = np.full(n, 2, np.int8)
        i = ball["n"] - n0
        b[i, 0], b[i, 1], b[i, 2] = (ball["x"] + L / 2) / L, (ball["y"] + W / 2) / W, ball["z"]
        lab[i] = np.where(ball["stat"] != 1, 2, np.where(ball["poss"] == 1, 0, 1))
        pl = np.full((n, len(persons), 3), np.nan, np.float32)
        for j, (t, p) in enumerate(persons):
            tr = sets.get((sec, t, p))
            if tr is None:
                continue
            i = tr["n"] - n0
            ok = (i >= 0) & (i < n)
            pl[i[ok], j, 0] = (tr["x"][ok] + L / 2) / L
            pl[i[ok], j, 1] = (tr["y"][ok] + W / 2) / W
            pl[i[ok], j, 2] = tr["s"][ok]
        halves.append((pl, b, lab))
        blocks.append((start, start + n - 1))
        start += n
    np.savez_compressed(out_path, players=np.concatenate([h[0] for h in halves]),
                        ball=np.concatenate([h[1] for h in halves]),
                        labels=np.concatenate([h[2] for h in halves]), slot_team=slot_team,
                        pitch=np.array([L, W]), blocks=np.array(blocks))
    return out_path


def _ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s)


def half_t0(match_id):
    """{section: (first ball frame N, its timestamp)} from the position file (streams it)."""
    out, want = {}, None
    with open(_file("positions_raw", match_id), encoding="utf-8") as fh:
        for line in fh:
            if want is not None and "<Frame " in line:
                m = re.search(r'N="(\d+)" T="([^"]+)"', line)
                out[want] = (int(m.group(1)), _ts(m.group(2)))
                want = None
                if len(out) == len(SECTIONS):
                    break
                continue
            if 'TeamId="BALL"' in line:
                want = re.search(r'GameSection="(\w+)"', line).group(1)
    return out


# Smart-Tagging-like set-piece tags from the DFL events (fair dead-ball bench, 2026-11-14).
# Kept per type at the rate Smart Tagging tagged them in BAR-ATH against the DFL mean per match:
# throw-in 26/44, corner 12/8.1 (all), goal kick 10/17, free kick 5/24.6, kick-off 6/4.6 (all).
TAG_TYPES = {"ThrowIn": "Throw-in", "CornerKick": "Corner Kick", "GoalKick": "Goal Kick",
             "FreeKick": "Free Kick", "KickOff": "Kick Off"}
TAG_KEEP = {"Throw-in": 26 / 44.0, "Corner Kick": 1.0, "Goal Kick": 10 / 17.0, "Free Kick": 5 / 24.6,
            "Kick Off": 1.0}
TAG_WINDOW_S = {"long": (5.0, 35.0), "kickoff": (3.4, 5.0)}    # (restart offset, window length)
TAG_JITTER_S = 1.0


def set_piece_tags(match_id, blocks, seed=0):
    rng = np.random.default_rng(seed)
    t0 = half_t0(match_id)
    text = open(_file("events_raw", match_id), encoding="utf-8").read()
    secs = [s for s in SECTIONS if s in t0]
    out = []
    for m in re.finditer(r'<Event [^>]*EventTime="([^"]+)"[^>]*>\s*<(\w+)', text):
        kind = TAG_TYPES.get(m.group(2))
        if kind is None or rng.random() >= TAG_KEEP[kind]:
            continue
        t = _ts(m.group(1))
        k = max(i for i, s in enumerate(secs) if i == 0 or t >= t0[s][1])
        frame = blocks[k][0] + int(round((t - t0[secs[k]][1]).total_seconds() * 25))
        if not blocks[k][0] <= frame <= blocks[k][1]:
            continue
        if kind == "Kick Off":
            first = not any(o["tags"]["Type"].startswith("Kick Off") and blocks[k][0] <= o["restart"] <= blocks[k][1]
                            for o in out)
            kind_out, (off, length) = ("Kick Off (Start)" if first else "Kick Off (After Goal)"), TAG_WINDOW_S["kickoff"]
        else:
            kind_out, (off, length) = kind, TAG_WINDOW_S["long"]
        start = frame - int(round((off + rng.uniform(-TAG_JITTER_S, TAG_JITTER_S)) * 25))
        out.append({"code": "SET PIECES", "start_frame": start, "end_frame": start + int(length * 25),
                    "start_s": start / 25, "end_s": (start + int(length * 25)) / 25, "restart": frame,
                    "tags": {"Team": "N/A", "Type": kind_out}})
    return out


CONTROLLED_ABSORB_S = 3.0   # brief flips back (H A H, A < 3 s) are not controlled changes; a guess
                            # that brings DFL's possession rate near the public GT's (see memory note)


def controlled_labels(labels, blocks, absorb_s=CONTROLLED_ABSORB_S, fps=25):
    """Approximate the competition's controlled possession: a team spell shorter than absorb_s
    between two spells of the other team goes to that team. Repeated until stable."""
    out = labels.copy()
    for a, b in blocks:
        w = out[a:b + 1]
        while True:
            cut = np.flatnonzero(w[1:] != w[:-1]) + 1
            bd = np.r_[0, cut, len(w)]
            r = [(s, e, int(w[s])) for s, e in zip(bd[:-1], bd[1:])]
            changed = False
            for i in range(1, len(r) - 1):
                s, e, k = r[i]
                if k != 2 and e - s < absorb_s * fps and r[i - 1][2] == r[i + 1][2] != 2 and r[i - 1][2] != k:
                    w[s:e] = r[i - 1][2]
                    changed = True
            if not changed:
                break
    return out


def load(match_id):
    with np.load(convert(match_id)) as z:
        d = {k: z[k] for k in z.files}
    d["blocks"] = [tuple(map(int, b)) for b in d["blocks"]]
    return d
