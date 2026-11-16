"""Parse the raw tactical ATD into dense numpy arrays, cached on disk.

Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<x,y,s>"  (coordinates normalised 0-1).
frame_id counts from 1; row i of every array here is video frame i (= submission row i).

The cache (cache/atd_tactical.npz) is derived match data. It must never be committed.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
KIT = os.path.join(PROJECT, "more-than-a-hack-2027")
sys.path.insert(0, KIT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.environ.get("BARCA_CACHE") or os.path.join(PROJECT, "cache")
N_SLOTS = 46
N_FIELDS = 1 + 3 * N_SLOTS + 3
CHUNK_LINES = 20000


def slot_teams(meta_path):
    """int8 array over the 46 track slots: 0 = home (first team in the file), 1 = away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    if len(slots) != N_SLOTS:
        raise ValueError("expected {} player slots, found {}".format(N_SLOTS, len(slots)))
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def _parse_chunk(lines, n_fields=N_FIELDS):
    text = ",".join(line.strip().replace(":", ",").replace(";", ",") for line in lines)
    values = np.array(text.split(","), dtype=np.float64)
    if values.size != len(lines) * n_fields:
        raise ValueError("malformed ATD chunk: {} values for {} lines".format(
            values.size, len(lines)))
    return values.reshape(len(lines), n_fields)


def parse(raw_path, n_frames, n_slots=N_SLOTS):
    """(players (n,slots,3) float32, ball (n,3) float32) with NaN where unobserved."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    n_fields = 1 + 3 * n_slots + 3

    def flush(buffer):
        block = _parse_chunk(buffer, n_fields)
        rows = block[:, 0].astype(np.int64) - 1
        if rows.min() < 0 or rows.max() >= n_frames:
            raise ValueError("frame id out of range")
        players[rows] = block[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
        ball[rows] = block[:, 1 + 3 * n_slots:]

    buffer = []
    with open(raw_path) as handle:
        for line in handle:
            if line.strip():
                buffer.append(line)
            if len(buffer) >= CHUNK_LINES:
                flush(buffer)
                buffer = []
    if buffer:
        flush(buffer)
    return players, ball


def load(game=GAME, data_dir=None, rebuild=False):
    """dict with players (n,46,3), ball (n,3), slot_team (46,), pitch (L, W). Cached."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(CACHE_DIR, "atd_tactical_{}.npz".format(game))
    if not rebuild and os.path.isfile(cache) \
            and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
    else:
        players, ball = parse(raw_path, source_frames("tactical", game))
        out = {"players": players, "ball": ball, "slot_team": slot_teams(meta_path)}
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    out["pitch"] = np.array(GAMES[game]["pitch"], dtype=np.float64)
    return out


BROADCAST_SLOTS = 51


def load_broadcast(game=GAME, data_dir=None, rebuild=False):
    """Broadcast ATD on ITS OWN timeline: dict with players (m,51,3), ball (m,3), pitch. Cached.

    Not aligned to the tactical video; see sync.py. Slot teams are not parsed (sync is
    team-agnostic).
    """
    (raw_path,) = require(["atd_raw"], game, data_dir, source="broadcast")
    cache = os.path.join(CACHE_DIR, "atd_broadcast_{}.npz".format(game))
    if not rebuild and os.path.isfile(cache)             and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
    else:
        players, ball = parse(raw_path, source_frames("broadcast", game), BROADCAST_SLOTS)
        out = {"players": players, "ball": ball}
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    out["pitch"] = np.array(GAMES[game]["pitch"], dtype=np.float64)
    return out
