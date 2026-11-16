"""Thin adapter giving the possession pipeline the helper API it was specified against.

The spec ("BAR-ATH possession pipeline: session summary and rebuild spec") was written in a repo
whose helpers lived in loaders.py. Here they wrap this repo's own pieces: atd.load (cached
tracking), the starter's homography / Smart Tagging loaders and games.py.

Nothing here prints or exports raw data; callers report aggregates only.
"""

import os
import sys

import numpy as np

from barca import atd

PROJECT_ROOT = atd.PROJECT
KIT = atd.KIT
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "starter"))
sys.path.insert(0, os.path.join(KIT, "evaluation"))

from games import GAME, half_video_blocks, source_frames  # noqa: E402
import load_homography as _hom  # noqa: E402
import load_smart_tagging as _tags  # noqa: E402

PITCH_M = (105.3, 68.0)
FPS = 25
CACHE_DIR = atd.CACHE_DIR
SUBMISSION_DIR = os.path.join(CACHE_DIR, "submissions")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PUBLIC_GT = os.path.join(KIT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def load_tracking(game=GAME):
    """dict: ball (N, 2) normalised, px / py / ps (N, S) normalised and m/s, slot_team (S,)."""
    data = atd.load(game)
    p = data["players"]
    return {"ball": data["ball"][:, :2], "px": p[:, :, 0], "py": p[:, :, 1], "ps": p[:, :, 2],
            "slot_team": data["slot_team"].astype(np.int64), "n_slots": p.shape[1]}


def load_homography(game=GAME):
    """dict with the per-frame translation terms h02, h12 (NaN where invalid)."""
    m, _, _ = _hom.load(game)
    return {"h02": m[:, 0, 2], "h12": m[:, 1, 2]}


def load_smart_tagging(game=GAME):
    return _tags.load(game)


def load_public_gt():
    """Label codes per frame: 0 H, 1 A, 2 D, 3 X."""
    import possession_metrics as pm
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def graded_runs(game=GAME):
    """[(first, last + 1)] per graded half."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def half_array(n, game=GAME):
    """(n,) int8: 1 or 2 inside each graded half, 0 elsewhere."""
    out = np.zeros(n, np.int8)
    for half, first, last in half_video_blocks(game):
        out[first:last + 1] = half
    return out


def run_lengths(mask):
    """(starts, lengths) of the True runs of a boolean array."""
    m = np.asarray(mask, bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts
