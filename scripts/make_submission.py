"""Regenerate the possession file from the provided data.

    barca\\Scripts\\python.exe scripts\\make_submission.py              # v5 (default)
    barca\\Scripts\\python.exe scripts\\make_submission.py --version v4   # previous default

Deterministic: same data in, same bytes out.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from barca import atd, ball_fusion, boundaries, control, decoder, features, shape, tags  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402


def blocks(game=GAME):
    return [(first, last) for _, first, last in half_video_blocks(game)]


def build_features(version="v1", verbose=True, extra_fill=None, post=None):
    """extra_fill (v4+ only, None = unchanged): fn(data, ball, filled) -> (ball, filled), applied
    after the broadcast fill; its frames count as filled (decoder fill_weight, never a touch).
    post (None = unchanged): fn(f, data) -> f, applied last (e.g. receiver.add_to_features)."""
    data = atd.load()
    f = features.compute(data)
    if version == "v0":
        return f
    n = len(f["seen"])
    home, mask = features.pseudo_ownership(f)
    tag = tags.evidence(n, home, mask)
    f.update({k: tag[k] for k in ("phase_team", "pre_restart", "restart_team", "restart_frames")})
    f["shape_logit"], shape_acc = shape.train_and_predict(data, blocks(), home, mask)
    if verbose:
        print("phase tag agreement with tracking (rate, frames):", json.dumps(tag["calibration"]))
        print("shape model held-out pseudo-label accuracy per half:",
              [round(a, 3) for a in shape_acc])
    if version in ("v4", "v5", "v6"):
        # calibration and the shape model above use native tracking only; the fill joins here
        ball, filled, info = ball_fusion.fused_ball(data)
        if extra_fill is not None:
            ball, filled = extra_fill(data, ball, filled)
        fused = features.compute(dict(data, ball=ball))
        fused["filled"] = filled
        f = ball_fusion.overlay(f, fused)
        if verbose:
            print("broadcast ball fill:", json.dumps(info))
    if version == "v6":
        f["shape_confirm"] = True     # ambiguous changes decided by the formation reaction
    if version in ("v5", "v6"):
        # touches need the ball moving with the player; filled frames never touch anyway
        f["ball_rel_speed"] = features.ball_rel_speed(data)
    if version in ("v3", "v4", "v5", "v6"):
        f["control_team"], f["control_changes"] = control.evidence(f, blocks(), n)
        if verbose:
            print("control changes:", len(f["control_changes"]),
                  "frames with control evidence: %.3f" % float((f["control_team"] != 0).mean()))
    if post is not None:
        f = post(f, data)
    return f


def write(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v5", choices=("v0", "v1", "v2", "v3", "v4", "v5", "v6"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(PROJECT, "submissions", "barca_{}.csv".format(args.version))

    f = build_features(args.version)
    params = decoder.PARAMS_V0 if args.version == "v0" else decoder.PARAMS
    labels = decoder.decode(f, blocks(), source_frames("tactical"), params)
    if args.version == "v2":
        labels, moved = boundaries.refine(labels, f, blocks(), f["restart_frames"])
        print("boundary refinement:", json.dumps(moved))
    if args.version in ("v3", "v4", "v5", "v6"):
        labels, moved = boundaries.refine(labels, f, blocks(), team_changes=f["control_changes"],
                                          snap_dead=False)
        print("snap to control changes:", json.dumps(moved))
    write(out, labels)
    print(json.dumps(decoder.spell_stats(labels, blocks())))
    print("wrote", out)


if __name__ == "__main__":
    main()
