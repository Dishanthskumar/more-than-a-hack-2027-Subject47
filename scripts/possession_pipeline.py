"""Rebuild the BAR-ATH possession pipeline and score every step.

    python scripts/possession_pipeline.py            # all steps + rejected variants + grid
    python scripts/possession_pipeline.py --quick    # skip the dist x dwell grid

Writes submission CSVs to cache/submissions/ (never share them) and one aggregates-only JSON to
reports/possession_pipeline_summary.json. The best submission is cache/submissions/poss_control_mid.csv.

Ground rules: constants are fixed in the modules before scoring; the public window is only a final
check of the variants named here. The grid is judged on full-match numbers (no ground truth) only.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from barca import dead_ball, loaders, nearest, possession  # noqa: E402
from barca.loaders import FPS  # noqa: E402
import baseline  # noqa: E402  (starter, on path via loaders)
import possession_metrics as pm  # noqa: E402

D = pm.LABELS.index("D")
X = pm.LABELS.index("X")
N_FRAMES = 199319


def setup():
    t0 = time.time()
    tr = loaders.load_tracking()
    _, team, dist, margin = nearest.from_tracking(tr)
    tags = loaders.load_smart_tagging()
    starts = dead_ball.tag_starts(tags)
    c = dead_ball.cues(tr, loaders.load_homography(), tags)
    base = pm.label_codes(baseline.smooth(team, 50))
    print("setup %.0f s: %d frames, %d set-piece tags" % (time.time() - t0, len(team), len(starts)))
    return {"team": team, "dist": dist, "margin": margin, "cues": c, "starts": starts,
            "base": base, "gt": loaders.load_public_gt(), "runs": loaders.graded_runs()}


def public_scores(S, pred, ref=None):
    gt = S["gt"]
    r = pm.evaluate(gt, pred)
    g = gt != X
    gl, pl = gt[g], pred[g]
    gt_dead, pr_dead = gl == D, pl == D
    out = {
        "macroF1": round(r["macro_f1"], 4),
        "spellF1": round(r["spell_f1_mean"], 4),
        "alive_dead_f1": round(r["alive_dead_f1"], 4),
        "dead_precision": round(r["per_class"]["D"]["precision"], 4),
        "dead_recall": round(r["per_class"]["D"]["recall"], 4),
        "dead_f1": round(r["per_class"]["D"]["f1"], 4),
        "n_gt_spells": r["n_gt_spells"], "n_pred_spells": r["n_pred_spells"],
        "wrong": int(np.sum(gl != pl)),
        "dead_called_live": int(np.sum(gt_dead & ~pr_dead)),
        "live_called_dead": int(np.sum(~gt_dead & pr_dead)),
        "team_swap": int(np.sum(~gt_dead & ~pr_dead & (gl != pl))),
    }
    if ref is not None:
        rl = ref[g]
        out["fixed_vs_ref"] = int(np.sum((rl != gl) & (pl == gl)))
        out["new_vs_ref"] = int(np.sum((rl == gl) & (pl != gl)))
    return out


def full_match(S, pred):
    runs = S["runs"]
    graded = np.zeros(len(pred), bool)
    for a, b in runs:
        graded[a:b] = True
    dead = (pred == D) & graded
    lengths = []
    for a, b in runs:
        s, k = loaders.run_lengths(dead[a:b])
        lengths += (k / FPS).tolist()
    pre = [bool(dead[max(s - 5 * FPS, 0):s].any()) for s in S["starts"] if graded[min(s, len(pred) - 1)]]
    st = pm.possession_stats(pred, runs)
    return {
        "dead_share": round(float(dead.sum() / graded.sum()), 4),
        "n_dead_spells": len(lengths),
        "dead_len_s_q10_50_90": [round(float(q), 1) for q in np.percentile(lengths, [10, 50, 90])]
        if lengths else None,
        "set_pieces_dead_in_5s_before": round(float(np.mean(pre)), 3) if pre else None,
        "n_possessions": st["n_spells"],
        "mean_possession_s": round(st["mean_duration_s"], 2),
    }


def write(name, pred):
    path = os.path.join(loaders.SUBMISSION_DIR, name + ".csv")
    baseline.write_submission(path, np.array(pm.LABELS)[pred])
    with open(path) as fh:
        rows = sum(1 for _ in fh) - 1
    assert rows == N_FRAMES, (path, rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the distance x dwell grid")
    args = ap.parse_args()

    S = setup()
    c, starts, base = S["cues"], S["starts"], S["base"]
    team, dist, margin = S["team"], S["dist"], S["margin"]

    dead_speed = dead_ball.detect(c, starts, snap=False, parts=("speed",))
    dead_all = dead_ball.detect(c, starts, snap=False)
    dead = dead_ball.detect(c, starts, snap=True)
    events = possession.control_events(team, dist, margin, c["ball_speed"])
    carry = possession.carry_labels(base, dead)

    def ctrl(switch_at, ev=events):
        return possession.labels_from_events(ev, dead, carry, switch_at, team=team)

    steps = [
        ("baseline", "Starter baseline: nearest team, majority over +/-50 frames", base, None),
        ("dead_speed", "+ dead-ball detector, player speed only",
         np.where(dead_speed, D, base).astype(np.int8), "dead_speed"),
        ("dead_all_cues", "+ all dead-ball cues",
         np.where(dead_all, D, base).astype(np.int8), "dead_all_cues"),
        ("dead_all_cues_snap", "+ restart snap at tagged set pieces",
         np.where(dead, D, base).astype(np.int8), "dead_all_cues_snap"),
        ("poss_carry", "+ carry the last team through live spells", carry, "poss_carry"),
        ("poss_control", "+ control events, switch at the new team's first control",
         ctrl("start"), "poss_control"),
        ("poss_control_mid", "+ switch halfway between the two teams' controls (BEST)",
         ctrl("mid"), "poss_control_mid"),
    ]
    rejected = [
        ("poss_control_handover", "switch at the old team's last control", ctrl("handover")),
        ("poss_control_nearest", "switch at the nearest-team flip", ctrl("nearest")),
        ("poss_control_confirm", "confirmation rule (1.5 s hold or 3 s repeat), mid switch",
         ctrl("mid", possession.confirmed(events, dead))),
    ]

    best = steps[-1][2]
    summary = {"n_control_events": len(events), "steps": [], "rejected": [], "grid": []}
    prev = None
    print("\n%-58s %8s %8s %6s %7s %6s" % ("step", "spellF1", "macroF1", "wrong", "poss", "mean s"))
    for key, title, pred, csv in steps:
        row = {"key": key, "title": title, "public": public_scores(S, pred, prev),
               "full_match": full_match(S, pred), "csv": (csv + ".csv") if csv else None}
        if csv:
            write(csv, pred)
        summary["steps"].append(row)
        p, f = row["public"], row["full_match"]
        print("%-58s %8.3f %8.3f %6d %7d %6.1f" % (title[:58], p["spellF1"], p["macroF1"],
                                                 p["wrong"], f["n_possessions"],
                                                 f["mean_possession_s"]))
        prev = pred
    for key, title, pred in rejected:
        write(key, pred)
        row = {"key": key, "title": title, "public": public_scores(S, pred, best),
               "full_match": full_match(S, pred)}
        summary["rejected"].append(row)
        p, f = row["public"], row["full_match"]
        print("  rejected: %-47s %8.3f %8.3f %6d %7d %6.1f" % (title[:47], p["spellF1"],
                                                             p["macroF1"], p["wrong"],
                                                             f["n_possessions"],
                                                             f["mean_possession_s"]))

    b = summary["steps"][-1]["public"]
    print("\nbest wrong frames: %d = %d team swap + %d dead called live + %d live called dead"
          % (b["wrong"], b["team_swap"], b["dead_called_live"], b["live_called_dead"]))
    fm = summary["steps"][-1]["full_match"]
    print("best full match: %.0f%% graded time dead, %d dead spells, %d possessions (%.1f s)"
          % (100 * fm["dead_share"], fm["n_dead_spells"], fm["n_possessions"],
             fm["mean_possession_s"]))

    if not args.quick:
        print("\ngrid (full match only, no ground truth): dist x dwell -> possessions, mean s")
        for d_m in (1.5, 2.0, 2.5):
            for dw in (0.25, 0.5, 1.0):
                ev = possession.control_events(team, dist, margin, c["ball_speed"], d_m, dw)
                pred = possession.labels_from_events(ev, dead, carry, "mid")
                f = full_match(S, pred)
                summary["grid"].append({"dist_m": d_m, "dwell_s": dw, "n_events": len(ev),
                                        "n_possessions": f["n_possessions"],
                                        "mean_possession_s": f["mean_possession_s"]})
                print("  %.2f m  %.2f s  events %5d  possessions %4d  mean %.1f s"
                      % (d_m, dw, len(ev), f["n_possessions"], f["mean_possession_s"]))

    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    out = os.path.join(loaders.REPORTS_DIR, "possession_pipeline_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote", os.path.relpath(out, loaders.PROJECT_ROOT))
    print("submit:", os.path.relpath(os.path.join(loaders.SUBMISSION_DIR, "poss_control_mid.csv"),
                                     loaders.PROJECT_ROOT))


if __name__ == "__main__":
    main()
