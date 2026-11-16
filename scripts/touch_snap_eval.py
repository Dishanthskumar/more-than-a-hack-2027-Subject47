"""Snap change times to learned touches (touch_model.snap_changes). Aggregates only.

    python scripts/touch_snap_eval.py            # DFL validation + BAR-ATH label-free checks
    python scripts/touch_snap_eval.py --score    # ... then the one public check

Pre-registered 2026-11-14, before running:
  DFL validation: the tracking-only rule pipeline (control events -> labels, switch at "mid") stands
  in for B2; its changes are snapped to out-of-fold detected touches (threshold 0.2, the all-match
  choice). Reference: DFL labels with brief flips (< 3 s) absorbed (dfl.controlled_labels).
  Reported before / after: spellF1 (competition code), live team accuracy, change timing (median
  |offset| of matched changes, share within 1 s).
  BAR-ATH: detector on the ATD (final model, frames with the tactical ball seen, B2 live frames),
  events per live minute vs DFL, how B2 changes move, Metrica agreement (receiver_eval yardsticks).
  Decision: the snapped B2 is preferred only if on DFL snapping raises spellF1 and lowers the
  median change offset, on BAR-ATH the event rate is within 2x of DFL's and Metrica agreement does
  not drop by more than 1 point, and the public spellF1 is not below B2's.
Writes reports/touch_snap_eval_summary.json; with --score submissions/barca_hybrid_ts.csv.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import make_hybrid as mh  # noqa: E402
import make_submission as ms  # noqa: E402
import possession_pipeline as pp  # noqa: E402
import receiver_eval as rx  # noqa: E402
import touch_model_eval as tme  # noqa: E402
from barca import atd, dfl, loaders, possession, touch_model as M  # noqa: E402
from games import source_frames  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "touch_snap_eval_summary.json")
TH = 0.2


def proxy_labels(m, dead):
    speed = np.full(m["n"], np.nan)
    speed[1:] = np.hypot(*np.diff(m["ball_m"], axis=0).T) * M.FPS
    ev = possession.control_events(m["team"], m["dist"], m["margin"], speed)
    base = np.where(m["team"] >= 0, m["team"], possession.DEAD).astype(np.int8)
    return possession.labels_from_events(ev, dead, possession.carry_labels(base, dead), "mid")


def timing(gt, pred, live):
    g, p = M.live_changes(gt, live), M.live_changes(pred, live)
    offs = []
    for c in g:
        k = p[(np.abs(p - c) <= 10 * M.FPS) & (pred[p] == gt[c])]
        if len(k):
            offs.append(int(k[np.argmin(np.abs(k - c))] - c))
    o = np.abs(np.array(offs)) / M.FPS
    return {"gt_changes": len(g), "matched": len(offs), "median_abs_offset_s": round(float(np.median(o)), 2),
            "within_1s": round(float(np.mean(o <= 1)), 3)}


def scores(gt, pred, live):
    runs = pm.valid_runs(gt)
    return {"spellF1": round(pm.spell_scores(gt, pred, runs)["mean_f1"], 4),
            "team_acc_live": round(float((pred[live] == gt[live]).mean()), 4),
            "timing": timing(gt, pred, live)}


def dfl_validation():
    ids = dfl.match_ids()
    data = {mid: tme.prepare(mid) for mid in ids}
    tot_flips = sum(len(d["flips"]) for d in data.values())
    live_min = sum(d["live"].sum() for d in data.values()) / M.FPS / 60
    rows, n_ev = [], 0
    agg = {"before": [], "after": []}
    for mid in ids:
        X, y = tme.stack([data[k] for k in ids if k != mid])
        clf = tme.model().fit(X, y)
        m = data[mid]
        p = tme.predict(clf, m)
        ev = M.events(p[0], p[1], m["live"], TH)
        n_ev += len(ev)
        d = dfl.load(mid)
        gt = dfl.controlled_labels(d["labels"], d["blocks"])
        dead = d["labels"] == 2
        live = ~dead
        before = proxy_labels(m, dead)
        after, moves = M.snap_changes(before, live, ev)
        r = {"match": mid, "before": scores(gt, before, live), "after": scores(gt, after, live),
             "changes_moved": len(moves), "changes": int(len(M.live_changes(before, live)))}
        rows.append(r)
        agg["before"].append(r["before"])
        agg["after"].append(r["after"])
        print(json.dumps(r), flush=True)
    mean = {k: {"spellF1": round(float(np.mean([x["spellF1"] for x in v])), 4),
                "team_acc_live": round(float(np.mean([x["team_acc_live"] for x in v])), 4),
                "median_abs_offset_s": round(float(np.mean([x["timing"]["median_abs_offset_s"] for x in v])), 2),
                "within_1s": round(float(np.mean([x["timing"]["within_1s"] for x in v])), 3)} for k, v in agg.items()}
    return {"events_per_live_min": round(n_ev / live_min, 2), "flips_per_live_min": round(tot_flips / live_min, 2),
            "mean_over_matches": mean, "per_match": rows}


def bar_atd(S, b2, dead, graded, score):
    live = graded & ~dead
    data = atd.load()
    with open(tme.MODEL, "rb") as fh:
        saved = pickle.load(fh)
    F = M.features(data["players"], data["ball"], data["slot_team"].astype(int), data["pitch"])
    seen = np.isfinite(data["ball"][:, 0])
    use = live & seen
    p = {}
    for t in (0, 1):
        q = np.zeros(len(b2))
        q[use] = saved["model"].predict_proba(F[t][use])[:, 1]
        p[t] = q
    ev = M.events(p[0], p[1], live, TH)
    snapped, moves = M.snap_changes(b2, live, ev)
    live_min = live.sum() / M.FPS / 60
    cov, trans = rx.metrica(len(b2))
    covered = live & (cov >= 0)
    mv = np.array(moves) / M.FPS
    out = {"events_per_live_min": round(len(ev) / live_min, 2),
           "events_per_live_min_ball_seen": round(len(ev) / (use.sum() / M.FPS / 60), 2),
           "b2_changes": int(len(M.live_changes(b2, live))), "changes_moved": len(moves),
           "move_s_q10_50_90": [round(float(v), 1) for v in np.percentile(mv, [10, 50, 90])] if len(mv) else None,
           "changed_s": round(float((snapped != b2).sum()) / M.FPS, 1)}
    for k, pred in (("B2", b2), ("B2_snap", snapped)):
        fm = pp.full_match(S, pred)
        out[k] = {"Y1_metrica_agree": round(100.0 * float((pred[covered] == cov[covered]).mean()), 1),
                  "Y2_transitions": rx.y2(pred, live, trans, len(b2)), "possessions": fm["n_possessions"]}
        if score:
            out[k]["public"] = pp.public_scores(S, pred, ref=None if k == "B2" else b2)
    if score:
        path = os.path.join(PROJECT, "submissions", "barca_hybrid_ts.csv")
        ms.write(path, np.array(pm.LABELS)[snapped])
        err = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical"))
        if err:
            sys.exit("invalid submission: " + err)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    out = {"dfl": dfl_validation()}
    print(json.dumps({k: v for k, v in out["dfl"].items() if k != "per_match"}, indent=1), flush=True)
    S = pp.setup()
    graded = np.zeros(len(S["team"]), bool)
    for s, e in S["runs"]:
        graded[s:e] = True
    dead, rb = mh.rebuilt(S, broadcast_rules=True)
    b2 = mh.hybrid(mh.v5_labels(), dead, rb)
    if not np.array_equal(b2, pm.label_codes(pm.read_label_csv(mh.BC_OUT))):
        sys.exit("B2 rebuilt here differs from " + mh.BC_OUT)
    out["bar_atd"] = bar_atd(S, b2, dead, graded, a.score)
    d, b = out["dfl"]["mean_over_matches"], out["bar_atd"]
    out["decision_label_free"] = bool(
        d["after"]["spellF1"] > d["before"]["spellF1"]
        and d["after"]["median_abs_offset_s"] < d["before"]["median_abs_offset_s"]
        and 0.5 <= b["events_per_live_min"] / out["dfl"]["flips_per_live_min"] <= 2
        and b["B2_snap"]["Y1_metrica_agree"] >= b["B2"]["Y1_metrica_agree"] - 1)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["bar_atd"], indent=1))
    print("decision_label_free", out["decision_label_free"])


if __name__ == "__main__":
    main()
