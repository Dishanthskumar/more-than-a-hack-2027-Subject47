"""Approach 1: next-receiver team signal in the v5 decoder (src/barca/receiver.py). Aggregates only.

    python scripts/receiver_eval.py            # label-free checks (full match)
    python scripts/receiver_eval.py --score    # ... then the one public check of R1 and R2

Same B2 dead mask and fallback; only the v5 team evidence changes. Label-free yardsticks:
  Y1  agreement with Metrica's phase-tag team (its own possession model, independent of ours) on
      live frames covered by exactly one team's tags
  Y2  Metrica ATTACKING TRANSITION tags: offset of our change to the tagged team (median, IQR)
  Y3  full-match possessions (B2 237; > 284 = over-splitting, as before)
  Y4  pass flights (ball > 8 m/s between two receptions): label = the first receiver's team,
      split into completed passes (same team receives) and turnovers (the other team receives)
Decision, fixed before running: a variant is preferred over B2 only if Y1 rises by >= 1 point,
Y3 stays under the alarm, and the public spellF1 is not below B2's.
Writes reports/receiver_eval_summary.json; with --score submissions/barca_hybrid_rx_<R>.csv.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import make_hybrid as mh  # noqa: E402
import make_submission as ms  # noqa: E402
import possession_pipeline as pp  # noqa: E402
from barca import loaders, possession, receiver, tags as T  # noqa: E402
from barca.loaders import FPS  # noqa: E402
from games import source_frames  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "receiver_eval_summary.json")
B2_POSSESSIONS, ALARM = 237, 1.2
FLIGHT_MS = possession.CONTROL_BALL_MAX_MS


def tag_team(x):
    return {T.HOME_TEAM_TAG: 0, T.AWAY_TEAM_TAG: 1}.get(x["tags"].get("Team"), -1)


def metrica(n):
    tg = loaders.load_smart_tagging()
    cov = np.full(n, -1)
    clash = np.zeros(n, bool)
    for x in tg:
        if x["code"] in T.PHASE_CODES and tag_team(x) >= 0:
            w = cov[x["start_frame"]:x["end_frame"]]
            clash[x["start_frame"]:x["end_frame"]] |= (w >= 0) & (w != tag_team(x))
            w[w < 0] = tag_team(x)
    cov[clash] = -1
    trans = [(x["start_frame"], tag_team(x)) for x in tg
             if x["code"] == "ATTACKING TRANSITION" and tag_team(x) >= 0]
    return cov, trans


def y2(pred, live, trans, n):
    ch = np.flatnonzero(live[1:] & live[:-1] & (pred[1:] != pred[:-1])) + 1
    offs = []
    for s, tm in trans:
        if not live[min(s, n - 1)]:
            continue
        k = [c for c in ch[np.abs(ch - s) <= 10 * FPS] if pred[c] == tm]
        if k:
            offs.append(min(((c - s) / FPS for c in k), key=abs))
    return {"matched": len(offs), "offset_s_q25_50_75": [round(float(v), 1) for v in np.percentile(offs, [25, 50, 75])]
            if offs else None}


def flights(rec, speed, live):
    """Pass-flight frames: between consecutive receptions, ball > FLIGHT_MS. (frames, first team, second team)."""
    idx = np.flatnonzero(rec != 0)
    fr, a_team, b_team = [], [], []
    for u, v in zip(idx[:-1], idx[1:]):
        if v - u <= 1 or v - u > 3 * FPS or not live[u:v + 1].all():
            continue
        w = np.arange(u + 1, v)
        w = w[np.nan_to_num(speed[w], nan=0) > FLIGHT_MS]
        fr.append(w)
        a_team.append(np.full(len(w), 0 if rec[u] > 0 else 1))
        b_team.append(np.full(len(w), 0 if rec[v] > 0 else 1))
    if not fr:
        return np.array([], int), np.array([], int), np.array([], int)
    return np.concatenate(fr), np.concatenate(a_team), np.concatenate(b_team)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    S = pp.setup()
    n = len(S["team"])
    graded = np.zeros(n, bool)
    for s, e in S["runs"]:
        graded[s:e] = True
    dead, rb = mh.rebuilt(S, broadcast_rules=True)
    b2 = mh.hybrid(mh.v5_labels(), dead, rb)
    if not np.array_equal(b2, pm.label_codes(pm.read_label_csv(mh.BC_OUT))):
        sys.exit("B2 rebuilt here differs from " + mh.BC_OUT)
    live = graded & ~dead

    kept = {}

    def hook(mode):
        def post(f, data):
            f = receiver.add_to_features(f, data, ms.blocks(), mode)
            kept["rec"], kept["speed"] = f["receptions"], f["ball_speed"]
            kept["signal"] = f["receiver_team"]
            return f
        return post

    preds = {"B2": b2,
             "R1": mh.hybrid(mh.v5_labels(post=hook("replace")), dead, rb),
             "R2": mh.hybrid(mh.v5_labels(post=hook("add")), dead, rb)}
    rec, speed, sig = kept["rec"], kept["speed"], kept["signal"]
    cov, trans = metrica(n)
    covered = live & (cov >= 0)
    fl, fa, fb = flights(rec, speed, live)
    same = fa == fb
    out = {"receptions": {"per_live_min": round(float((rec[live] != 0).sum()) / (live.sum() / FPS / 60), 1),
                          "reception_runs": int(np.sum((rec[1:] != 0) & (rec[1:] != rec[:-1]))),
                          "signal_nonzero_live": round(float((sig[live] != 0).mean()), 3)},
           "metrica_covered_live_share": round(float(covered.sum() / live.sum()), 3),
           "flight_frames": {"completed": int(same.sum()), "turnover": int((~same).sum())},
           "variants": {}}
    for k, p in preds.items():
        fm = pp.full_match(S, p)
        r = {"Y1_metrica_agree": round(100.0 * float((p[covered] == cov[covered]).mean()), 1),
             "Y2_transitions": y2(p, live, trans, n),
             "Y3_possessions": fm["n_possessions"],
             "Y3_over_splitting": bool(fm["n_possessions"] > ALARM * B2_POSSESSIONS),
             "Y4_flight_label_is_first_team_pct": {
                 "completed": round(100.0 * float((p[fl[same]] == fa[same]).mean()), 1) if same.any() else None,
                 "turnover": round(100.0 * float((p[fl[~same]] == fa[~same]).mean()), 1) if (~same).any() else None},
             "changed_vs_B2_s": round(float((graded & (p != b2)).sum()) / FPS, 1),
             "mean_possession_s": fm["mean_possession_s"]}
        if a.score:
            r["public"] = pp.public_scores(S, p, ref=None if k == "B2" else b2)
            if k != "B2":
                path = os.path.join(PROJECT, "submissions", "barca_hybrid_rx_{}.csv".format(k))
                ms.write(path, np.array(pm.LABELS)[p])
                err = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical"))
                if err:
                    sys.exit("invalid submission {}: {}".format(path, err))
        out["variants"][k] = r
    y1 = out["variants"]["B2"]["Y1_metrica_agree"]
    out["decision_label_free"] = {k: bool(out["variants"][k]["Y1_metrica_agree"] - y1 >= 1
                                          and not out["variants"][k]["Y3_over_splitting"]) for k in ("R1", "R2")}
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
