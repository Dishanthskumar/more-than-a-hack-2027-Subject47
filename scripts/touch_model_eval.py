"""Train and test the learned touch detector on the 7 DFL matches, leave one match out. Aggregates only.

    python scripts/touch_model_eval.py [--degrade]

--degrade (2026-11-14): train AND test on DFL made to look like the BAR-ATH ATD (src/barca/degrade.py,
seed 0); the all-match model then also runs on the ATD to compare event rates (transfer check).

Fixed before running:
  Model: sklearn HistGradientBoostingClassifier (defaults except max_iter=300, early stopping
  on a 10% internal split, seed 0), live frames only, rows for both candidate teams stacked.
  Threshold for a held-out match: the one maximising pooled F1 (tolerance 12 frames) on the
  out-of-fold predictions of the OTHER six matches (grid 0.10-0.90 by 0.05).
  Scoring: true events = DFL last-touch flips with the ball alive (touch_model.flips);
  predicted events matched one-to-one, same team, within 5 frames (0.2 s) and 12 (0.5 s).
  Baselines on the same events, no training:
    kink   touches.kinks (dv >= 3 m/s) whose nearest player is within 3 m, the other side >= 1 m
           further, dv < 8 m/s (the T2 rule), team = nearest team; then alternation
    control  starts of possession.control_events (2 m, margin 1 m, ball <= 8 m/s, 0.5 s); then
           alternation (a change of controlling team)
Writes reports/touch_model_eval_summary.json and the all-match model to cache/dfl/touch_model.pkl.
"""

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from barca import dfl, loaders, nearest, possession, touch_model as M, touches  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "touch_model_eval_summary.json")
MODEL = os.path.join(dfl.CACHE, "touch_model.pkl")
GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)
TOLS = (5, 12)


def model():
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, validation_fraction=0.1,
                                          random_state=0)


DEG = None          # (gap_rate, gap_lengths) when --degrade


def prepare(mid):
    d = dfl.load(mid)
    if DEG is not None:
        from barca import degrade
        d = degrade.degrade(d, DEG[0], DEG[1], seed=0)
    n = len(d["labels"])
    live = d["labels"] != 2
    F = M.features(d["players"], d["ball"], d["slot_team"], d["pitch"])
    fl = M.flips(d["labels"], d["blocks"])
    y = M.targets(n, fl)
    L, W = d["pitch"]
    ball_m = d["ball"][:, :2].astype(np.float64) * [L, W]
    _, team, dist, margin = nearest.nearest_team(ball_m.astype(np.float32), d["players"][..., 0] * np.float32(L),
                                                 d["players"][..., 1] * np.float32(W), d["slot_team"])
    return {"F": F, "y": y, "live": live, "flips": fl, "n": n, "ball_m": ball_m,
            "team": team, "dist": dist, "margin": margin}


def stack(ms):
    X = np.concatenate([m["F"][t][m["live"]] for m in ms for t in (0, 1)])
    y = np.concatenate([m["y"][t][m["live"]] for m in ms for t in (0, 1)])
    return X, y


def predict(clf, m):
    p = {}
    for t in (0, 1):
        q = np.zeros(m["n"])
        q[m["live"]] = clf.predict_proba(m["F"][t][m["live"]])[:, 1]
        p[t] = q
    return p


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def score(m, ev):
    return {tol: M.match_events(m["flips"], ev, tol) for tol in TOLS}


def kink_events(m):
    dv = touches.velocity_change_fast(m["ball_m"])
    ks = touches.kinks(dv)
    ok = [int(k) for k in ks if m["live"][k] and m["team"][k] >= 0 and m["dist"][k] < touches.TOUCH_DIST_M
          and m["margin"][k] >= touches.TOUCH_MARGIN_M and dv[k] < touches.TOUCH_DV_MAX_MS]
    return M.alternate(ok, m["team"].astype(int), m["live"])


def control_events(m):
    speed = np.full(m["n"], np.nan)
    speed[1:] = np.hypot(*np.diff(m["ball_m"], axis=0).T) * M.FPS
    ev = possession.control_events(m["team"], m["dist"], m["margin"], speed)
    starts = [s for s, _, _ in ev if m["live"][s]]
    team = np.full(m["n"], -1)
    for s, _, t in ev:
        team[s] = t
    return M.alternate(starts, team, m["live"])


def atd_transfer(saved_model, threshold):
    """Event rate of the all-match model on the BAR-ATH ATD (B2 live frames, tactical ball seen)."""
    import possession_metrics as pm
    from barca import atd
    b2 = pm.label_codes(pm.read_label_csv(os.path.join(PROJECT, "submissions", "barca_hybrid_bc.csv")))
    live = b2 < 2
    data = atd.load()
    F = M.features(data["players"], data["ball"], data["slot_team"].astype(int), data["pitch"])
    p = {}
    for t in (0, 1):
        q = np.zeros(len(b2))
        q[live] = saved_model.predict_proba(F[t][live])[:, 1]
        p[t] = q
    ev = M.events(p[0], p[1], live, threshold)
    return {"events_per_live_min": round(len(ev) / (live.sum() / M.FPS / 60), 2)}


def main():
    global DEG, OUT, MODEL
    if "--degrade" in sys.argv[1:]:
        import possession_metrics as pm
        from barca import atd, degrade
        b2 = pm.label_codes(pm.read_label_csv(os.path.join(PROJECT, "submissions", "barca_hybrid_bc.csv")))
        DEG = degrade.atd_gap_profile(atd.load()["ball"], b2 < 2)
        OUT = OUT.replace(".json", "_degraded.json")
        MODEL = MODEL.replace(".pkl", "_degraded.pkl")
    t0 = time.time()
    ids = dfl.match_ids()
    data = {mid: prepare(mid) for mid in ids}
    print("prepared %d matches, %d flips, %.0f s" % (len(ids), sum(len(d["flips"]) for d in data.values()),
                                                     time.time() - t0), flush=True)
    oof = {}
    for mid in ids:
        X, y = stack([data[k] for k in ids if k != mid])
        clf = model().fit(X, y)
        oof[mid] = predict(clf, data[mid])
        print("fold", mid, "fitted, %.0f s" % (time.time() - t0), flush=True)

    memo = {}

    def counts(k, th):
        if (k, th) not in memo:
            memo[(k, th)] = M.match_events(data[k]["flips"],
                                           M.events(oof[k][0], oof[k][1], data[k]["live"], th), 12)
        return memo[(k, th)]

    def pooled_f1(mids, th):
        tp = fp = fn = 0
        for k in mids:
            a, b, c = counts(k, th)
            tp, fp, fn = tp + a, fp + b, fn + c
        return prf(tp, fp, fn)["f1"]

    rows, tot = [], {name: {tol: [0, 0, 0] for tol in TOLS} for name in ("model", "kink", "control")}
    for mid in ids:
        others = [k for k in ids if k != mid]
        th = float(max(GRID, key=lambda g: pooled_f1(others, g)))
        m = data[mid]
        res = {"model": score(m, M.events(oof[mid][0], oof[mid][1], m["live"], th)),
               "kink": score(m, kink_events(m)), "control": score(m, control_events(m))}
        row = {"match": mid, "flips": len(m["flips"]), "threshold": th}
        for name, r in res.items():
            for tol in TOLS:
                tot[name][tol] = [a + b for a, b in zip(tot[name][tol], r[tol])]
            row[name] = {"tol%d" % tol: prf(*r[tol]) for tol in TOLS}
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = {"matches": len(ids), "flips": sum(len(d["flips"]) for d in data.values()),
           "live_min": round(sum(d["live"].sum() for d in data.values()) / M.FPS / 60, 1),
           "pooled": {name: {"tol%d" % tol: prf(*tot[name][tol]) for tol in TOLS} for name in tot},
           "per_match": rows}
    X, y = stack(list(data.values()))
    final = model().fit(X, y)
    th_all = float(max(GRID, key=lambda g: pooled_f1(ids, g)))
    os.makedirs(dfl.CACHE, exist_ok=True)
    with open(MODEL, "wb") as fh:
        pickle.dump({"model": final, "threshold": th_all, "features": M.FEATURES}, fh)
    out["final_threshold"] = th_all
    out["degraded"] = DEG is not None
    out["oof_events_per_live_min"] = round(sum(len(M.events(oof[k][0], oof[k][1], data[k]["live"], th_all))
                                               for k in ids) / out["live_min"], 2)
    out["flips_per_live_min"] = round(out["flips"] / out["live_min"], 2)
    out["atd_transfer"] = atd_transfer(final, th_all)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("matches", "flips", "live_min", "pooled", "final_threshold", "degraded",
                                          "flips_per_live_min", "oof_events_per_live_min", "atd_transfer")}, indent=1))


if __name__ == "__main__":
    main()
