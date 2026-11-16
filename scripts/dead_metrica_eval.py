"""Dead-ball model with Metrica: players-only vs ball + players, on held-out Metrica and the DFL fair bench.

    python scripts/dead_metrica_eval.py

Pre-registered 2026-11-14, before running:
  Metrica games 1-3, smoothed (metrica_eval.smooth); labels from metrica.labels_from_events
  (amended: ball out at the BALL OUT end frame, goal kicks as restarts, unexplained restarts dead
  from the previous event's end); stand-in tags metrica.set_piece_tags (BAR tagging rates).
  DFL: the fair bench (stand-in tags, simulated dead-time ball, degraded), as dead_model_eval --fair.
  Rules R0: B2's dead layer minus broadcast (set_pieces.detect with tags, control live term).
  Models (HistGradientBoosting as before), evidence w = 4 (the value used so far):
    P   players-only, trained on DFL only                       (the deployed kind)
    PM  players-only, DFL + Metrica (Metrica weighted to DFL's total)
    BM  ball + players (dead_model.FULL_FEATURES), DFL + Metrica; ball features NaN on DFL rows
  Each test match is predicted by a model that never saw it. Variants per model:
    E   extend-only (dead_model.extend_only: R0 + model spells overlapping R0's)
    F   full R0 + w * logit(p) (may add, shorten or remove stoppages)
  Metrics: alive/dead F1, dead F1, false / missed stoppages, boundary errors, spellF1 with perfect
  team labels (dead-mask effect only).
  Go rule: a variant of PM or BM replaces P-E if its mean held-out Metrica alive/dead F1 >= P-E's
  + 0.01, its Metrica false stoppages <= P-E's, and its DFL mean alive/dead F1 >= P-E's - 0.01.
  Winner = the best Metrica alive/dead F1 among those passing; it is refitted on all data and saved
  as cache/dfl/dead_<variant>_model.pkl.
Writes reports/dead_metrica_eval_summary.json.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import dead_model_eval as dme  # noqa: E402
import metrica_eval as me  # noqa: E402
from barca import dead_ball as db, dead_model as DM, degrade, dfl, loaders, metrica, set_pieces  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "dead_metrica_eval_summary.json")
W = 4.0
BALL_IDX = [DM.FULL_FEATURES.index(f) for f in DM.BALL_FEATURES]


def item(d, tags, team_labels, source):
    tr = DM.tr_from(d)
    n = len(d["labels"])
    c = db.cues(tr, {"h02": np.full(n, np.nan), "h12": np.full(n, np.nan)}, tags)
    st = d["slot_team"].astype(int)
    Fp = DM.features(d["players"], d["ball"], d["pitch"], st, "players")
    Ff = DM.features(d["players"], d["ball"], d["pitch"], st, "full")
    if source == "dfl":
        Ff[:, BALL_IDX] = np.nan
    team = team_labels.astype(float)
    team[team_labels == 2] = np.nan
    team = np.asarray(pd.Series(team).ffill().bfill(), dtype=np.int8)
    return {"tr": tr, "c": c, "tags": tags, "gt": d["labels"] == 2, "lab": team_labels, "team": team,
            "Fp": Fp, "Ff": Ff, "extra": dme.ctrl_extra(tr, c, d["pitch"]), "src": source}


def fit(items, kind):
    X = np.concatenate([it["F" + kind] for it in items])
    y = np.concatenate([it["gt"] for it in items])
    nd = sum(len(it["gt"]) for it in items if it["src"] == "dfl")
    nm = sum(len(it["gt"]) for it in items if it["src"] == "met")
    w = np.concatenate([np.full(len(it["gt"]), 1.0 if it["src"] == "dfl" or not nm else nd / nm) for it in items])
    return dme.model().fit(X, y, sample_weight=w)


def rules(it, add=None):
    e = it["extra"] if add is None else it["extra"] + add
    return set_pieces.detect(it["c"], it["tr"], it["tags"], onset=2, extra=e)[0]


def evaluate(it, mask):
    m = dme.metrics(it["gt"], mask)
    m["spellF1_perfect_teams"] = round(pm.spell_scores(it["lab"], np.where(mask, 2, it["team"]).astype(np.int8),
                                                       pm.valid_runs(it["lab"]))["mean_f1"], 4)
    return m


def main():
    t0 = time.time()
    gap = dme.atd_gap()
    items = {}
    for mid in dfl.match_ids():
        d = dict(dfl.load(mid))
        tags = dfl.set_piece_tags(mid, d["blocks"], seed=0)
        d = degrade.degrade(degrade.simulate_dead_ball(d, seed=0), gap[0], gap[1], seed=0)
        items[("dfl", mid)] = item(d, tags, dfl.controlled_labels(d["labels"], d["blocks"], 3.0), "dfl")
    for g in metrica.GAMES:
        d = me.smooth(metrica.load(g))
        tags = metrica.set_piece_tags(g, len(d["labels"]), d["blocks"], seed=0)
        items[("met", str(g))] = item(d, tags, d["labels"], "met")
    print("prepared %.0f s" % (time.time() - t0), flush=True)

    keys = list(items)
    res = {}
    for k in keys:
        it = items[k]
        others = [items[j] for j in keys if j != k]
        dfl_others = [o for o in others if o["src"] == "dfl"]
        models = {"P": (fit(dfl_others, "p"), "p"), "PM": (fit(others, "p"), "p"), "BM": (fit(others, "f"), "f")}
        r0 = rules(it)
        row = {"R0": evaluate(it, r0)}
        for name, (clf, kind) in models.items():
            p = clf.predict_proba(it["F" + kind])[:, 1]
            r1 = rules(it, W * DM.logit(p))
            row[name + "-F"] = evaluate(it, r1)
            row[name + "-E"] = evaluate(it, DM.extend_only(r1, r0))
        res[k] = row
        print(k, json.dumps({v: (row[v]["ad_f1"], row[v]["false_stops"], row[v]["missed"]) for v in row}),
              "%.0f s" % (time.time() - t0), flush=True)

    variants = list(res[keys[0]])

    def mean(src, v, key):
        vals = [res[k][v][key] for k in keys if k[0] == src and res[k][v][key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    def total(src, v, key):
        return int(sum(res[k][v][key] for k in keys if k[0] == src))

    summary = {src: {v: {"ad_f1": mean(src, v, "ad_f1"), "dead_f1": mean(src, v, "dead_f1"),
                         "spellF1_perfect_teams": mean(src, v, "spellF1_perfect_teams"),
                         "false_stops": total(src, v, "false_stops"), "missed": total(src, v, "missed"),
                         "start_err_s": mean(src, v, "start_err_s"), "end_err_s": mean(src, v, "end_err_s")}
                     for v in variants} for src in ("met", "dfl")}
    base = "P-E"
    passing = [v for v in variants if v[:2] in ("PM", "BM")
               and summary["met"][v]["ad_f1"] >= summary["met"][base]["ad_f1"] + 0.01
               and summary["met"][v]["false_stops"] <= summary["met"][base]["false_stops"]
               and summary["dfl"][v]["ad_f1"] >= summary["dfl"][base]["ad_f1"] - 0.01]
    winner = max(passing, key=lambda v: summary["met"][v]["ad_f1"], default=None)
    out = {"summary": summary, "passing": passing, "winner": winner,
           "detail": {"%s_%s" % k: v for k, v in res.items()}}
    print(json.dumps({"summary": summary, "passing": passing, "winner": winner}, indent=1), flush=True)
    if winner:
        name, mode = winner.split("-")
        kind = "f" if name == "BM" else "p"
        final = fit(list(items.values()), kind) if name != "P" else fit([v for v in items.values() if v["src"] == "dfl"], kind)
        path = os.path.join(dfl.CACHE, "dead_%s_model.pkl" % winner.replace("-", "_"))
        with open(path, "wb") as fh:
            pickle.dump({"model": final, "wc": W, "w": W, "features": "full" if kind == "f" else "players",
                         "mode": "extend" if mode == "E" else "full"}, fh)
        out["saved"] = path
        print("saved", path)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
