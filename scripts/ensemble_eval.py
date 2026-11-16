"""Ensembles of possession models (A = DFL only, C = DFL + Metrica). Aggregates only.

    python scripts/ensemble_eval.py

Pre-registered 2026-11-15, before running:
  Data, truth and touch probabilities as in decoder_tune / metrica_train_eval (DFL controlled labels
  absorb 3 s; Metrica event labels, amended; Metrica smoothed).
  Members: single A and single C (HistGradientBoosting, seed 0, all rows, as before), and 3 bagged
  copies of each (80% row subsample, max_features 0.8, seeds 1-3).
  Variants (probabilities averaged): A, C (singles), E_AC (A + C), E_A3, E_C3 (the bags), E_AC6.
  Decoder D1 (8, 3). Each test match is predicted only by members that never saw it: a Metrica game
  by A (all DFL) / C (DFL + the other Metrica); a DFL match by A (other 6 DFL) / C (other 6 DFL + all
  Metrica).
  Objective: 0.5 * mean DFL spellF1 + 0.5 * mean Metrica spellF1.
  Go rule: an ensemble replaces the better single (A or C by the objective) if its objective is
  >= that single's + 0.01 and neither source is below the single's by more than 0.005. The winner's
  members are refitted on all data and saved as cache/dfl/poss_model_<variant>.pkl (a list of
  models; poss_model.apply_bar averages them).
Writes reports/ensemble_eval_summary.json.
"""

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

import metrica_train_eval as mt  # noqa: E402
import poss_model_eval as pe  # noqa: E402
from barca import dfl, loaders, metrica, poss_model as P  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "ensemble_eval_summary.json")
BAGS = (1, 2, 3)
FPS = 25


def model(seed):
    if seed == 0:
        return pe.hgb()
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, validation_fraction=0.1,
                                          random_state=seed, max_features=0.8)


def fit(items, weights, seed):
    X = np.concatenate([it["F"][it["live"]] for it in items])
    y = np.concatenate([it["y"][it["live"]] for it in items])
    w = np.concatenate([np.full(int(it["live"].sum()), wt) for it, wt in zip(items, weights)])
    if seed:
        keep = np.random.default_rng(seed).random(len(y)) < 0.8
        X, y, w = X[keep], y[keep], w[keep]
    return model(seed).fit(X, y, sample_weight=w)


def members(dfl_tr, met_tr):
    """{name: fitted model} for one test match: A0, C0 and the bags A1-3, C1-3."""
    out = {}
    a_w = [1.0] * len(dfl_tr)
    c_items = dfl_tr + met_tr
    c_w = mt.weights_for(dfl_tr, met_tr)
    for s in (0,) + BAGS:
        out["A%d" % s] = fit(dfl_tr, a_w, s)
        out["C%d" % s] = fit(c_items, c_w, s)
    return out


VARIANTS = {"A": ["A0"], "C": ["C0"], "E_AC": ["A0", "C0"], "E_A3": ["A1", "A2", "A3"],
            "E_C3": ["C1", "C2", "C3"], "E_AC6": ["A1", "A2", "A3", "C1", "C2", "C3"]}


def score(it, probs):
    out = {}
    runs = pm.valid_runs(it["gt"])
    for v, names in VARIANTS.items():
        q = np.mean([probs[nm] for nm in names], axis=0)
        pred = P.absorb_short(P.decode(q, it["live"], P.D1[0]), P.D1[1])
        out[v] = {"spellF1": round(pm.spell_scores(it["gt"], pred, runs)["mean_f1"], 4),
                  "acc": round(float((pred[it["live"]] == it["gt"][it["live"]]).mean()), 4)}
    return out


def predict(models, it):
    probs = {}
    for nm, m in models.items():
        q = np.zeros(len(it["gt"]))
        q[it["live"]] = m.predict_proba(it["F"][it["live"]])[:, 1]
        probs[nm] = q
    return probs


def main():
    t0 = time.time()
    gap = pe.gap_profile()
    ids = dfl.match_ids()
    data = {mid: pe.prepare(mid, gap) for mid in ids}
    p_touch = {}
    for mid in ids:
        clf = pe.hgb().fit(*pe.stack_touch([data[k] for k in ids if k != mid]))
        p_touch[mid] = P.touch_probs(clf, data[mid]["Ft"], data[mid]["live"])
    touch_final = pe.hgb().fit(*pe.stack_touch(list(data.values())))
    D = {}
    for mid in ids:
        gt = dfl.controlled_labels(data[mid]["d"]["labels"], data[mid]["d"]["blocks"], 3.0)
        D[mid] = {"F": pe.poss_features(data[mid], p_touch[mid]), "y": gt == 0, "gt": gt, "live": data[mid]["live"]}
    M = {g: mt.metrica_item(g, touch_final) for g in metrica.GAMES}
    print("prepared %.0f s" % (time.time() - t0), flush=True)

    res = {}
    for g in metrica.GAMES:
        ms = members(list(D.values()), [M[k] for k in metrica.GAMES if k != g])
        res[("met", g)] = score(M[g], predict(ms, M[g]))
        print("metrica", g, json.dumps({v: r["spellF1"] for v, r in res[("met", g)].items()}), "%.0f s" % (time.time() - t0), flush=True)
    for mid in ids:
        ms = members([D[k] for k in ids if k != mid], list(M.values()))
        res[("dfl", mid)] = score(D[mid], predict(ms, D[mid]))
        print("dfl", mid, json.dumps({v: r["spellF1"] for v, r in res[("dfl", mid)].items()}), "%.0f s" % (time.time() - t0), flush=True)

    def mean(src, v):
        return round(float(np.mean([r[v]["spellF1"] for k, r in res.items() if k[0] == src])), 4)
    summary = {v: {"dfl": mean("dfl", v), "met": mean("met", v)} for v in VARIANTS}
    for v in summary:
        summary[v]["objective"] = round(0.5 * summary[v]["dfl"] + 0.5 * summary[v]["met"], 4)
    best_single = max(("A", "C"), key=lambda v: summary[v]["objective"])
    b = summary[best_single]
    passing = [v for v in VARIANTS if v.startswith("E_") and summary[v]["objective"] >= b["objective"] + 0.01
               and summary[v]["dfl"] >= b["dfl"] - 0.005 and summary[v]["met"] >= b["met"] - 0.005]
    winner = max(passing, key=lambda v: summary[v]["objective"], default=None)
    out = {"summary": summary, "best_single": best_single, "passing": passing, "winner": winner,
           "detail": {"%s_%s" % k: v for k, v in res.items()}}
    print(json.dumps({k: out[k] for k in ("summary", "best_single", "passing", "winner")}, indent=1), flush=True)
    if winner:
        ms = members(list(D.values()), list(M.values()))
        path = os.path.join(dfl.CACHE, "poss_model_%s.pkl" % winner)
        with open(path, "wb") as fh:
            pickle.dump({"touch": touch_final, "poss": [ms[nm] for nm in VARIANTS[winner]], "cost": 64,
                         "touch_th": pe.TOUCH_TH, "absorb": 3.0, "training": winner}, fh)
        out["saved"] = path
        print("saved", path)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
