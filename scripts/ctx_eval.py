"""Multi-scale context features for the possession model (C training set). Aggregates only.

    python scripts/ctx_eval.py                      # level 0 (25 features) vs level 1 (47)
    python scripts/ctx_eval.py --levels 1 2         # round 2 (2026-11-15): 47 vs 66 features
                                                    #   (poss_model.CTX2_FEATURES, same go rule)

Pre-registered 2026-11-15, before running:
  Training set C (DFL + Metrica, Metrica weighted to DFL's total), single HistGradientBoosting
  (seed 0, sample weights given for both variants alike), decoder D1 (8, 3). Data, truth and touch
  probabilities as in ensemble_eval.
    base  poss_model.FEATURES (25)
    ctx   + poss_model.CTX_FEATURES (22): touch-probability maxima, distance margin and nearer-team
          share over +-0.5 / 1 / 2 / 4 s; ball velocity along the attack axis (now, 1 s); players
          of each team ahead of the ball; 1 s trend of each team's distance to the ball
  Tests: Metrica games held out (trained on all DFL + the other Metrica games); DFL leave one match
  out (other 6 DFL + all Metrica).
  Objective 0.5 * mean DFL spellF1 + 0.5 * mean Metrica spellF1.
  Go rule (raised for run-to-run noise ~0.02-0.03 per game): ctx replaces base if its objective
  >= base + 0.015 AND it is better on >= 2 of 3 Metrica games AND >= 4 of 7 DFL matches. Then
  refitted on all data and saved as cache/dfl/poss_model_C_ctx.pkl (context=True).
Writes reports/ctx_eval_summary.json.
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

import metrica_eval as me  # noqa: E402
import metrica_train_eval as mt  # noqa: E402
import poss_model_eval as pe  # noqa: E402
from barca import dfl, loaders, metrica, nearest, possession, poss_model as P, touch_model as TM  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "ctx_eval_summary.json")
FPS = 25


LEVELS = (0, 1)


def feats(players, ball, st, pitch, blocks, live, pt, ctrl):
    ev = P.touch_event_list(pt, live, pe.TOUCH_TH)
    return tuple(P.features(players, ball, st, pitch, blocks, live, pt, ev, ctrl, context=lv) for lv in LEVELS)


def metrica_item(g, touch_final):
    d = me.smooth(metrica.load(g))
    gt = d["labels"]
    live = gt != 2
    L, W = d["pitch"]
    st = d["slot_team"].astype(int)
    _, team, dist, margin = nearest.nearest_team(
        (d["ball"][:, :2].astype(np.float64) * [L, W]).astype(np.float32),
        d["players"][..., 0] * np.float32(L), d["players"][..., 1] * np.float32(W), st)
    ctrl = possession.control_events(team, dist, margin, pe.ball_speed(d["ball"], d["pitch"]))
    pt = P.touch_probs(touch_final, TM.features(d["players"], d["ball"], st, d["pitch"]), live)
    Fb, Fc = feats(d["players"], d["ball"], st, d["pitch"], d["blocks"], live, pt, ctrl)
    return {"Fb": Fb, "Fc": Fc, "y": gt == 0, "gt": gt, "live": live, "src": "met"}


def fit(items, key):
    X = np.concatenate([it[key][it["live"]] for it in items])
    y = np.concatenate([it["y"][it["live"]] for it in items])
    nd = sum(int(it["live"].sum()) for it in items if it["src"] == "dfl")
    nm = sum(int(it["live"].sum()) for it in items if it["src"] == "met")
    w = np.concatenate([np.full(int(it["live"].sum()), 1.0 if it["src"] == "dfl" or not nm else nd / nm) for it in items])
    return pe.hgb().fit(X, y, sample_weight=w)


def score(clf, it, key):
    q = np.zeros(len(it["gt"]))
    q[it["live"]] = clf.predict_proba(it[key][it["live"]])[:, 1]
    pred = P.absorb_short(P.decode(q, it["live"], P.D1[0]), P.D1[1])
    runs = pm.valid_runs(it["gt"])
    return {"spellF1": round(pm.spell_scores(it["gt"], pred, runs)["mean_f1"], 4),
            "acc": round(float((pred[it["live"]] == it["gt"][it["live"]]).mean()), 4),
            "poss_per_min": round(len(pm.possession_spells(pred, runs)) / (it["live"].sum() / FPS / 60), 2)}


def main():
    global LEVELS, OUT
    if "--levels" in sys.argv[1:]:
        i = sys.argv.index("--levels")
        LEVELS = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
        OUT = OUT.replace("ctx_eval_summary", "ctx_eval_%d_%d_summary" % LEVELS)
    t0 = time.time()
    gap = pe.gap_profile()
    ids = dfl.match_ids()
    data = {mid: pe.prepare(mid, gap) for mid in ids}
    p_touch = {}
    for mid in ids:
        clf = pe.hgb().fit(*pe.stack_touch([data[k] for k in ids if k != mid]))
        p_touch[mid] = P.touch_probs(clf, data[mid]["Ft"], data[mid]["live"])
    touch_final = pe.hgb().fit(*pe.stack_touch(list(data.values())))
    items = {}
    for mid in ids:
        m = data[mid]
        dd = m["d"]
        gt = dfl.controlled_labels(dd["labels"], dd["blocks"], 3.0)
        Fb, Fc = feats(dd["players"], dd["ball"], m["st"], dd["pitch"], dd["blocks"], m["live"], p_touch[mid], m["ctrl"])
        items[("dfl", mid)] = {"Fb": Fb, "Fc": Fc, "y": gt == 0, "gt": gt, "live": m["live"], "src": "dfl"}
    for g in metrica.GAMES:
        items[("met", str(g))] = metrica_item(g, touch_final)
    print("prepared %.0f s" % (time.time() - t0), flush=True)

    keys = list(items)
    res = {}
    for k in keys:
        tr = [items[j] for j in keys if j != k]
        res[k] = {"base": score(fit(tr, "Fb"), items[k], "Fb"), "ctx": score(fit(tr, "Fc"), items[k], "Fc")}
        print(k, json.dumps(res[k]), "%.0f s" % (time.time() - t0), flush=True)

    def mean(src, v):
        return round(float(np.mean([r[v]["spellF1"] for kk, r in res.items() if kk[0] == src])), 4)
    summary = {v: {"dfl": mean("dfl", v), "met": mean("met", v)} for v in ("base", "ctx")}
    for v in summary:
        summary[v]["objective"] = round(0.5 * summary[v]["dfl"] + 0.5 * summary[v]["met"], 4)
    wins = {src: int(sum(r["ctx"]["spellF1"] > r["base"]["spellF1"] for kk, r in res.items() if kk[0] == src))
            for src in ("met", "dfl")}
    go = (summary["ctx"]["objective"] >= summary["base"]["objective"] + 0.015 and wins["met"] >= 2 and wins["dfl"] >= 4)
    out = {"summary": summary, "ctx_wins": wins, "go": bool(go), "detail": {"%s_%s" % k: v for k, v in res.items()}}
    print(json.dumps({k: out[k] for k in ("summary", "ctx_wins", "go")}, indent=1), flush=True)
    if go:
        final = fit(list(items.values()), "Fc")
        path = os.path.join(dfl.CACHE, "poss_model_C_ctx.pkl" if LEVELS[1] == 1 else "poss_model_C_ctx%d.pkl" % LEVELS[1])
        with open(path, "wb") as fh:
            pickle.dump({"touch": touch_final, "poss": final, "cost": 64, "touch_th": pe.TOUCH_TH,
                         "absorb": 3.0, "training": "C", "context": True if LEVELS[1] == 1 else LEVELS[1]}, fh)
        out["saved"] = path
        print("saved", path)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
