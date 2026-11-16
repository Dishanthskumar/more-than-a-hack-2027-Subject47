"""Context features for the touch detector, judged through the possession model. Aggregates only.

    python scripts/touchctx_eval.py

Pre-registered 2026-11-16, before running:
  Touch detector: touch_model.FEATURES (15) vs + CTX_FEATURES (16: over +-6 / 12 / 25 frames the max
  velocity change and turn, min home / away distance, min relative speed; and the velocity change
  over its 2 s mean). Trained on degraded DFL as before (out of fold for DFL, all 7 for Metrica).
  Touch F1 reported (events at 0.1, matched within 12 frames, out of fold).
  Possession model: context level 1 (47 features), training set C, decoder D1 (8, 3), Metrica
  games held out and DFL leave one match out, exactly as ctx_eval; only the touch probabilities
  (and the touch events built from them) differ between the two arms.
  Go rule: objective (0.5 DFL + 0.5 Metrica) >= base + 0.015 AND better on >= 2 of 3 Metrica games
  AND >= 4 of 7 DFL matches. Then refitted on all data and saved as
  cache/dfl/poss_model_C_ctx_tctx.pkl (touch_context=True, context=1).
Writes reports/touchctx_eval_summary.json.
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

import ctx_eval as ce  # noqa: E402
import metrica_eval as me  # noqa: E402
import poss_model_eval as pe  # noqa: E402
from barca import dfl, loaders, metrica, nearest, possession, poss_model as P, touch_model as TM  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "touchctx_eval_summary.json")
FPS = 25


def stack(ms, key):
    X = np.concatenate([m[key][t][m["live"]] for m in ms for t in (0, 1)])
    y = np.concatenate([m["yt"][t][m["live"]] for m in ms for t in (0, 1)])
    return X, y


def poss_F(players, ball, st, pitch, blocks, live, pt, ctrl):
    ev = P.touch_event_list(pt, live, pe.TOUCH_TH)
    return P.features(players, ball, st, pitch, blocks, live, pt, ev, ctrl, context=1), ev


def main():
    t0 = time.time()
    gap = pe.gap_profile()
    ids = dfl.match_ids()
    data = {mid: pe.prepare(mid, gap) for mid in ids}
    for mid in ids:
        dd = data[mid]["d"]
        data[mid]["Ftc"] = TM.features(dd["players"], dd["ball"], data[mid]["st"], dd["pitch"], context=True)
        data[mid]["flips"] = TM.flips(dd["labels"], dd["blocks"])
    arms = {"base": "Ft", "tctx": "Ftc"}
    touch_oof, touch_final, touch_f1 = {a: {} for a in arms}, {}, {}
    for a, key in arms.items():
        tp = fp = fn = 0
        for mid in ids:
            clf = pe.hgb().fit(*stack([data[k] for k in ids if k != mid], key))
            touch_oof[a][mid] = P.touch_probs(clf, data[mid][key], data[mid]["live"])
            ev = TM.events(touch_oof[a][mid][0], touch_oof[a][mid][1], data[mid]["live"], pe.TOUCH_TH)
            x, y_, z = TM.match_events(data[mid]["flips"], ev, 12)
            tp, fp, fn = tp + x, fp + y_, fn + z
        touch_final[a] = pe.hgb().fit(*stack(list(data.values()), key))
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        touch_f1[a] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3)}
        print("touch", a, json.dumps(touch_f1[a]), "%.0f s" % (time.time() - t0), flush=True)

    items = {a: {} for a in arms}
    for mid in ids:
        m, dd = data[mid], data[mid]["d"]
        gt = dfl.controlled_labels(dd["labels"], dd["blocks"], 3.0)
        for a in arms:
            F, _ = poss_F(dd["players"], dd["ball"], m["st"], dd["pitch"], dd["blocks"], m["live"], touch_oof[a][mid], m["ctrl"])
            items[a][("dfl", mid)] = {"Fb": F, "y": gt == 0, "gt": gt, "live": m["live"], "src": "dfl"}
    for g in metrica.GAMES:
        d = me.smooth(metrica.load(g))
        gt = d["labels"]
        live = gt != 2
        L, W = d["pitch"]
        st = d["slot_team"].astype(int)
        _, team, dist, margin = nearest.nearest_team(
            (d["ball"][:, :2].astype(np.float64) * [L, W]).astype(np.float32),
            d["players"][..., 0] * np.float32(L), d["players"][..., 1] * np.float32(W), st)
        ctrl = possession.control_events(team, dist, margin, pe.ball_speed(d["ball"], d["pitch"]))
        for a in arms:
            Ft = TM.features(d["players"], d["ball"], st, d["pitch"], context=(a == "tctx"))
            pt = P.touch_probs(touch_final[a], Ft, live)
            F, _ = poss_F(d["players"], d["ball"], st, d["pitch"], d["blocks"], live, pt, ctrl)
            items[a][("met", str(g))] = {"Fb": F, "y": gt == 0, "gt": gt, "live": live, "src": "met"}
    print("prepared possession features %.0f s" % (time.time() - t0), flush=True)

    keys = list(items["base"])
    res = {}
    for k in keys:
        res[k] = {}
        for a in arms:
            tr = [items[a][j] for j in keys if j != k]
            res[k][a] = ce.score(ce.fit(tr, "Fb"), items[a][k], "Fb")
        print(k, json.dumps(res[k]), "%.0f s" % (time.time() - t0), flush=True)

    def mean(src, a):
        return round(float(np.mean([r[a]["spellF1"] for kk, r in res.items() if kk[0] == src])), 4)
    summary = {a: {"dfl": mean("dfl", a), "met": mean("met", a)} for a in arms}
    for a in summary:
        summary[a]["objective"] = round(0.5 * summary[a]["dfl"] + 0.5 * summary[a]["met"], 4)
    wins = {src: int(sum(r["tctx"]["spellF1"] > r["base"]["spellF1"] for kk, r in res.items() if kk[0] == src))
            for src in ("met", "dfl")}
    go = summary["tctx"]["objective"] >= summary["base"]["objective"] + 0.015 and wins["met"] >= 2 and wins["dfl"] >= 4
    out = {"touch_f1_out_of_fold": touch_f1, "summary": summary, "tctx_wins": wins, "go": bool(go),
           "detail": {"%s_%s" % k: v for k, v in res.items()}}
    print(json.dumps({k: out[k] for k in ("touch_f1_out_of_fold", "summary", "tctx_wins", "go")}, indent=1), flush=True)
    if go:
        final = ce.fit(list(items["tctx"].values()), "Fb")
        path = os.path.join(dfl.CACHE, "poss_model_C_ctx_tctx.pkl")
        with open(path, "wb") as fh:
            pickle.dump({"touch": touch_final["tctx"], "touch_context": True, "poss": final, "cost": 64,
                         "touch_th": pe.TOUCH_TH, "absorb": 3.0, "training": "C", "context": True}, fh)
        out["saved"] = path
        print("saved", path)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
