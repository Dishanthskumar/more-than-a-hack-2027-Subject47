"""Train the possession model with Metrica too? DFL-only vs Metrica-only vs both. Aggregates only.

    python scripts/metrica_train_eval.py
    python scripts/metrica_train_eval.py --save C     # skip the tests; fit C on all data and save it

--save C (2026-11-14, user decision): C narrowly missed the +0.02 bar (+0.0186) while beating A on
all 3 Metrica games and on DFL; the user chose to keep it. C then replaces the DFL-only model in the
main candidate only if its public spellF1 >= the DFL-only build's (same decoder).

Pre-registered 2026-11-14, before running:
  Data: DFL 7 matches as in poss_model_eval (simulated dead-time ball, degraded; truth: controlled
  labels, absorb 3 s). Metrica games 1-3, smoothed as in metrica_eval (truth: event-based labels).
  Touch probabilities: DFL out of fold (as before); Metrica from the touch model fitted on all
  7 DFL matches (Metrica has no touch labels).
  Training sets for the possession model (HistGradientBoosting as before):
    A  DFL only
    B  Metrica only
    C  DFL + Metrica, Metrica rows weighted so their total weight equals DFL's
  Decoder: D1 (switch cost 8, min hand-back spell 3 s) for all; D0 (cost 64) reported only.
  Tests: each Metrica game held out (trained on the other sets' data minus that game); and DFL
  leave one match out for A and C (C: 6 DFL + all 3 Metrica), and B (all 3 Metrica) on each DFL match.
  Go rule: B or C replaces A if its mean held-out Metrica spellF1 >= A's + 0.02 and its mean
  DFL spellF1 >= A's - 0.01 (if both pass, the higher Metrica score). The winner is then refitted
  on all its data and saved as cache/dfl/poss_model_<set>.pkl for make_hybrid --poss-model-file.
Writes reports/metrica_train_eval_summary.json.
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
import poss_model_eval as pe  # noqa: E402
from barca import dfl, loaders, metrica, nearest, possession, poss_model as P, touch_model as TM  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "metrica_train_eval_summary.json")
D1 = P.D1
D0_COST = 64
FPS = 25


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
    ev = P.touch_event_list(pt, live, pe.TOUCH_TH)
    F = P.features(d["players"], d["ball"], st, d["pitch"], d["blocks"], live, pt, ev, ctrl)
    return {"F": F, "y": gt == 0, "gt": gt, "live": live}


def fit(items, weights=None):
    X = np.concatenate([it["F"][it["live"]] for it in items])
    y = np.concatenate([it["y"][it["live"]] for it in items])
    w = None if weights is None else np.concatenate([np.full(int(it["live"].sum()), wt) for it, wt in zip(items, weights)])
    return pe.hgb().fit(X, y, sample_weight=w)


def weights_for(dfl_items, met_items):
    nd = sum(int(it["live"].sum()) for it in dfl_items)
    nm = sum(int(it["live"].sum()) for it in met_items)
    return [1.0] * len(dfl_items) + [nd / max(nm, 1)] * len(met_items)


def score(clf, it):
    q = np.zeros(len(it["gt"]))
    q[it["live"]] = clf.predict_proba(it["F"][it["live"]])[:, 1]
    runs = pm.valid_runs(it["gt"])
    out = {}
    for name, pred in (("D1", P.absorb_short(P.decode(q, it["live"], D1[0]), D1[1])),
                       ("D0", P.decode(q, it["live"], D0_COST))):
        out[name] = {"spellF1": round(pm.spell_scores(it["gt"], pred, runs)["mean_f1"], 4),
                     "acc": round(float((pred[it["live"]] == it["gt"][it["live"]]).mean()), 4),
                     "poss_per_min": round(len(pm.possession_spells(pred, runs)) / (it["live"].sum() / FPS / 60), 2)}
    return out


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
    dfl_items = {}
    for mid in ids:
        gt = dfl.controlled_labels(data[mid]["d"]["labels"], data[mid]["d"]["blocks"], 3.0)
        dfl_items[mid] = {"F": pe.poss_features(data[mid], p_touch[mid]), "y": gt == 0, "gt": gt,
                          "live": data[mid]["live"]}
    met = {g: metrica_item(g, touch_final) for g in metrica.GAMES}
    print("prepared %.0f s" % (time.time() - t0), flush=True)
    if "--save" in sys.argv[1:]:
        which = sys.argv[sys.argv.index("--save") + 1]
        items = list(dfl_items.values()) + list(met.values())
        final = fit(list(met.values())) if which == "B" else fit(items, weights_for(list(dfl_items.values()), list(met.values())))
        path = os.path.join(dfl.CACHE, "poss_model_%s.pkl" % which)
        with open(path, "wb") as fh:
            pickle.dump({"touch": touch_final, "poss": final, "cost": D0_COST, "touch_th": pe.TOUCH_TH,
                         "absorb": 3.0, "training": which}, fh)
        print("saved", path, "%.0f s" % (time.time() - t0))
        return

    res = {"metrica": {}, "dfl": {}}
    A = fit(list(dfl_items.values()))
    for g in metrica.GAMES:
        others = [met[k] for k in metrica.GAMES if k != g]
        B = fit(others)
        C = fit(list(dfl_items.values()) + others, weights_for(list(dfl_items.values()), others))
        res["metrica"]["game%d" % g] = {k: score(m, met[g]) for k, m in (("A", A), ("B", B), ("C", C))}
        print("metrica game", g, json.dumps(res["metrica"]["game%d" % g]), "%.0f s" % (time.time() - t0), flush=True)
    B_all = fit(list(met.values()))
    for mid in ids:
        tr = [dfl_items[k] for k in ids if k != mid]
        A = fit(tr)
        C = fit(tr + list(met.values()), weights_for(tr, list(met.values())))
        res["dfl"][mid] = {k: score(m, dfl_items[mid]) for k, m in (("A", A), ("B", B_all), ("C", C))}
    print("dfl checks %.0f s" % (time.time() - t0), flush=True)

    mean = {}
    for part in ("metrica", "dfl"):
        mean[part] = {k: round(float(np.mean([v[k]["D1"]["spellF1"] for v in res[part].values()])), 4) for k in "ABC"}
        mean[part + "_D0"] = {k: round(float(np.mean([v[k]["D0"]["spellF1"] for v in res[part].values()])), 4) for k in "ABC"}
        mean[part + "_poss_per_min"] = {k: round(float(np.mean([v[k]["D1"]["poss_per_min"] for v in res[part].values()])), 2) for k in "ABC"}
    passes = {k: bool(mean["metrica"][k] >= mean["metrica"]["A"] + 0.02 and mean["dfl"][k] >= mean["dfl"]["A"] - 0.01)
              for k in "BC"}
    winner = max((k for k in "BC" if passes[k]), key=lambda k: mean["metrica"][k], default="A")
    out = {"mean_spellF1": mean, "passes": passes, "winner": winner, "detail": res}
    print(json.dumps({k: out[k] for k in ("mean_spellF1", "passes", "winner")}, indent=1), flush=True)
    if winner != "A":
        items = list(dfl_items.values()) + list(met.values())
        final = fit(list(met.values())) if winner == "B" else fit(items, weights_for(list(dfl_items.values()), list(met.values())))
        path = os.path.join(dfl.CACHE, "poss_model_%s.pkl" % winner)
        with open(path, "wb") as fh:
            pickle.dump({"touch": touch_final, "poss": final, "cost": D0_COST, "touch_th": pe.TOUCH_TH,
                         "absorb": 3.0, "training": winner}, fh)
        out["saved"] = path
        print("saved", path)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
