"""Tune the possession decoder (switch cost, minimum hand-back spell) on DFL + Metrica. Aggregates only.

    python scripts/decoder_tune.py          (first run computes and caches the out-of-fold probabilities)
    python scripts/decoder_tune.py --context 1 --dfl-absorbs 2 3 5
        # 2026-11-15 amendment, fixed before running: the DFL truth was built by merging hand-backs
        # < 3 s, which favours a 3 s minimum by construction. With --dfl-absorbs the DFL score is
        # the mean spellF1 over truths built with 2, 3 and 5 s. Go: honest objective >= D1 + 0.01,
        # Metrica not below D1's, DFL (absorb mean) not below D1's by more than 0.005.
        # Report: decoder_tune_ctx1_abs2-3-5_summary.json. Re-scores the cache; no retraining.
    python scripts/decoder_tune.py --context 1   # 2026-11-15 re-tune for the context model (C_ctx), same
                                                 #   grid and go rule; cache poss_oof_C_ctx1.npz

Pre-registered 2026-11-14, before running:
  Probabilities: the C-style model (DFL + Metrica, Metrica weighted to DFL's total) out of fold
  for all 10 matches: a DFL match from 6 DFL + 3 Metrica, a Metrica game from 7 DFL + 2 Metrica.
  Truth: DFL controlled labels (absorb 3 s); Metrica event labels. Cached: cache/dfl/poss_oof_C.npz.
  Decoder: poss_model.absorb_short(poss_model.decode(p, live, cost), min_s) over cost in
  {2, 4, 6, 8, 12, 16, 24, 32} x min_s in {0, 1, 2, 3, 4, 5}.
  Objective: 0.5 * mean DFL spellF1 + 0.5 * mean Metrica spellF1.
  Honest estimate: each held-out match decoded with the setting best on the other nine.
  Go rule: the tuned setting (best on all 10) replaces D1 (8, 3) if the honest objective gains
  >= 0.01 over D1's and neither source loses more than 0.005. Then make_hybrid --poss-decoder d1
  --d1-cost C --d1-min M; the public check only guards against breakage (reject if spellF1 falls
  more than 0.04 below the current 0.6533).
Writes reports/decoder_tune_summary.json.
"""

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import metrica_train_eval as mt  # noqa: E402
import poss_model_eval as pe  # noqa: E402
from barca import dfl, loaders, metrica, poss_model as P  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "decoder_tune_summary.json")
CACHE = os.path.join(dfl.CACHE, "poss_oof_C.npz")
COSTS = (2, 4, 6, 8, 12, 16, 24, 32)
MINS = (0, 1, 2, 3, 4, 5)
D1 = tuple(P.D1)
FPS = 25


CONTEXT = 0
DFL_ABSORBS = None


def build_oof():
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
        gt = dfl.controlled_labels(data[mid]["d"]["labels"], data[mid]["d"]["blocks"], 3.0)
        if CONTEXT:
            import ctx_eval as ce
            ce.LEVELS = (CONTEXT, CONTEXT)            # metrica_item unpacks two feature sets
            m, dd = data[mid], data[mid]["d"]
            F, _ = ce.feats(dd["players"], dd["ball"], m["st"], dd["pitch"], dd["blocks"], m["live"], p_touch[mid], m["ctrl"])
        else:
            F = pe.poss_features(data[mid], p_touch[mid])
        items[("dfl", mid)] = {"F": F, "y": gt == 0, "gt": gt, "live": data[mid]["live"]}
    for g in metrica.GAMES:
        if CONTEXT:
            it = ce.metrica_item(g, touch_final)
            items[("met", str(g))] = {"F": it["Fb"], "y": it["y"], "gt": it["gt"], "live": it["live"]}
        else:
            items[("met", str(g))] = mt.metrica_item(g, touch_final)
    print("prepared %.0f s" % (time.time() - t0), flush=True)
    keys = list(items)
    saved = {}
    for i, k in enumerate(keys):
        tr = [items[j] for j in keys if j != k]
        d_tr = [items[j] for j in keys if j != k and j[0] == "dfl"]
        m_tr = [items[j] for j in keys if j != k and j[0] == "met"]
        clf = mt.fit(d_tr + m_tr, mt.weights_for(d_tr, m_tr))
        q = np.zeros(len(items[k]["gt"]))
        q[items[k]["live"]] = clf.predict_proba(items[k]["F"][items[k]["live"]])[:, 1]
        saved["src_%d" % i], saved["name_%d" % i] = k[0], k[1]
        saved["p_%d" % i], saved["gt_%d" % i], saved["live_%d" % i] = q.astype(np.float32), items[k]["gt"], items[k]["live"]
        print("out of fold", k, "%.0f s" % (time.time() - t0), flush=True)
    saved["n"] = len(keys)
    np.savez_compressed(CACHE, **saved)


def main():
    global CONTEXT, CACHE, OUT
    if "--context" in sys.argv[1:]:
        CONTEXT = int(sys.argv[sys.argv.index("--context") + 1])
        CACHE = CACHE.replace("poss_oof_C.npz", "poss_oof_C_ctx%d.npz" % CONTEXT)
        OUT = OUT.replace("decoder_tune_summary", "decoder_tune_ctx%d_summary" % CONTEXT)
    global DFL_ABSORBS
    if "--dfl-absorbs" in sys.argv[1:]:
        i = sys.argv.index("--dfl-absorbs")
        DFL_ABSORBS = [float(x) for x in sys.argv[i + 1:] if not x.startswith("--")]
        OUT = OUT.replace("_summary.json", "_abs%s_summary.json" % "-".join("%g" % a for a in DFL_ABSORBS))
    if not os.path.isfile(CACHE):
        build_oof()
    z = np.load(CACHE)
    n = int(z["n"])
    src = [str(z["src_%d" % i]) for i in range(n)]
    t0 = time.time()
    res = {}
    per_absorb = {}
    for i in range(n):
        p, gt, live = z["p_%d" % i], z["gt_%d" % i], z["live_%d" % i]
        runs = pm.valid_runs(gt)
        gts = [gt]
        if DFL_ABSORBS and src[i] == "dfl":
            d = dfl.load(str(z["name_%d" % i]))
            gts = [dfl.controlled_labels(d["labels"], d["blocks"], a) for a in DFL_ABSORBS]
        for c in COSTS:
            base = P.decode(p, live, c)
            for m in MINS:
                pred = P.absorb_short(base, m)
                f1s = [pm.spell_scores(g, pred, pm.valid_runs(g))["mean_f1"] for g in gts]
                if len(gts) > 1:
                    per_absorb[(i, c, m)] = f1s
                res[(i, c, m)] = (float(np.mean(f1s)),
                                  len(pm.possession_spells(pred, runs)) / (live.sum() / FPS / 60))
        res[(i, "truth")] = len(pm.possession_spells(gt, runs)) / (live.sum() / FPS / 60)
    print("grid scored %.0f s" % (time.time() - t0), flush=True)
    grid = list(itertools.product(COSTS, MINS))

    def objective(setting, rows):
        d = [res[(i,) + setting][0] for i in rows if src[i] == "dfl"]
        m = [res[(i,) + setting][0] for i in rows if src[i] == "met"]
        return 0.5 * np.mean(d) + 0.5 * np.mean(m)

    honest = {}
    for i in range(n):
        others = [k for k in range(n) if k != i]
        best = max(grid, key=lambda s: objective(s, others))
        honest[i] = (best, res[(i,) + best][0], res[(i,) + D1][0])

    def by_src(idx):
        return {s: round(float(np.mean([honest[i][idx] for i in range(n) if src[i] == s])), 4) for s in ("dfl", "met")}
    tuned_src, d1_src = by_src(1), by_src(2)
    tuned_obj = 0.5 * tuned_src["dfl"] + 0.5 * tuned_src["met"]
    d1_obj = 0.5 * d1_src["dfl"] + 0.5 * d1_src["met"]
    best_all = max(grid, key=lambda s: objective(s, range(n)))
    go = (tuned_obj - d1_obj >= 0.01 and all(tuned_src[s] >= d1_src[s] - 0.005 for s in ("dfl", "met")))
    if DFL_ABSORBS:   # amended rule: Metrica must not be lower at all
        go = go and tuned_src["met"] >= d1_src["met"]
    top = sorted(grid, key=lambda s: -objective(s, range(n)))[:8]
    out = {"honest": {"tuned": tuned_src | {"objective": round(tuned_obj, 4)},
                      "D1_8_3": d1_src | {"objective": round(d1_obj, 4)},
                      "chosen_per_match": [list(honest[i][0]) for i in range(n)]},
           "best_on_all_10": list(best_all), "go": bool(go),
           "top8_on_all_10": [{"cost": c, "min_s": m, "objective": round(objective((c, m), range(n)), 4),
                               "poss_per_min_dfl": round(float(np.mean([res[(i, c, m)][1] for i in range(n) if src[i] == "dfl"])), 2),
                               "poss_per_min_met": round(float(np.mean([res[(i, c, m)][1] for i in range(n) if src[i] == "met"])), 2)}
                              for c, m in top],
           "D1_on_all_10": {"objective": round(objective(D1, range(n)), 4)},
           "truth_poss_per_min": {s: round(float(np.mean([res[(i, "truth")] for i in range(n) if src[i] == s])), 2)
                                  for s in ("dfl", "met")}}
    if per_absorb:
        dfl_idx = [i for i in range(n) if src[i] == "dfl"]
        out["dfl_by_absorb"] = {"%g" % a: {"D1": round(float(np.mean([per_absorb[(i,) + D1][k] for i in dfl_idx])), 4),
                                           "best_on_all_10": round(float(np.mean([per_absorb[(i,) + tuple(best_all)][k] for i in dfl_idx])), 4)}
                                for k, a in enumerate(DFL_ABSORBS)}
    print(json.dumps(out, indent=1))
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
