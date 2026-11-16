"""Self-training (pseudo-labels on the target match) for the possession model. Aggregates only.

    python scripts/selftrain_eval.py            # test on the 10 labelled matches
    python scripts/selftrain_eval.py --bar      # ... and, if it passes, build the BAR-ATH version

Pre-registered 2026-11-16, before running:
  Base model: the C_ctx recipe (47 features, training set C, HistGradientBoosting seed 0), D1 decoder.
  Test: each of the 10 DFL / Metrica matches in turn is the "unlabelled" target:
    1. fit the base model on the other 9 matches; predict p(home) on the target's live frames
    2. pseudo-labels: frames with p >= 0.9 (home) or p <= 0.1 (away); nothing else is used
    3. refit with the pseudo-labelled target frames added (weight 1 per frame, like a DFL match)
    4. score both models against the target's TRUE labels (spellF1 with D1)
  Go rule: mean spellF1 gain >= 0.01 over the 10 targets AND better on >= 6 of 10.
  --bar (only if it passes): the same steps with BAR-ATH as the target (base = all 10 matches,
  BAR features from poss_model.bar_features, live = B2's live frames); saved as
  cache/dfl/poss_model_C_ctx_st.pkl. Allowed by RULES.md: pseudo-labels produced by an algorithm on
  the provided game, no human decision.
Writes reports/selftrain_eval_summary.json.
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
import poss_model_eval as pe  # noqa: E402
from barca import dfl, loaders, metrica, poss_model as P  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "selftrain_eval_summary.json")
CONF = 0.9


def pseudo_item(F, live, q):
    conf = live & ((q >= CONF) | (q <= 1 - CONF))
    return {"Fb": F, "y": q >= 0.5, "live": conf, "src": "dfl"}, float(conf.sum() / max(live.sum(), 1))


def predict(clf, F, live):
    q = np.zeros(len(live))
    q[live] = clf.predict_proba(F[live])[:, 1]
    return q


def build_items():
    t0 = time.time()
    gap = pe.gap_profile()
    ids = dfl.match_ids()
    data = {mid: pe.prepare(mid, gap) for mid in ids}
    p_touch = {}
    for mid in ids:
        clf = pe.hgb().fit(*pe.stack_touch([data[k] for k in ids if k != mid]))
        p_touch[mid] = P.touch_probs(clf, data[mid]["Ft"], data[mid]["live"])
    touch_final = pe.hgb().fit(*pe.stack_touch(list(data.values())))
    ce.LEVELS = (1, 1)
    items = {}
    for mid in ids:
        m, dd = data[mid], data[mid]["d"]
        gt = dfl.controlled_labels(dd["labels"], dd["blocks"], 3.0)
        F, _ = ce.feats(dd["players"], dd["ball"], m["st"], dd["pitch"], dd["blocks"], m["live"], p_touch[mid], m["ctrl"])
        items[("dfl", mid)] = {"Fb": F, "y": gt == 0, "gt": gt, "live": m["live"], "src": "dfl"}
    for g in metrica.GAMES:
        items[("met", str(g))] = ce.metrica_item(g, touch_final)
    print("prepared %.0f s" % (time.time() - t0), flush=True)
    return items, touch_final


def main():
    t0 = time.time()
    items, touch_final = build_items()
    keys = list(items)
    res = {}
    for k in keys:
        tgt = items[k]
        base = ce.fit([items[j] for j in keys if j != k], "Fb")
        q = predict(base, tgt["Fb"], tgt["live"])
        ps, share = pseudo_item(tgt["Fb"], tgt["live"], q)
        st = ce.fit([items[j] for j in keys if j != k] + [ps], "Fb")
        res[k] = {"base": ce.score(base, tgt, "Fb"), "self": ce.score(st, tgt, "Fb"), "pseudo_share": round(share, 3),
                  "pseudo_acc": round(float(((q >= 0.5) == tgt["y"])[ps["live"]].mean()), 4)}
        print(k, json.dumps(res[k]), "%.0f s" % (time.time() - t0), flush=True)
    gains = [r["self"]["spellF1"] - r["base"]["spellF1"] for r in res.values()]
    wins = int(sum(g > 0 for g in gains))
    go = float(np.mean(gains)) >= 0.01 and wins >= 6
    out = {"mean_gain": round(float(np.mean(gains)), 4), "wins": wins, "go": bool(go),
           "mean_base": round(float(np.mean([r["base"]["spellF1"] for r in res.values()])), 4),
           "mean_self": round(float(np.mean([r["self"]["spellF1"] for r in res.values()])), 4),
           "detail": {"%s_%s" % k: v for k, v in res.items()}}
    print(json.dumps({k: out[k] for k in ("mean_base", "mean_self", "mean_gain", "wins", "go")}, indent=1), flush=True)
    if go and "--bar" in sys.argv[1:]:
        import make_hybrid as mh
        import possession_metrics as pm
        import possession_pipeline as pp
        S = pp.setup()
        b2 = pm.label_codes(pm.read_label_csv(mh.BC_OUT))
        base = ce.fit(list(items.values()), "Fb")
        saved = {"touch": touch_final, "poss": base, "cost": 64, "touch_th": pe.TOUCH_TH, "absorb": 3.0,
                 "training": "C", "context": True}
        F, live = P.bar_features(S, b2, saved)
        q = predict(base, F, live)
        ps, share = pseudo_item(F, live, q)
        final = ce.fit(list(items.values()) + [ps], "Fb")
        saved["poss"] = final
        saved["self_training"] = {"conf": CONF, "pseudo_share": round(share, 3)}
        path = os.path.join(dfl.CACHE, "poss_model_C_ctx_st.pkl")
        with open(path, "wb") as fh:
            pickle.dump(saved, fh)
        out["bar"] = {"pseudo_share": round(share, 3), "saved": path}
        print("BAR pseudo share %.3f, saved %s" % (share, path), flush=True)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
