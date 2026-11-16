"""Possession count: decode the learned model's out-of-fold DFL probabilities differently. Aggregates only.

    python scripts/poss_decode_study.py     (needs cache/dfl/poss_oof.npz from poss_model_eval.py)

Pre-registered 2026-11-14, before running. The current decoder (2-state Viterbi, switch cost chosen
on spellF1, cost 64 at absorb 3) gives 3.9 possessions per live min against the DFL truth's 5.3:
a high cost kills flicker but also real short possessions. Decoders compared:
  D0  Viterbi, cost in {2, 4, 8, 16, 32, 64}            (the current one)
  D1  Viterbi, cost in {1, 2, 4, 8, 16, 32}, then poss_model.absorb_short with min length
      in {1, 2, 3, 4} s: a short spell handing the ball straight back is merged, like the
      "controlled" rule; other changes may happen freely
For each held-out match, the parameters with the best mean spellF1 on the other six (leave one
match out), per absorb time (2, 3, 5 s). Reported: spellF1, possessions per live min against the
truth, live team accuracy.
Go rule: D1 replaces D0 if its mean spellF1 is >= D0's + 0.01 at absorb 3 and not lower at 2 and
5. The BAR-ATH rebuild then uses D1 with the parameters chosen on all 7 matches at absorb 3.
Writes reports/poss_decode_study_summary.json.
"""

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from barca import dfl, loaders, poss_model as P  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "poss_decode_study_summary.json")
ABSORB = (2.0, 3.0, 5.0)
D0 = [(c, 0) for c in (2, 4, 8, 16, 32, 64)]
D1 = [(c, m) for c, m in itertools.product((1, 2, 4, 8, 16, 32), (1, 2, 3, 4))]
FPS = 25


def evaluate(gt, pred, live):
    runs = pm.valid_runs(gt)
    return {"spellF1": pm.spell_scores(gt, pred, runs)["mean_f1"],
            "poss_per_min": len(pm.possession_spells(pred, runs)) / (live.sum() / FPS / 60),
            "acc": float((pred[live] == gt[live]).mean())}


def main():
    z = np.load(os.path.join(dfl.CACHE, "poss_oof.npz"))
    ids = [str(x) for x in z["ids"]]
    out = {}
    for absorb in ABSORB:
        res = {}
        for i in range(len(ids)):
            p, gt, live = z["p_%g_%d" % (absorb, i)], z["gt_%g_%d" % (absorb, i)], z["live_%d" % i]
            base = {}
            for c, m in D0 + D1:
                if c not in base:
                    base[c] = P.decode(p, live, c)
                res[(i, c, m)] = evaluate(gt, P.absorb_short(base[c], m), live)
            res[(i, "truth")] = evaluate(gt, gt, live)
        summary = {}
        for name, grid in (("D0", D0), ("D1", D1)):
            rows = []
            for i in range(len(ids)):
                others = [k for k in range(len(ids)) if k != i]
                best = max(grid, key=lambda g: np.mean([res[(k,) + g]["spellF1"] for k in others]))
                rows.append(dict(res[(i,) + best], params=list(best)))
            summary[name] = {"spellF1": round(float(np.mean([r["spellF1"] for r in rows])), 4),
                             "poss_per_min": round(float(np.mean([r["poss_per_min"] for r in rows])), 2),
                             "acc": round(float(np.mean([r["acc"] for r in rows])), 4),
                             "params_chosen": sorted({tuple(r["params"]) for r in rows}),
                             "params_all": list(max(grid, key=lambda g: np.mean([res[(k,) + g]["spellF1"] for k in range(len(ids))])))}
        summary["truth_poss_per_min"] = round(float(np.mean([res[(i, "truth")]["poss_per_min"] for i in range(len(ids))])), 2)
        out[str(absorb)] = summary
        print("absorb %g:" % absorb, json.dumps(summary), flush=True)
    d = {a: out[str(a)]["D1"]["spellF1"] - out[str(a)]["D0"]["spellF1"] for a in ABSORB}
    out["go_D1"] = bool(d[3.0] >= 0.01 and d[2.0] >= 0 and d[5.0] >= 0)
    out["D1_params_for_bar"] = out["3.0"]["D1"]["params_all"]
    print("go_D1", out["go_D1"], "params", out["D1_params_for_bar"])
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=list)


if __name__ == "__main__":
    main()
