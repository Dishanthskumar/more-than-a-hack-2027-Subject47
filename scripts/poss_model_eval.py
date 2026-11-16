"""Learned possession (team layer) on degraded DFL, then BAR-ATH. Aggregates only.

    python scripts/poss_model_eval.py                    # DFL bench at absorb 2, 3, 5 s
    python scripts/poss_model_eval.py --bar              # ... then BAR-ATH label-free checks
    python scripts/poss_model_eval.py --bar --score      # ... and the public check, if the go rule passes

Pre-registered 2026-11-14, before running:
  Data: every DFL match with the simulated dead-time ball and then degraded like the ATD
  (degrade.simulate_dead_ball, degrade.degrade, seed 0). Truth: dfl.controlled_labels with flips
  back shorter than the absorb time folded in (2, 3, 5 s; 3 s is the main one).
  Touch probabilities: the touch_model features and target, leave one match out (out of fold on
  DFL; refit on all 7 for BAR-ATH); touch events at threshold 0.1 (the degraded choice).
  Model: HistGradientBoosting (max_iter 300, early stopping, seed 0) on poss_model.FEATURES, live
  frames, target = home in possession. Decoding: poss_model.decode; switch cost from {2, 4, 8, 16,
  32, 64} chosen for a held-out match by the mean spellF1 of the other six.
  Baseline on the same frames and dead mask (DFL truth): the tracking-only rule pipeline (control
  events -> labels_from_events "mid", nearest-team fallback), as in touch_snap_eval.
  Go rule: model spellF1 >= baseline + 0.04 (the noise band) at all three absorb times; on BAR-ATH
  (absorb 3 model) no over-splitting (<= 284 possessions) and Metrica phase agreement no more than
  1 point below B2's; only then --score, and the variant is preferred only if public spellF1 >= B2's.
Writes reports/poss_model_eval_summary.json; with --score submissions/barca_hybrid_pm.csv.
Also saves cache/dfl/poss_oof.npz (2026-11-14): per absorb and match the out-of-fold p(home), the
truth and the live mask, for scripts/poss_decode_study.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from barca import degrade, dfl, loaders, nearest, possession, poss_model as P, touch_model as TM  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "poss_model_eval_summary.json")
ABSORB = (2.0, 3.0, 5.0)
MAIN_ABSORB = 3.0
COSTS = (2, 4, 8, 16, 32, 64)
TOUCH_TH = 0.1
FPS = 25


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, validation_fraction=0.1,
                                          random_state=0)


def gap_profile():
    from barca import atd
    b2 = pm.label_codes(pm.read_label_csv(os.path.join(PROJECT, "submissions", "barca_hybrid_bc.csv")))
    return degrade.atd_gap_profile(atd.load()["ball"], b2 < 2)


def ball_speed(ball, pitch):
    b = ball[:, :2].astype(np.float64) * np.asarray(pitch, float)
    s = np.full(len(b), np.nan)
    s[1:] = np.hypot(*np.diff(b, axis=0).T) * FPS
    return s


def prepare(mid, gap):
    d = degrade.degrade(degrade.simulate_dead_ball(dfl.load(mid), seed=0), gap[0], gap[1], seed=0)
    live = d["labels"] != 2
    L, W = d["pitch"]
    st = d["slot_team"].astype(int)
    _, team, dist, margin = nearest.nearest_team(
        (d["ball"][:, :2].astype(np.float64) * [L, W]).astype(np.float32),
        d["players"][..., 0] * np.float32(L), d["players"][..., 1] * np.float32(W), st)
    ctrl = possession.control_events(team, dist, margin, ball_speed(d["ball"], d["pitch"]))
    dead = ~live
    base = np.where(team >= 0, team, possession.DEAD).astype(np.int8)
    baseline = possession.labels_from_events(ctrl, dead, possession.carry_labels(base, dead), "mid")
    return {"d": d, "live": live, "st": st, "ctrl": ctrl, "baseline": baseline,
            "Ft": TM.features(d["players"], d["ball"], st, d["pitch"]),
            "yt": TM.targets(len(live), TM.flips(d["labels"], d["blocks"]))}


def stack_touch(ms):
    X = np.concatenate([m["Ft"][t][m["live"]] for m in ms for t in (0, 1)])
    y = np.concatenate([m["yt"][t][m["live"]] for m in ms for t in (0, 1)])
    return X, y


def poss_features(m, p_touch):
    d = m["d"]
    ev = P.touch_event_list(p_touch, m["live"], TOUCH_TH)
    return P.features(d["players"], d["ball"], m["st"], d["pitch"], d["blocks"], m["live"], p_touch, ev, m["ctrl"])


def spell_f1(gt, pred):
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]


def scores(gt, pred, live):
    import touch_snap_eval as tse
    n_sp = len(pm.possession_spells(pred, pm.valid_runs(gt)))
    return {"spellF1": round(spell_f1(gt, pred), 4), "team_acc_live": round(float((pred[live] == gt[live]).mean()), 4),
            "possessions_per_live_min": round(n_sp / (live.sum() / FPS / 60), 2),
            "timing": tse.timing(gt, pred, live)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    gap = gap_profile()
    ids = dfl.match_ids()
    data = {mid: prepare(mid, gap) for mid in ids}
    print("prepared %.0f s" % (time.time() - t0), flush=True)
    p_touch = {}
    for mid in ids:
        clf = hgb().fit(*stack_touch([data[k] for k in ids if k != mid]))
        p_touch[mid] = P.touch_probs(clf, data[mid]["Ft"], data[mid]["live"])
    for mid in ids:
        data[mid]["F"] = poss_features(data[mid], p_touch[mid])
    print("touch model out of fold + features %.0f s" % (time.time() - t0), flush=True)

    out = {"absorb": {}}
    finals = {}
    saved = {"ids": np.array(ids)}
    for i, mid in enumerate(ids):
        saved["live_%d" % i] = data[mid]["live"]
    for absorb in ABSORB:
        gts = {mid: dfl.controlled_labels(data[mid]["d"]["labels"], data[mid]["d"]["blocks"], absorb) for mid in ids}
        oof = {}
        for mid in ids:
            tr = [k for k in ids if k != mid]
            X = np.concatenate([data[k]["F"][data[k]["live"]] for k in tr])
            y = np.concatenate([(gts[k] == 0)[data[k]["live"]] for k in tr])
            q = np.zeros(len(data[mid]["live"]))
            q[data[mid]["live"]] = hgb().fit(X, y).predict_proba(data[mid]["F"][data[mid]["live"]])[:, 1]
            oof[mid] = q
        for i, mid in enumerate(ids):
            saved["p_%g_%d" % (absorb, i)] = oof[mid].astype(np.float32)
            saved["gt_%g_%d" % (absorb, i)] = gts[mid]
        dec = {(mid, c): P.decode(oof[mid], data[mid]["live"], c) for mid in ids for c in COSTS}
        sf = {(mid, c): spell_f1(gts[mid], dec[(mid, c)]) for mid in ids for c in COSTS}
        rows = []
        for mid in ids:
            others = [k for k in ids if k != mid]
            c = max(COSTS, key=lambda g: np.mean([sf[(k, g)] for k in others]))
            m = data[mid]
            rows.append({"match": mid, "cost": c,
                         "model": scores(gts[mid], dec[(mid, c)], m["live"]),
                         "baseline": scores(gts[mid], m["baseline"], m["live"])})
        mean = {k: {"spellF1": round(float(np.mean([r[k]["spellF1"] for r in rows])), 4),
                    "team_acc_live": round(float(np.mean([r[k]["team_acc_live"] for r in rows])), 4),
                    "possessions_per_live_min": round(float(np.mean([r[k]["possessions_per_live_min"] for r in rows])), 2),
                    "median_abs_offset_s": round(float(np.mean([r[k]["timing"]["median_abs_offset_s"] for r in rows])), 2)}
                for k in ("model", "baseline")}
        mean["truth_possessions_per_live_min"] = round(float(np.mean([
            len(pm.possession_spells(gts[k], pm.valid_runs(gts[k]))) / (data[k]["live"].sum() / FPS / 60) for k in ids])), 2)
        c_all = max(COSTS, key=lambda g: np.mean([sf[(k, g)] for k in ids]))
        go = mean["model"]["spellF1"] - mean["baseline"]["spellF1"] >= 0.04
        out["absorb"][str(absorb)] = {"mean": mean, "cost_all": c_all, "go": bool(go), "per_match": rows}
        finals[absorb] = (gts, c_all)
        print("absorb %.0f s:" % absorb, json.dumps(mean), "cost", c_all, "go", go, "%.0f s" % (time.time() - t0), flush=True)
    out["go_dfl"] = bool(all(v["go"] for v in out["absorb"].values()))
    os.makedirs(dfl.CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(dfl.CACHE, "poss_oof.npz"), **saved)
    print("go_dfl", out["go_dfl"], flush=True)

    if a.bar:
        gts, c_all = finals[MAIN_ABSORB]
        touch_final = hgb().fit(*stack_touch(list(data.values())))
        X = np.concatenate([data[k]["F"][data[k]["live"]] for k in ids])
        y = np.concatenate([(gts[k] == 0)[data[k]["live"]] for k in ids])
        poss_final = hgb().fit(X, y)
        import pickle
        with open(os.path.join(dfl.CACHE, "poss_model.pkl"), "wb") as fh:
            pickle.dump({"touch": touch_final, "poss": poss_final, "cost": c_all, "touch_th": TOUCH_TH,
                         "absorb": MAIN_ABSORB}, fh)
        out["bar"] = bar(touch_final, poss_final, c_all, a.score and out["go_dfl"])
        print(json.dumps(out["bar"], indent=1), flush=True)
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


def bar(touch_model, poss_model, cost, score):
    import make_hybrid as mh
    import make_submission as ms
    import possession_pipeline as pp
    import receiver_eval as rx
    from barca import atd
    from games import source_frames
    S = pp.setup()
    b2 = pm.label_codes(pm.read_label_csv(mh.BC_OUT))
    n = len(b2)
    graded = np.zeros(n, bool)
    for s, e in S["runs"]:
        graded[s:e] = True
    dead = b2 == 2
    live = graded & ~dead
    blocks = [(s, e - 1) for s, e in S["runs"]]
    data = atd.load()
    st = data["slot_team"].astype(int)
    Ft = TM.features(data["players"], data["ball"], st, data["pitch"])
    pt = P.touch_probs(touch_model, Ft, live)
    ctrl = possession.control_events(S["team"], S["dist"], S["margin"], S["cues"]["ball_speed"])
    ev = P.touch_event_list(pt, live, TOUCH_TH)
    F = P.features(data["players"], data["ball"], st, data["pitch"], blocks, live, pt, ev, ctrl)
    q = np.zeros(n)
    q[live] = poss_model.predict_proba(F[live])[:, 1]
    team = P.decode(q, live, cost)
    pred = np.where(graded, np.where(dead, 2, team), b2).astype(np.int8)
    cov, trans = rx.metrica(n)
    covered = live & (cov >= 0)
    out = {"touch_events_per_live_min": round(len(ev) / (live.sum() / FPS / 60), 2),
           "p_home_mean_live": round(float(q[live].mean()), 3), "cost": cost}
    for k, p in (("B2", b2), ("PM", pred)):
        fm = pp.full_match(S, p)
        out[k] = {"possessions": fm["n_possessions"], "mean_possession_s": fm["mean_possession_s"],
                  "Y1_metrica_agree": round(100.0 * float((p[covered] == cov[covered]).mean()), 1),
                  "Y2_transitions": rx.y2(p, live, trans, n)}
    out["PM"]["changed_vs_B2_s"] = round(float((graded & (pred != b2)).sum()) / FPS, 1)
    out["go_bar"] = bool(out["PM"]["possessions"] <= 284
                         and out["PM"]["Y1_metrica_agree"] >= out["B2"]["Y1_metrica_agree"] - 1)
    if score and out["go_bar"]:
        out["B2"]["public"] = pp.public_scores(S, b2)
        out["PM"]["public"] = pp.public_scores(S, pred, ref=b2)
        path = os.path.join(PROJECT, "submissions", "barca_hybrid_pm.csv")
        ms.write(path, np.array(pm.LABELS)[pred])
        err = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical"))
        if err:
            sys.exit("invalid submission: " + err)
    return out


if __name__ == "__main__":
    main()
