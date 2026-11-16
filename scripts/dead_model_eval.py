"""Learned dead-ball evidence: bench on degraded DFL, then BAR-ATH. Aggregates only.

    python scripts/dead_model_eval.py [--features players|all]   # DFL bench (leave one match out)
    python scripts/dead_model_eval.py --bar        # ... then BAR-ATH label-free checks
    python scripts/dead_model_eval.py --bar --score  # ... and the one public check

Pre-registered 2026-11-14, before running. All DFL matches degraded (degrade.py, seed 0).
  R0  B2's dead layer without what DFL lacks (Smart Tagging, broadcast): set_pieces.detect(c, tr,
      tags=[], onset=2) with the live term -1 inside control events (make_hybrid.CTRL_LIVE_W)
  M1  the model alone: dead_ball.min_durations(viterbi_dead(logit p))
  R1  R0 + w * logit(p) as extra evidence; w from {0.25, 0.5, 1, 2, 4}, chosen for a held-out match
      by the mean alive/dead F1 of the other six (their out-of-fold p)
  Model: sklearn HistGradientBoostingClassifier (max_iter 300, early stopping, seed 0) on
  dead_model.FEATURES, all frames of the halves; out-of-fold p for every match.
  Metrics: dead F1, alive/dead F1 (mean of dead and live F1, frames), dead spells true / predicted,
  false stoppages (predicted dead spells touching no true dead frame), missed stoppages, median
  |start| and |end| error of matched spells (s).
  BAR-ATH (--bar): model refit on all 7, w = the best pooled value; B2's dead layer gets
  + w * logit(p) on top of its own extra terms. Go on to the public check only if on DFL R1 beats
  R0 by >= 0.01 mean alive/dead F1 without more false stoppages.
Amended 2026-11-14 after the first run: DFL freezes the ball through every dead spell, so the
first run (ball features, ball unmasked) was invalid. Now the DFL ball is hidden (NaN) on dead
frames BEFORE degrading, for every variant including R0. --features players (option 1) uses
dead_model.PLAYER_FEATURES; --features all (option 2) keeps the ball features on the masked ball.
Same go rule.
--fair (2026-11-14, pre-registered before running): the DFL bench gets what B2 has on BAR-ATH,
simulated: Smart-Tagging-like set-piece tags from the DFL events (dfl.set_piece_tags: kept per type
at BAR-ATH's tagging rate, restart 5 s into a 35 s window / 3.4 s into 5 s for kick-offs, +-1 s
jitter) and an ATD-like dead-time ball (degrade.simulate_dead_ball) instead of the masked one. R0
is then B2's full dead layer minus broadcast. Implies --features players. Same go rule.
Also saves (2026-11-14) cache/dfl/dead_<features>[_fair]_oof.npz (per DFL match: out-of-fold p,
truth, R0 and R1 masks) and, with --bar, cache/dfl/dead_<...>_bar.npz (p on the ATD, B2's dead mask
with and without the model), and reports where the BAR-ATH changes fall: inside / outside the
public window (its location only; no scores unless --score).
R1c (2026-11-14, option 2; named before its numbers were known, as "let the model only extend or
confirm stoppages the rules already see"): R1, keeping only dead spells that overlap a dead spell of
R0. It cannot add stoppages, only move their edges and fill them. Same go rule; --score is allowed
when R1c passes it. --bar adds B2_dead_confirm and saves the all-match model to
cache/dfl/dead_<...>_model.pkl.
Writes reports/dead_model_eval_<features>[_fair]_summary.json; with --score submissions/barca_hybrid_dm_<features>.csv.
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

from barca import dead_ball as db, dead_model as DM, degrade, dfl, loaders, nearest, possession, set_pieces  # noqa: E402
from barca.loaders import run_lengths  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "dead_model_eval_summary.json")
W_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
CTRL_LIVE_W = 1.0
FPS = 25


def model():
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, validation_fraction=0.1,
                                          random_state=0)


def ctrl_extra(tr, c, pitch):
    _, team, dist, margin = nearest.from_tracking(tr, pitch=tuple(pitch))
    ev = possession.control_events(team, dist, margin, c["ball_speed"])
    e = np.zeros(len(team))
    for s, t, _ in ev:
        e[s:t] = -CTRL_LIVE_W
    return e


def metrics(gt, pred):
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())
    f_dead = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f_live = 2 * tn / (2 * tn + fp + fn) if tn else 0.0
    ts, tk = run_lengths(gt)
    ps, pk = run_lengths(pred)
    false_stop = sum(1 for s, k in zip(ps, pk) if not gt[s:s + k].any())
    missed = sum(1 for s, k in zip(ts, tk) if not pred[s:s + k].any())
    st, en = [], []
    for s, k in zip(ts, tk):
        o = [(a, b) for a, b in zip(ps, pk) if a < s + k and a + b > s]
        if o:
            a, b = max(o, key=lambda x: min(x[0] + x[1], s + k) - max(x[0], s))
            st.append(abs(a - s) / FPS)
            en.append(abs(a + b - (s + k)) / FPS)
    return {"dead_f1": round(f_dead, 4), "ad_f1": round((f_dead + f_live) / 2, 4),
            "true_spells": len(ts), "pred_spells": len(ps), "false_stops": false_stop, "missed": missed,
            "start_err_s": round(float(np.median(st)), 2) if st else None,
            "end_err_s": round(float(np.median(en)), 2) if en else None}


KIND = "all"
FAIR = False
OUT_NPZ = None


def prepare(mid, gap):
    d = dict(dfl.load(mid))
    tags = []
    if FAIR:
        tags = dfl.set_piece_tags(mid, d["blocks"], seed=0)
        d = degrade.simulate_dead_ball(d, seed=0)
    else:
        ball = d["ball"].copy()
        ball[d["labels"] == 2] = np.nan           # DFL freezes the ball while dead: hide it
        d["ball"] = ball
    d = degrade.degrade(d, gap[0], gap[1], seed=0)
    tr = DM.tr_from(d)
    n = len(d["labels"])
    c = db.cues(tr, {"h02": np.full(n, np.nan), "h12": np.full(n, np.nan)}, tags)
    return {"tr": tr, "c": c, "F": DM.features(d["players"], d["ball"], d["pitch"], d["slot_team"], KIND),
            "gt": d["labels"] == 2, "extra": ctrl_extra(tr, c, d["pitch"]), "n": n, "tags": tags}


def confirm_only(r1, r0):
    """Dead spells of r1 that overlap a dead spell of r0 (no new stoppages)."""
    out = np.zeros_like(r1)
    for s, k in zip(*run_lengths(r1)):
        if r0[s:s + k].any():
            out[s:s + k] = True
    return out


def rules(m, add=None):
    e = m["extra"] if add is None else m["extra"] + add
    return set_pieces.detect(m["c"], m["tr"], m["tags"], onset=2, extra=e)[0]


def mean(rows, key, sub):
    return round(float(np.mean([r[key][sub] for r in rows])), 4)


def atd_gap():
    import possession_metrics as pm
    from barca import atd
    b2 = pm.label_codes(pm.read_label_csv(os.path.join(PROJECT, "submissions", "barca_hybrid_bc.csv")))
    return degrade.atd_gap_profile(atd.load()["ball"], b2 < 2)


def bar(final, w, score):
    import make_hybrid as mh
    import make_submission as ms
    import possession_metrics as pm
    import possession_pipeline as pp
    from barca import atd, broadcast
    from games import source_frames
    S = pp.setup()
    n = len(S["team"])
    data = atd.load()
    p = final.predict_proba(DM.features(data["players"], data["ball"], data["pitch"],
                                        data["slot_team"].astype(int), KIND))[:, 1]
    add = w * DM.logit(p)
    c = S["cues"]
    events = possession.control_events(S["team"], S["dist"], S["margin"], c["ball_speed"])
    in_ctrl = np.zeros(n, bool)
    for s, e, _ in events:
        in_ctrl[s:e] = True
    graded = np.zeros(n, bool)
    for a, b in S["runs"]:
        graded[a:b] = True
    extra = np.where(in_ctrl, -mh.CTRL_LIVE_W, 0.0)
    extra = extra + broadcast.evidence(broadcast.cut_onsets(broadcast.coverage(n), graded),
                                       broadcast.restart_cue(c, in_ctrl), n)
    v5 = mh.v5_labels()
    out, masks = {}, {}
    for key, ex in (("B2", extra), ("B2_dead_model", extra + add), ("B2_dead_confirm", extra + add)):
        dead, _ = set_pieces.detect(c, loaders.load_tracking(), loaders.load_smart_tagging(), onset=2, extra=ex)
        if key == "B2_dead_confirm":
            dead = confirm_only(dead, masks["B2"])
        masks[key] = dead
        rb = possession.labels_from_events(events, dead, possession.carry_labels(S["base"], dead), "mid")
        hy = mh.hybrid(v5, dead, rb)
        if key == "B2" and not np.array_equal(hy, pm.label_codes(pm.read_label_csv(mh.BC_OUT))):
            sys.exit("B2 rebuilt here differs from " + mh.BC_OUT)
        r = {"full_match": pp.full_match(S, hy)}
        if key != "B2":
            r["changed_vs_B2_s"] = round(float((hy != out["B2"]["_hy"]).sum()) / FPS, 1)
        if score and key in ("B2", score):
            r["public"] = pp.public_scores(S, hy, ref=None if key == "B2" else out["B2"]["_hy"])
            if key != "B2":
                path = os.path.join(PROJECT, "submissions", "barca_hybrid_{}.csv".format(
                    "dm_confirm" if key == "B2_dead_confirm" else "dm_" + KIND))
                ms.write(path, np.array(pm.LABELS)[hy])
                err = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical"))
                if err:
                    sys.exit("invalid submission: " + err)
        r["_hy"] = hy
        out[key] = r
    win = S["gt"] != 3
    d0, d1 = masks["B2"], masks["B2_dead_model"]
    ch = graded & (d0 != d1)
    lab = graded & (out["B2"]["_hy"] != out["B2_dead_model"]["_hy"])
    out["where"] = {"dead_mask_changed_s": round(ch.sum() / FPS, 1),
                    "live_to_dead_s": round((ch & d1).sum() / FPS, 1), "dead_to_live_s": round((ch & d0).sum() / FPS, 1),
                    "public_window_s": round(win.sum() / FPS, 1),
                    "dead_mask_changed_in_public_s": round((ch & win).sum() / FPS, 1),
                    "labels_changed_in_public_s": round((lab & win).sum() / FPS, 1),
                    "labels_changed_outside_public_s": round((lab & ~win).sum() / FPS, 1)}
    out["where_confirm"] = {"dead_mask_changed_s": round((graded & (masks["B2_dead_confirm"] != d0)).sum() / FPS, 1),
                            "dead_mask_changed_in_public_s": round((graded & win & (masks["B2_dead_confirm"] != d0)).sum() / FPS, 1)}
    np.savez_compressed(OUT_NPZ + "_bar.npz", p=p, dead_b2=d0, dead_model=d1, dead_confirm=masks["B2_dead_confirm"])
    for r in out.values():
        r.pop("_hy", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--features", choices=("players", "all"), default="players")
    ap.add_argument("--fair", action="store_true")
    a = ap.parse_args()
    global KIND, OUT, FAIR
    FAIR = a.fair
    KIND = "players" if FAIR else a.features
    OUT = OUT.replace("eval_summary", "eval_{}{}_summary".format(KIND, "_fair" if FAIR else ""))
    global OUT_NPZ
    OUT_NPZ = os.path.join(dfl.CACHE, "dead_{}{}".format(KIND, "_fair" if FAIR else ""))
    t0 = time.time()
    gap = atd_gap()
    ids = dfl.match_ids()
    data = {mid: prepare(mid, gap) for mid in ids}
    print("prepared %.0f s" % (time.time() - t0), flush=True)
    oof = {}
    for mid in ids:
        X = np.concatenate([data[k]["F"] for k in ids if k != mid])
        y = np.concatenate([data[k]["gt"] for k in ids if k != mid])
        oof[mid] = model().fit(X, y).predict_proba(data[mid]["F"])[:, 1]
    print("fitted %.0f s" % (time.time() - t0), flush=True)
    res, keep = {}, {}
    for mid in ids:
        m = data[mid]
        r0_mask = rules(m)
        r = {"R0": metrics(m["gt"], r0_mask),
             "M1": metrics(m["gt"], db.min_durations(db.viterbi_dead(DM.logit(oof[mid]))))}
        for w in W_GRID:
            r1_mask = rules(m, w * DM.logit(oof[mid]))
            r["R1_w%g" % w] = metrics(m["gt"], r1_mask)
            r["R1c_w%g" % w] = metrics(m["gt"], confirm_only(r1_mask, r0_mask))
        res[mid] = r
        keep[mid] = m
        print(mid, json.dumps({k: (v["ad_f1"], v["false_stops"], v["pred_spells"], v["true_spells"]) for k, v in r.items()}),
              flush=True)
    rows = []
    for mid in ids:
        others = [k for k in ids if k != mid]
        w = max(W_GRID, key=lambda g: np.mean([res[k]["R1_w%g" % g]["ad_f1"] for k in others]))
        wc = max(W_GRID, key=lambda g: np.mean([res[k]["R1c_w%g" % g]["ad_f1"] for k in others]))
        rows.append({"match": mid, "w": w, "wc": wc, "R0": res[mid]["R0"], "M1": res[mid]["M1"],
                     "R1": res[mid]["R1_w%g" % w], "R1c": res[mid]["R1c_w%g" % wc]})
    summary = {k: {s: mean(rows, k, s) for s in ("dead_f1", "ad_f1")} | {
        s: int(sum(r[k][s] for r in rows)) for s in ("true_spells", "pred_spells", "false_stops", "missed")} | {
        s: round(float(np.median([r[k][s] for r in rows if r[k][s] is not None])), 2) for s in ("start_err_s", "end_err_s")}
        for k in ("R0", "M1", "R1", "R1c")}
    w_all = max(W_GRID, key=lambda g: np.mean([res[k]["R1_w%g" % g]["ad_f1"] for k in ids]))
    arrays = {}
    for i, mid in enumerate(ids):
        w = rows[i]["w"]
        arrays["p_%d" % i] = oof[mid].astype(np.float32)
        arrays["gt_%d" % i] = keep[mid]["gt"]
        arrays["r0_%d" % i] = rules(keep[mid])
        arrays["r1_%d" % i] = rules(keep[mid], w * DM.logit(oof[mid]))
    os.makedirs(dfl.CACHE, exist_ok=True)
    np.savez_compressed(OUT_NPZ + "_oof.npz", ids=np.array(ids), **arrays)
    go = (summary["R1"]["ad_f1"] - summary["R0"]["ad_f1"] >= 0.01
          and summary["R1"]["false_stops"] <= summary["R0"]["false_stops"])
    go_c = (summary["R1c"]["ad_f1"] - summary["R0"]["ad_f1"] >= 0.01
            and summary["R1c"]["false_stops"] <= summary["R0"]["false_stops"])
    wc_all = max(W_GRID, key=lambda g: np.mean([res[k]["R1c_w%g" % g]["ad_f1"] for k in ids]))
    out = {"go_confirm": bool(go_c), "wc_all": wc_all, "features": KIND, "fair": FAIR, "tags_per_match": round(float(np.mean([len(data[k]["tags"]) for k in ids])), 1),
           "dfl_bench": summary, "per_match": rows, "w_all": w_all, "go_to_bar": bool(go)}
    print(json.dumps({k: out[k] for k in ("dfl_bench", "w_all", "go_to_bar", "wc_all", "go_confirm")}, indent=1), flush=True)
    if a.bar:
        final = model().fit(np.concatenate([data[k]["F"] for k in ids]), np.concatenate([data[k]["gt"] for k in ids]))
        import pickle
        with open(OUT_NPZ + "_model.pkl", "wb") as fh:
            pickle.dump({"model": final, "w": w_all, "wc": wc_all, "features": KIND}, fh)
        score = ("B2_dead_confirm" if go_c else ("B2_dead_model" if go else None)) if a.score else None
        out["bar"] = bar(final, wc_all if go_c else w_all, score)
        print(json.dumps(out["bar"], indent=1))
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
