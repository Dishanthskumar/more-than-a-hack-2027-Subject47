"""External test on Metrica sample games 1 and 2 of the DFL-trained possession model. Aggregates only.

    python scripts/metrica_eval.py

Pre-registered 2026-11-14, before running:
  Truth: metrica.labels_from_events (event-based; possession shape close to our public GT:
  5.0-5.2 per live min, median 6.7-7.2 s, 33-35% under 4 s; public GT 4.2 / 7.6 s / 35%).
  Tracking: "smoothed" = ball Gaussian sigma 6 and players sigma 2 (degrade.py constants, no extra
  gaps: Metrica's ball is already missing 7-8% of live and ~70% of dead time, like the ATD);
  "raw" reported too. The dead mask is Metrica's truth, so only the team layer is scored.
  Model: cache/dfl/poss_model.pkl as saved (touch model, possession model, switch cost); no refit.
  Also the D1 decoder (cost 8, min 3 s) from the DFL study, reported only.
  Baseline: the tracking-only rule pipeline (control events -> labels_from_events "mid").
  Go rule: the model beats the baseline by >= 0.04 spellF1 on both games (smoothed tracking).
Writes reports/metrica_eval_summary.json.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from barca import degrade, dfl, loaders, metrica, nearest, possession, poss_model as P, touch_model as TM  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(loaders.REPORTS_DIR, "metrica_eval_summary.json")
FPS = 25


def smooth(d):
    out = dict(d)
    ball = d["ball"].astype(np.float64)
    ball[:, :2] = degrade._smooth(ball[:, :2], degrade.BALL_SIGMA, d["blocks"])
    out["ball"] = ball.astype(np.float32)
    pl = d["players"].astype(np.float64)
    pl[..., :2] = degrade._smooth(pl[..., :2], degrade.PLAYER_SIGMA, d["blocks"])
    out["players"] = pl.astype(np.float32)
    return out


def scores(gt, pred, live):
    import touch_snap_eval as tse
    runs = pm.valid_runs(gt)
    return {"spellF1": round(pm.spell_scores(gt, pred, runs)["mean_f1"], 4),
            "team_acc_live": round(float((pred[live] == gt[live]).mean()), 4),
            "possessions_per_live_min": round(len(pm.possession_spells(pred, runs)) / (live.sum() / FPS / 60), 2),
            "timing": tse.timing(gt, pred, live)}


def run(d, saved):
    gt = d["labels"]
    live = gt != 2
    dead = ~live
    L, W = d["pitch"]
    st = d["slot_team"].astype(int)
    _, team, dist, margin = nearest.nearest_team(
        (d["ball"][:, :2].astype(np.float64) * [L, W]).astype(np.float32),
        d["players"][..., 0] * np.float32(L), d["players"][..., 1] * np.float32(W), st)
    b = d["ball"][:, :2].astype(np.float64) * [L, W]
    speed = np.full(len(b), np.nan)
    speed[1:] = np.hypot(*np.diff(b, axis=0).T) * FPS
    ctrl = possession.control_events(team, dist, margin, speed)
    base = np.where(team >= 0, team, possession.DEAD).astype(np.int8)
    baseline = possession.labels_from_events(ctrl, dead, possession.carry_labels(base, dead), "mid")
    pt = P.touch_probs(saved["touch"], TM.features(d["players"], d["ball"], st, d["pitch"]), live)
    ev = P.touch_event_list(pt, live, saved["touch_th"])
    F = P.features(d["players"], d["ball"], st, d["pitch"], d["blocks"], live, pt, ev, ctrl)
    q = np.zeros(len(gt))
    q[live] = saved["poss"].predict_proba(F[live])[:, 1]
    model = P.decode(q, live, saved["cost"])
    d1 = P.absorb_short(P.decode(q, live, 8), 3)
    return {"truth_possessions_per_live_min": round(len(pm.possession_spells(gt, pm.valid_runs(gt))) / (live.sum() / FPS / 60), 2),
            "touch_events_per_live_min": round(len(ev) / (live.sum() / FPS / 60), 2),
            "model": scores(gt, model, live), "model_D1": scores(gt, d1, live), "baseline": scores(gt, baseline, live)}


def main():
    with open(os.path.join(dfl.CACHE, P.MODEL_FILE), "rb") as fh:
        saved = pickle.load(fh)
    out = {}
    for g in metrica.GAMES:
        d = metrica.load(g)
        out["game%d" % g] = {"smoothed": run(smooth(d), saved), "raw": run(d, saved)}
        print("game", g, json.dumps(out["game%d" % g]), flush=True)
    out["go"] = bool(all(out["game%d" % g]["smoothed"]["model"]["spellF1"]
                         - out["game%d" % g]["smoothed"]["baseline"]["spellF1"] >= 0.04 for g in metrica.GAMES))
    print("go", out["go"])
    os.makedirs(loaders.REPORTS_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
