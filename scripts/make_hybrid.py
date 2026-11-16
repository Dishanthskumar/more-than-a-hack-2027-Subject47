"""Build the combined (hybrid) submission from the match data, in one command.

    python scripts/make_hybrid.py                 # -> submissions/barca_hybrid.csv
    python scripts/make_hybrid.py --set-pieces    # -> submissions/barca_hybrid_sp.csv
    python scripts/make_hybrid.py --stoppages     # -> submissions/barca_hybrid_st.csv
    python scripts/make_hybrid.py --broadcast     # -> submissions/barca_hybrid_bc.csv
    python scripts/make_hybrid.py --dead-model    # -> submissions/barca_hybrid_dmx.csv
    python scripts/make_hybrid.py --poss-model    # -> submissions/barca_hybrid_pm.csv
    python scripts/make_hybrid.py --poss-model --dead-model   # -> submissions/barca_hybrid_pmdx.csv
    add --poss-decoder d1 to either for the D1 decoder        # -> ..._pm_d1.csv / ..._pmdx_d1.csv
    python scripts/make_hybrid.py --final         # -> submissions/subject47.csv (THE SUBMISSION)
        = --poss-model --dead-model --poss-decoder d1 --poss-model-file cache/dfl/poss_model_C_ctx.pkl;
          that configuration writes subject47.csv whenever --out is not given
    python scripts/make_hybrid.py --out other.csv

The hybrid (pre-registered 2026-11-09, see scripts/compare_pipelines.py):
  * dead ball: the rebuilt rule pipeline's detector (src/barca/dead_ball.py, with restart snap)
  * team:      Track A v5 inside each live spell; v5's D frames there are filled with the last
               v5 team (head back-filled)
  * fallback:  a live spell where v5 names no team keeps the rebuilt label (poss_control_mid)

--set-pieces swaps in the set-piece-aware dead mask (src/barca/set_pieces.py): line restarts
dead from the ball exit instead of a fixed 10 s before the tag, and restarts located by tag type.
See scripts/set_piece_eval.py for the checks behind it.

--stoppages (implies --set-pieces) adds onset-aware speed + slow-player share evidence for untagged
stoppages, with a live term inside control events (src/barca/stoppages.py; variant S3 of
scripts/stoppage_eval.py). Experimental: frame-level gains, public spellF1 0.533 vs 0.552.

--broadcast (implies --stoppages) adds dead evidence from the moment the broadcast cuts away from
live play, stopped by tracking restart cues (src/barca/broadcast.py; variant B2 of
scripts/broadcast_eval.py). Public spellF1 0.560, macroF1 0.905.

--dead-model (implies --broadcast) lets a players-only stoppage model trained on DFL open data
(src/barca/dead_model.py; scripts/dead_model_eval.py --fair --bar writes the model) EXTEND B2's
stoppages: the dead layer is run again with + w * logit(p_dead) and only its spells that overlap a
B2 stoppage are added (dead_model.extend_only). It never removes or shortens a stoppage.
DFL fair bench: alive/dead F1 0.695 -> 0.819, false stoppages 7 -> 7.

--poss-model (implies --broadcast) replaces the team on every live frame with a learned
possession model trained on degraded DFL open data (src/barca/poss_model.py; model written by
scripts/poss_model_eval.py --bar to cache/dfl/poss_model.pkl; needs scikit-learn). The dead mask is
unchanged. DFL bench spellF1 0.671 vs 0.526 for the rules; public spellF1 0.627 vs B2's 0.560.

Both inputs are rebuilt from the data here, so this does not depend on any CSV being current.
Deterministic: same data in, same bytes out. Prints aggregates only.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import make_submission as ms  # noqa: E402
import possession_pipeline as pp  # noqa: E402
from barca import boundaries, broadcast, dead_ball, decoder, loaders, possession, set_pieces  # noqa: E402
from games import source_frames  # noqa: E402
import possession_metrics as pm  # noqa: E402

D = pm.LABELS.index("D")
DEFAULT_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid.csv")
SP_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid_sp.csv")
ST_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid_st.csv")
BC_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid_bc.csv")
DMX_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid_dmx.csv")
PM_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid_pm.csv")
PMDX_OUT = os.path.join(PROJECT, "submissions", "barca_hybrid_pmdx.csv")
FINAL_OUT = os.path.join(PROJECT, "submissions", "subject47.csv")      # team subject47's submission
FINAL_POSS_MODEL = os.path.join("cache", "dfl", "poss_model_C_ctx.pkl")
CTRL_LIVE_W = 1.0


def v5_labels(extra_fill=None, post=None):
    """Track A v5 codes, exactly as make_submission.py --version v5 builds them (extra_fill, post:
    see make_submission.build_features; None = unchanged)."""
    f = ms.build_features("v5", verbose=False, extra_fill=extra_fill, post=post)
    labels = decoder.decode(f, ms.blocks(), source_frames("tactical"), decoder.PARAMS)
    labels, _ = boundaries.refine(labels, f, ms.blocks(), team_changes=f["control_changes"],
                                  snap_dead=False)
    return pm.label_codes(np.asarray(labels))


def rebuilt(S, set_piece_rules=False, stoppage_rules=False, broadcast_rules=False, dead_model=False,
            dead_model_file=None, dead_mode=None):
    """(dead mask, poss_control_mid codes) from the rebuilt rule pipeline."""
    c = S["cues"]
    events = possession.control_events(S["team"], S["dist"], S["margin"], c["ball_speed"])
    if stoppage_rules or broadcast_rules:
        n = len(c["ball_speed"])
        in_ctrl = np.zeros(n, bool)
        for s, e, _ in events:
            in_ctrl[s:e] = True
        extra = np.where(in_ctrl, -CTRL_LIVE_W, 0.0)
        if broadcast_rules:
            graded = np.zeros(n, bool)
            for a, b in S["runs"]:
                graded[a:b] = True
            onsets = broadcast.cut_onsets(broadcast.coverage(n), graded)
            extra = extra + broadcast.evidence(onsets, broadcast.restart_cue(c, in_ctrl), n)
        dead, _ = set_pieces.detect(c, loaders.load_tracking(), loaders.load_smart_tagging(),
                                    onset=2, extra=extra)
        if dead_model:
            from barca import atd, dead_model as dm
            with_model, _ = set_pieces.detect(c, loaders.load_tracking(), loaders.load_smart_tagging(),
                                              onset=2, extra=extra + dm.bar_evidence(atd.load(), dead_model_file))
            mode = dead_mode or dm.saved_mode(dead_model_file)
            dead = (with_model if mode == "full" else dm.extend_trim_end(with_model, dead) if mode == "trim"
                    else dm.extend_only(with_model, dead))
    elif set_piece_rules:
        dead, _ = set_pieces.detect(c, loaders.load_tracking(), loaders.load_smart_tagging())
    else:
        dead = dead_ball.detect(c, S["starts"], snap=True)
    carry = possession.carry_labels(S["base"], dead)
    return dead, possession.labels_from_events(events, dead, carry, "mid")


def hybrid(v5, dead, rb):
    hy = possession.carry_labels(v5, dead)
    return np.where((hy == D) & ~dead, rb, hy).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-pieces", action="store_true",
                    help="use the set-piece-aware dead mask (default output barca_hybrid_sp.csv)")
    ap.add_argument("--stoppages", action="store_true",
                    help="also detect untagged stoppages (default output barca_hybrid_st.csv)")
    ap.add_argument("--broadcast", action="store_true",
                    help="also use broadcast cut-aways (default output barca_hybrid_bc.csv)")
    ap.add_argument("--dead-model", action="store_true",
                    help="also let the DFL-trained players-only stoppage model extend stoppages (barca_hybrid_dmx.csv)")
    ap.add_argument("--poss-model", action="store_true",
                    help="team from the DFL-trained learned possession model (barca_hybrid_pm.csv)")
    ap.add_argument("--poss-decoder", choices=("d0", "d1"), default="d0",
                    help="d1 = switch cost 8 + minimum hand-back spell 3 s (more possessions)")
    ap.add_argument("--d1-cost", type=float, default=None, help="D1 switch cost (default 8)")
    ap.add_argument("--d1-min", type=float, default=None, help="D1 minimum hand-back spell in s (default 3)")
    ap.add_argument("--dead-mode", choices=("extend", "trim", "full"), default=None,
                    help="how the stoppage model is used (default: the model file's mode, extend)")
    ap.add_argument("--dead-model-file", default=None,
                    help="alternative stoppage model pickle (default cache/dfl/dead_players_fair_model.pkl)")
    ap.add_argument("--poss-model-file", default=None,
                    help="alternative possession model pickle (default cache/dfl/poss_model.pkl)")
    ap.add_argument("--final", action="store_true",
                    help="the submitted configuration (writes submissions/subject47.csv)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.final:
        args.poss_model, args.dead_model, args.poss_decoder = True, True, "d1"
        args.poss_model_file = args.poss_model_file or FINAL_POSS_MODEL
    is_final = (args.poss_model and args.dead_model and args.poss_decoder == "d1"
                and args.poss_model_file is not None
                and os.path.basename(args.poss_model_file) == os.path.basename(FINAL_POSS_MODEL)
                and args.d1_cost is None and args.d1_min is None
                and args.dead_model_file is None and args.dead_mode is None)
    args.broadcast = args.broadcast or args.dead_model or args.poss_model
    args.out = args.out or (FINAL_OUT if is_final else PMDX_OUT if args.poss_model and args.dead_model else PM_OUT if args.poss_model
                            else DMX_OUT if args.dead_model else BC_OUT if args.broadcast else ST_OUT if args.stoppages
                            else SP_OUT if args.set_pieces else DEFAULT_OUT)
    if args.poss_model and args.poss_decoder == "d1" and args.out in (PM_OUT, PMDX_OUT):
        args.out = args.out.replace(".csv", "_d1.csv")
    if args.poss_model_file and not os.path.isabs(args.poss_model_file) and not os.path.isfile(args.poss_model_file):
        args.poss_model_file = os.path.join(PROJECT, args.poss_model_file)   # relative to the project, from anywhere

    S = pp.setup()
    dead, rb = rebuilt(S, args.set_pieces, args.stoppages, args.broadcast, args.dead_model, args.dead_model_file,
                       args.dead_mode)
    hy = hybrid(v5_labels(), dead, rb)
    if args.poss_model:
        from barca import poss_model
        d1 = None if args.d1_cost is None and args.d1_min is None else (
            args.d1_cost if args.d1_cost is not None else poss_model.D1[0],
            args.d1_min if args.d1_min is not None else poss_model.D1[1])
        hy = poss_model.apply_bar(S, hy, model_path=args.poss_model_file, decoder=args.poss_decoder, d1=d1)

    ms.write(args.out, np.array(pm.LABELS)[hy])
    err = pm.validate_submission(pm.read_label_csv(args.out), source_frames("tactical"))
    if err:
        raise SystemExit("invalid submission: " + err)
    print(json.dumps({"public": pp.public_scores(S, hy), "full_match": pp.full_match(S, hy)}))
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
