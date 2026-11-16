"""Three-state (H, A, D) Viterbi decoder with switching costs.

Emissions are hand-set log-scores built from generic football physics (who is nearer the ball,
whether the ball is on the pitch, whether the players are moving). Nothing here is fitted to any
label of the provided game. Switching costs make a possession something that has to be earned
by several seconds of consistent evidence, which is what spellF1 rewards.
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = {
    # ownership: tanh((d_away - d_home) / own_scale), weighted by how close anyone is
    "own_gain": 0.4,
    "own_scale": 1.5,
    "own_reach": 1.5,       # metres within which the nearest player counts fully
    "own_decay": 3.0,       # metres over which the weight falls off beyond that
    # dead ball
    "d_bias": 0.0,
    "d_speed_mid": 1.5,     # m/s mean player speed where D and live are equally likely
    "d_speed_gain": 0.3,    # nats per m/s below the mid point
    "d_speed_cap": 0.6,
    "d_off_pitch": 1.5,
    "d_missing": 0.05,
    "d_close_ball": -0.2,   # ball on the pitch at a player's feet argues against D
    # v1: Smart Tagging and team shape (each used only if present in the feature dict)
    "phase_gain": 0.03,     # nats per frame per unit of calibrated phase log-odds
    "shape_gain": 0.03,     # nats per frame per unit of shape-model log-odds
    "shape_clip": 3.0,
    "pre_restart_d": 0.3,   # before a tagged restart: dead
    "restart_team": 0.3,    # just after a tagged restart: the named team, and live
    # v3: controlled possession from touch sequences (control.py)
    "control_gain": 0.3,
    # v4: ball filled from the broadcast ATD counts at this fraction of a native observation
    "fill_weight": 0.5,
    # switching costs, in nats
    "switch_team": 8.0,     # H <-> A
    "switch_dead": 8.0,     # live <-> D
}


# v0 is frozen so its file stays reproducible; v1-only terms are inert without their features.
PARAMS_V0 = dict(PARAMS, switch_team=8.0, switch_dead=8.0)


def emissions(f, params=PARAMS):
    """(n, 3) log-scores for H, A, D."""
    p = params
    n = len(f["seen"])
    out = np.zeros((n, 3))

    seen = f["seen"] & ~f["off_pitch"]
    with np.errstate(invalid="ignore"):
        margin = np.where(seen, f["d_away"] - f["d_home"], 0.0)
        margin = np.where(np.isfinite(margin), margin, 0.0)
        reach = np.exp(-np.clip(f["d_min"] - p["own_reach"], 0, None) / p["own_decay"])
    reach = np.where(seen & np.isfinite(reach), reach, 0.0)
    own = p["own_gain"] * reach * np.tanh(margin / p["own_scale"])
    if "receiver_team" in f:
        # approach 1 (receiver.py): the team of the next reception along the real ball path
        rec = p["own_gain"] * f["receiver_team"]
        if f.get("receiver_mode") == "replace":
            own = np.where(f["filled"], own, rec) if "filled" in f else rec
        else:
            own = own + rec
    if "filled" in f:
        own = np.where(f["filled"], p["fill_weight"] * own, own)
    if "phase_team" in f:
        own = own + p["phase_gain"] * f["phase_team"]
    if "shape_logit" in f:
        # formation matters most when the ball tells us little
        own = own + (1.0 - reach) * p["shape_gain"] * np.clip(
            f["shape_logit"], -p["shape_clip"], p["shape_clip"])
    if "restart_team" in f:
        own = own + p["restart_team"] * f["restart_team"]
    if "control_team" in f:
        own = own + p["control_gain"] * f["control_team"]
    out[:, H] = own
    out[:, A] = -own

    speed = np.where(np.isfinite(f["player_speed"]), f["player_speed"], p["d_speed_mid"])
    dead = p["d_bias"] + np.clip((p["d_speed_mid"] - speed) * p["d_speed_gain"],
                                 -p["d_speed_cap"], p["d_speed_cap"])
    dead = dead + np.where(f["off_pitch"], p["d_off_pitch"], 0.0)
    dead = dead + np.where(~f["seen"], p["d_missing"], 0.0)
    dead = dead + np.where(seen & (f["d_min"] < p["own_reach"]), p["d_close_ball"], 0.0)
    if "pre_restart" in f:
        dead = dead + p["pre_restart_d"] * f["pre_restart"]
    if "restart_team" in f:
        dead = dead - p["restart_team"] * np.abs(f["restart_team"])
    # D is scored against the better of the two live states, so ownership evidence does not
    # also count as evidence against a stoppage.
    out[:, D] = dead + np.maximum(out[:, H], out[:, A])
    return out


def transition_costs(params=PARAMS):
    c = np.full((3, 3), params["switch_dead"])
    c[H, A] = c[A, H] = params["switch_team"]
    np.fill_diagonal(c, 0.0)
    return c


def viterbi(scores, costs):
    """State path maximising sum(scores) - sum(switch costs). Deterministic tie-breaking."""
    n = len(scores)
    s = scores.tolist()
    neg = (-costs).tolist()
    back = np.zeros((n, 3), dtype=np.int8)
    v = list(s[0])
    for t in range(1, n):
        st = s[t]
        new = [0.0, 0.0, 0.0]
        bt = back[t]
        row = [0, 0, 0]
        for j in range(3):
            best_i, best = 0, v[0] + neg[0][j]
            for i in (1, 2):
                c = v[i] + neg[i][j]
                if c > best:
                    best_i, best = i, c
            new[j] = best + st[j]
            row[j] = best_i
        bt[:] = row
        v = new
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(f, blocks, n_frames, params=PARAMS):
    """Labels for every frame: decoded inside each [first, last] block, X elsewhere."""
    scores = emissions(f, params)
    costs = transition_costs(params)
    labels = np.full(n_frames, "X", dtype="<U1")
    for first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(scores[first:last + 1], costs)]
    return labels


def spell_stats(labels, blocks, fps=25):
    """Possession count and duration percentiles. No ground truth needed."""
    lengths = []
    for first, last in blocks:
        w = labels[first:last + 1]
        cut = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], cut, [len(w)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if w[a] in ("H", "A"):
                lengths.append(b - a)
    lengths = np.array(lengths, dtype=float) / fps
    if not len(lengths):
        return {"n_spells": 0}
    return {
        "n_spells": int(len(lengths)),
        "p10_s": round(float(np.percentile(lengths, 10)), 1),
        "median_s": round(float(np.median(lengths)), 1),
        "p90_s": round(float(np.percentile(lengths, 90)), 1),
        "share_D": round(float(np.mean(np.concatenate(
            [labels[a:b + 1] for a, b in blocks]) == "D")), 3),
    }
