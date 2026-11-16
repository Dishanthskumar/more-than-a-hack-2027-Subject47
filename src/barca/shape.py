"""Who has the ball, from team shape alone. For the long stretches where the ball is missing.

Self-supervised: trained on pseudo-labels from the tracking itself (frames where the ball is on
the pitch, within reach of a player, and clearly nearer one team), then applied everywhere. No
human label of any kind is involved. Logistic regression, solved by Newton's method from a zero
start, so the fit is deterministic.

Attack direction is found per half from the data: each team's goalkeeper is the player who sits
deepest, so the team whose deepest player is nearer x=0 defends the left goal.
"""

import warnings

import numpy as np

VEL_HALF_FRAMES = 6      # velocity from positions +/- 0.24 s
L2 = 1.0
NEWTON_ITERS = 25


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def attack_sign(data, first, last):
    """+1 if home attacks towards x=1 in this half, -1 otherwise."""
    x = data["players"][first:last + 1, :, 0].astype(np.float64)
    team = data["slot_team"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_min, home_max = np.nanmin(x[:, team == 0], 1), np.nanmax(x[:, team == 0], 1)
        away_min, away_max = np.nanmin(x[:, team == 1], 1), np.nanmax(x[:, team == 1], 1)
    # home defends left if its deepest-left player is nearer the left line than away's is
    left = np.nanmedian(home_min) < np.nanmedian(away_min)
    right = np.nanmedian(home_max) < np.nanmedian(away_max)
    if left != right:
        return 1.0 if left else -1.0
    return 1.0 if np.nanmedian(home_min + home_max) < np.nanmedian(away_min + away_max) else -1.0


def features(data, blocks):
    """(n, k) feature matrix, NaN-free (0 where undefined), and the feature names."""
    L, W = data["pitch"]
    x = data["players"][:, :, 0].astype(np.float64) * L
    y = data["players"][:, :, 1].astype(np.float64) * W
    team = data["slot_team"]
    n = len(x)
    sign = np.zeros(n)
    for first, last in blocks:
        sign[first:last + 1] = attack_sign(data, first, last)

    k = VEL_HALF_FRAMES
    vx = np.full_like(x, np.nan)
    vx[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k / 25.0)

    cols = {}
    for t, s in ((0, 1.0), (1, -1.0)):           # away attacks the opposite way to home
        m = team == t
        d = sign * s
        cols["x%d" % t] = d * (_nanmean(x[:, m], 1) - L / 2)
        cols["vx%d" % t] = d * _nanmean(vx[:, m], 1)
        cols["sx%d" % t] = _nanstd(x[:, m], 1)
        cols["sy%d" % t] = _nanstd(y[:, m], 1)
        # most advanced player, in that team's attacking direction, relative to halfway
        adv = np.where(np.isfinite(x[:, m]), d[:, None] * (x[:, m] - L / 2), -np.inf)
        cols["front%d" % t] = adv.max(axis=1)
    names = sorted(cols)
    X = np.column_stack([cols[c] for c in names])
    X[~np.isfinite(X)] = 0.0
    # pairwise differences carry most of the signal; give them explicitly
    diff = np.column_stack([X[:, names.index(a + "0")] - X[:, names.index(a + "1")]
                            for a in ("x", "vx", "sx", "sy", "front")])
    return np.hstack([X, diff]), names + ["d_" + a for a in ("x", "vx", "sx", "sy", "front")]


def fit(X, y):
    """L2 logistic regression by Newton. Returns (mean, scale, weights incl. bias)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], L2)
    reg[0] = 0.0
    for _ in range(NEWTON_ITERS):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + reg * w
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return mu, sd, w


def logit(model, X):
    mu, sd, w = model
    return np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ w


def train_and_predict(data, blocks, pseudo_home, pseudo_mask):
    """Log-odds of home possession per frame, plus GT-free held-out accuracy (half vs half).

    pseudo_home: 1 where the tracking says home is on the ball, 0 away; used only on pseudo_mask.
    The returned log-odds for each half come from a model trained on the OTHER half, so a frame's
    own pseudo-label never feeds its own prediction.
    """
    X, _ = features(data, blocks)
    out = np.zeros(len(X))
    acc = []
    for i, (first, last) in enumerate(blocks):
        train = np.zeros(len(X), bool)
        for j, (a, b) in enumerate(blocks):
            if j != i:
                train[a:b + 1] = True
        train &= pseudo_mask
        model = fit(X[train], pseudo_home[train].astype(float))
        out[first:last + 1] = logit(model, X[first:last + 1])
        test = np.zeros(len(X), bool)
        test[first:last + 1] = True
        test &= pseudo_mask
        acc.append(float(np.mean((out[test] > 0) == (pseudo_home[test] == 1))))
    return out, acc
