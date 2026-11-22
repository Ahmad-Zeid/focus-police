"""Leave-one-session-out evaluation on continuous streams, scored the way a user experiences Focus Police.

For each held-out session: train on the other sessions' labelled windows, predict every window of the
held-out stream (every 0.5 s, transitions included), smooth over time, then simulate the nudge.

Scores (all on the held-out session):
    balanced     mean of phone caught and on-task left alone, over labelled windows
    caught       share of phone windows flagged off task
    false_alarm  share of on-task windows flagged off task
    false_nudges false nudges per hour of on-task time (a nudge fires after NUDGE_S s off task in a row;
                 it is false if it fires during an on-task block)
    nudged       share of phone blocks that got a nudge
    delay        median seconds from the start of a phone block to its nudge
"""
import os

import numpy as np
from joblib import Memory
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import feats as FE
from data import F, HOP, on_task, window_ends, window_labels

NUDGE_S = 10
CACHE = Memory(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"), verbose=0)

MODELS = {
    "rf": lambda: RandomForestClassifier(400, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=0,
                                         class_weight="balanced_subsample"),
    "et": lambda: ExtraTreesClassifier(600, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=0,
                                       class_weight="balanced_subsample"),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, class_weight="balanced", random_state=0),
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")),
}


def session_table(sess, window_s, families, base_sensors="all", hop=HOP, R=None, aug_key="none"):
    """(features, names, activity labels, on/off truth, block ids, window end times) for one stream."""
    return _table(sess["folder"], aug_key, None if R is None else np.round(R, 6).tolist(), window_s,
                  tuple(sorted(families)), base_sensors, hop, sess)


@CACHE.cache(ignore=["sess"])
def _table(folder, aug_key, R, window_s, families, base_sensors, hop, sess):
    X = sess["X"] if R is None else FE.rotate(sess["X"], np.array(R))
    ends = window_ends(sess, window_s, hop)
    M, names = FE.session_features(X, ends, window_s, families, base_sensors, sess)
    labels, blocks = window_labels(sess, ends, window_s)
    return M, names, labels, on_task(labels), blocks, sess["t"][ends]


def augment_rotations(n, max_deg, seed=0):
    """Small random rotations about random axes: the strap sitting differently, a different arm pose."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ax = rng.normal(size=3)
        out.append(FE.rot_matrix(ax, rng.uniform(-max_deg, max_deg)))
    return out


# ------------------------------------------------------------------ smoothing

def smooth(p, method, **kw):
    """p: probability of OFF task per window, in time order. Returns smoothed probability."""
    if method == "none":
        return p
    if method == "mean":
        k = kw.get("k", 5)
        c = np.cumsum(np.r_[0, p]); i = np.arange(len(p))
        lo = np.maximum(0, i + 1 - k)
        return (c[i + 1] - c[lo]) / (i + 1 - lo)
    if method == "hmm":
        # two-state forward filter (causal): the user stays in the same state from one window to the next
        # with probability 1 - s. The classifier's probability is the evidence for the current window.
        s, prior_off = kw.get("s", 0.02), kw.get("prior_off", 0.5)
        out = np.empty_like(p)
        post = 0.5
        for i, q in enumerate(np.clip(p, 1e-4, 1 - 1e-4)):
            pr = (1 - s) * post + s * (1 - post)
            a = pr * q / prior_off; b = (1 - pr) * (1 - q) / (1 - prior_off)
            post = a / (a + b)
            out[i] = post
        return out
    raise ValueError(method)


# ------------------------------------------------------------------ scoring

def nudges(off, t, nudge_s=NUDGE_S):
    """Indices where a nudge fires: off task for nudge_s seconds in a row, once per unbroken stretch."""
    fired, start, done = [], None, False
    for i, o in enumerate(off):
        if not o or (i > 0 and t[i] - t[i - 1] > 2 * HOP + 0.25):   # stretch broken (or a gap in the data)
            start, done = None, False
        if o:
            if start is None:
                start = t[i]
            if not done and t[i] - start >= nudge_s:
                fired.append(i); done = True
    return np.array(fired, dtype=int)


def score(truth, off, t, blocks, nudge_s=NUDGE_S):
    lab = truth >= 0
    on_m, off_m = truth == 1, truth == 0
    caught = float(off[off_m].mean()) if off_m.any() else np.nan
    fa = float(off[on_m].mean()) if on_m.any() else np.nan
    fired = nudges(off, t, nudge_s)
    false_n = int(sum(truth[i] == 1 for i in fired))
    on_hours = on_m.sum() * HOP / 3600
    phone_blocks = np.unique(blocks[off_m])
    delays, hit = [], 0
    for b in phone_blocks:
        idx = np.where(blocks == b)[0]
        f = [i for i in fired if idx[0] <= i <= idx[-1] + int(5 / HOP)]
        # phone use straight after phone use: the nudge already fired in the stretch that runs into this block
        j = idx[0]
        while j > 0 and off[j - 1] and t[j] - t[j - 1] <= 2 * HOP + 0.25:
            j -= 1
        earlier = [i for i in fired if j <= i < idx[0]]
        if f or earlier:
            hit += 1; delays.append(0.0 if earlier else t[f[0]] - t[idx[0]])
    return {"balanced": float(np.nanmean([caught, 1 - fa])), "caught": caught, "false_alarm": fa,
            "acc": float((off[lab] == (truth[lab] == 0)).mean()),
            "false_nudges": false_n / on_hours if on_hours else np.nan,
            "nudged": hit / len(phone_blocks) if len(phone_blocks) else np.nan,
            "delay": float(np.median(delays)) if delays else np.nan, "n_false": false_n}


# ------------------------------------------------------------------ one experiment

def loso(sessions, window_s=2.0, families=("base",), base_sensors="all", model="rf", target="multi",
         n_aug=0, aug_deg=15, train_hop=1.0, smooth_method="none", smooth_kw=None, threshold=0.5,
         return_probs=False):
    smooth_kw = smooth_kw or {}
    rots = augment_rotations(n_aug, aug_deg)
    folds, probs = [], {}
    for k, test in enumerate(sessions):
        Xs, ys = [], []
        for s in sessions:
            if s is test:
                continue
            versions = [(None, "none")] + [(R, f"rot{aug_deg}_{j}") for j, R in enumerate(rots)]
            for R, key in versions:
                M, _, labels, _, _, _ = session_table(s, window_s, families, base_sensors, train_hop, R, key)
                keep = np.array([l is not None for l in labels])
                Xs.append(M[keep]); ys.append(labels[keep])
        Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
        if target == "binary":
            ytr = np.where([F.is_on_task(l) is False for l in ytr], "off", "on")
        m = MODELS[model]()
        m.fit(Xtr, ytr)
        M, _, labels, truth, blocks, t = session_table(test, window_s, families, base_sensors, HOP)
        P = m.predict_proba(M)
        off_cols = [i for i, c in enumerate(m.classes_) if (c == "off" or F.is_on_task(c) is False)]
        p_off = P[:, off_cols].sum(1)
        prior_off = float(np.mean([F.is_on_task(l) is False or l == "off" for l in ytr]))
        ps = smooth(p_off, smooth_method, prior_off=prior_off, **smooth_kw)
        off = ps >= threshold
        sc = score(truth, off, t, blocks)
        sc.update(held_out=test["id"])
        folds.append(sc)
        if return_probs:
            probs[test["id"]] = (p_off, truth, t, blocks, prior_off)
    return (folds, probs) if return_probs else folds


def summarize(folds):
    keys = ["balanced", "caught", "false_alarm", "acc", "false_nudges", "nudged", "delay"]
    return {k: float(np.nanmean([f[k] for f in folds])) for k in keys}
