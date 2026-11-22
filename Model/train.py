"""Train and compare Focus Police models on the recorded sessions, then report and deploy the best one.

    python train.py                 # everything, a few minutes
    python train.py --quick         # one window length, extra trees only: a fast sanity run
    python train.py --no-deploy     # don't touch the live page's model
    python train.py --test-participant P-XXXX --deploy-all
                                    # score on P-XXXX as usual, then deploy a model refit on EVERY session
                                    # (test people included) for live use; the report's score stays honest

Version 2 (22 Nov 2026). What changed and why is measured in lab/FINDINGS.md; in short:
  - Every session is handled as ONE CONTINUOUS STREAM, the way the live page sees it: a prediction every
    0.5 s, transitions between activities included. Only windows inside one labelled activity are used for
    training and for window scores; the nudge simulation runs over everything.
  - CALIBRATION: each session is rotated so its calibration pose (hands resting on the keyboard) points the
    watch face up. The pose comes from the session's calibration.json (written when the runner or the live
    page calibrates), or, for older sessions, from the first 5 s of its typing activity.
  - Features add WRIST ANGLES and 10 s / 30 s CONTEXT (features.py, version 2 configs).
  - A SHORT candidate list (2 windows x 2 feature sets x 3 models), so the choice has little room to
    overfit a handful of sessions.

What it does, in order:
  1. Finds every labelled session in ../Data Collection/sessions. The session name decides its role:
        S1, S2, ...  training        TEST  final score only        NATURAL, OTHERHAND  extra tests
     A role.txt in the session folder overrides the name (e.g. "test" for a session recorded as S2 that is
     kept aside because it comes from a person the model has never seen).
  2. Leave-one-session-out over the training sessions for every candidate: train on all but one, predict
     the held-out stream, score phone-vs-work. Whole sessions are held out because windows from the same
     sitting look alike and a random split gives flattering numbers.
  3. Picks the best balanced on/off accuracy, then fewest false alarms, then activity macro-F1 (rank()).
     Then picks the live smoothing length by fewest simulated false nudges. TEST plays no part.
  4. Retrains the choice on all training sessions and scores it ONCE on TEST, next to an ablation (what each
     part adds) and the reference paper's setting. Also NATURAL and OTHERHAND if recorded.
  5. Writes results/<time>/ (report.md, all_combinations.csv, figures), copies the report to REPORT.md, and
     saves ../WristStream/model.pkl so the live page uses the new model.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baseline as B  # noqa: E402
import features as F  # noqa: E402

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
SESSIONS = os.path.join(HERE, "..", "Data Collection", "sessions")
SEGMENT = os.path.join(HERE, "..", "Data Collection", "segment.py")
DEPLOY = os.path.join(HERE, "..", "WristStream", "model.pkl")

HZ = F.HZ
HOP = 0.5            # the live page predicts every 0.5 s
TRAIN_EVERY = 2      # training uses every 2nd window (1 s apart): half-overlapping windows add little
LABEL_SHARE = 0.9    # a window belongs to an activity when 90% of its samples do
STREAM_GAP_S = 3.0   # the receiver forgets its history after a gap this long; so do we
NUDGE_S = 10         # same as NUDGE_S in WristStream/receiver.py
WATCH_NUDGE_S = 120  # same as WATCH_NUDGE_S in WristStream/receiver.py: the watch buzz
NEAR_PHONE_S = 15    # a nudge is correct when labelled phone use lies within this many seconds of it
SMOOTH = [1, 5, 9]   # average the last k predictions (k x 0.5 s)

MODELS = {
    "Extra trees": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    "Random forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000,
                                                                                      class_weight="balanced")),
    "Naive Bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
}
CANDIDATE_MODELS = ["Extra trees", "Random forest", "Logistic regression"]
WINDOWS = [2.0, 4.0]
FULL = {"sensors": "all", "freq": True, "orient": True, "context": True, "calibrated": True}
FEATURE_SETS = {
    "motion only": {"sensors": "all", "freq": True},
    "motion + wrist angle + context, calibrated": FULL,
}
ABLATION = {   # reported on TEST for the chosen model and window, never used to choose
    "motion only (version 1 features)": {"sensors": "all", "freq": True},
    "+ wrist angle + context, not calibrated": {"sensors": "all", "freq": True, "orient": True, "context": True},
    "+ calibration (full)": FULL,
}
REFERENCE = ("Naive Bayes", {"sensors": "acc", "freq": False})   # stand-in for the reference paper's setting
# The INSTANT model, shown under the main prediction on the live pages: the same classifier on a short window
# with no 10/30 s context and no averaging, so it reacts within about two seconds (a sip, a switch of activity).
# The main model stays the decision the nudge uses; this one shows what is happening right now.
INSTANT = {"window_s": 2.0, "sensors": "all", "freq": True, "orient": True, "calibrated": True}
DEPLOY_INSTANT = os.path.join(HERE, "..", "WristStream", "model_instant.pkl")


def role_of(name):
    n = str(name).upper()
    if n == "TEST" or n.startswith("TEST"):
        return "test"
    if n.startswith("NATURAL"):
        return "natural"
    if n.startswith("OTHERHAND"):
        return "otherhand"
    if (len(n) >= 2 and n[0] == "S" and n[1:].isdigit()) or n.startswith("STUDY"):
        return "train"
    return "ignored"


# ------------------------------------------------------------------ data

def session_calibration(folder, df):
    """(rotation, how it was obtained). Prefers the recorded calibration step."""
    path = os.path.join(folder, "calibration.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            g = json.load(f)["gravity"]
        return F.calibration_rotation([g[0]], [g[1]], [g[2]]), "calibration step"
    typing = np.where(df.label.to_numpy() == "typing")[0]
    if len(typing):
        a = typing[0] + 3 * HZ
        p = df.iloc[a:a + 5 * HZ]
        return F.calibration_rotation(p.gravityX, p.gravityY, p.gravityZ), "first 5 s of typing (no calibration step)"
    return np.eye(3), "none (no calibration step and no typing)"


# Moving the watch to the other wrist mirrors the motion across the body's midline: the x axis of every
# direction (movement, gravity) flips, and so do the y and z axes of rotation. Training on mirrored copies of
# every session lets one model serve either wrist (tested 22 Nov: no cost on the left wrist, held-out sessions
# mirrored to the right went from 85% to 95% balanced).
MIRROR_FLIP = ("userAccelerationX", "gravityX", "accX", "rotationRateY", "rotationRateZ")


def mirrored(sess):
    """A copy of a session as if the watch had been on the other wrist, recalibrated on the mirrored pose."""
    df = sess["df"].copy()
    for c in MIRROR_FLIP:
        if c in df:
            df[c] = -df[c]
    R, how = session_calibration(os.path.join(SESSIONS, sess["folder"]), df)
    if how == "calibration step":                      # the recorded pose, mirrored the same way
        R = np.diag([-1.0, 1.0, 1.0]) @ R @ np.diag([-1.0, 1.0, 1.0])
    return {**sess, "df": df, "R": R, "folder": sess["folder"] + " (mirrored)"}


def load_sessions(root):
    out = []
    need = set(F.V2_COLUMNS) | set(F.SENSORS["acc"])
    for d in sorted(os.listdir(root)):
        folder = os.path.join(root, d)
        if d.startswith(("_", ".")) or not os.path.isdir(folder):
            continue
        lab = os.path.join(folder, "labelled.csv")
        has_runner = any(f.endswith("_runner.csv") for f in os.listdir(folder))
        if not os.path.exists(lab) and has_runner and os.path.exists(SEGMENT):
            print(f"labelling {d} ...")
            subprocess.run([sys.executable, SEGMENT, folder], capture_output=True, text=True)
        if not os.path.exists(lab):
            continue
        df = pd.read_csv(lab)
        if not need <= set(df.columns) or "block" not in df.columns:
            print(f"skipping {d}: not a Galaxy recording with block labels")
            continue
        labelled = df[df.label != "none"]
        if labelled.empty:
            continue
        name = str(labelled.session.iloc[0])
        R, how = session_calibration(folder, df)
        role = role_of(name)
        role_file = os.path.join(folder, "role.txt")
        if os.path.exists(role_file):
            with open(role_file, encoding="utf-8") as f:
                role = f.read().split()[0].strip().lower()
        out.append({"folder": d, "name": name, "role": role,
                    "participant": str(labelled.participant.iloc[0]) if "participant" in df else "?",
                    "df": df, "R": R, "calibration": how})
    return out


def window_ends(t, window_s, hop=HOP):
    """Indices where a window ends, every hop seconds, never across a time gap over 0.5 s."""
    n = int(round(window_s * HZ))
    gap = np.r_[True, np.diff(t) > 0.5]
    seg_start = np.maximum.accumulate(np.where(gap, np.arange(len(t)), 0))
    ends, next_t = [], -1e18
    for i in range(n - 1, len(t)):
        if i - seg_start[i] + 1 >= n and t[i] >= next_t:
            ends.append(i)
            next_t = t[i] + hop - 1e-6
    return np.array(ends, dtype=int)


def onoff(labels):
    """1 = on task, 0 = off task (phone), -1 = no label (transition or unlabelled time)."""
    return np.array([-1 if l is None else (0 if F.is_on_task(l) is False else 1) for l in labels])


def table(sess, config, groups=False):
    """Every window of one continuous session: features, label the model learns (the activity's group, see
    features.GROUPS, or the activity itself with groups=False), the recorded activity, on/off, block, time."""
    df = sess["df"]
    t = df.t.to_numpy(float)
    cols = sorted(set(F.V2_COLUMNS) | set(F.SENSORS[config["sensors"]]))
    arr = {c: df[c].to_numpy(float) for c in cols}
    lab, blk = df.label.to_numpy(object), df.block.to_numpy(int)
    n = int(round(config["window_s"] * HZ))
    hist = int(round(F.history_seconds(config) * HZ))
    big_gap = np.r_[True, np.diff(t) > STREAM_GAP_S]
    hist_start = np.maximum.accumulate(np.where(big_gap, np.arange(len(t)), 0))
    X, labels, blocks = [], [], []
    ends = window_ends(t, config["window_s"])
    for e in ends:
        lo = max(hist_start[e], e - hist + 1)
        X.append(F.features({c: a[lo:e + 1] for c, a in arr.items()}, config, sess["R"]))
        b = blk[e - n + 1:e + 1]
        vals, cnt = np.unique(b, return_counts=True)
        k = cnt.argmax()
        if vals[k] >= 0 and cnt[k] >= LABEL_SHARE * n:
            labels.append(lab[e - n + 1 + int(np.argmax(b == vals[k]))]); blocks.append(int(vals[k]))
        else:
            labels.append(None); blocks.append(-1)
    activity = np.array(labels, dtype=object)
    labels = np.array([F.group(l) if groups and l is not None else l for l in labels], dtype=object)
    return {"X": np.nan_to_num(np.array(X, float)), "label": labels, "activity": activity, "truth": onoff(labels),
            "block": np.array(blocks), "t": t[ends]}


# ------------------------------------------------------------------ scoring

def smooth_mean(p, k):
    """Causal average of the last k predictions, as the live page does."""
    if k <= 1:
        return p
    c = np.cumsum(np.r_[0, p]); i = np.arange(len(p)); lo = np.maximum(0, i + 1 - k)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def nudges(off, t, nudge_s=NUDGE_S):
    """Indices where a nudge fires: off task for nudge_s seconds in a row, once per unbroken stretch."""
    fired, start, done = [], None, False
    for i, o in enumerate(off):
        if not o or (i > 0 and t[i] - t[i - 1] > 2 * HOP + 0.25):
            start, done = None, False
        if o:
            if start is None:
                start = t[i]
            if not done and t[i] - start >= nudge_s:
                fired.append(i); done = True
    return np.array(fired, dtype=int)


def false_count(fired, truth, t):
    """Nudges with no labelled phone use within NEAR_PHONE_S of them. This includes nudges that fire while
    switching between activities (unlabelled time), which a person feels just the same."""
    phone_t = t[truth == 0]
    return int(sum(not (np.abs(phone_t - t[i]) <= NEAR_PHONE_S).any() for i in fired))


def score(truth, off, t, blocks):
    """Phone-vs-work scores over labelled windows, plus the simulated nudge and watch buzz."""
    on_m, off_m = truth == 1, truth == 0
    caught = float(off[off_m].mean()) if off_m.any() else np.nan
    fa = float(off[on_m].mean()) if on_m.any() else np.nan
    fired = nudges(off, t)
    false_n = false_count(fired, truth, t)
    false_b = false_count(nudges(off, t, WATCH_NUDGE_S), truth, t)
    on_hours = (truth != 0).sum() * HOP / 3600          # all time that is not phone use, transitions included
    phone_blocks = np.unique(blocks[off_m])
    delays, hit = [], 0
    for b in phone_blocks:
        idx = np.where(blocks == b)[0]
        inside = [i for i in fired if idx[0] <= i <= idx[-1] + int(5 / HOP)]
        j = idx[0]                                      # phone straight after phone: one nudge covers both
        while j > 0 and off[j - 1] and t[j] - t[j - 1] <= 2 * HOP + 0.25:
            j -= 1
        earlier = [i for i in fired if j <= i < idx[0]]
        if inside or earlier:
            hit += 1; delays.append(0.0 if earlier else t[inside[0]] - t[idx[0]])
    lab = truth >= 0
    return {"balanced": float(np.nanmean([caught, 1 - fa])), "caught": caught, "false_alarm": fa,
            "onoff": float((off[lab] == (truth[lab] == 0)).mean()) if lab.any() else np.nan,
            "false_nudges_per_h": false_n / on_hours if on_hours else np.nan, "false_nudges": false_n,
            "false_buzzes": false_b, "false_buzzes_per_h": false_b / on_hours if on_hours else np.nan,
            "on_task_min": on_m.sum() * HOP / 60,
            "phone_nudged": hit / len(phone_blocks) if len(phone_blocks) else np.nan,
            "median_delay_s": float(np.median(delays)) if delays else np.nan}


def p_off(model, X):
    P = model.predict_proba(X)
    cols = [i for i, c in enumerate(model.classes_) if F.is_on_task(c) is False]
    return P[:, cols].sum(1), P


def fit(model_name, tables):
    X = np.vstack([tb["X"][::TRAIN_EVERY][tb["label"][::TRAIN_EVERY] != None] for tb in tables])  # noqa: E711
    y = np.concatenate([tb["label"][::TRAIN_EVERY][tb["label"][::TRAIN_EVERY] != None] for tb in tables])  # noqa: E711
    m = (MODELS[model_name] if isinstance(model_name, str) else model_name)(); m.fit(X, y)
    return m


def halves(tb):
    """Fallback with one training session: its first and second half, by time."""
    mid = np.median(tb["t"])
    return [{k: v[tb["t"] < mid] for k, v in tb.items()}, {k: v[tb["t"] >= mid] for k, v in tb.items()}]


def cross_validate(model_name, tables, extra=None, groups=None):
    """Leave-one-session-out, or leave-one-PARTICIPANT-out when `groups` (one participant per table) is given:
    every session of the held-out person is scored by a model that never saw that person.
    extra: tables aligned with `tables` (the mirrored copies) that join training, except the held-out ones'.
    Returns one dict per held-out session with p_off and truth for later scoring."""
    folds = []
    if len(tables) < 2:
        tables, extra, groups = halves(tables[0]), None, None
    extra = extra or []
    if groups is None:
        groups = list(range(len(tables)))
    for k, te in enumerate(tables):
        m = fit(model_name, [tb for j, tb in enumerate(tables) if groups[j] != groups[k]]
                + [tb for j, tb in enumerate(extra) if groups[j] != groups[k]])
        p, P = p_off(m, te["X"])
        lab = te["label"] != None  # noqa: E711
        pred = m.classes_[P.argmax(1)]
        folds.append({"p": p, "truth": te["truth"], "t": te["t"], "block": te["block"],
                      "macro_f1": f1_score(te["label"][lab].astype(str), pred[lab], average="macro"),
                      "activity_acc": accuracy_score(te["label"][lab].astype(str), pred[lab])})
    return folds


def summarize(folds, k=1, threshold=0.5):
    s = [score(f["truth"], smooth_mean(f["p"], k) >= threshold, f["t"], f["block"]) for f in folds]
    out = {key: float(np.nanmean([x[key] for x in s])) for key in s[0]}
    out["false_nudges"] = int(sum(x["false_nudges"] for x in s))
    out["false_buzzes"] = int(sum(x["false_buzzes"] for x in s))
    out["macro_f1"] = float(np.mean([f["macro_f1"] for f in folds]))
    out["activity_acc"] = float(np.mean([f["activity_acc"] for f in folds]))
    out["per_held_out"] = [round(x["balanced"], 3) for x in s]
    return out


def rank(res):
    """Best balanced on/off accuracy (rounded: the 4th decimal doesn't decide), then fewest false alarms
    (these become false nudges), then activity macro-F1."""
    key = res.assign(_b=res.cv_balanced.round(3), _fa=res.cv_false_alarm.round(3))
    order = key.sort_values(["_b", "_fa", "cv_macro_f1"], ascending=[False, True, False]).index
    return res.loc[order].reset_index(drop=True)


# ------------------------------------------------------------------ figures and tables

def plot_confusion(y, p, labels, path, title, cols=None):
    """Rows: actual (labels); columns: predicted (cols, default the same labels)."""
    cols = labels if cols is None else cols
    cm = pd.crosstab(pd.Categorical(y, labels), pd.Categorical(p, cols), dropna=False).to_numpy()
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(2.2 + 0.75 * len(cols), 0.8 + 0.6 * len(labels)))
    ax.imshow(pct, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(cols)):
            if cm[i, j]:
                ax.text(j, i, f"{pct[i, j] * 100:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_timeline(t, truth, p, threshold, fired, path, title):
    """The held-out stream as the live page would have shown it: truth, P(phone), nudges."""
    fig, ax = plt.subplots(2, 1, figsize=(12, 3.2), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    m = t / 60
    col = np.where(truth == 1, "#2f6b3a", np.where(truth == 0, "#b3261e", "#cccccc"))
    ax[0].bar(m, np.ones(len(m)), width=HOP / 60 * 1.05, color=col)
    ax[0].set_yticks([]); ax[0].set_ylabel("actual", rotation=0, ha="right", va="center")
    ax[1].plot(m, p, color="#222", lw=0.8); ax[1].axhline(threshold, color="#888", ls=":", lw=0.8)
    for i in fired:
        ax[1].axvline(m[i], color="#b3261e", lw=1.2)
    ax[1].set_ylim(-0.02, 1.02); ax[1].set_ylabel("P(phone)")
    ax[1].set_xlabel("minutes   (top: green = work, red = phone, grey = transition;  red lines = nudges)")
    ax[0].set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_baselines(names, cv, test, path):
    """Balanced phone-vs-work accuracy per method: held-out training sessions vs the TEST session."""
    ink, muted, grid = "#2b2b2a", "#6b6a64", "#e4e3dd"
    series = [("held-out training sessions (mean)", cv, "#2a78d6"), ("TEST: unseen person", test, "#eb6834")]
    n, h = len(names), 0.36
    fig, ax = plt.subplots(figsize=(8, 0.75 * n + 1.1))
    y = np.arange(n)[::-1]
    for j, (lab, vals, col) in enumerate(series):
        yy = y + (0.5 - j) * h
        ax.barh(yy, vals, height=h - 0.04, color=col, label=lab)
        for v, yv in zip(vals, yy):
            ax.text(v + 0.01, yv, f"{v:.0%}", va="center", fontsize=9, color=ink)
    ax.axvline(0.5, color=muted, lw=0.8, ls=":")
    ax.text(0.505, y[-1] - 0.6, "chance", fontsize=8, color=muted, ha="left")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9, color=ink)
    ax.set_xlim(0, 1.08); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.tick_params(colors=muted, length=0); ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("balanced phone-vs-work accuracy", color=muted, fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.3f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


LOOKALIKES = ["drinking", "hold_cup", "calculator", "other_desk", "phone_call", "phone_text", "phone_scroll"]


def flagged_by_activity(ps, tables, k=1):
    """Share of each activity's windows flagged as phone (after the live smoothing)."""
    flag = np.concatenate([smooth_mean(p, k) >= 0.5 for p in ps])
    lab = np.concatenate([tb["activity"] for tb in tables])
    return {a: float(flag[lab == a].mean()) for a in LOOKALIKES if (lab == a).any()}


def pct(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=SESSIONS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--quick", action="store_true", help="4 s windows, extra trees only")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--deploy-to", default=DEPLOY, help="where to save the live model (testing)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--test-participant", default="", help="comma-separated participant codes set aside as the "
                    "test set: all their sessions, never used for any choice")
    ap.add_argument("--by-session", action="store_true", help="hold out sessions, not participants")
    ap.add_argument("--deploy-all", action="store_true", help="after scoring, refit the chosen model on all "
                    "sessions, test and NATURAL included, and deploy that (more people in the live model; no score "
                    "for it)")
    ap.add_argument("--no-mirror", action="store_true", help="don't add mirrored (other-wrist) copies to training")
    ap.add_argument("--train-on-groups", action="store_true",
                    help="learn the classes in features.GROUPS directly instead of every recorded activity "
                         "(tried and dropped on 22 Nov; kept to reproduce the comparison)")
    args = ap.parse_args()
    t_start = time.time()
    notes = []

    sessions = load_sessions(args.sessions)
    if not sessions:
        sys.exit("No labelled sessions found. Record with the runner (http://localhost:8000/runner) first.")
    inv = pd.DataFrame([{"folder": s["folder"], "session": s["name"], "participant": s["participant"],
                         "role": s["role"], "labelled min": round((s["df"].label != "none").sum() / HZ / 60, 1),
                         "activities": s["df"].label[s["df"].label != "none"].nunique(),
                         "calibration": s["calibration"]} for s in sessions])
    print(inv.to_string(index=False))

    train = [s for s in sessions if s["role"] == "train"]
    test = [s for s in sessions if s["role"] == "test"]
    natural = [s for s in sessions if s["role"] == "natural"]
    otherhand = [s for s in sessions if s["role"] == "otherhand"]
    held = [c.strip() for c in args.test_participant.split(",") if c.strip()]
    if held:
        moved = [s for s in train if s["participant"] in held]
        if not moved:
            sys.exit(f"--test-participant: no training session belongs to {held}")
        train = [s for s in train if s["participant"] not in held]
        test += moved
        notes.append(f"Test set: every session of participant(s) {', '.join(held)}, set aside before any choice. "
                     "The model never saw these people.")
    if not train:
        sys.exit("No training sessions (S1, S2, ... or STUDY) found.")
    if not test and len(train) >= 3 and not held:
        test = [train.pop()]
        notes.append(f"No TEST session was recorded, so the last training session ({test[0]['name']}, "
                     f"{test[0]['folder']}) was set aside as the test set.")
    if len(train) < 2:
        notes.append("Only ONE training session, so candidates were compared by training on one half of it and "
                     "scoring the other half. These scores are optimistic: the same sitting and watch placement.")
    if not test:
        notes.append("No test session: the final score below is the cross-validation score, not a held-out test.")
    if any(s["calibration"] != "calibration step" for s in sessions if s["role"] != "ignored"):
        notes.append("Some sessions have no recorded calibration step; their calibration pose was taken from the "
                     "first 5 s of typing, which is the same pose (hands on the keyboard). See the table below.")
    participants = sorted({s["participant"] for s in train})
    by_person = len(participants) >= 3 and not args.by_session
    cv_groups = [s["participant"] for s in train] if by_person else None
    if by_person:
        notes.append(f"Candidates compared by leaving one PARTICIPANT out at a time ({len(participants)} people): "
                     "every score is on a person the model never saw.")
    if len(participants) < 3:
        notes.append(f"Training covers {len(participants)} participant(s) ({', '.join(participants)}). "
                     "Accuracy on a new person is the open question; record more people to measure it.")
    classes = sorted({F.group(l) for s in train for l in s["df"].label.unique() if l != "none"})

    windows_ = [4.0] if args.quick else WINDOWS
    models_ = ["Extra trees"] if args.quick else CANDIDATE_MODELS
    train_m = [] if args.no_mirror else [mirrored(s) for s in train]
    test_m = [] if args.no_mirror else [mirrored(s) for s in test]
    everything = train + test + natural + otherhand + train_m + test_m
    where = {id(s): i for i, s in enumerate(everything)}

    # ---- features: one table per (session, window, feature set), computed in parallel
    configs = {(w, fs): {"window_s": w, **cfg} for w in windows_ for fs, cfg in FEATURE_SETS.items()}
    configs.update({(w, f"ablation:{a}"): {"window_s": w, **cfg} for w in windows_ for a, cfg in ABLATION.items()})
    configs.update({(w, "reference"): {"window_s": w, **REFERENCE[1]} for w in windows_})
    configs[(INSTANT["window_s"], "instant")] = dict(INSTANT)
    configs.update({(w, f"baseline:{b}"): {"window_s": w, **c} for w in windows_ for b, (c, _) in B.RULES.items()})
    uniq = {}
    for key, cfg in configs.items():
        uniq.setdefault(json.dumps(cfg, sort_keys=True), cfg)
    print(f"\ncomputing features: {len(uniq)} feature settings x {len(everything)} sessions ...")
    jobs = [(ck, i) for ck in uniq for i in range(len(everything))]
    groups = args.train_on_groups
    done = Parallel(n_jobs=args.jobs)(delayed(table)(everything[i], uniq[ck], groups) for ck, i in jobs)
    TB = {(ck, i): tb for (ck, i), tb in zip(jobs, done)}

    def tabs(cfg, group):
        ck = json.dumps(cfg, sort_keys=True)
        return [TB[(ck, where[id(s)])] for s in group]

    def trn(cfg):
        """Training tables: every training session and, unless --no-mirror, its mirrored copy."""
        return tabs(cfg, train) + tabs(cfg, train_m)

    # ---- candidates, leave-one-session-out
    rows, cv_folds = [], {}
    for w in windows_:
        for fs, cfg in FEATURE_SETS.items():
            for mname in models_:
                folds = cross_validate(mname, tabs(configs[(w, fs)], train), tabs(configs[(w, fs)], train_m),
                                       groups=cv_groups)
                s = summarize(folds)
                cv_folds[(w, fs, mname)] = folds
                rows.append({"window_s": w, "features": fs, "model": mname, "cv_balanced": s["balanced"],
                             "cv_false_alarm": s["false_alarm"], "cv_phone_caught": s["caught"],
                             "cv_macro_f1": s["macro_f1"], "cv_activity_acc": s["activity_acc"],
                             "cv_false_nudges": s["false_nudges"], "per_held_out": s["per_held_out"]})
                print(f"  {w:g} s | {fs:45s} | {mname:20s} balanced {s['balanced']:.3f}  "
                      f"false alarms {s['false_alarm']:.3f}  per held-out {s['per_held_out']}", flush=True)
    res = rank(pd.DataFrame(rows))
    best = res.iloc[0]
    cfg = configs[(best.window_s, best.features)]

    # ---- smoothing: fewest false nudges, then best balanced, then the shortest (fastest) average
    sm = {k: summarize(cv_folds[(best.window_s, best.features, best.model)], k) for k in SMOOTH}
    smooth_n = min(SMOOTH, key=lambda k: (sm[k]["false_nudges"], -round(sm[k]["balanced"], 3), k))

    # ---- the proposal's baseline: one threshold on the wrist, fitted on training windows only
    w = best.window_s
    base_cv = [(f"**Chosen model** ({best.model}, smoothed over {smooth_n})",
                summarize(cv_folds[(w, best.features, best.model)], smooth_n))]
    rules = {}
    for b, (bcfg, feat) in B.RULES.items():
        c = configs[(w, f"baseline:{b}")]
        rules[b] = (c, B.make(c, feat), feat)
        base_cv.append((b, summarize(cross_validate(rules[b][1], tabs(c, train), tabs(c, train_m), groups=cv_groups))))
    rc = configs[(w, "reference")]
    base_cv.append(("Reference-paper setting (raw accelerometer, Naive Bayes)",
                    summarize(cross_validate(REFERENCE[0], tabs(rc, train), tabs(rc, train_m), groups=cv_groups))))

    # ---- final model
    final = fit(best.model, trn(cfg))
    stamp = time.strftime("%Y%m%d-%H%M")
    out = os.path.join(args.out, stamp); os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "all_combinations.csv"), index=False)

    lines, R = [], {}
    if test:
        te = tabs(cfg, test)
        ts = []
        for tb in te:
            p, P = p_off(final, tb["X"])
            ts.append((tb, p, P))
        y = np.concatenate([tb["label"] for tb, _, _ in ts])
        ya = np.concatenate([tb["activity"] for tb, _, _ in ts])
        P = np.vstack([P for _, _, P in ts])
        lab = y != None  # noqa: E711
        pred = final.classes_[P.argmax(1)]
        al = ya[lab].astype(str)
        yl = np.array([F.group(a) for a in al]); pl = np.array([F.group(x) for x in pred[lab]])   # reported classes
        test_acc, test_f1 = accuracy_score(yl, pl), f1_score(yl, pl, average="macro")

        def test_score(model, tables, k):
            out_ = []
            for tb in tables:
                pp, _ = p_off(model, tb["X"])
                out_.append(score(tb["truth"], smooth_mean(pp, k) >= 0.5, tb["t"], tb["block"]))
            agg = {key: float(np.nanmean([o[key] for o in out_])) for key in out_[0]}
            agg["false_nudges"] = int(sum(o["false_nudges"] for o in out_))
            agg["false_buzzes"] = int(sum(o["false_buzzes"] for o in out_))
            return agg

        main_raw, main_sm = test_score(final, te, 1), test_score(final, te, smooth_n)
        right = test_score(final, tabs(cfg, test_m), smooth_n) if test_m else None
        icfg = configs[(INSTANT["window_s"], "instant")]
        inst_model = fit(best.model, trn(icfg))
        inst = test_score(inst_model, tabs(icfg, test), 1)
        it = tabs(icfg, test)
        ia = np.concatenate([tb["activity"] for tb in it]); il = ia != None  # noqa: E711
        ip = np.array([F.group(x) for x in inst_model.classes_[np.vstack([p_off(inst_model, tb["X"])[1] for tb in it]).argmax(1)]])
        inst_rec = {a: float((ip[il][ia[il] == a] == F.group(a)).mean()) for a in sorted(set(ia[il]))}
        R.update(test_acc=test_acc, test_bin=main_sm["balanced"])
        abl_rows = []
        for a, acfg in ABLATION.items():
            c = configs[(best.window_s, f"ablation:{a}")]
            m = fit(best.model, trn(c))
            abl_rows.append((a, test_score(m, tabs(c, test), smooth_n)))
        ref_model = fit(REFERENCE[0], trn(rc))
        ref = test_score(ref_model, tabs(rc, test), 1)
        rule_rows, look = [], {}
        look[f"Chosen model ({best.model})"] = flagged_by_activity([p for _, p, _ in ts], te, smooth_n)
        for b, (c, factory, feat) in rules.items():
            m = fit(factory, trn(c))
            rule_rows.append((f"{b}: {m.describe(feat)}", test_score(m, tabs(c, test), 1)))
            look[b] = flagged_by_activity([p_off(m, tb["X"])[0] for tb in tabs(c, test)], tabs(c, test))
        look["Reference-paper setting"] = flagged_by_activity([p_off(ref_model, tb["X"])[0] for tb in tabs(rc, test)],
                                                              tabs(rc, test))
        look_df = pd.DataFrame(look).T.reset_index().rename(columns={"index": "model"})
        cv_by = dict(base_cv)
        short = [(f"Classifier ({best.model})", base_cv[0][1], main_sm)] + [
            (b.replace(" (calibrated)", ""), cv_by[b], s_) for b, (_, s_) in zip(rules, rule_rows)] + [
            ("Reference paper setting (Naive Bayes)", base_cv[-1][1], ref)]
        plot_baselines([n_ for n_, _, _ in short], [c["balanced"] for _, c, _ in short],
                       [t_["balanced"] for _, _, t_ in short], os.path.join(out, "baselines.png"))

        def row(name, s):
            return (f"| {name} | **{pct(s['balanced'])}** | {pct(s['caught'])} | {pct(s['false_alarm'])} | "
                    f"{s['false_nudges']} ({s['false_nudges_per_h']:.1f}/h) | "
                    f"{s['false_buzzes']} ({s['false_buzzes_per_h']:.1f}/h) | {pct(s['phone_nudged'])} | "
                    f"{s['median_delay_s']:.0f} s |")
        head_row = ("| | Balanced on/off | Phone caught | False alarms | False nudges (10 s) | False watch buzzes "
                    "(2 min) | Phone activities nudged | Median nudge delay |\n|---|---|---|---|---|---|---|---|")
        labs = sorted(set(yl) | set(pl))
        acts = sorted(set(al))
        plot_confusion(al, pl, acts, os.path.join(out, "confusion_test.png"),
                       "TEST: % of each recorded activity per predicted class", cols=sorted(set(pl) | set(yl)))
        flag = smooth_mean(np.concatenate([p for _, p, _ in ts]), smooth_n)[lab] >= 0.5
        per = pd.DataFrame({"activity": acts, "reported class": [F.group(l) for l in acts],
                            "flagged as phone": [float(flag[al == l].mean()) for l in acts],
                            "recognised as its class": [float((pl[al == l] == F.group(l)).mean()) for l in acts],
                            "windows": [int((al == l).sum()) for l in acts]})
        tb0, p0, _ = ts[0]
        ps0 = smooth_mean(p0, smooth_n)
        plot_timeline(tb0["t"], tb0["truth"], ps0, 0.5, nudges(ps0 >= 0.5, tb0["t"]), os.path.join(out, "timeline_test.png"),
                      f"TEST ({test[0]['name']}): the held-out session as the live page would have seen it")
        test_names = ", ".join("{} ({}, {})".format(s["name"], s["participant"], s["folder"]) for s in test)
        lines += [
            "## 2. The final score (TEST, never used for any choice)", "",
            f"TEST: {test_names}. Scored on the continuous recording, every {HOP:g} s, as the live page runs.", "",
            head_row,
            row(f"**Chosen model**, smoothed over {smooth_n} predictions as on the live page", main_sm),
            row("Chosen model, each prediction on its own", main_raw)] + (
            [row("Chosen model, watch on the other wrist (TEST mirrored, simulated)", right)] if right else []) + [
            row(f"Instant model ({INSTANT['window_s']:g} s window, no context, no averaging), shown under the main "
                "prediction", inst)] + [
            row("Reference-paper setting: raw accelerometer, simple features, Naive Bayes", ref)] + [
            row(n, s_) for n, s_ in rule_rows] + ["",
            "Instant model, share of each activity recognised as its class on TEST: " + ", ".join(
                f"{a} {v:.0%}" for a, v in inst_rec.items()) + ". It is not used for the nudge.", "",
            "The threshold rows are the proposal's baseline: one number per window and one cut-off, fitted on the "
            "training sessions (`Model/baseline.py`).", "", "![baselines](baselines.png)", "",
            f"Classes ({', '.join(labs)}): accuracy {test_acc:.1%}, macro-F1 {test_f1:.3f}. "
            + "The model is trained on every recorded activity; typing, writing, reading and the calculator are "
            "reported as one class, desk work, and the phone activities as phone, because one wrist cannot tell "
            "them apart and they need the same response (features.GROUPS). The per-activity table below scores "
            "each recorded activity against its class.", "",
            "**What each part adds** (same model and window, retrained without the part, scored on TEST):", "",
            head_row] + [row(a, s) for a, s in abl_rows] + [
            "", "![timeline](timeline_test.png)", "", "![confusion](confusion_test.png)", "",
            "Per activity on TEST:", "", md_table(per), "",
            "**The one challenge: look-alikes** (proposal). Share of each activity's windows flagged as phone on "
            "TEST. Drinking, holding a cup and the calculator are work and should stay near 0; the phone rows "
            "should be near 1.", "", md_table(look_df), "",
            "*Balanced on/off* is the mean of *phone caught* (phone windows flagged) and work time left alone "
            "(1 − *false alarms*). A *nudge* fires after "
            f"{NUDGE_S} s flagged as phone in a row (the watch buzzes after {WATCH_NUDGE_S} s); it is false when no "
            f"labelled phone use lies within {NEAR_PHONE_S} s of it, so nudges while switching between activities "
            "count as false. Per hour means per hour of time not on the phone. The delay counts from the start of "
            "the labelled phone activity, which begins 3 s after the runner click.", ""]
    else:
        s = summarize(cv_folds[(best.window_s, best.features, best.model)], smooth_n)
        lines += ["## 2. The final score", "", "No test session. Best cross-validation result (held-out sessions): "
                  f"balanced on/off {pct(s['balanced'])}, false alarms {pct(s['false_alarm'])}, "
                  f"false nudges {s['false_nudges']} ({s['false_nudges_per_h']:.1f}/h).", ""]

    for group, title, text in ((natural, "## 3. NATURAL (unscripted work and phone use)",
                                "Closest to real use. The phone may have been in either hand."),
                               (otherhand, "## 4. OTHERHAND (phone in the hand without the watch)",
                                "The expected blind spot of one wrist: the watch hand rests, which looks like work.")):
        if group:
            gt = tabs(cfg, group)
            s = summarize([{"p": p_off(final, tb["X"])[0], "truth": tb["truth"], "t": tb["t"], "block": tb["block"],
                            "macro_f1": np.nan, "activity_acc": np.nan} for tb in gt], smooth_n)
            who = ", ".join(f"{g['name']} ({g['participant']}, {g['folder']})" for g in group)
            lines += [title, "", text, f"Sessions: {who}.", "",
                      f"Phone caught {pct(s['caught'])}, false alarms {pct(s['false_alarm'])}, "
                      f"false nudges {s['false_nudges']} ({s['false_nudges_per_h']:.1f}/h), false watch buzzes "
                      f"{s['false_buzzes']}, phone activities nudged {pct(s['phone_nudged'])}, median nudge delay "
                      f"{s['median_delay_s']:.0f} s, over {s['on_task_min']:.1f} min not on the phone.", ""]
            fig = "timeline_" + title.split()[2].lower() + ".png"
            ps = smooth_mean(p_off(final, gt[0]["X"])[0], smooth_n)
            plot_timeline(gt[0]["t"], gt[0]["truth"], ps, 0.5, nudges(ps >= 0.5, gt[0]["t"]), os.path.join(out, fig),
                          f"{group[0]['name']} ({group[0]['participant']}): unscripted work and phone use, as the live page saw it")
            lines += [f"![{fig}]({fig})", ""]

    show = ["window_s", "features", "model", "cv_balanced", "cv_false_alarm", "cv_phone_caught", "cv_false_nudges",
            "cv_macro_f1"]
    imp = ""
    if hasattr(final, "feature_importances_"):
        names = F.feature_names(cfg)
        top = np.argsort(final.feature_importances_)[::-1][:12]
        imp = ("Features the model leans on most: " + ", ".join(f"`{names[i]}`" for i in top) + ".\n")
    how = "one half of the single training session at a time" if len(train) < 2 else "one whole training session at a time"
    if by_person:
        how = "one participant (all their sessions) at a time"
    head = [f"# Focus Police model report, {time.strftime('%d %b %Y %H:%M')}", "",
            "Generated by `Model/train.py` (version 2). Everything below is measured on the recordings in "
            "`Data Collection/sessions/`. Why version 2 looks the way it does: `Model/lab/FINDINGS.md`.", ""]
    if notes:
        head += ["**Read this first:**", ""] + [f"- {n}" for n in notes] + [""]
    head += [
        "## 1. What was chosen", "",
        f"- **Model:** {best.model}",
        f"- **Features:** {best.features}; {cfg['window_s']:g} s window ({len(F.feature_names(cfg))} numbers)",
        f"- **Live smoothing:** average of the last {smooth_n} predictions "
        "(held-out false nudges for 1 / 5 / 9: " + " / ".join(str(sm[k]["false_nudges"]) for k in SMOOTH) + ")",
        "- **Selection rule:** best balanced on/off accuracy, then fewest false alarms, then activity macro-F1, "
        f"all from holding out {how}.",
        f"- Held-out score of the choice: balanced {pct(best.cv_balanced)} (phone caught {pct(best.cv_phone_caught)}, "
        f"false alarms {pct(best.cv_false_alarm)}); per held-out session {best.per_held_out}.",
        f"- {len(res)} candidates compared in {time.time() - t_start:.0f} s.", "",
        "Sessions:", "", md_table(inv), ""]
    def cv_row(name, s_):
        return (f"| {name} | **{pct(s_['balanced'])}** | {pct(s_['caught'])} | {pct(s_['false_alarm'])} | "
                f"{s_['false_nudges']} | {s_['false_buzzes']} | {pct(s_['phone_nudged'])} | {s_['per_held_out']} |")
    held = ", ".join(f"{s['name']} ({s['participant']})" for s in train)
    base_lines = ["## 2b. Classifier vs the baselines, holding out one training session at a time", "",
                  f"Held-out sessions, in order: {held}. A session from a person missing from the other "
                  "sessions is a cross-person test.", "",
                  "| | Balanced on/off | Phone caught | False alarms | False nudges (10 s) | False watch buzzes "
                  "(2 min) | Phone activities nudged | Balanced per held-out session |", "|---|---|---|---|---|---|---|---|"] + [
                  cv_row(n, s_) for n, s_ in base_cv] + [""]
    lines = lines + base_lines
    per = summarize(cv_folds[(best.window_s, best.features, best.model)], smooth_n)["per_held_out"]
    who = pd.DataFrame([{"held-out session": f"{x['name']} ({x['folder']})", "participant": x["participant"],
                         "balanced on/off": pct(b)} for x, b in zip(train, per)])
    tail = ["## 4. Each held-out session, chosen model", "", f"Held out {how}.", "", md_table(who), "",
            "## 5. All candidates", "", f"Held out {how}. TEST was not involved.", "",
            md_table(res[show]), "", imp, "",
            "## 6. How to read this", "",
            "- **Balanced on/off accuracy** averages phone caught and work left alone, so a model that never flags "
            "anything can't look good. It decides the choice.",
            "- **False alarms** are work moments flagged as phone; ten seconds of them in a row make a false nudge, "
            "which is what makes people switch a focus tool off.",
            "- **Held-out sessions** are the honest test: the model is judged on a sitting it never saw.",
            "- Idle (reading, hands resting) counts as on task here. The reference paper counted sitting idle as distracted.", ""]
    report = "\n".join(head + lines + tail) + "\n"
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    real = os.path.abspath(args.out) == os.path.abspath(os.path.join(HERE, "results"))
    latest = os.path.join(HERE if real else args.out, "REPORT.md")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(report.replace("](", f"](results/{stamp}/" if real else f"]({stamp}/"))

    if not args.no_deploy:
        if args.deploy_all and test:
            everyone = train + test
            # test sessions join (without mirrored copies). NATURAL does not: its labels are only "natural work" /
            # "natural phone", so the sips and cups inside it would teach the model that drinking is desk work.
            final = fit(best.model, trn(cfg) + tabs(cfg, test))
            if test:
                inst_model = fit(best.model, trn(icfg) + tabs(icfg, test))
            train = train + test                      # only for the name below; scoring is already done
            kind = "balanced on/off on the test person, measured before they were added to this model"
        else:
            kind = "balanced on/off task, TEST session"
        name = (f"{best.model}, {best.features}, {cfg['window_s']:g} s, {len(train)} training session(s), "
                f"{len(classes)} classes (trained {time.strftime('%d %b %H:%M')})")
        blob = {"model": final, "config": cfg, "smooth_n": int(smooth_n), "name": name,
                "test_accuracy": float(R["test_bin"]) if R else None,
                "test_accuracy_kind": kind,
                "test_activity_accuracy": float(R["test_acc"]) if R else None,
                "classes": list(final.classes_), "needs_calibration": bool(cfg.get("calibrated"))}
        with open(args.deploy_to, "wb") as f:
            pickle.dump(blob, f)
        print(f"\nlive model saved: {os.path.normpath(args.deploy_to)}")
        if test:
            iblob = {**blob, "model": inst_model, "config": dict(INSTANT), "smooth_n": 1,
                     "name": f"instant: {best.model}, {INSTANT['window_s']:g} s, no context, no averaging",
                     "classes": list(inst_model.classes_),
                     "test_accuracy": float(inst["balanced"]), "test_accuracy_kind": "balanced on/off, TEST"}
            ipath = os.path.join(os.path.dirname(os.path.abspath(args.deploy_to)), os.path.basename(DEPLOY_INSTANT))
            with open(ipath, "wb") as f:
                pickle.dump(iblob, f)
            print(f"instant model saved: {os.path.normpath(ipath)}")
    print(f"report: {os.path.normpath(latest)}   ({time.time() - t_start:.0f} s)")
    if R:
        print(f"TEST: balanced on/off {R['test_bin']:.1%}, activity accuracy {R['test_acc']:.1%}")


if __name__ == "__main__":
    main()
