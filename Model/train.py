"""Train and compare activity models on the recorded sessions, then report and deploy the best one.

    python3 train.py                 # everything, 5-15 min
    python3 train.py --quick         # fewer combinations, about a minute
    python3 train.py --no-deploy     # don't touch the live page's model

What it does, in order:
  1. Finds every labelled session in ../Data Collection/sessions (labels any that have a runner
     file but were never labelled). The session name from the runner decides its role:
        S1, S2, ...  training        TEST  final score only        NATURAL, OTHERHAND  extra tests
  2. Cuts each labelled block into windows and computes features (features.py).
  3. Tries every combination of  window length x sensors x feature set x model  using
     leave-one-session-out: train on all training sessions but one, score on the one left out,
     rotate. Whole sessions are held out because windows from the same session look alike, and
     a random split would give flattering, wrong numbers.
  4. Picks the combination with the best macro-F1 in step 3. TEST plays no part in the choice.
  5. Retrains that one on all training sessions and scores it ONCE on TEST, plus NATURAL and OTHERHAND.
  6. Writes results/<time>/ (report.md, all_combinations.csv, figures), copies the report to REPORT.md,
     and saves ../WristStream/model.pkl so the live page uses the new model.
"""
import argparse
import itertools
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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"        # worker processes too
SESSIONS = os.path.join(HERE, "..", "Data Collection", "sessions")
SEGMENT = os.path.join(HERE, "..", "Data Collection", "segment.py")
DEPLOY = os.path.join(HERE, "..", "WristStream", "model.pkl")

MODELS = {
    "Naive Bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
    "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "k-nearest neighbours": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0)),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=0),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0),
    "Neural net (MLP)": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                                           early_stopping=True, random_state=0)),
}
WINDOWS = [1.0, 2.0, 4.0]
SMOOTH = [1, 3, 5]


def role_of(name):
    n = str(name).upper()
    if n == "TEST" or n.startswith("TEST"):
        return "test"
    if n.startswith("NATURAL"):
        return "natural"
    if n.startswith("OTHERHAND"):
        return "otherhand"
    if len(n) >= 2 and n[0] == "S" and n[1:].isdigit():
        return "train"
    return "ignored"


# ------------------------------------------------------------------ data

def load_sessions(root):
    out = []
    for d in sorted(os.listdir(root)):
        folder = os.path.join(root, d)
        if d.startswith("_") or d.startswith(".") or not os.path.isdir(folder):
            continue
        lab = os.path.join(folder, "labelled.csv")
        has_runner = any(f.endswith("_runner.csv") for f in os.listdir(folder))
        if not os.path.exists(lab) and has_runner and os.path.exists(SEGMENT):
            print(f"labelling {d} ...")
            subprocess.run([sys.executable, SEGMENT, folder], capture_output=True, text=True)
        if not os.path.exists(lab):
            continue
        df = pd.read_csv(lab)
        if "block" not in df.columns:                      # labelled before the block column existed
            subprocess.run([sys.executable, SEGMENT, folder], capture_output=True, text=True)
            df = pd.read_csv(lab)
        need = set(F.SENSORS["all"]) | set(F.SENSORS["acc"])
        if not need <= set(df.columns):
            print(f"skipping {d}: not a Galaxy recording (missing {sorted(need - set(df.columns))[:3]}...)")
            continue
        df = df[df.label != "none"]
        if df.empty:
            continue
        name = str(df.session.iloc[0])
        out.append({"folder": d, "name": name, "role": role_of(name), "df": df})
    return out


def windows(sess, window_s):
    """(start index, end index) pairs inside each labelled block, half-overlapping. Never across blocks."""
    n, hop = int(round(window_s * F.HZ)), int(round(window_s * F.HZ / 2))
    df = sess["df"]
    spans = []
    for (_, label), g in df.groupby(["block", "label"], sort=False):
        t = g["t"].to_numpy()
        # a watch restart or dropout inside a block would leave a time jump: split there
        cuts = np.where(np.diff(t) > 0.5)[0] + 1
        for part in np.split(np.arange(len(g)), cuts):
            for s in range(0, len(part) - n + 1, hop):
                spans.append((g.index[part[s]], label, int(g["block"].iloc[0]), t[part[s]]))
    return spans, n


def build(sessions, window_s, sensors):
    """Feature table for one (window, sensors) pair with the frequency features included.
    The time-only table is its first columns, so it is not computed twice."""
    cfg = {"window_s": window_s, "sensors": sensors, "freq": True}
    cols = F.SENSORS[sensors]
    X, meta = [], []
    for si, s in enumerate(sessions):
        df = s["df"]
        arr = {c: df[c].to_numpy(dtype=float) for c in cols}
        pos = {ix: k for k, ix in enumerate(df.index)}
        spans, n = windows(s, window_s)
        for ix, label, block, t in spans:
            k = pos[ix]
            X.append(F.features({c: a[k:k + n] for c, a in arr.items()}, cfg))
            meta.append((si, label, block, t))
    X = np.nan_to_num(np.array(X, dtype=float))
    meta = pd.DataFrame(meta, columns=["sess", "label", "block", "t"])
    n_time = len(F.feature_names({"sensors": sensors, "freq": False}))
    return X, meta, n_time


def chunk_groups(meta, k=4):
    """Fallback when there is only one training session: cut every block into k pieces in time order
    and hold out one piece of every block at a time. Every activity stays in both sides."""
    g = np.zeros(len(meta), dtype=int)
    for _, idx in meta.groupby(["sess", "block"]).indices.items():
        order = np.argsort(meta.t.to_numpy()[idx])
        g[idx[order]] = np.minimum(k - 1, (np.arange(len(idx)) * k) // max(len(idx), 1))
    return g


def onoff(labels):
    """True = on task. A label that is neither (unknown) counts as on task: it is not phone use."""
    return np.array([F.is_on_task(l) is not False for l in labels], dtype=bool)


def smooth_predict(proba, classes, order_groups, k):
    """Average the last k windows' probabilities (within a session, in time order), as the live page does."""
    out = np.empty(len(proba), dtype=object)
    for g in np.unique(order_groups):
        idx = np.where(order_groups == g)[0]
        p = proba[idx]
        if k > 1:
            c = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), axis=0)
            lo = np.maximum(0, np.arange(len(p)) + 1 - k)
            p = (c[np.arange(len(p)) + 1] - c[lo]) / (np.arange(len(p)) + 1 - lo)[:, None]
        out[idx] = classes[p.argmax(1)]
    return out


def cv_one(model_name, X, y, groups):
    """Leave-one-group-out. Returns mean macro-F1, accuracy, on/off accuracy, and per-group macro-F1."""
    f1s, accs, bins = [], [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = MODELS[model_name]()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        f1s.append(f1_score(y[te], p, average="macro"))
        accs.append(accuracy_score(y[te], p))
        bins.append(accuracy_score(onoff(y[te]), onoff(p)))
    return float(np.mean(f1s)), float(np.mean(accs)), float(np.mean(bins)), [round(float(v), 3) for v in f1s]


# ------------------------------------------------------------------ figures

def plot_confusion(y, p, labels, path, title):
    cm = confusion_matrix(y, p, labels=labels)
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(labels), 0.8 + 0.7 * len(labels)))
    ax.imshow(pct, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{pct[i, j] * 100:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontfamily="serif")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontfamily="serif")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title, fontsize=10, fontfamily="serif")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_bars(series, path, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(series) + 1.2))
    ax.barh(list(series.index)[::-1], list(series.values)[::-1], color="#444")
    for i, v in enumerate(list(series.values)[::-1]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10, fontfamily="serif")
    for lab in ax.get_yticklabels():
        lab.set_fontfamily("serif")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_timeline(t, truth, pred, path, title):
    fig, ax = plt.subplots(2, 1, figsize=(12, 2.4), sharex=True)
    for a, v, name in ((ax[0], truth, "actual"), (ax[1], pred, "predicted")):
        col = np.where(v == True, "#2f6b3a", np.where(v == False, "#b3261e", "#bbbbbb"))  # noqa: E712
        a.bar(t / 60, np.ones(len(t)), width=(np.median(np.diff(t)) if len(t) > 1 else 1) / 60 * 1.05, color=col)
        a.set_yticks([]); a.set_ylabel(name, rotation=0, ha="right", va="center", fontfamily="serif")
    ax[1].set_xlabel("minutes  (green = on task, red = off task)"); ax[0].set_title(title, fontsize=10, fontfamily="serif")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=SESSIONS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--quick", action="store_true", help="2 s windows, 3 models: a fast sanity run")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--deploy-to", default=DEPLOY, help="where to save the live model (testing)")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    t_start = time.time()
    notes = []                                             # caveats that go at the top of the report

    sessions = load_sessions(args.sessions)
    if not sessions:
        sys.exit("No labelled sessions found. Record with the runner (http://localhost:8000/runner) first.")
    inv = []
    for s in sessions:
        secs = s["df"].groupby("label").size() / F.HZ
        inv.append({"folder": s["folder"], "session": s["name"], "role": s["role"],
                    "labelled min": round(len(s["df"]) / F.HZ / 60, 1), "activities": len(secs),
                    "shortest activity (s)": int(secs.min())})
    inv = pd.DataFrame(inv)
    print(inv.to_string(index=False))

    train = [s for s in sessions if s["role"] == "train"]
    test = [s for s in sessions if s["role"] == "test"]
    natural = [s for s in sessions if s["role"] == "natural"]
    otherhand = [s for s in sessions if s["role"] == "otherhand"]
    if not train:
        sys.exit("No training sessions (S1, S2, ...) found.")
    if not test and len(train) >= 3:
        test = [train.pop()]
        notes.append(f"No TEST session was recorded, so the last training session ({test[0]['name']}, "
                     f"{test[0]['folder']}) was set aside as the test set.")
    group_by_block = len(train) < 2
    if group_by_block:
        notes.append("Only ONE training session, so models were compared by holding out a quarter of every "
                     "block at a time, inside that one session. These scores are OPTIMISTIC: the same sitting, "
                     "the same watch placement, and the held-out quarter sits right next to the training part. "
                     "They show whether the activities are separable at all, not how the model does on a new day.")
    if not test:
        notes.append("No test session: the final score below is the cross-validation score, not a held-out test.")
    classes = sorted({l for s in train for l in s["df"].label.unique()})
    for s in test:
        extra = set(s["df"].label.unique()) - set(classes)
        if extra:
            notes.append(f"TEST contains activities never trained on ({sorted(extra)}); they count as errors.")

    windows_ = [2.0] if args.quick else WINDOWS
    models_ = ["Naive Bayes", "Random forest", "Logistic regression"] if args.quick else list(MODELS)
    sensors_ = list(F.SENSORS)

    # ---- features, once per (window, sensors)
    print("\ncomputing features ...")
    everything = train + test + natural + otherhand
    feats = {}
    for w, sn in itertools.product(windows_, sensors_):
        feats[(w, sn)] = build(everything, w, sn)
    idx_of = {id(s): i for i, s in enumerate(everything)}
    tr_ids = [idx_of[id(s)] for s in train]

    def subset(w, sn, freq, ids):
        X, meta, n_time = feats[(w, sn)]
        m = meta.sess.isin(ids).to_numpy()
        return (X[m] if freq else X[m][:, :n_time]), meta[m].reset_index(drop=True)

    # ---- every combination, leave-one-session-out
    combos = list(itertools.product(windows_, sensors_, [False, True], models_))
    print(f"comparing {len(combos)} combinations ...")

    def run(w, sn, freq, mname):
        X, meta = subset(w, sn, freq, tr_ids)
        y = meta.label.to_numpy()
        groups = (chunk_groups(meta) if group_by_block else meta.sess.to_numpy())
        f1, acc, binacc, per = cv_one(mname, X, y, groups)
        return {"window_s": w, "sensors": sn, "features": "time+frequency" if freq else "time",
                "model": mname, "cv_macro_f1": f1, "cv_accuracy": acc, "cv_onoff_accuracy": binacc,
                "per_held_out": per, "n_features": X.shape[1], "n_windows": len(y)}

    res = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(run)(*c) for c in combos))
    res = res.sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    best = res.iloc[0]
    cfg = {"window_s": float(best.window_s), "sensors": best.sensors, "freq": best.features != "time"}
    print(res.head(10).drop(columns=["per_held_out"]).to_string(index=False))

    # ---- smoothing length, chosen on the training sessions only
    Xtr, mtr = subset(cfg["window_s"], cfg["sensors"], cfg["freq"], tr_ids)
    ytr = mtr.label.to_numpy()
    groups = chunk_groups(mtr) if group_by_block else mtr.sess.to_numpy()
    sm_scores = {k: [] for k in SMOOTH}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = MODELS[best.model](); m.fit(Xtr[tr], ytr[tr])
        proba = m.predict_proba(Xtr[te])
        for k in SMOOTH:
            p = smooth_predict(proba, m.classes_, mtr.sess.to_numpy()[te], k)
            sm_scores[k].append(accuracy_score(onoff(ytr[te]), onoff(p)))
    sm_scores = {k: float(np.mean(v)) for k, v in sm_scores.items()}
    smooth_n = max(SMOOTH, key=lambda k: (round(sm_scores[k], 3), -k))

    # ---- final model on all training sessions
    final = MODELS[best.model](); final.fit(Xtr, ytr)
    stamp = time.strftime("%Y%m%d-%H%M")
    out = os.path.join(args.out, stamp); os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "all_combinations.csv"), index=False)

    def evaluate(sess_list, tag):
        ids = [idx_of[id(s)] for s in sess_list]
        X, meta = subset(cfg["window_s"], cfg["sensors"], cfg["freq"], ids)
        proba = final.predict_proba(X)
        raw = final.classes_[proba.argmax(1)]
        sm = smooth_predict(proba, final.classes_, meta.sess.to_numpy(), smooth_n)
        return X, meta, raw, sm

    R = {"cfg": cfg, "model": best.model}
    lines = []
    test_acc = test_bin = None
    if test:
        X, meta, raw, sm = evaluate(test, "test")
        y = meta.label.to_numpy()
        test_acc, test_f1 = accuracy_score(y, raw), f1_score(y, raw, average="macro")
        test_bin = accuracy_score(onoff(y), onoff(raw))
        test_acc_s, test_bin_s = accuracy_score(y, sm), accuracy_score(onoff(y), onoff(sm))
        labs = sorted(set(y) | set(raw))
        plot_confusion(y, raw, labs, os.path.join(out, "confusion_test.png"),
                       f"TEST session, {best.model}: % of each actual activity")
        yb, pb = np.where(onoff(y), "on task", "off task"), np.where(onoff(raw), "on task", "off task")
        plot_confusion(yb, pb, ["on task", "off task"], os.path.join(out, "confusion_test_onoff.png"), "TEST: on vs off task (%)")
        per = pd.DataFrame({"activity": labs,
                            "recall": [float((raw[y == l] == l).mean()) if (y == l).any() else float("nan") for l in labs],
                            "precision": [float((y[raw == l] == l).mean()) if (raw == l).any() else float("nan") for l in labs],
                            "windows": [int((y == l).sum()) for l in labs]})
        cm = confusion_matrix(y, raw, labels=labs); np.fill_diagonal(cm, 0)
        worst = sorted(((cm[i, j], labs[i], labs[j]) for i in range(len(labs)) for j in range(len(labs)) if cm[i, j]),
                       reverse=True)[:5]
        # the reference paper's setting, for comparison: accelerometer only, simple features, Naive Bayes
        bw = 2.0 if 2.0 in windows_ else windows_[0]
        Xb, mb = subset(bw, "acc", False, tr_ids)
        base = MODELS["Naive Bayes"](); base.fit(Xb, mb.label.to_numpy())
        Xbt, mbt = subset(bw, "acc", False, [idx_of[id(s)] for s in test])
        pbase = base.predict(Xbt)
        base_acc = accuracy_score(mbt.label.to_numpy(), pbase)
        base_bin = accuracy_score(onoff(mbt.label.to_numpy()), onoff(pbase))
        R.update(test_acc=test_acc, test_f1=test_f1, test_bin=test_bin)
        lines += [
            "## 2. The final score (TEST session, never used for any choice)", "",
            "| | Activity accuracy | Macro-F1 | On/off-task accuracy |", "|---|---|---|---|",
            f"| **Chosen model**, each window on its own | **{test_acc:.1%}** | {test_f1:.3f} | **{test_bin:.1%}** |",
            f"| Chosen model, averaged over the last {smooth_n} window(s), as on the live page | {test_acc_s:.1%} | | {test_bin_s:.1%} |",
            f"| Reference-paper setting: accelerometer only, simple features, Naive Bayes | {base_acc:.1%} | | {base_bin:.1%} |",
            "", f"Chance level for {len(classes)} activities is {1 / len(classes):.0%}. "
            f"TEST has {len(y)} windows of {cfg['window_s']:g} s.", "",
            "![confusion](confusion_test.png)", "", "![on/off](confusion_test_onoff.png)", "",
            "Per activity:", "", md_table(per), "",
            "Most common mistakes (actual → predicted, windows): " +
            ("; ".join(f"{a} → {b} ({n})" for n, a, b in worst) if worst else "none") + ".", ""]
    else:
        lines += ["## 2. The final score", "", "No test session. Best cross-validation result: "
                  f"macro-F1 {best.cv_macro_f1:.3f}, accuracy {best.cv_accuracy:.1%}, on/off {best.cv_onoff_accuracy:.1%}.", ""]

    if natural:
        X, meta, raw, sm = evaluate(natural, "natural")
        yb = onoff(meta.label.to_numpy())
        a_raw, a_sm = accuracy_score(yb, onoff(raw)), accuracy_score(yb, onoff(sm))
        plot_timeline(np.arange(len(yb)) * cfg["window_s"] / 2, yb, onoff(sm), os.path.join(out, "natural_timeline.png"),
                      "NATURAL session: actual vs predicted (smoothed)")
        lines += ["## 3. NATURAL session (unscripted work and phone use)", "",
                  f"On/off-task accuracy: **{a_raw:.1%}** per window, **{a_sm:.1%}** smoothed over {smooth_n} windows. "
                  f"Share of real work flagged as off task (false alarms): {float((~onoff(sm)[yb]).mean()):.1%}. "
                  f"Share of real phone use caught: {float((~onoff(sm)[~yb]).mean()):.1%}.", "",
                  "This is closer to the demo than the scripted TEST. The phone may have been in either hand.", "",
                  "![natural](natural_timeline.png)", ""]
        R.update(natural=a_sm)
    if otherhand:
        X, meta, raw, sm = evaluate(otherhand, "otherhand")
        caught = float((~onoff(raw)).mean())
        top = pd.Series(raw).value_counts(normalize=True).head(3)
        lines += ["## 4. OTHERHAND (phone in the hand without the watch)", "",
                  f"Only **{caught:.1%}** of these windows were recognised as off task. Most were seen as: " +
                  ", ".join(f"{k} ({v:.0%})" for k, v in top.items()) + ". This is the expected blind spot of a "
                  "single wrist: the watch hand is resting, so it looks like on-task stillness.", ""]
        R.update(otherhand=caught)

    # ---- comparison tables
    def best_by(col):
        g = res.loc[res.groupby(col).cv_macro_f1.idxmax()].sort_values("cv_macro_f1", ascending=False)
        return g
    by_sensor, by_model, by_window, by_feat = best_by("sensors"), best_by("model"), best_by("window_s"), best_by("features")
    plot_bars(by_sensor.set_index("sensors").cv_macro_f1, os.path.join(out, "by_sensors.png"),
              "Best result per sensor choice", "macro-F1, held-out training session")
    plot_bars(by_model.set_index("model").cv_macro_f1, os.path.join(out, "by_model.png"),
              "Best result per model", "macro-F1, held-out training session")
    show = ["window_s", "sensors", "features", "model", "cv_macro_f1", "cv_accuracy", "cv_onoff_accuracy"]
    imp = ""
    if hasattr(final, "feature_importances_"):
        names = F.feature_names(cfg)
        top = np.argsort(final.feature_importances_)[::-1][:12]
        imp = ("Features the model leans on most: " +
               ", ".join(f"`{names[i]}` ({final.feature_importances_[i]:.3f})" for i in top) + ".\n")

    how = "a quarter of every block at a time, within the single training session" if group_by_block else "one whole training session at a time"
    head = [
        f"# Focus Police model report, {time.strftime('%d %b %Y %H:%M')}", "",
        "Generated by `Model/train.py`. Everything below is measured on the recordings in `Data Collection/sessions/`.", ""]
    if notes:
        head += ["**Read this first:**", ""] + [f"- {n}" for n in notes] + [""]
    head += [
        "## 1. What was chosen", "",
        f"- **Model:** {best.model}",
        f"- **Sensors:** {best.sensors} ({', '.join(F.SENSORS[best.sensors])})",
        f"- **Window:** {cfg['window_s']:g} s, half-overlapping; **features:** {best.features} ({int(best.n_features)} numbers per window)",
        f"- **Live smoothing:** average of the last {smooth_n} windows "
        f"(on/off accuracy when holding out {how}: " + ", ".join(f"{k} → {v:.1%}" for k, v in sm_scores.items()) + ")",
        f"- Chosen by holding out {how}: macro-F1 {best.cv_macro_f1:.3f}, accuracy {best.cv_accuracy:.1%}, "
        f"on/off {best.cv_onoff_accuracy:.1%}. Score on each held-out part: {best.per_held_out}.",
        f"- {len(combos)} combinations compared; {int(best.n_windows)} training windows; {time.time() - t_start:.0f} s to run.", "",
        "Sessions used:", "", md_table(inv), ""]
    tail = [
        "## 5. What works best (the comparison)", "",
        f"All numbers here come from holding out {how}. TEST was not involved.", "",
        "**By sensors** (best combination for each):", "", md_table(by_sensor[show]), "", "![sensors](by_sensors.png)", "",
        "**By model:**", "", md_table(by_model[show]), "", "![models](by_model.png)", "",
        "**By window length:**", "", md_table(by_window[show]), "",
        "**By feature set:**", "", md_table(by_feat[show]), "",
        "**Top 15 combinations overall:**", "", md_table(res.head(15)[show]), "",
        imp,
        "## 6. How to read this", "",
        "- **Macro-F1** averages the score of every activity equally, so a model can't look good by only getting the easy ones right.",
        "- **On/off-task accuracy** is what the nudge depends on: typing mistaken for writing costs nothing there, phone mistaken for typing does.",
        "- **Held-out session** scores are lower than a random split would give, and more honest: the model is judged on a sitting it never saw.",
        "- All of this is one person. Accuracy on someone else will be lower until they record their own sessions.",
        "- Idle (reading, hands resting) counts as on task here. The reference paper counted sitting idle as distracted.", "",
        f"Every combination is in `all_combinations.csv` ({len(res)} rows)."]
    report = "\n".join(head + lines + tail) + "\n"
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report)
    real = os.path.abspath(args.out) == os.path.abspath(os.path.join(HERE, "results"))
    latest = os.path.join(HERE if real else args.out, "REPORT.md")
    with open(latest, "w") as f:                               # the latest report, with image links fixed
        f.write(report.replace("](", f"](results/{stamp}/" if real else f"]({stamp}/"))

    if not args.no_deploy:
        name = (f"{best.model}, {best.sensors}, {cfg['window_s']:g} s windows, {len(train)} training session(s), "
                f"{len(classes)} activities (trained {time.strftime('%d %b %H:%M')})")
        blob = {"model": final, "config": cfg, "smooth_n": int(smooth_n), "name": name,
                "test_accuracy": float(test_bin) if test_bin is not None else None,
                "test_accuracy_kind": "on/off task, TEST session",
                "test_activity_accuracy": float(test_acc) if test_acc is not None else None,
                "classes": list(final.classes_)}
        with open(args.deploy_to, "wb") as f:
            pickle.dump(blob, f)
        print(f"\nlive model saved: {os.path.normpath(args.deploy_to)}")
    print(f"report: {os.path.normpath(latest)}   ({time.time() - t_start:.0f} s)")
    if test:
        print(f"TEST: activity accuracy {test_acc:.1%}, on/off accuracy {test_bin:.1%}")


if __name__ == "__main__":
    main()
