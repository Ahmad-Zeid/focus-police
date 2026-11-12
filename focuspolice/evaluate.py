"""Offline evaluation with whole sessions (and whole people) held out.

    python -m focuspolice.evaluate --data data/raw --out results

Reports what the proposal asks for: accuracy per activity, false nudges per hour
and nudge delay, for the movement-threshold baseline and the trained classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import CLASSES, HOP_S, MIN_PURITY, focus_of
from .dataio import load_all
from .features import session_windows
from .models import ActivityClassifier, MovementThreshold, smooth_labels
from .nudge import nudge_times, score_nudges

METHODS = ["baseline", "classifier", "classifier_smoothed"]


def prepare(sessions):
    """Windows for every session, computed once."""
    items = []
    for s in sessions:
        w = session_windows(s)
        w["session"] = s
        w["train_mask"] = (w["purity"] >= MIN_PURITY) & (w["y"] != "")
        items.append(w)
    return items


def _stack(items):
    X = np.vstack([w["X"][w["train_mask"]] for w in items])
    y = np.concatenate([w["y"][w["train_mask"]] for w in items])
    return X, y


def run_split(items, group_of):
    """Hold out one group at a time. Returns per-session predictions."""
    groups = sorted({group_of(w["session"]) for w in items})
    if len(groups) < 2:
        return None
    out = []
    for g in groups:
        train = [w for w in items if group_of(w["session"]) != g]
        test = [w for w in items if group_of(w["session"]) == g]
        X, y = _stack(train)
        names = train[0]["names"]
        clf = ActivityClassifier().fit(X, names, y)
        base = MovementThreshold().fit(X, names, y)
        for w in test:
            pred = clf.predict(w["X"])
            smoothed = smooth_labels(pred)
            out.append({
                "session": w["session"], "t": w["t"], "y": w["y"], "mask": w["train_mask"],
                "pred": pred, "pred_smoothed": smoothed,
                "focus": {
                    "baseline": base.predict_focus(w["X"]),
                    "classifier": np.array([focus_of(c) for c in pred], dtype=object),
                    "classifier_smoothed": np.array([focus_of(c) for c in smoothed], dtype=object),
                },
                "threshold": base.threshold,
            })
    return out


def activity_metrics(preds, key):
    y = np.concatenate([p["y"][p["mask"]] for p in preds]).astype(str)
    yhat = np.concatenate([p[key][p["mask"]] for p in preds]).astype(str)
    labels = [c for c in CLASSES if c in set(y)]
    prec, rec, f1, sup = precision_recall_fscore_support(y, yhat, labels=labels, zero_division=0)
    return {
        "labels": labels,
        "windows": int(len(y)),
        "accuracy": float((y == yhat).mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            c: {"accuracy": float(r), "precision": float(p), "f1": float(f), "windows": int(n)}
            for c, p, r, f, n in zip(labels, prec, rec, f1, sup)
        },
        "confusion": confusion_matrix(y, yhat, labels=labels).tolist(),
    }


def focus_metrics(preds, method):
    truth = np.concatenate([[focus_of(c) for c in p["y"][p["mask"]]] for p in preds])
    pred = np.concatenate([p["focus"][method][p["mask"]] for p in preds]).astype(str)
    keep = truth != "neutral"
    truth, pred = truth[keep], pred[keep]
    is_off, said_off = truth == "off", pred == "off"
    tp = int((is_off & said_off).sum())
    precision = tp / max(1, int(said_off.sum()))
    recall = tp / max(1, int(is_off.sum()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    binary_pred = np.where(said_off, "off", "on")
    return {
        "windows": int(len(truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, binary_pred)),
        "off_task_precision": precision,
        "off_task_recall": recall,
        "off_task_f1": f1,
    }


def nudge_metrics(preds, method, kind=None):
    total = None
    for p in preds:
        s = p["session"]
        if kind and s.kind != kind:
            continue
        truth_focus = [focus_of(c) for c in p["y"]]
        ref = nudge_times(p["t"], truth_focus, HOP_S)
        got = nudge_times(p["t"], p["focus"][method], HOP_S)
        sc = score_nudges(got, ref, s.duration_s)
        if total is None:
            total = {k: sc[k] for k in ("reference_nudges", "predicted_nudges", "matched",
                                         "missed", "false_nudges", "hours")}
            total["delays_s"] = list(sc["delays_s"])
        else:
            for k in ("reference_nudges", "predicted_nudges", "matched", "missed", "false_nudges", "hours"):
                total[k] += sc[k]
            total["delays_s"] += sc["delays_s"]
    if total is None:
        return None
    d = total.pop("delays_s")
    total["false_nudges_per_hour"] = total["false_nudges"] / total["hours"] if total["hours"] else 0.0
    total["mean_delay_s"] = float(np.mean(d)) if d else None
    total["mean_abs_delay_s"] = float(np.mean(np.abs(d))) if d else None
    return total


def plot_confusion(metrics, title, path):
    cm = np.array(metrics["confusion"], dtype=float)
    labels = metrics["labels"]
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{share[i, j]:.0%}\n{int(cm[i, j])}", ha="center", va="center",
                        fontsize=8, color="white" if share[i, j] > 0.55 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarise(preds):
    res = {
        "activity": {"classifier": activity_metrics(preds, "pred"),
                     "classifier_smoothed": activity_metrics(preds, "pred_smoothed")},
        "focus": {m: focus_metrics(preds, m) for m in METHODS},
        "nudges_all": {m: nudge_metrics(preds, m) for m in METHODS},
        "nudges_unscripted": {m: nudge_metrics(preds, m, kind="unscripted") for m in METHODS},
    }
    return res


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def write_report(results, sessions, out: Path, data_dir: str):
    lines = ["# Focus Police evaluation", "",
             f"Data: `{data_dir}`. {len(sessions)} sessions, "
             f"{len({s.person for s in sessions})} people, "
             f"{sum(s.duration_s for s in sessions) / 60:.1f} minutes in total.", ""]
    lines += ["| Session | Person | Kind | Minutes |", "|---|---|---|---|"]
    lines += [f"| {s.name} | {s.person} | {s.kind} | {s.duration_s / 60:.1f} |" for s in sessions]
    for split, res in results.items():
        if res is None:
            lines += ["", f"## {split}", "", "Skipped: needs at least two groups."]
            continue
        act = res["activity"]["classifier"]
        act_s = res["activity"]["classifier_smoothed"]
        lines += ["", f"## {split}", "",
                  f"Classifier on {act['windows']} held-out windows: accuracy {_fmt(act['accuracy'], True)}, "
                  f"macro-F1 {_fmt(act['macro_f1'])} "
                  f"(with smoothing: {_fmt(act_s['accuracy'], True)}, {_fmt(act_s['macro_f1'])}).", "",
                  "### Accuracy per activity", "",
                  "| Activity | Windows | Accuracy | Precision | F1 |", "|---|---|---|---|---|"]
        for c, m in act["per_class"].items():
            lines.append(f"| {c} | {m['windows']} | {_fmt(m['accuracy'], True)} | "
                         f"{_fmt(m['precision'], True)} | {_fmt(m['f1'])} |")
        lines += ["", f"![confusion matrix]({split}/confusion.png)", "",
                  "### Off-task detection per window (baseline vs classifier)", "",
                  "| Method | Balanced accuracy | Off-task precision | Off-task recall | F1 |",
                  "|---|---|---|---|---|"]
        for m in METHODS:
            f = res["focus"][m]
            lines.append(f"| {m} | {_fmt(f['balanced_accuracy'], True)} | {_fmt(f['off_task_precision'], True)} | "
                         f"{_fmt(f['off_task_recall'], True)} | {_fmt(f['off_task_f1'])} |")
        for key, title in (("nudges_unscripted", "Nudges, unscripted sessions only"),
                           ("nudges_all", "Nudges, all sessions")):
            lines += ["", f"### {title}", ""]
            if res[key][METHODS[0]] is None:
                lines.append("No sessions of this kind.")
                continue
            lines += ["| Method | Hours | Reference | Matched | Missed | False | False per hour | Mean delay (s) |",
                      "|---|---|---|---|---|---|---|---|"]
            for m in METHODS:
                n = res[key][m]
                lines.append(f"| {m} | {n['hours']:.2f} | {n['reference_nudges']} | {n['matched']} | "
                             f"{n['missed']} | {n['false_nudges']} | {_fmt(n['false_nudges_per_hour'])} | "
                             f"{_fmt(n['mean_delay_s'])} |")
    lines += ["", "Reference nudges are the ones the true labels would trigger under the same "
              "two-minute rule. Delay is predicted nudge time minus reference nudge time.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def evaluate(data_dir: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sessions = load_all(data_dir)
    items = prepare(sessions)
    splits = {
        "leave-one-session-out": lambda s: s.name,
        "leave-one-person-out": lambda s: s.person,
    }
    results = {}
    for split, group_of in splits.items():
        preds = run_split(items, group_of)
        if preds is None:
            results[split] = None
            continue
        results[split] = summarise(preds)
        (out / split).mkdir(exist_ok=True)
        plot_confusion(results[split]["activity"]["classifier"],
                       f"{split}: classifier, per 5 s window", out / split / "confusion.png")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    write_report(results, sessions, out, data_dir)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    evaluate(args.data, args.out)
    print((Path(args.out) / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
