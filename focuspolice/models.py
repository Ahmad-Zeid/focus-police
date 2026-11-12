"""The two methods the proposal compares.

1. MovementThreshold: "a simple wrist-movement threshold". Little wrist movement
   means out of focus (idle or holding a phone), more movement means working.
2. ActivityClassifier: a random forest over the window features that predicts the
   desk activity; the focus state follows from the activity.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from .config import SMOOTH_WINDOWS, focus_of
from .features import MOVEMENT_FEATURE


class MovementThreshold:
    """Baseline: movement below a threshold -> off-task, otherwise on-task."""

    def __init__(self):
        self.threshold = None
        self.col = None

    def fit(self, X, names, y_cls):
        self.col = names.index(MOVEMENT_FEATURE)
        focus = np.array([focus_of(c) for c in y_cls])
        keep = focus != "neutral"
        move, truth = X[keep, self.col], focus[keep]
        if len(set(truth)) < 2:
            self.threshold = float(np.median(move))
            return self
        best, best_score = None, -1.0
        for thr in np.quantile(move, np.linspace(0.02, 0.98, 97)):
            pred = np.where(move < thr, "off", "on")
            score = balanced_accuracy_score(truth, pred)
            if score > best_score:
                best, best_score = float(thr), score
        self.threshold = best
        return self

    def predict_focus(self, X):
        return np.where(X[:, self.col] < self.threshold, "off", "on").astype(object)


class ActivityClassifier:
    """Random forest over window features."""

    def __init__(self, n_estimators: int = 300, seed: int = 0):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1, random_state=seed,
        )
        self.names = None

    def fit(self, X, names, y_cls):
        self.names = list(names)
        self.model.fit(X, np.asarray(y_cls, dtype=str))
        return self

    @property
    def classes(self):
        return list(self.model.classes_)

    def predict(self, X):
        return self.model.predict(X).astype(object)

    def predict_focus(self, X):
        return np.array([focus_of(c) for c in self.predict(X)], dtype=object)

    def importances(self, top: int = 15):
        order = np.argsort(self.model.feature_importances_)[::-1][:top]
        return [(self.names[i], float(self.model.feature_importances_[i])) for i in order]


def smooth_labels(labels, k: int = SMOOTH_WINDOWS):
    """Causal majority vote over the last k decisions (ties keep the newest)."""
    out, recent = [], deque(maxlen=k)
    for lab in labels:
        recent.append(lab)
        counts = Counter(recent)
        best = max(counts.values())
        out.append(next(x for x in reversed(recent) if counts[x] == best))
    return np.array(out, dtype=object)


def save_bundle(path, classifier: ActivityClassifier, baseline: MovementThreshold, info: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "baseline": baseline, "info": info}, path)


def load_bundle(path):
    return joblib.load(path)
