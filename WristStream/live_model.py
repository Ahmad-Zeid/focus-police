"""The live classifier that receiver.py calls.

The features and the on/off-task split live in ../Model/features.py, shared with training
(../Model/train.py), so the live model sees exactly what it was trained on.

Contract (receiver.py relies on exactly this):
    load() -> Model       called at start and again whenever this file or model.pkl changes
    Model.name, Model.window_s, Model.test_accuracy (0-1 or None), Model.accuracy_kind
    Model.predict(win) -> (label, {label: prob})     win = dict column name -> numpy array

Until train.py has written model.pkl, load() returns a PLACEHOLDER that guesses from motion and
wrist angle with hand-set thresholds. The page labels it clearly. It is not a result.
"""
import os
import pickle
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Model"))
import features as F  # noqa: E402

MODEL_PATH = os.environ.get("FOCUS_MODEL", os.path.join(HERE, "model.pkl"))   # the env var is for testing
is_on_task = F.is_on_task


class Trained:
    def __init__(self, blob):
        self.m = blob["model"]
        self.config = blob["config"]
        self.window_s = float(self.config["window_s"])
        self.name = blob.get("name", "trained model")
        self.test_accuracy = blob.get("test_accuracy")
        self.accuracy_kind = blob.get("test_accuracy_kind", "TEST session")
        self.last = deque(maxlen=max(1, int(blob.get("smooth_n", 1))))   # same smoothing as in training

    def predict(self, win):
        x = np.nan_to_num(np.array([F.features(win, self.config)], dtype=float))
        self.last.append(self.m.predict_proba(x)[0])
        p = np.mean(self.last, axis=0)
        probs = {str(c): float(v) for c, v in zip(self.m.classes_, p)}
        return max(probs, key=probs.get), probs


class Placeholder:
    name = "PLACEHOLDER: hand-set thresholds, not trained"
    test_accuracy = None
    accuracy_kind = ""
    window_s = 2.0

    def predict(self, win):
        ua = np.sqrt(win["userAccelerationX"] ** 2 + win["userAccelerationY"] ** 2 + win["userAccelerationZ"] ** 2).std()
        gy = np.sqrt(win["rotationRateX"] ** 2 + win["rotationRateY"] ** 2 + win["rotationRateZ"] ** 2).mean()
        if ua < 0.03 and gy < 0.25:
            label = "idle"
        elif win["gravityZ"].mean() < 0.3:      # wrist turned well away from flat
            label = "phone_scroll"
        else:
            label = "typing"
        probs = {k: 0.1 for k in ("idle", "typing", "phone_scroll")}
        probs[label] = 0.8
        return label, probs


def load():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return Trained(pickle.load(f))
    return Placeholder()
