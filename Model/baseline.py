"""The proposal's baseline: one simple threshold on the wrist, compared with the trained classifier.

    "Compare a simple wrist-movement threshold with your trained classifier." (proposal, How to test, 3)

A ThresholdRule looks at ONE number per window and says "phone" when it is above (or below) a cut-off.
fit() picks the cut-off and its direction on the training windows only, by balanced phone-vs-work
accuracy, so the rule gets the same fair chance as the classifier and never sees TEST.

Two rules are reported, both computed from the same windows as the classifier:
  - movement: how much the wrist moves (std of the watch's linear-acceleration magnitude in the window).
    The literal "wrist-movement threshold": phone use is quieter than typing or writing.
  - wrist angle: the calibrated forearm pitch (mean over the window). Holding a phone up to look at it
    tilts the wrist; so does a cup, which is the proposal's "one challenge".

The class behaves like a scikit-learn classifier (fit, predict_proba, classes_), so train.py scores it with
exactly the same code as the models: continuous stream, nudge simulation, held-out sessions.
"""
import numpy as np

import features as F

# name -> (feature config, feature name). Window length is added by train.py.
RULES = {
    "Threshold on wrist movement": ({"sensors": "all", "freq": False}, "userAccelerationMag_std"),
    "Threshold on wrist angle (calibrated)": ({"sensors": "all", "freq": False, "orient": True, "calibrated": True},
                                              "win_pitch_mean"),
}


class ThresholdRule:
    """One feature, one cut-off. classes_ uses two labels that features.is_on_task() already knows."""

    classes_ = np.array(["natural_work", "natural_phone"])

    def __init__(self, column):
        self.column = column
        self.cut, self.above = None, True

    def fit(self, X, y):
        x = np.asarray(X, float)[:, self.column]
        phone = np.array([F.is_on_task(l) is False for l in y])
        if phone.all() or not phone.any():
            raise ValueError("the threshold needs both work and phone windows")
        best = (-1.0, None, True)
        for cut in np.unique(np.quantile(x, np.linspace(0.01, 0.99, 197))):
            for above in (True, False):
                flag = x > cut if above else x < cut
                bal = 0.5 * (flag[phone].mean() + (1 - flag[~phone].mean()))
                if bal > best[0]:
                    best = (bal, cut, above)
        _, self.cut, self.above = best
        return self

    def predict_proba(self, X):
        x = np.asarray(X, float)[:, self.column]
        flag = (x > self.cut if self.above else x < self.cut).astype(float)
        return np.column_stack([1 - flag, flag])

    def describe(self, name):
        return f"phone when `{name}` {'>' if self.above else '<'} {self.cut:.3g}"


def make(config, feature):
    """A factory for train.py: ThresholdRule on the named feature of this config."""
    names = F.feature_names(config)
    if feature not in names:
        raise KeyError(f"{feature} is not a feature of {config}")
    col = names.index(feature)
    return lambda: ThresholdRule(col)
