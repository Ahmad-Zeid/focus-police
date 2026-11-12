"""Shared by training (train.py) and the live page (WristStream/live_model.py).
One definition of windows -> features, so the live model sees exactly what it was trained on.

A "config" is a small dict saved inside model.pkl:
    {"window_s": 2.0, "sensors": "acc+gyro", "freq": True}
"""
import numpy as np

HZ = 50

# which recorded columns each sensor option uses
SENSORS = {
    # raw accelerometer only: what a plain wrist accelerometer gives (the reference paper's setting)
    "acc": ["accX", "accY", "accZ"],
    "gyro": ["rotationRateX", "rotationRateY", "rotationRateZ"],
    "acc+gyro": ["accX", "accY", "accZ", "rotationRateX", "rotationRateY", "rotationRateZ"],
    # the watch's own split of the accelerometer into movement and wrist angle
    "motion+angle": ["userAccelerationX", "userAccelerationY", "userAccelerationZ",
                     "gravityX", "gravityY", "gravityZ"],
    "all": ["userAccelerationX", "userAccelerationY", "userAccelerationZ",
            "gravityX", "gravityY", "gravityZ", "rotationRateX", "rotationRateY", "rotationRateZ"],
}

ON_TASK = {"typing", "writing", "idle", "drinking", "hold_cup", "calculator", "other_desk", "natural_work"}
OFF_TASK = {"phone_scroll", "phone_text", "phone_call", "natural_phone",
            "phone_scroll_other", "phone_text_other", "phone_call_other"}


def is_on_task(label):
    """True / False for known labels, None otherwise."""
    return True if label in ON_TASK else False if label in OFF_TASK else None


BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 25.0)]


def _time(x):
    return [x.mean(), x.std(), x.min(), x.max(), np.percentile(x, 75) - np.percentile(x, 25),
            np.abs(np.diff(x)).mean()]


def _freq(x, hz=HZ):
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / hz)
    tot = p.sum() + 1e-12
    out = [p[(f >= lo) & (f < hi)].sum() / tot for lo, hi in BANDS]
    out.append(f[1:][np.argmax(p[1:])] if len(p) > 1 else 0.0)        # dominant frequency
    q = p / tot
    out.append(float(-(q * np.log(q + 1e-12)).sum()))                 # spectral entropy
    out.append(np.log(tot))                                           # log energy
    return out


def feature_names(config):
    cols, names = SENSORS[config["sensors"]], []
    groups = sorted({c[:-1] for c in cols})
    for c in cols:
        names += [f"{c}_{s}" for s in ("mean", "std", "min", "max", "iqr", "jerk")]
    for g in groups:
        names += [f"{g}Mag_{s}" for s in ("mean", "std", "min", "max", "iqr", "jerk")]
        names += [f"{g}_corr{a}{b}" for a, b in ("XY", "XZ", "YZ")]
    if config.get("freq"):
        for c in cols:
            names += [f"{c}_{s}" for s in ("b0-1", "b1-3", "b3-8", "b8-25", "domf", "entropy", "logE")]
        for g in groups:
            names += [f"{g}Mag_{s}" for s in ("b0-1", "b1-3", "b3-8", "b8-25", "domf", "entropy", "logE")]
    return names


def features(win, config):
    """win: dict column name -> 1-D numpy array (one window). Returns a flat list of floats."""
    cols = SENSORS[config["sensors"]]
    groups = sorted({c[:-1] for c in cols})
    f = []
    for c in cols:
        f += _time(win[c])
    mags = {}
    for g in groups:
        x, y, z = win[g + "X"], win[g + "Y"], win[g + "Z"]
        mags[g] = np.sqrt(x * x + y * y + z * z)
        f += _time(mags[g])
        for a, b in ((x, y), (x, z), (y, z)):
            sa, sb = a.std(), b.std()
            f.append(float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 1e-9 and sb > 1e-9 else 0.0)
    if config.get("freq"):
        for c in cols:
            f += _freq(win[c])
        for g in groups:
            f += _freq(mags[g])
    return f
