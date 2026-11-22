"""Shared by training (train.py) and the live page (WristStream/live_model.py).
One definition of windows -> features, so the live model sees exactly what it was trained on.

A "config" is a small dict saved inside model.pkl:
    {"window_s": 2.0, "sensors": "acc+gyro", "freq": True}
Version 2 configs (22 Nov 2026) add three switches; a config without them behaves exactly as before:
    "calibrated": True   every 3-axis signal is first rotated so the user's calibration pose (hands resting
                         on the keyboard) points the watch face up. Removes how the strap sits and each
                         person's resting posture. Biggest single gain measured: Model/lab/FINDINGS.md.
    "orient": True       wrist angles in the window: pitch, angle from face-up, roll
    "context": True      the last 10 s and 30 s: average wrist angle and its spread, movement, stillness.
                         Needs CONTEXT_S seconds of history, not just the window (history_seconds()).
features(win, config, R) takes `win` covering history_seconds(config) and the calibration rotation R.
"""
import numpy as np

HZ = 50
CONTEXT_S = (10, 30)
TRIPLES = [("userAccelerationX", "userAccelerationY", "userAccelerationZ"), ("gravityX", "gravityY", "gravityZ"),
           ("rotationRateX", "rotationRateY", "rotationRateZ"), ("accX", "accY", "accZ")]
V2_COLUMNS = ["userAccelerationX", "userAccelerationY", "userAccelerationZ", "gravityX", "gravityY", "gravityZ",
              "rotationRateX", "rotationRateY", "rotationRateZ"]

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

ON_TASK = {"typing", "writing", "idle", "drinking", "hold_cup", "calculator", "other_desk", "natural_work",
           "desk_work"}
OFF_TASK = {"phone_scroll", "phone_text", "phone_call", "natural_phone", "phone",
            "phone_scroll_other", "phone_text_other", "phone_call_other"}


# What the system reports: only what one wrist can tell apart. With the watch on the wrist that does not hold
# the pen, typing, writing and reading all look like a resting wrist (on the unseen-person TEST, idle was read
# as writing 71% of the time), and they all need the same response. The model is still trained on every
# recorded activity; group() adds its probabilities up into these classes for the live page and the report.
# (Training on the groups directly was tried on 22 Nov: no better on held-out training sessions, and held
# cups were flagged as phone more often, so it was dropped.)
GROUPS = {"typing": "desk_work", "writing": "desk_work", "idle": "desk_work", "calculator": "desk_work",
          "other_desk": "desk_work", "natural_work": "desk_work",
          # drinking and holding a cup stay apart: holding still is a distinct pose and the key look-alike
          "drinking": "drinking", "hold_cup": "hold_cup",
          "phone_scroll": "phone", "phone_text": "phone", "phone_call": "phone", "natural_phone": "phone",
          "phone_scroll_other": "phone", "phone_text_other": "phone", "phone_call_other": "phone"}


def group(label):
    """The reported class of a recorded activity (unknown labels, and group names, pass through)."""
    return GROUPS.get(label, label)


def group_probs(probs):
    """{activity: probability} -> {class: probability}, summing the activities of each class."""
    out = {}
    for k, v in probs.items():
        out[group(k)] = out.get(group(k), 0.0) + v
    return out


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


def is_v2(config):
    return bool(config.get("calibrated") or config.get("orient") or config.get("context"))


def history_seconds(config):
    """How much signal features() needs: the window, or the longest context."""
    return max(float(config["window_s"]), max(CONTEXT_S) if config.get("context") else 0.0)


# ------------------------------------------------------------------ calibration

def unit_gravity(gx, gy, gz):
    g = np.column_stack([gx, gy, gz]).astype(float)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)


def calibration_rotation(gx, gy, gz):
    """From the gravity samples of the calibration pose (about 5 s, hands resting on the keyboard): the
    rotation that turns that pose into 'watch face up'. Apply it with rotate()."""
    base = np.median(unit_gravity(gx, gy, gz), axis=0)
    base /= np.linalg.norm(base)
    target = np.array([0.0, 0.0, np.sign(base[2]) or 1.0])
    v = np.cross(base, target); s = np.linalg.norm(v); c = float(base @ target)
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def rotate(win, R):
    """Copy of win with every complete 3-axis sensor rotated by R."""
    out = dict(win)
    for tri in TRIPLES:
        if all(c in win for c in tri):
            v = np.column_stack([win[c] for c in tri]) @ np.asarray(R).T
            for i, c in enumerate(tri):
                out[c] = v[:, i]
    return out


# ------------------------------------------------------------------ version 2 features

def _per_sample(win):
    gu = unit_gravity(win["gravityX"], win["gravityY"], win["gravityZ"])
    roll = np.arctan2(gu[:, 0], gu[:, 2])
    move = np.sqrt(win["userAccelerationX"] ** 2 + win["userAccelerationY"] ** 2 + win["userAccelerationZ"] ** 2)
    turn = np.sqrt(win["rotationRateX"] ** 2 + win["rotationRateY"] ** 2 + win["rotationRateZ"] ** 2)
    k = 25                                                   # 0.5 s running means for "still"
    ker = np.ones(k) / k
    sm = lambda x: np.convolve(np.r_[np.full(k - 1, x[0]), x], ker, "valid")   # noqa: E731  causal
    still = ((sm(turn) < 0.15) & (sm(move) < 0.03)).astype(float)
    return {"gx": gu[:, 0], "gy": gu[:, 1], "gz": gu[:, 2],
            "pitch": np.degrees(np.arcsin(np.clip(gu[:, 1], -1, 1))),
            "face": np.degrees(np.arccos(np.clip(gu[:, 2], -1, 1))),
            "roll_s": np.sin(roll), "roll_c": np.cos(roll), "move": move, "turn": turn, "still": still}


ORIENT = ("pitch", "face", "roll_s", "roll_c")
CTX_SPREAD = ("gx", "gy", "gz", "pitch", "face", "roll_s", "roll_c")
CTX_MEAN = ("move", "turn", "still")


def _v2_extra(sig, n_win):
    f = []
    for c in ORIENT:
        v = sig[c][-n_win:]
        f += [v.mean(), v.std(), v.min(), v.max()]
    f.append(sig["still"][-n_win:].mean())
    for s in CONTEXT_S:
        n = int(s * HZ)
        part = {c: v[-n:] for c, v in sig.items()}
        for c in CTX_SPREAD:
            v = part[c]
            f += [v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0]
        for c in CTX_MEAN:
            f.append(part[c].mean())
        f += [part["pitch"].max(), part["pitch"].min()]
    return f


def _v2_names(config):
    names = []
    if config.get("orient"):
        names += [f"win_{c}_{s}" for c in ORIENT for s in ("mean", "std", "min", "max")] + ["win_still"]
    if config.get("context"):
        for s in CONTEXT_S:
            for c in CTX_SPREAD:
                names += [f"ctx{s}_{c}_mean", f"ctx{s}_{c}_std"]
            names += [f"ctx{s}_{c}_mean" for c in CTX_MEAN] + [f"ctx{s}_pitch_max", f"ctx{s}_pitch_min"]
    return names


def feature_names(config):
    return _v1_names(config) + (_v2_names(config) if is_v2(config) else [])


def _v1_names(config):
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


def features(win, config, R=None):
    """win: dict column name -> 1-D numpy array ending now. For a version 1 config it is exactly one window;
    for version 2 it covers history_seconds(config) (shorter is fine at the start of a recording).
    R: calibration rotation (calibration_rotation()); None means not calibrated (identity).
    Returns a flat list of floats."""
    if not is_v2(config):
        return _v1_features(win, config)
    if config.get("calibrated") and R is not None:
        win = rotate(win, R)
    n_win = int(round(config["window_s"] * HZ))
    f = _v1_features({c: v[-n_win:] for c, v in win.items()}, config)
    if config.get("orient") or config.get("context"):
        sig = _per_sample(win)
        extra = _v2_extra(sig, n_win)
        n_orient = 4 * len(ORIENT) + 1
        if config.get("orient"):
            f += extra[:n_orient]
        if config.get("context"):
            f += extra[n_orient:]
    return f


def _v1_features(win, config):
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
