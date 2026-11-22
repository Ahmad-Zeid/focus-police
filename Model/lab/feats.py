"""Feature families for the experiments. Every feature at time T uses only samples up to T (causal),
so anything that wins here can run live.

    base     the existing features.py features on the last window (all sensors, time + frequency)
    orient   wrist angles from the gravity vector: pitch (forearm up/down), roll (wrist turned), and how
             far the watch face is from pointing at the ceiling
    context  the last 10 s and 30 s: average wrist angle and its spread, how much the hand moved, and how
             long the wrist has been held still. A sip is short; holding a phone lasts.
    calib    gravity expressed relative to the session's own resting pose (the watch strapped slightly
             rotated, a different desk height), estimated from the session's still moments
"""
import numpy as np
import pandas as pd

from data import F, HZ, RAW

IDX = {c: i for i, c in enumerate(RAW)}
G = [IDX["gravityX"], IDX["gravityY"], IDX["gravityZ"]]
UA = [IDX["userAccelerationX"], IDX["userAccelerationY"], IDX["userAccelerationZ"]]
GY = [IDX["rotationRateX"], IDX["rotationRateY"], IDX["rotationRateZ"]]
AC = [IDX["accX"], IDX["accY"], IDX["accZ"]]
TRIPLES = [UA, G, GY, AC]


def rotate(X, R):
    """Apply a 3x3 rotation to every 3-axis sensor in a stream (the watch turned on the wrist)."""
    Y = X.copy()
    for tri in TRIPLES:
        Y[:, tri] = X[:, tri] @ R.T
    return Y


def rot_matrix(axis, deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def unit_gravity(X):
    g = X[:, G]
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)


def angles(gu):
    """pitch: forearm tilt along the watch's y axis; roll: rotation about it; face_up: angle from ceiling."""
    pitch = np.degrees(np.arcsin(np.clip(gu[:, 1], -1, 1)))
    roll = np.degrees(np.arctan2(gu[:, 0], gu[:, 2]))
    face = np.degrees(np.arccos(np.clip(gu[:, 2], -1, 1)))
    return pitch, roll, face


def still_mask(X, thr_gyro=0.15, thr_acc=0.03, smooth=25):
    gy = pd.Series(np.linalg.norm(X[:, GY], axis=1)).rolling(smooth, min_periods=1).mean().to_numpy()
    ua = pd.Series(np.linalg.norm(X[:, UA], axis=1)).rolling(smooth, min_periods=1).mean().to_numpy()
    return (gy < thr_gyro) & (ua < thr_acc)


def calibration_rotation(X):
    """Rotation taking the session's typical still wrist pose to 'face up', estimated from ALL the
    session's still samples (uses the future: an upper bound, not usable live)."""
    gu = unit_gravity(X)
    m = still_mask(X)
    return align(np.median(gu[m] if m.sum() > HZ * 5 else gu, axis=0))


def calibrate(X, sess, method):
    """Returns X expressed relative to a reference wrist pose.
        still    reference = median pose of all still moments in the session (offline upper bound)
        typing5  reference = the first 5 s of typing: an explicit 'hands on the keyboard' calibration step
        causal   reference = median pose of the still moments seen SO FAR (updated every 10 s, no user step)
    """
    if method == "still":
        return rotate(X, calibration_rotation(X))
    gu = unit_gravity(X)
    if method == "typing5":
        i = int(np.argmax(sess["label"] == "typing"))
        start = i + 3 * HZ                                 # skip the moment of settling in
        return rotate(X, align(np.median(gu[start:start + 5 * HZ], axis=0)))
    if method == "causal":
        m = still_mask(X)
        Y = X.copy()
        chunk = 10 * HZ
        R = np.eye(3)
        for a in range(0, len(X), chunk):
            seen = np.where(m[:a])[0][-30 * 60 * HZ:]       # still samples so far, last 30 min
            if len(seen) >= 20 * HZ:
                R = align(np.median(gu[seen], axis=0))
            Y[a:a + chunk] = rotate(X[a:a + chunk], R)
        return Y
    raise ValueError(method)


def align(base):
    """Rotation taking the reference gravity direction to 'face up' (+z or -z, whichever is closer)."""
    base = np.asarray(base, float) / np.linalg.norm(base)
    target = np.array([0.0, 0.0, np.sign(base[2]) or 1.0])
    v = np.cross(base, target); s = np.linalg.norm(v); c = float(base @ target)
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def per_sample(X):
    """Derived per-sample signals used by orient and context."""
    gu = unit_gravity(X)
    pitch, roll, face = angles(gu)
    return {"gx": gu[:, 0], "gy": gu[:, 1], "gz": gu[:, 2], "pitch": pitch, "face": face,
            "roll_s": np.sin(np.radians(roll)), "roll_c": np.cos(np.radians(roll)),
            "move": np.linalg.norm(X[:, UA], axis=1), "turn": np.linalg.norm(X[:, GY], axis=1),
            "still": still_mask(X).astype(float)}


def rolling_context(sig, seconds=(10, 30)):
    """Causal rolling mean/std of the per-sample signals over the last N seconds, for every sample."""
    out = {}
    df = pd.DataFrame(sig)
    for s in seconds:
        n = int(s * HZ)
        r = df.rolling(n, min_periods=HZ)
        m, sd = r.mean(), r.std()
        for c in ("gx", "gy", "gz", "pitch", "face", "roll_s", "roll_c"):
            out[f"ctx{s}_{c}_mean"] = m[c].to_numpy(); out[f"ctx{s}_{c}_std"] = sd[c].to_numpy()
        for c in ("move", "turn", "still"):
            out[f"ctx{s}_{c}_mean"] = m[c].to_numpy()
        out[f"ctx{s}_pitch_max"] = r.max()["pitch"].to_numpy()
        out[f"ctx{s}_pitch_min"] = r.min()["pitch"].to_numpy()
    return out


def dwell(gu, ends, max_s=120, tol_deg=25):
    """Seconds the wrist has stayed within tol_deg of its current pose (looking back up to max_s), and
    seconds since it was last in the 'face up, resting' pose. A call or a phone in hand holds a raised
    pose for a long time; a sip leaves it after a few seconds."""
    cos_tol = np.cos(np.radians(tol_deg))
    step = HZ // 5                                         # look back in 0.2 s steps
    look = int(max_s * HZ)
    out = []
    for e in ends:
        lo = max(0, e - look)
        seg = gu[lo:e + 1:step][::-1]
        cur = gu[max(0, e - HZ // 2):e + 1].mean(0); cur /= np.linalg.norm(cur)
        away = np.where(seg @ cur < cos_tol)[0]
        same_s = (away[0] if len(away) else len(seg)) * step / HZ
        rest = np.where(np.abs(seg[:, 2]) > np.cos(np.radians(35)))[0]   # face within 35 deg of up/down
        since_rest = (rest[0] if len(rest) else len(seg)) * step / HZ
        out.append([same_s, since_rest])
    return np.array(out, float)


def session_features(X, ends, window_s, families, base_sensors="all", sess=None):
    """Feature matrix (len(ends) x k) and names for one stream. A family 'calib' / 'calib:<method>'
    expresses everything relative to a reference wrist pose first (see calibrate)."""
    n = int(round(window_s * HZ))
    for fam in families:
        if fam == "calib" or fam.startswith("calib:"):
            X = calibrate(X, sess, fam.split(":")[1] if ":" in fam else "still")
    cols, names = [], []
    if "base" in families:
        cfg = {"window_s": window_s, "sensors": base_sensors, "freq": True}
        cmap = {c: X[:, IDX[c]] for c in F.SENSORS[base_sensors]}
        cols.append(np.array([F.features({c: a[e - n + 1:e + 1] for c, a in cmap.items()}, cfg) for e in ends]))
        names += F.feature_names(cfg)
    if {"orient", "context", "context60"} & set(families):
        sig = per_sample(X)
    if "orient" in families:
        rows = []
        for e in ends:
            sl = slice(e - n + 1, e + 1)
            r = []
            for c in ("pitch", "face", "roll_s", "roll_c"):
                v = sig[c][sl]; r += [v.mean(), v.std(), v.min(), v.max()]
            r += [sig["still"][sl].mean()]
            rows.append(r)
        cols.append(np.array(rows))
        names += [f"win_{c}_{s}" for c in ("pitch", "face", "roll_s", "roll_c") for s in ("mean", "std", "min", "max")]
        names += ["win_still"]
    if "context" in families or "context60" in families:
        ctx = rolling_context(sig, (10, 30, 60) if "context60" in families else (10, 30))
        cols.append(np.column_stack([v[ends] for v in ctx.values()]))
        names += list(ctx)
    if "dwell" in families:
        cols.append(dwell(unit_gravity(X), ends))
        names += ["dwell_same_pose_s", "dwell_since_rest_s"]
    M = np.nan_to_num(np.hstack(cols).astype(float))
    return M, names
