"""Windowing and hand-crafted features for wrist accelerometer + gyroscope data."""

from __future__ import annotations

import numpy as np

from .config import FS, HOP_S, WINDOW_S

BANDS_HZ = [(0.3, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 20.0)]
MOVEMENT_FEATURE = "acc_dyn_std"  # the single feature the threshold baseline uses


def window_starts(n: int, fs: float = FS, window_s: float = WINDOW_S, hop_s: float = HOP_S):
    w, h = int(round(window_s * fs)), int(round(hop_s * fs))
    if n < w:
        return np.array([], dtype=int), w
    return np.arange(0, n - w + 1, h), w


def _spectral(mag: np.ndarray, fs: float, prefix: str):
    """Dominant frequency and band-energy shares of a (N, n) magnitude signal."""
    sig = mag - mag.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(sig * np.hanning(sig.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(sig.shape[1], 1.0 / fs)
    total = spec[:, 1:].sum(axis=1) + 1e-12
    names, cols = [f"{prefix}_dom_freq"], [freqs[1:][spec[:, 1:].argmax(axis=1)]]
    for lo, hi in BANDS_HZ:
        band = (freqs >= lo) & (freqs < hi)
        names.append(f"{prefix}_band_{lo:g}_{hi:g}")
        cols.append(spec[:, band].sum(axis=1) / total)
    names.append(f"{prefix}_log_energy")
    cols.append(np.log10(total))
    return names, cols


def compute_features(windows: np.ndarray, fs: float = FS):
    """Features for windows of shape (N, n, 6). Returns (N, F) array and F names."""
    acc, gyr = windows[:, :, :3], windows[:, :, 3:6]
    names: list[str] = []
    cols: list[np.ndarray] = []

    def add(name, col):
        names.append(name)
        cols.append(col)

    for tag, sig in (("acc", acc), ("gyr", gyr)):
        for i, axis in enumerate("xyz"):
            a = sig[:, :, i]
            add(f"{tag}_{axis}_mean", a.mean(axis=1))
            add(f"{tag}_{axis}_std", a.std(axis=1))
            add(f"{tag}_{axis}_range", a.max(axis=1) - a.min(axis=1))
        mag = np.linalg.norm(sig, axis=2)
        add(f"{tag}_mag_mean", mag.mean(axis=1))
        add(f"{tag}_mag_std", mag.std(axis=1))
        n, c = _spectral(mag, fs, tag)
        names += n
        cols += c

    # Pose: direction of gravity in the sensor frame (mean acceleration, normalised).
    g = acc.mean(axis=1)
    g_unit = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    for i, axis in enumerate("xyz"):
        add(f"grav_{axis}", g_unit[:, i])

    # Movement: acceleration with the window's mean (gravity) removed.
    dyn = np.linalg.norm(acc - g[:, None, :], axis=2)
    add("acc_dyn_mean", dyn.mean(axis=1))
    add(MOVEMENT_FEATURE, dyn.std(axis=1))
    add("acc_jerk", np.abs(np.diff(acc, axis=1)).mean(axis=(1, 2)) * fs)

    # How much the pose itself moves within the window (sips, lifts).
    half = acc.shape[1] // 2
    g1, g2 = acc[:, :half].mean(axis=1), acc[:, half:].mean(axis=1)
    add("grav_shift", np.linalg.norm(g1 - g2, axis=1))

    for (i, a), (j, b) in ((p, q) for p in enumerate("xyz") for q in enumerate("xyz") if p[0] < q[0]):
        u = acc[:, :, i] - acc[:, :, i].mean(axis=1, keepdims=True)
        v = acc[:, :, j] - acc[:, :, j].mean(axis=1, keepdims=True)
        denom = np.sqrt((u**2).sum(axis=1) * (v**2).sum(axis=1)) + 1e-9
        add(f"acc_corr_{a}{b}", (u * v).sum(axis=1) / denom)

    return np.column_stack(cols), names


def session_windows(session, hop_s: float = HOP_S):
    """Cut a session into windows.

    Returns a dict with features `X`, `names`, window centre times `t`, the
    majority class per window `y` ("" if mostly unlabeled) and its `purity`.
    """
    starts, w = window_starts(len(session.t), hop_s=hop_s)
    if len(starts) == 0:
        return {"X": np.empty((0, 0)), "names": [], "t": np.array([]),
                "y": np.array([], dtype=object), "purity": np.array([])}
    idx = starts[:, None] + np.arange(w)[None, :]
    X, names = compute_features(session.x[idx])
    y = np.empty(len(starts), dtype=object)
    purity = np.zeros(len(starts))
    for k, s in enumerate(starts):
        vals, counts = np.unique(session.cls[s:s + w], return_counts=True)
        top = counts.argmax()
        y[k], purity[k] = vals[top], counts[top] / w
    return {"X": X, "names": names, "t": session.t[starts] + WINDOW_S / 2, "y": y, "purity": purity}
