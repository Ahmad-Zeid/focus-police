"""Loading recorded sessions.

A session is a folder:

    data/raw/<person>_<session>/
        imu.csv      t,ax,ay,az,gx,gy,gz   (t in seconds, acceleration INCLUDING gravity)
        labels.csv   start,end,label       (same time base as imu.csv, fine labels)
        meta.json    {"person": "...", "session": "...", "wrist": "...", "desk": "...",
                      "device": "...", "kind": "scripted" | "unscripted"}

Acceleration may be in g or m/s^2 (detected automatically), rotation in rad/s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FINE_TO_CLASS, FS

G = 9.80665
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


@dataclass
class Session:
    name: str
    t: np.ndarray            # (n,) seconds from session start, uniform at FS
    x: np.ndarray            # (n, 6) ax ay az [m/s^2], gx gy gz [rad/s]
    fine: np.ndarray         # (n,) fine label per sample, "" where unlabeled
    cls: np.ndarray          # (n,) class label per sample, "" where unlabeled
    meta: dict = field(default_factory=dict)

    @property
    def person(self) -> str:
        return str(self.meta.get("person", self.name.split("_")[0]))

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind", "scripted"))

    @property
    def duration_s(self) -> float:
        return float(len(self.t) / FS)


def to_ms2(acc: np.ndarray) -> np.ndarray:
    """Convert acceleration to m/s^2 if it looks like it is expressed in g."""
    med = float(np.median(np.linalg.norm(acc, axis=1)))
    if 0.5 < med < 2.0:
        return acc * G
    return acc


def gravity_missing(acc_ms2: np.ndarray) -> bool:
    """True if the signal looks like gravity was removed by the recording app."""
    return float(np.median(np.linalg.norm(acc_ms2, axis=1))) < 5.0


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float = FS):
    """Linear interpolation of irregular samples onto a uniform grid starting at 0."""
    order = np.argsort(t)
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.column_stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])])
    return grid - t[0], out, float(t[0])


def load_session(path: str | Path) -> Session:
    path = Path(path)
    df = pd.read_csv(path / "imu.csv")
    missing = [c for c in ["t", *IMU_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}/imu.csv is missing columns {missing}")
    df = df.dropna(subset=["t", *IMU_COLS])
    raw_t = df["t"].to_numpy(float)
    x = df[IMU_COLS].to_numpy(float)
    x[:, :3] = to_ms2(x[:, :3])
    if gravity_missing(x[:, :3]):
        raise ValueError(
            f"{path}: acceleration has no gravity component. Record total/raw "
            "acceleration, or add the gravity stream back when importing."
        )
    t, x, t0 = resample_uniform(raw_t, x)

    fine = np.full(len(t), "", dtype=object)
    labels_file = path / "labels.csv"
    if labels_file.exists():
        lab = pd.read_csv(labels_file)
        for row in lab.itertuples(index=False):
            label = str(row.label).strip()
            if label not in FINE_TO_CLASS:
                raise ValueError(f"{labels_file}: unknown label '{label}'")
            mask = (t >= row.start - t0) & (t < row.end - t0)
            fine[mask] = label
    cls = np.array([FINE_TO_CLASS.get(f, "") for f in fine], dtype=object)

    meta = {}
    if (path / "meta.json").exists():
        meta = json.loads((path / "meta.json").read_text())
    return Session(name=path.name, t=t, x=x, fine=fine, cls=cls, meta=meta)


def load_all(root: str | Path) -> list[Session]:
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if (p / "imu.csv").exists())
    if not dirs:
        raise FileNotFoundError(f"no sessions (folders with imu.csv) under {root}")
    return [load_session(p) for p in dirs]


def write_session(path: str | Path, t, x, intervals, meta: dict) -> None:
    """Write a session folder. `intervals` is a list of (start, end, fine_label)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.column_stack([t, x]), columns=["t", *IMU_COLS])
    df.to_csv(path / "imu.csv", index=False, float_format="%.5f")
    pd.DataFrame(intervals, columns=["start", "end", "label"]).to_csv(
        path / "labels.csv", index=False, float_format="%.3f"
    )
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
