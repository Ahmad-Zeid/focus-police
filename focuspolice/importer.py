"""Turn an offline recording into a session folder.

    python -m focuspolice.importer path/to/export --person mohamad --session scripted1 \
        --labels path/to/labels.csv --kind scripted --device "Galaxy Watch" --desk library

`export` is a folder of per-sensor CSV files as written by Sensor Logger
(Accelerometer.csv, Gyroscope.csv, optionally Gravity.csv / TotalAcceleration.csv),
each with a time column and x, y, z columns. Other apps work if their files
contain "accel" / "gyro" in the name and have those columns.

`labels.csv` has columns start,end,label in seconds from the start of the
recording (use the three desk taps to line the video up with the signal, and
pass --label-offset if the video clock starts earlier or later).

NOT YET VERIFIED against a real Galaxy Watch export. Run it on a short test
recording first and check the printed sample rate and gravity magnitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FINE_TO_CLASS
from .dataio import G, to_ms2, write_session

TIME_COLS = ["seconds_elapsed", "time", "timestamp", "t"]


def _read(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None or not all(c in df.columns for c in "xyz"):
        raise ValueError(f"{path.name}: need a time column and x, y, z; found {list(df.columns)}")
    t = df[tcol].to_numpy(float)
    span = t[-1] - t[0]
    for scale in (1e9, 1e6, 1e3):  # ns / us / ms timestamps -> seconds
        if span > 50 * scale:
            t = t / scale
            break
    return t, df[["x", "y", "z"]].to_numpy(float)


def _find(folder: Path, *needles, exclude=("uncal",)):
    for p in sorted(folder.glob("*.csv")):
        name = p.name.lower()
        if all(n in name for n in needles) and not any(e in name for e in exclude):
            return p
    return None


def import_export(folder, out_dir, labels_csv, meta, label_offset=0.0):
    folder = Path(folder)
    gyro_file = _find(folder, "gyro")
    total_file = _find(folder, "total")
    acc_file = total_file or _find(folder, "accel")
    if acc_file is None or gyro_file is None:
        raise FileNotFoundError(f"need accelerometer and gyroscope CSV files in {folder}")
    t, acc = _read(acc_file)
    tg, gyr = _read(gyro_file)
    if total_file is None and np.median(np.linalg.norm(acc, axis=1)) < 0.5:
        grav_file = _find(folder, "gravity")
        if grav_file is None:
            raise ValueError("accelerometer has no gravity and there is no Gravity.csv to add back")
        tv, grav = _read(grav_file)
        acc = acc + np.column_stack([np.interp(t, tv, grav[:, i]) for i in range(3)])
    acc = to_ms2(acc)
    gyr_on_acc = np.column_stack([np.interp(t, tg, gyr[:, i]) for i in range(3)])
    t0 = t[0]
    intervals = []
    if labels_csv:
        lab = pd.read_csv(labels_csv)
        for row in lab.itertuples(index=False):
            if row.label not in FINE_TO_CLASS:
                raise ValueError(f"unknown label '{row.label}' in {labels_csv}")
            intervals.append((row.start + label_offset, row.end + label_offset, row.label))
    write_session(out_dir, t - t0, np.column_stack([acc, gyr_on_acc]), intervals, meta)
    dur = t[-1] - t0
    print(f"wrote {out_dir}")
    print(f"  duration {dur / 60:.1f} min, accel {len(t) / dur:.1f} Hz, gyro {len(tg) / (tg[-1] - tg[0]):.1f} Hz")
    print(f"  median |acc| = {np.median(np.linalg.norm(acc, axis=1)):.2f} m/s^2 (expect about {G:.1f})")
    print(f"  {len(intervals)} labelled blocks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("--person", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--label-offset", type=float, default=0.0)
    ap.add_argument("--kind", default="scripted", choices=["scripted", "unscripted"])
    ap.add_argument("--wrist", default="dominant")
    ap.add_argument("--desk", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    meta = {"person": args.person, "session": args.session, "kind": args.kind,
            "wrist": args.wrist, "desk": args.desk, "device": args.device}
    import_export(args.export, Path(args.out) / f"{args.person}_{args.session}", args.labels, meta,
                  args.label_offset)


if __name__ == "__main__":
    main()
