"""Cut one watch recording into labelled activity segments.

Usage:
    python3 segment.py sessions/S1
    python3 segment.py sessions/S1 --threshold 1.2

The session folder holds the watch export (unzipped) and the runner file
<SESSION>_<PARTICIPANT>_runner.csv from session-runner.html.

How the labels are made:
  1. Your clicks in session-runner.html give each activity's start and end.
     Takes you redid ("Redo this one") are dropped.
  2. The triple claps (3+ sharp peaks within 1.5 s) happen just before each click.
     Each activity ends 1 s before the clap that closes it, so the claps never get a label.
  3. The claps also check the two clocks: if clap-to-click gaps look impossible,
     the laptop times are shifted to fit.
  4. 3 s are trimmed from the start of every activity (settling in).

Output, written to the session folder:
  labelled.csv   one row per sample: t, epoch_s, acc x/y/z (g), gyro x/y/z and grav x/y/z if present,
                 then EVERY original column of the recording unchanged, then label, session.
                 The recording itself (stream.csv / the export) is never modified.
  segments.csv   one row per activity with start/end and how it was found
  check.png      the signal with the segments drawn on, for a visual check
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

G = 9.80665
TRIM_S = 3.0


# ---------- loading ----------

def find_motion_file(folder):
    """Pick the CSV with the wrist accelerometer. Prefers watch files."""
    cands = [f for f in glob.glob(os.path.join(glob.escape(folder), "**", "*.csv"), recursive=True)
             if "runner" not in os.path.basename(f).lower()
             and os.path.basename(f) not in ("labelled.csv", "segments.csv")]
    if not cands:
        sys.exit(f"No CSV files in {folder}. Unzip the watch export there.")

    def score(f):
        name = os.path.basename(f).lower()
        s = 0
        if "wrist" in name or "watch" in name:
            s += 10
        if "motion" in name or "acc" in name:
            s += 5
        if name.startswith("meta") or "metadata" in name or "annotation" in name:
            s -= 20
        return s
    # HemiPhysioData files (header has TimeSince1970): use all of them, stitched by clock time,
    # so a session may be one long recording or one file per activity
    hemi = [f for f in cands if "TimeSince1970" in open(f).readline()]
    if hemi:
        return sorted(hemi)
    cands.sort(key=score, reverse=True)
    return [cands[0]]


def pick(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None


def load_motion(paths, units):
    if isinstance(paths, str):
        paths = [paths]
    parts = []
    for p in paths:
        part = pd.read_csv(p, skipinitialspace=True)
        part.columns = [c.strip() for c in part.columns]
        if "TimeSince1970(ms)" in part.columns and "TimeStamp" in part.columns:
            # the wall-clock column jitters and jumps; the watch uptime (TimeStamp, s) is steady.
            # Rebuild clock time as uptime + one robust offset per file.
            off = np.median(part["TimeSince1970(ms)"] / 1000.0 - part["TimeStamp"])
            part["TimeSince1970(ms)"] = (part["TimeStamp"] + off) * 1000.0
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    cols = list(df.columns)
    if "TimeSince1970(ms)" in cols:
        df = df.sort_values("TimeSince1970(ms)").drop_duplicates("TimeSince1970(ms)").reset_index(drop=True)
    path = paths[0] if len(paths) == 1 else f"{len(paths)} files"

    # time: epoch ns ("time"), seconds_elapsed, or phyphox "Time (s)"
    tcol = pick(cols, [r"^timesince1970", r"^time$", r"epoch", r"timestamp"])
    ecol = pick(cols, [r"seconds_elapsed", r"^time \(s\)$", r"^t$"])
    if tcol is not None and df[tcol].dtype.kind in "if":
        raw = df[tcol].astype(float)
        scale = 1e9 if raw.iloc[0] > 1e17 else 1e3 if raw.iloc[0] > 1e11 else 1.0
        epoch = raw / scale
    else:
        epoch = None
    if ecol is not None:
        t = df[ecol].astype(float) - float(df[ecol].iloc[0])
    elif epoch is not None:
        t = epoch - epoch.iloc[0]
    else:
        sys.exit(f"Could not find a time column in {path}: {cols}")

    # acceleration: prefer user acceleration (gravity removed); otherwise total
    ax = pick(cols, [r"^dmuaccelx$", r"^(user)?acceleration_?x", r"^accelerationx", r"^acc.*x", r"^x$", r"acceleration x"])
    ay = pick(cols, [r"^dmuaccely$", r"^(user)?acceleration_?y", r"^accelerationy", r"^acc.*y", r"^y$", r"acceleration y"])
    az = pick(cols, [r"^dmuaccelz$", r"^(user)?acceleration_?z", r"^accelerationz", r"^acc.*z", r"^z$", r"acceleration z"])
    if not (ax and ay and az):
        sys.exit(f"Could not find acceleration columns in {path}: {cols}")
    acc = df[[ax, ay, az]].astype(float).to_numpy()

    grav_cols = [pick(cols, [rf"^gravity_?{a}", rf"^dmgrv{a}$"]) for a in "xyz"]
    has_grav = all(grav_cols)

    # units: decide from gravity if present, else from the magnitude
    if units == "auto":
        ref = np.linalg.norm(df[grav_cols].astype(float).to_numpy(), axis=1) if has_grav \
            else np.linalg.norm(acc, axis=1)
        med = np.median(ref)
        units = "ms2" if med > 4 else "g"
    if units == "ms2":
        acc = acc / G

    # dynamic part: if it still contains gravity, take it out with a 1 s moving mean
    mag = np.linalg.norm(acc, axis=1)
    fs = 1.0 / np.median(np.diff(t))
    if np.median(mag) > 0.5:  # total acceleration, gravity included
        k = max(1, int(round(fs)))
        baseline = pd.DataFrame(acc).rolling(k, center=True, min_periods=1).mean().to_numpy()
        dyn = np.linalg.norm(acc - baseline, axis=1)
    else:
        dyn = mag

    if has_grav:
        for a, gc in zip("xyz", grav_cols):
            extra_grav = df[gc].astype(float).to_numpy() / (G if units == "ms2" else 1.0)
            df[f"__grav_{a}"] = extra_grav
    out = pd.DataFrame({"t": t.to_numpy(), "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2]})
    if epoch is not None:
        out.insert(1, "epoch_s", epoch.to_numpy())
    for a in "xyz":
        c = pick(cols, [rf"^dmrot{a}$", rf"^rotationrate_?{a}", rf"^gyro.*{a}", rf"rotation.*{a}"])
        if c:
            out[f"gyro_{a}"] = df[c].astype(float).to_numpy()
    if has_grav:
        for a in "xyz":
            out[f"grav_{a}"] = df[f"__grav_{a}"].to_numpy()
    # every original column too, untouched, under its own name (raw accelerometer, orientation...)
    for c in cols:
        if not c.startswith("__") and c not in out.columns:
            out[c] = df[c].to_numpy()
    return out, dyn, fs, units, os.path.basename(path)


def load_runner(folder):
    files = glob.glob(os.path.join(glob.escape(folder), "*runner*.csv"))
    if not files:
        return None
    r = pd.read_csv(files[0])
    r["epoch_s"] = r["epoch_ms"] / 1000.0
    return r


# ---------- clap detection ----------

def find_bursts(t, dyn, fs, threshold, min_peaks=3, window=1.5):
    peaks, _ = find_peaks(dyn, height=threshold, distance=max(1, int(0.12 * fs)))
    pt = t[peaks]
    bursts, cur = [], []
    for x in pt:
        if cur and x - cur[0] > window:
            if len(cur) >= min_peaks:
                bursts.append((cur[0], cur[-1]))
            cur = []
        cur.append(x)
    if len(cur) >= min_peaks:
        bursts.append((cur[0], cur[-1]))
    return bursts, pt


# ---------- runner events ----------

def runner_segments(runner):
    """Valid (index, activity, start_epoch, end_epoch) from the clicks, ignoring redone takes.
    Also returns every boundary click (each start, plus the final end), where claps are expected."""
    ev = runner[runner.event.isin(["start", "end", "discard"])].sort_values("epoch_ms")
    segs, open_ = [], {}
    for _, r in ev.iterrows():
        if r.event == "start":
            open_[r["index"]] = r
        elif r.event == "discard":
            open_.pop(r["index"], None)
        elif r.event == "end" and r["index"] in open_:
            s = open_.pop(r["index"])
            segs.append((int(r["index"]), s.activity, s.epoch_s, r.epoch_s))
    starts = ev[ev.event == "start"].epoch_s.tolist()
    ends = ev[ev.event == "end"].epoch_s.tolist()
    boundaries = sorted(set(starts + ends[-1:]))
    return segs, boundaries


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--threshold", type=float, default=1.5, help="clap peak height in g (dynamic acceleration)")
    ap.add_argument("--units", choices=["auto", "g", "ms2"], default="auto")
    ap.add_argument("--file", nargs="+", help="motion CSV(s) to use, if the guess is wrong")
    args = ap.parse_args()

    folder = args.folder.rstrip("/")
    path = args.file or find_motion_file(folder)
    data, dyn, fs, units, fname = load_motion(path, args.units)
    t = data["t"].to_numpy()
    runner = load_runner(folder)
    print(f"motion file : {fname}  ({len(data)} samples, {fs:.1f} Hz, {t[-1]/60:.1f} min, units read as {units})")
    if runner is None:
        sys.exit("No *_runner.csv in the folder. Put the file from session-runner.html next to the export.")

    segs, boundaries = runner_segments(runner)
    session = str(runner["session"].iloc[0])
    n_discard = int((runner.event == "discard").sum())
    print(f"runner file : {len(segs)} activities kept, {n_discard} redone, order {[s[1] for s in segs]}")

    bursts, peak_times = find_bursts(t, dyn, fs, args.threshold)
    print(f"clap bursts : {len(bursts)} found for {len(boundaries)} switches  (peaks over {args.threshold} g: {len(peak_times)})")

    # laptop clock -> watch time axis
    if "epoch_s" in data:
        w0 = data["epoch_s"].iloc[0]
        offset = 0.0
    else:
        if not bursts:
            sys.exit("Export has no clock time and no claps were found; cannot align. Try --threshold.")
        w0 = boundaries[0] - (bursts[0][1] + 2.0)
        offset = 0.0
        print("  no clock time in export: aligned the first click to the first clap burst")

    def match(offset):
        """For each switch click, the last clap burst that ended in the 30 s before it."""
        # a clap only counts for a click if it comes after the previous click and at most 15 s
        # before this one, so a missing clap never borrows the previous activity's clap
        out, prev = [], -1e18
        for b in boundaries:
            tc = b - w0 + offset
            c = [x for x in bursts if max(prev + 1.0, tc - 15) <= x[1] <= tc + 3]
            out.append(c[-1] if c else None)
            prev = tc
        return out

    # clock check: find the shift (within ±2 min) at which the most clicks have a
    # clap burst ending 0.3–6 s before them. Keep 0 if it is already as good as any.
    if bursts:
        ends_ = np.array([x[1] for x in bursts])
        bnd = np.array(boundaries) - w0
        def score(o):
            return sum(np.any((ends_ >= c + o - 6) & (ends_ <= c + o - 0.3)) for c in bnd)
        grid = np.round(np.arange(-120, 120.01, 0.1), 1)
        sc = np.array([score(o) for o in grid])
        best = sc.max()
        if score(0.0) < best:
            # centre of the best run of shifts closest to zero
            idx = np.where(sc == best)[0]
            k = idx[np.argmin(np.abs(grid[idx]))]
            lo = hi = k
            while lo - 1 >= 0 and sc[lo - 1] == best: lo -= 1
            while hi + 1 < len(sc) and sc[hi + 1] == best: hi += 1
            offset = float((grid[lo] + grid[hi]) / 2)
            print(f"  clocks disagree: shifting the laptop times by {offset:+.1f} s ({best}/{len(bnd)} claps line up)")
    matched = match(offset)
    gaps = [b - w0 + offset - m[1] for b, m in zip(boundaries, matched) if m]
    if gaps:
        print(f"clap → click: median {float(np.median(gaps)):.1f} s over {len(gaps)}/{len(boundaries)} switches")
    missing = sum(m is None for m in matched)
    if missing:
        print(f"  {missing} switch(es) without a clap: those ends are trimmed by a fixed 6 s instead")
    burst_before = {b: m for b, m in zip(boundaries, matched)}

    data["label"] = "none"
    data["block"] = -1
    rows = []
    for idx, a, s_ep, e_ep in segs:
        s = s_ep - w0 + offset + TRIM_S
        m = burst_before.get(e_ep)
        e = (m[0] - 1.0) if m else (e_ep - w0 + offset - 6.0)
        if e - s < 5:
            print(f"  skipped {a}: under 5 s left after trimming")
            continue
        sel = (t >= s) & (t < e)
        data.loc[sel, "label"] = a
        data.loc[sel, "block"] = idx                # which runner block (an activity repeats once per round)
        rows.append({"index": idx, "activity": a, "start_s": round(s, 2), "end_s": round(e, 2),
                     "duration_s": round(e - s, 1), "end_from": "clap" if m else "click-6s"})
    seg_df = pd.DataFrame(rows)
    data["session"] = session
    data["watch_wrist"] = runner["watch_wrist"].iloc[0] if "watch_wrist" in runner else ""
    data["participant"] = runner["participant"].iloc[0]

    data.to_csv(os.path.join(folder, "labelled.csv"), index=False)
    seg_df.to_csv(os.path.join(folder, "segments.csv"), index=False)
    print(seg_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(t, dyn, lw=0.4, color="#333")
    ax.axhline(args.threshold, color="#a5521b", ls="--", lw=0.8)
    cmap = plt.get_cmap("tab10")
    for k, r in seg_df.iterrows():
        ax.axvspan(r.start_s, r.end_s, color=cmap(k % 10), alpha=0.2)
        ax.text((r.start_s + r.end_s) / 2, ax.get_ylim()[1] * 0.92, r.activity, ha="center", fontsize=8)
    for b in bursts:
        ax.axvspan(b[0], b[1], color="#a5521b", alpha=0.6)
    for b in boundaries:
        ax.axvline(b - w0 + offset, color="#1f4e9c", lw=0.8, ls=":")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dynamic acceleration (g)")
    ax.set_title(f"{session}: shaded = labelled, orange = clap bursts, blue dotted = your clicks")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "check.png"), dpi=110)
    print(f"wrote labelled.csv, segments.csv, check.png in {folder}")
    print(f"labelled share: {(data.label != 'none').mean():.0%} of samples")


if __name__ == "__main__":
    main()
