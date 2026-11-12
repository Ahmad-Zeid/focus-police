"""Synthetic wrist data for testing the code path ONLY.

    python -m focuspolice.synth --out data/synth

These signals are invented. They exist so the pipeline can be exercised before
real recordings arrive. Numbers computed from them say nothing about how the
system performs and must never appear in the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import FS
from .dataio import G, write_session

# fine label -> (gravity direction, accel noise, gyro noise, extra behaviour)
POSES = {
    "typing":          ((0.00, 0.05, -1.00), 0.25, 0.25, "jitter"),
    "writing":         ((0.30, -0.20, -0.93), 0.10, 0.15, "strokes"),
    "calculator":      ((0.05, 0.10, -0.99), 0.05, 0.05, "taps"),
    "drinking":        ((0.00, -0.30, -0.95), 0.05, 0.05, "sips"),
    "hold_cup":        ((0.20, -0.85, -0.50), 0.06, 0.05, None),
    "hold_bottle":     ((0.15, -0.80, -0.58), 0.06, 0.05, None),
    "phone_typing":    ((0.10, -0.75, -0.65), 0.09, 0.10, "jitter_small"),
    "phone_scrolling": ((0.10, -0.72, -0.68), 0.04, 0.04, "flicks"),
    "phone_calling":   ((-0.90, -0.30, 0.20), 0.04, 0.03, None),
    "phone_table":     ((0.05, 0.00, -1.00), 0.04, 0.05, "flicks"),
    "idle":            ((0.00, 0.00, -1.00), 0.02, 0.01, None),
}


def _segment(label, seconds, rng, person_tilt):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    pose, a_noise, g_noise, extra = POSES[label]
    g = np.array(pose, float) + person_tilt
    g = g / np.linalg.norm(g) * G
    acc = np.tile(g, (n, 1)) + rng.normal(0, a_noise, (n, 3))
    gyr = rng.normal(0, g_noise, (n, 3))
    if extra == "strokes":
        f = rng.uniform(3.5, 5.0)
        acc[:, 0] += 0.5 * np.sin(2 * np.pi * f * t)
        acc[:, 1] += 0.35 * np.cos(2 * np.pi * f * t)
        gyr[:, 2] += 0.4 * np.sin(2 * np.pi * f * t)
    elif extra in ("taps", "flicks"):
        period = 0.7 if extra == "taps" else 2.0
        for start in np.arange(rng.uniform(0, period), seconds, period * rng.uniform(0.8, 1.2)):
            i = int(start * FS)
            acc[i:i + 4, 2] += rng.uniform(0.8, 1.5) if extra == "taps" else 0.2
            gyr[i:i + 6, 0] += rng.uniform(0.3, 0.6) if extra == "flicks" else 0.1
    elif extra == "sips":
        up = np.array([-0.2, -0.95, 0.1]) * G
        for start in np.arange(rng.uniform(2, 6), seconds - 4, rng.uniform(10, 18)):
            i, j = int(start * FS), int((start + 3) * FS)
            ramp = np.sin(np.linspace(0, np.pi, j - i))[:, None]
            acc[i:j] = acc[i:j] * (1 - ramp) + up * ramp + rng.normal(0, 0.1, (j - i, 3))
            gyr[i:j, 0] += 1.5 * np.cos(np.linspace(0, np.pi, j - i))
    return acc, gyr


def make_session(plan, rng, person_tilt):
    """plan: list of (fine_label, seconds). Returns t, x (n,6), label intervals."""
    accs, gyrs, intervals, now = [], [], [], 0.0
    for label, seconds in plan:
        acc, gyr = _segment(label, seconds, rng, person_tilt)
        accs.append(acc)
        gyrs.append(gyr)
        intervals.append((now, now + seconds, label))
        now += seconds
    x = np.column_stack([np.vstack(accs), np.vstack(gyrs)])
    t = np.arange(len(x)) / FS + rng.normal(0, 0.001, len(x))  # slightly irregular clock
    return np.sort(t), x, intervals


def scripted_plan(rng, seconds=90):
    labels = list(POSES)
    rng.shuffle(labels)
    return [(lab, seconds) for lab in labels]


def unscripted_plan(rng, minutes=20):
    """A study session: mostly work, short sips and phone checks, two long drifts."""
    plan, total = [], 0.0
    long_drifts = 2
    while total < minutes * 60:
        work = rng.choice(["typing", "writing", "calculator"], p=[0.5, 0.4, 0.1])
        dur = float(rng.uniform(60, 180))
        plan.append((work, dur))
        total += dur
        r = rng.random()
        if long_drifts and r < 0.45:
            off = rng.choice(["phone_scrolling", "phone_typing", "idle"])
            dur = float(rng.uniform(170, 260))
            long_drifts -= 1
        elif r < 0.75:
            off, dur = rng.choice(["phone_scrolling", "idle"]), float(rng.uniform(10, 30))
        else:
            off, dur = rng.choice(["drinking", "hold_cup", "hold_bottle"]), float(rng.uniform(15, 40))
        plan.append((off, dur))
        total += dur
    return plan


def generate(out_dir: str, people=("p1", "p2", "p3"), seed: int = 0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for person in people:
        tilt = rng.normal(0, 0.06, 3)
        for k in (1, 2):
            t, x, iv = make_session(scripted_plan(rng), rng, tilt + rng.normal(0, 0.02, 3))
            write_session(out / f"{person}_scripted{k}", t, x, iv,
                          {"person": person, "session": f"scripted{k}", "kind": "scripted",
                           "device": "SYNTHETIC", "wrist": "dominant", "desk": f"desk{k}"})
        t, x, iv = make_session(unscripted_plan(rng), rng, tilt + rng.normal(0, 0.02, 3))
        write_session(out / f"{person}_unscripted", t, x, iv,
                      {"person": person, "session": "unscripted", "kind": "unscripted",
                       "device": "SYNTHETIC", "wrist": "dominant", "desk": "desk1"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, seed=args.seed)
    print(f"wrote synthetic sessions to {args.out} (testing only, not real data)")


if __name__ == "__main__":
    main()
