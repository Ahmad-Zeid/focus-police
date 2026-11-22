"""Runs a named group of experiments and appends the results to results.csv (one row per experiment).

    python run.py features      # which feature families help
    python run.py models        # which model, binary vs activity target
    python run.py augment       # rotation augmentation
    python run.py smoothing     # temporal smoothing and nudge behaviour
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from data import load
from evaluate import loso, summarize

CAL = ("base", "orient", "context", "calib:typing5")

GROUPS = {
    "features": [
        dict(name="baseline (current pipeline features, RF)", families=("base",)),
        dict(name="+ orient", families=("base", "orient")),
        dict(name="+ context", families=("base", "context")),
        dict(name="+ orient + context", families=("base", "orient", "context")),
        dict(name="orient + context only", families=("orient", "context")),
        dict(name="+ orient + context, calibrated", families=("base", "orient", "context", "calib")),
        dict(name="4 s window + orient + context", families=("base", "orient", "context"), window_s=4.0),
    ],
    "calibration": [
        dict(name="no calibration", families=("base", "orient", "context")),
        dict(name="calib: whole session (offline bound)", families=("base", "orient", "context", "calib:still")),
        dict(name="calib: 5 s typing pose at start", families=("base", "orient", "context", "calib:typing5")),
        dict(name="calib: running estimate, no step", families=("base", "orient", "context", "calib:causal")),
        dict(name="calib typing5, 4 s window", families=("base", "orient", "context", "calib:typing5"), window_s=4.0),
        dict(name="calib typing5, base only", families=("base", "calib:typing5")),
        dict(name="calib typing5, orient+context only", families=("orient", "context", "calib:typing5")),
    ],
    "models": [
        dict(name=f"{m}, {tg}", families=CAL, window_s=4.0, model=m, target=tg)
        for m in ("rf", "et", "hgb", "lr") for tg in ("multi", "binary")
    ],
    "duration": [
        dict(name=f"et, {name}", families=fam, window_s=4.0, model="et", smooth_method="mean", smooth_kw={"k": 9})
        for name, fam in (("current best", CAL),
                          ("+ dwell", CAL + ("dwell",)),
                          ("+ 60 s context", ("base", "orient", "context60", "calib:typing5")),
                          ("+ dwell + 60 s context", ("base", "orient", "context60", "dwell", "calib:typing5")))
    ],
    "augment": [
        dict(name=f"et, {n} rotated copies +-{d} deg", families=CAL, window_s=4.0, model="et",
             n_aug=n, aug_deg=d)
        for n, d in ((0, 0), (2, 10), (4, 15), (4, 25))
    ],
}


def main():
    group = sys.argv[1]
    sessions = load(("train",))
    print("sessions:", [s["id"] for s in sessions])
    rows = []
    for exp in GROUPS[group]:
        exp = dict(exp)
        name = exp.pop("name")
        t0 = time.time()
        folds = loso(sessions, **exp)
        s = summarize(folds)
        per = {f["held_out"]: round(f["balanced"], 3) for f in folds}
        print(f"{name:45s} bal={s['balanced']:.3f} caught={s['caught']:.3f} fa={s['false_alarm']:.3f} "
              f"fn/h={s['false_nudges']:.1f} nudged={s['nudged']:.2f} delay={s['delay']:.1f}  {per}  ({time.time()-t0:.0f}s)",
              flush=True)
        rows.append({"group": group, "name": name, **s, "per_fold": json.dumps(per),
                     "config": json.dumps(exp, default=str), "when": time.strftime("%Y-%m-%d %H:%M")})
    out = pd.DataFrame(rows)
    path = "results.csv"
    try:
        out = pd.concat([pd.read_csv(path), out])
    except FileNotFoundError:
        pass
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
