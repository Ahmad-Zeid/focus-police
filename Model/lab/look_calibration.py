"""Why does calibration help? Prints each session's resting wrist pose and the per-activity wrist angle,
before and after calibration."""
import numpy as np

import feats as FE
from data import load

for s in load(("train",)):
    X = s["X"]
    gu = FE.unit_gravity(X)
    m = FE.still_mask(X)
    base = np.median(gu[m], axis=0)
    R = FE.calibration_rotation(X)
    gc = FE.unit_gravity(FE.rotate(X, R))
    print(f"\n{s['id']}: still {m.mean():.0%} of samples, resting gravity {np.round(base, 2)}, "
          f"rotation {np.degrees(np.arccos((np.trace(R) - 1) / 2)):.0f} deg")
    for lab in sorted(set(s["label"]) - {"none"}):
        k = s["label"] == lab
        p0, _, f0 = FE.angles(gu[k]); p1, _, f1 = FE.angles(gc[k])
        print(f"  {lab:13s} pitch {np.median(p0):6.1f} -> {np.median(p1):6.1f}   face-from-ceiling {np.median(f0):6.1f} -> {np.median(f1):6.1f}")
