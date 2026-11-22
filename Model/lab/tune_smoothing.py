"""Temporal smoothing and nudge behaviour for the best window model, on the held-out probabilities.
Also lists every phone block that never got a nudge and every false nudge, so they can be inspected."""
import itertools

import numpy as np

from data import load
from evaluate import loso, nudges, score, smooth

CFG = dict(families=("base", "orient", "context", "calib:typing5"), window_s=4.0, model="et")


def main():
    sessions = load(("train",))
    _, probs = loso(sessions, return_probs=True, **CFG)
    options = [("none", {}, th) for th in (0.4, 0.5, 0.6)]
    options += [("mean", {"k": k}, th) for k, th in itertools.product((3, 5, 9), (0.4, 0.5, 0.6))]
    options += [("hmm", {"s": s}, th) for s, th in itertools.product((0.005, 0.01, 0.02, 0.05), (0.4, 0.5, 0.6))]
    print(f"{'method':22s} {'bal':>6s} {'caught':>7s} {'fa':>6s} {'fn/h':>6s} {'nudged':>7s} {'delay':>6s}  per fold balanced")
    best = None
    for method, kw, th in options:
        folds = []
        for sid, (p, truth, t, blocks, prior) in probs.items():
            off = smooth(p, method, prior_off=prior, **kw) >= th
            folds.append(score(truth, off, t, blocks))
        m = {k: np.nanmean([f[k] for f in folds]) for k in folds[0]}
        tag = f"{method} {kw} th={th}"
        print(f"{tag:22s} {m['balanced']:6.3f} {m['caught']:7.3f} {m['false_alarm']:6.3f} {m['false_nudges']:6.1f} "
              f"{m['nudged']:7.2f} {m['delay']:6.1f}  {[round(f['balanced'], 3) for f in folds]}")
        # what a user feels: no false nudges first, then every phone block nudged, then fast
        key = (-m["false_nudges"], m["nudged"], m["balanced"], -m["delay"])
        if best is None or key > best[0]:
            best = (key, method, kw, th)
    _, method, kw, th = best
    print(f"\nbest for the user: {method} {kw} threshold {th}")
    sess_by = {s["id"]: s for s in sessions}
    for sid, (p, truth, t, blocks, prior) in probs.items():
        ps = smooth(p, method, prior_off=prior, **kw)
        off = ps >= th
        fired = nudges(off, t)
        s = sess_by[sid]
        lab_of = {}
        for b in np.unique(blocks[blocks >= 0]):
            lab_of[b] = s["label"][s["block"] == b][0]
        for b in np.unique(blocks[truth == 0]):
            idx = np.where(blocks == b)[0]
            if not any(idx[0] <= i <= idx[-1] + 10 for i in fired):
                print(f"  MISSED  {sid} block {b} {lab_of[b]}: mean P(off) {p[idx].mean():.2f}, flagged {off[idx].mean():.0%}")
        for i in fired:
            if truth[i] == 1:
                print(f"  FALSE   {sid} at {t[i]:.0f} s during {lab_of[blocks[i]]}")


if __name__ == "__main__":
    main()
