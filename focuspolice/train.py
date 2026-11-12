"""Train the final model on every session in a folder and save it for the live system.

    python -m focuspolice.train --data data/raw --out models/focuspolice.joblib
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime

from .config import FS, WINDOW_S
from .dataio import load_all
from .evaluate import _stack, prepare
from .models import ActivityClassifier, MovementThreshold, save_bundle


def train(data_dir: str, out_path: str):
    sessions = load_all(data_dir)
    items = prepare(sessions)
    X, y = _stack(items)
    names = items[0]["names"]
    clf = ActivityClassifier().fit(X, names, y)
    base = MovementThreshold().fit(X, names, y)
    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "sessions": [s.name for s in sessions],
        "people": sorted({s.person for s in sessions}),
        "windows_per_class": dict(Counter(y)),
        "fs": FS, "window_s": WINDOW_S,
        "baseline_threshold": base.threshold,
    }
    save_bundle(out_path, clf, base, info)
    return clf, base, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="models/focuspolice.joblib")
    args = ap.parse_args()
    clf, base, info = train(args.data, args.out)
    print(f"saved {args.out}")
    print(f"sessions: {len(info['sessions'])}, people: {info['people']}")
    print(f"windows per class: {info['windows_per_class']}")
    print(f"baseline movement threshold: {base.threshold:.4f} m/s^2")
    print("top features:")
    for name, imp in clf.importances(10):
        print(f"  {name:22s} {imp:.3f}")


if __name__ == "__main__":
    main()
