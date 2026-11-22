"""Where the best setup still goes wrong: per activity, the share of windows flagged as phone (after
smoothing), held out session by session. Also the activity accuracy, and the most important features."""
import numpy as np
import pandas as pd

from data import load
from evaluate import MODELS, session_table, smooth

FAM = ("base", "orient", "context", "calib:typing5")


def main():
    sessions = load(("train",))
    rows, acc = [], []
    for s in sessions:
        Xs, ys = [], []
        for o in sessions:
            if o is s:
                continue
            M, names, labels, *_ = session_table(o, 4.0, FAM, "all", 1.0)
            k = np.array([l is not None for l in labels]); Xs.append(M[k]); ys.append(labels[k])
        m = MODELS["et"](); m.fit(np.vstack(Xs), np.concatenate(ys))
        M, names, labels, truth, blocks, t = session_table(s, 4.0, FAM)
        P = m.predict_proba(M)
        off_cols = [i for i, c in enumerate(m.classes_) if c.startswith("phone")]
        off = smooth(P[:, off_cols].sum(1), "mean", k=9) >= 0.5
        pred = m.classes_[P.argmax(1)]
        k = np.array([l is not None for l in labels])
        for lab in sorted(set(labels[k])):
            j = labels == lab
            rows.append({"held_out": s["id"], "activity": lab, "flagged_phone": off[j].mean(),
                         "activity_correct": (pred[j] == lab).mean() if lab in m.classes_ else np.nan})
        acc.append((s["id"], (pred[k] == labels[k]).mean()))
    df = pd.DataFrame(rows)
    print(df.pivot(index="activity", columns="held_out", values="flagged_phone").round(3).to_string())
    print("\nactivity recall:")
    print(df.pivot(index="activity", columns="held_out", values="activity_correct").round(2).to_string())
    print("\nactivity accuracy per held-out session:", [(a, round(b, 3)) for a, b in acc])
    imp = pd.Series(m.feature_importances_, index=names).sort_values(ascending=False)
    print("\ntop features (last fold):\n", imp.head(20).round(4).to_string())


if __name__ == "__main__":
    main()
