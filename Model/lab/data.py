"""Loads labelled sessions as CONTINUOUS streams, the way the live page sees them.

Each session keeps every sample, including the unlabelled gaps between activities (the real transitions:
putting the phone down, reaching for the cup). Windows are cut every HOP seconds over the whole stream,
exactly like the receiver, which predicts every 0.5 s. A window gets an activity label only when at least
LABEL_SHARE of its samples belong to one labelled block; otherwise its label is None (a transition or
unlabelled time). Training uses labelled windows only; evaluation runs the smoother over everything.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SESSIONS = os.path.join(ROOT, "Data Collection", "sessions")
sys.path.insert(0, os.path.join(ROOT, "Model"))
import features as F  # noqa: E402

HZ = 50
HOP = 0.5
LABEL_SHARE = 0.9
RAW = ["userAccelerationX", "userAccelerationY", "userAccelerationZ", "gravityX", "gravityY", "gravityZ",
       "rotationRateX", "rotationRateY", "rotationRateZ", "accX", "accY", "accZ"]


def load(roles=("train", "test"), root=SESSIONS):
    """List of sessions: dict(name, folder, participant, role, t, X (n x len(RAW)), label (object), block)."""
    sys.path.insert(0, os.path.join(ROOT, "Model"))
    from train import role_of
    out = []
    for d in sorted(os.listdir(root)):
        f = os.path.join(root, d, "labelled.csv")
        if d.startswith(("_", ".")) or not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        lab = df[df.label != "none"]
        if lab.empty or not set(RAW) <= set(df.columns):
            continue
        name = str(lab.session.iloc[0])
        role = role_of(name)
        if role not in roles:
            continue
        out.append({"folder": d, "name": name, "participant": str(lab.participant.iloc[0]), "role": role,
                    "id": f"{name}-{lab.participant.iloc[0]}",
                    "t": df.t.to_numpy(float), "X": df[RAW].to_numpy(float),
                    "label": df.label.to_numpy(object), "block": df.block.to_numpy(int)})
    return out


def window_ends(sess, window_s, hop=HOP):
    """Sample indices where a window of window_s seconds ends, every hop seconds, never across a time gap."""
    t = sess["t"]
    n = int(round(window_s * HZ))
    gap = np.r_[True, np.diff(t) > 0.5]
    seg_start = np.maximum.accumulate(np.where(gap, np.arange(len(t)), 0))
    ends, next_t = [], -1e9
    for i in range(n - 1, len(t)):
        if i - seg_start[i] + 1 >= n and t[i] >= next_t:
            ends.append(i)
            next_t = t[i] + hop - 1e-6
    return np.array(ends, dtype=int)


def window_labels(sess, ends, window_s):
    n = int(round(window_s * HZ))
    lab, blk = sess["label"], sess["block"]
    out, blocks = [], []
    for e in ends:
        b = blk[e - n + 1:e + 1]
        vals, cnt = np.unique(b, return_counts=True)
        k = cnt.argmax()
        if vals[k] >= 0 and cnt[k] >= LABEL_SHARE * n:
            out.append(lab[e - n + 1 + int(np.argmax(b == vals[k]))]); blocks.append(int(vals[k]))
        else:
            out.append(None); blocks.append(-1)
    return np.array(out, dtype=object), np.array(blocks)


def on_task(labels):
    """1 = on task, 0 = off task (phone), -1 = unlabelled."""
    return np.array([-1 if l is None else (0 if F.is_on_task(l) is False else 1) for l in labels])
