"""Does a small 1D CNN on the raw (calibrated) signals beat hand-made features + extra trees, alone or
averaged with it? Same leave-one-session-out protocol, same scoring, same smoothing."""
import numpy as np
import torch
from torch import nn

import feats as FE
from data import F, HOP, HZ, load, on_task, window_ends, window_labels
from evaluate import loso, score, smooth

torch.set_num_threads(8)
CH = FE.UA + FE.G + FE.GY
WIN = 4.0


def windows(sess, hop):
    X = FE.calibrate(sess["X"], sess, "typing5")[:, CH].astype(np.float32)
    ends = window_ends(sess, WIN, hop)
    n = int(WIN * HZ)
    W = np.stack([X[e - n + 1:e + 1].T for e in ends])
    labels, blocks = window_labels(sess, ends, WIN)
    return W, labels, on_task(labels), blocks, sess["t"][ends]


class Net(nn.Module):
    def __init__(self, c_in, n_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c_in, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_out))

    def forward(self, x):
        return self.f(x)


def fit_predict(Wtr, ytr, Wte, seed=0, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    classes = np.array(sorted(set(ytr)))
    yi = np.searchsorted(classes, ytr)
    mu, sd = Wtr.mean((0, 2), keepdims=True), Wtr.std((0, 2), keepdims=True) + 1e-6
    Xt = torch.tensor((Wtr - mu) / sd); Yt = torch.tensor(yi)
    net = Net(Wtr.shape[1], len(classes))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.tensor(len(yi) / (len(classes) * np.bincount(yi, minlength=len(classes))), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            x = Xt[b]
            x = x * (1 + 0.1 * torch.randn(len(b), x.shape[1], 1)) + 0.05 * torch.randn_like(x)  # gain + noise
            shift = np.random.randint(-20, 21)
            x = torch.roll(x, shift, dims=2)
            opt.zero_grad(); lossf(net(x), Yt[b]).backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        P = torch.softmax(net(torch.tensor((Wte - mu) / sd)), 1).numpy()
    return P, classes


def main():
    sessions = load(("train",))
    _, et_probs = loso(sessions, families=("base", "orient", "context", "calib:typing5"), window_s=4.0,
                       model="et", return_probs=True)
    data_tr = {s["id"]: windows(s, 1.0) for s in sessions}
    data_te = {s["id"]: windows(s, HOP) for s in sessions}
    res = {"cnn": [], "cnn+et": [], "et": []}
    for s in sessions:
        sid = s["id"]
        Wtr = np.concatenate([data_tr[o["id"]][0][np.array([l is not None for l in data_tr[o["id"]][1]])]
                              for o in sessions if o is not s])
        ytr = np.concatenate([data_tr[o["id"]][1][np.array([l is not None for l in data_tr[o["id"]][1]])]
                              for o in sessions if o is not s])
        Wte, _, truth, blocks, t = data_te[sid]
        Ps = [fit_predict(Wtr, ytr, Wte, seed) for seed in range(3)]      # 3 seeds averaged
        classes = Ps[0][1]
        P = np.mean([p for p, _ in Ps], 0)
        p_cnn = P[:, [i for i, c in enumerate(classes) if F.is_on_task(c) is False]].sum(1)
        p_et, truth_et, t_et, blocks_et, _ = et_probs[sid]
        assert len(p_et) == len(p_cnn)
        for name, p in (("cnn", p_cnn), ("cnn+et", (p_cnn + p_et) / 2), ("et", p_et)):
            off = smooth(p, "mean", k=9) >= 0.5
            sc = score(truth, off, t, blocks); sc["held_out"] = sid
            res[name].append(sc)
        print(sid, {k: round(v[-1]["balanced"], 3) for k, v in res.items()}, flush=True)
    for name, folds in res.items():
        m = {k: np.nanmean([f[k] for f in folds]) for k in ("balanced", "caught", "false_alarm", "false_nudges", "nudged", "delay")}
        print(f"{name:7s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
