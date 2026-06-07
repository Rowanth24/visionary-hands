#!/usr/bin/env python3
"""Train a small MLP fingerspelling classifier on handshape features.

Usage:
  python models/fingerspelling/train_fingerspelling.py \
    --train data/fingerspelling/set_a.npz data/fingerspelling/set_b.npz \
    --val data/fingerspelling/set_c.npz \
    --out models/fingerspelling/checkpoints/mlp.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.fingerspelling.labels import LETTERS


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_loop(model, Xtr, ytr, Xval, yval, epochs=60, lr=1e-3, batch_size=128):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval)
    n = len(Xtr_t)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(Xval_t).argmax(1) == yval_t).float().mean())
    return acc


def _load_concat(paths):
    Xs, ys = [], []
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"].astype("float32")); ys.append(d["y"].astype("int64"))
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    from models.fingerspelling.features import augment_features

    Xtr, ytr = _load_concat(args.train)
    Xval, yval = _load_concat([args.val])
    # Append appearance-invariant geometric finger-state features, then standardize (mu/sd from train).
    Xtr, Xval = augment_features(Xtr), augment_features(Xval)
    mean = Xtr.mean(axis=0); std = Xtr.std(axis=0) + 1e-6
    Xtr = ((Xtr - mean) / std).astype("float32"); Xval = ((Xval - mean) / std).astype("float32")

    torch.manual_seed(42)
    model = MLP(Xtr.shape[1], len(LETTERS))
    acc = train_loop(model, Xtr, ytr, Xval, yval, epochs=args.epochs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": Xtr.shape[1],
                "num_classes": len(LETTERS), "letters": LETTERS,
                "geometric": True, "mean": mean.astype("float32"), "std": std.astype("float32")}, out)
    print(f"cross-dataset val acc (train={args.train}, val={args.val}) = {round(acc, 4)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
