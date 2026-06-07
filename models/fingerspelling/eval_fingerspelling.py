#!/usr/bin/env python3
"""Evaluate a fingerspelling classifier: overall + per-letter accuracy, confusion matrix,
and per-sample top-k (for abstention calibration). This is the cross-dataset gate.

Usage:
  python models/fingerspelling/eval_fingerspelling.py \
    --model models/fingerspelling/checkpoints/mlp.pt --test data/fingerspelling/set_c.npz \
    --out data/fingerspelling/eval_on_C.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from models.fingerspelling.labels import LETTERS


def evaluate(probs: np.ndarray, y: np.ndarray, top_k: int = 5) -> dict:
    pred = probs.argmax(axis=1)
    n = len(y)
    overall = float((pred == y).mean())
    per_letter, confusion = {}, np.zeros((24, 24), dtype=int)
    for true_i in range(24):
        mask = y == true_i
        per_letter[LETTERS[true_i]] = {
            "n": int(mask.sum()),
            "accuracy": round(float((pred[mask] == true_i).mean()), 4) if mask.any() else None,
        }
        for p in pred[mask]:
            confusion[true_i, p] += 1
    order = np.argsort(-probs, axis=1)[:, :top_k]
    results = [{"expected": LETTERS[int(y[i])],
                "topk": [{"label": LETTERS[int(j)], "score": float(probs[i, j])} for j in order[i]]}
               for i in range(n)]
    accs = [v["accuracy"] for v in per_letter.values() if v["accuracy"] is not None]
    return {"overall_accuracy": round(overall, 4), "n": n,
            "per_letter": per_letter, "confusion": confusion.tolist(),
            "min_letter_accuracy": round(min(accs), 4) if accs else None,
            "results": results}


def top_confusions(report: dict, k: int = 8):
    """Largest off-diagonal confusion pairs (true -> predicted, count)."""
    conf = np.array(report["confusion"])
    pairs = []
    for i in range(24):
        for j in range(24):
            if i != j and conf[i, j] > 0:
                pairs.append((LETTERS[i], LETTERS[j], int(conf[i, j])))
    return sorted(pairs, key=lambda t: -t[2])[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = np.load(args.test, allow_pickle=True)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    import torch
    from models.fingerspelling.features import augment_features
    from models.fingerspelling.train_fingerspelling import MLP
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)  # trusted local ckpt w/ numpy mean/std
    if ckpt.get("geometric"):
        X = augment_features(X)
        X = ((X - ckpt["mean"]) / ckpt["std"]).astype(np.float32)
    model = MLP(ckpt["input_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()

    report = evaluate(probs, y)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"overall={report['overall_accuracy']}  min_letter={report['min_letter_accuracy']}  n={report['n']}")
    print("top confusions (true->pred, count):", top_confusions(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
