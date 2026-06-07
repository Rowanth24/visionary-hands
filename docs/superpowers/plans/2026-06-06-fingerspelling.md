# Fingerspelling-first (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An offline, isolated letter-at-a-time recognizer for the 24 static ASL letters (A–Z minus J, Z) that serves as the OOV/names/meds fallback, built on MediaPipe hand landmarks with abstention.

**Architecture:** Public RGB hand images → MediaPipe hand landmarks (`extract_landmarks_from_frames`) → `handshape_features` (wrist-centred, scale-normalized, handedness-canonical, 63-D xyz) → small torch MLP → softmax(24) → hold-window majority vote → `AbstentionPolicy`. Cross-dataset eval detects overfit; abstention bounds confident-wrong.

**Tech Stack:** Python 3.11, `.venv-oracle` (torch, opencv, mediapipe 0.10.9, scikit-learn, numpy), reuse `landmark_baseline/extraction.py` + `constants.py` + `eval/abstention.py`. Foreground execution only (no background jobs).

**Spec:** `docs/superpowers/specs/2026-06-06-fingerspelling-design.md`

**Env for every task:** `cd ~/visionary-hands && . .venv-oracle/bin/activate && source env.sh`

---

## File structure

```
models/fingerspelling/
  __init__.py
  labels.py                    # LETTERS = 24 static letters
  features.py                  # handshape_features() — shared train+serve
  prepare_alphabet_dataset.py  # images -> landmarks -> features -> npz + yield report
  train_fingerspelling.py      # small torch MLP (+ sklearn baseline floor)
  eval_fingerspelling.py       # cross-dataset acc + per-letter confusion matrix + abstention items
  checkpoints/                 # gitignored
tests/test_fingerspelling.py   # feature invariances + classifier I/O + abstention wiring
docs/fingerspelling.md         # usage + data sourcing + medical-trust gate
data/fingerspelling/           # gitignored: downloaded images + extracted .npz
```

---

# PHASE F0 — scaffold + data

### Task 1: Scaffold package + labels

**Files:**
- Create: `models/fingerspelling/__init__.py` (empty)
- Create: `models/fingerspelling/labels.py`
- Modify: `.gitignore` (ignore data + checkpoints)

- [ ] **Step 1: Create the labels module**

```python
# models/fingerspelling/labels.py
"""The 24 static ASL fingerspelling letters (J and Z are dynamic — excluded in Phase 1)."""

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]

assert len(LETTERS) == 24 and "j" not in LETTERS and "z" not in LETTERS
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(LETTERS)}
```

- [ ] **Step 2: Create the empty package init**

```python
# models/fingerspelling/__init__.py
```

- [ ] **Step 3: Add gitignore entries**

Append to `.gitignore`:
```gitignore
# Fingerspelling data + weights
data/fingerspelling/
models/fingerspelling/checkpoints/
```

- [ ] **Step 4: Verify import**

Run: `python -c "from models.fingerspelling.labels import LETTERS; print(len(LETTERS))"`
Expected: `24`

- [ ] **Step 5: Commit**

```bash
git add models/fingerspelling/__init__.py models/fingerspelling/labels.py .gitignore
git commit -m "feat(fingerspelling): scaffold package + 24-letter labels"
```

---

### Task 2: `handshape_features()` — shared train/serve feature (TDD)

**Files:**
- Create: `models/fingerspelling/features.py`
- Test: `tests/test_fingerspelling.py`

- [ ] **Step 1: Write the failing tests** (invariances are the spec's correctness criteria)

```python
# tests/test_fingerspelling.py
import numpy as np
from models.fingerspelling.features import handshape_features, HAND_FEATURE_DIM
from landmark_baseline.constants import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

def _frame_with_left_hand(seed=0):
    rng = np.random.default_rng(seed)
    arr = np.full((543, 3), np.nan, dtype=np.float32)
    arr[LEFT_HAND_SLICE] = rng.random((21, 3), dtype=np.float32) * 0.3 + 0.3  # plausible 0..1 coords
    return arr

def test_feature_dim_is_63():
    f = handshape_features(_frame_with_left_hand())
    assert f.shape == (HAND_FEATURE_DIM,) == (63,)

def test_translation_invariant():
    arr = _frame_with_left_hand()
    a = handshape_features(arr)
    arr2 = arr.copy(); arr2[LEFT_HAND_SLICE] += np.array([0.1, -0.05, 0.02], dtype=np.float32)
    b = handshape_features(arr2)
    np.testing.assert_allclose(a, b, atol=1e-5)

def test_scale_invariant():
    arr = _frame_with_left_hand()
    a = handshape_features(arr)
    arr2 = arr.copy()
    wrist = arr2[LEFT_HAND_SLICE.start].copy()
    arr2[LEFT_HAND_SLICE] = wrist + (arr2[LEFT_HAND_SLICE] - wrist) * 2.0   # scale 2x about wrist
    b = handshape_features(arr2)
    np.testing.assert_allclose(a, b, atol=1e-4)

def test_handedness_canonical_left_equals_mirrored_right():
    left = _frame_with_left_hand(seed=1)
    right = np.full((543, 3), np.nan, dtype=np.float32)
    r = left[LEFT_HAND_SLICE].copy(); r[:, 0] = 1.0 - r[:, 0]   # mirror x into a right hand
    right[RIGHT_HAND_SLICE] = r
    np.testing.assert_allclose(handshape_features(left), handshape_features(right), atol=1e-4)

def test_no_hand_returns_none():
    assert handshape_features(np.full((543, 3), np.nan, dtype=np.float32)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fingerspelling.py -q`
Expected: FAIL (`ModuleNotFoundError: models.fingerspelling.features`)

- [ ] **Step 3: Implement `handshape_features`**

```python
# models/fingerspelling/features.py
"""Static handshape feature: dominant 21 hand landmarks, wrist-centred, scale-normalized,
handedness-canonical (mirror right->left), xyz. ONE function used at train and serve (no skew)."""
from __future__ import annotations

import numpy as np

from landmark_baseline.constants import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

HAND_POINTS = 21
HAND_FEATURE_DIM = HAND_POINTS * 3  # 63


def _hand_present(points: np.ndarray) -> bool:
    return points.shape[0] == HAND_POINTS and not np.all(np.isnan(points))


def handshape_features(landmarks: np.ndarray) -> np.ndarray | None:
    """landmarks: (543, 3) one MediaPipe-Holistic frame. Returns (63,) float32 or None if no hand."""
    arr = np.asarray(landmarks, dtype=np.float32)
    left = arr[LEFT_HAND_SLICE]
    right = arr[RIGHT_HAND_SLICE]

    left_n = int(np.sum(~np.isnan(left).any(axis=1)))
    right_n = int(np.sum(~np.isnan(right).any(axis=1)))
    if left_n == 0 and right_n == 0:
        return None

    if right_n > left_n:
        hand = right.copy()
        hand[:, 0] = 1.0 - hand[:, 0]   # mirror right -> left (handedness-canonical)
    else:
        hand = left.copy()
    if not _hand_present(hand):
        return None

    hand = np.nan_to_num(hand, nan=0.0)
    wrist = hand[0].copy()             # MediaPipe hand landmark 0 = wrist
    centred = hand - wrist             # translation invariance
    scale = float(np.max(np.linalg.norm(centred, axis=1)))   # scale = farthest point from wrist
    if scale < 1e-6:
        scale = 1e-6
    normalized = centred / scale       # scale invariance
    return normalized.reshape(-1).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fingerspelling.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add models/fingerspelling/features.py tests/test_fingerspelling.py
git commit -m "feat(fingerspelling): shared handshape_features (wrist-centred, scale+handedness invariant) + tests"
```

---

### Task 3: Dataset acquisition (downloads APPROVED)

**Files:**
- Create: `docs/fingerspelling.md` (data sourcing section)
- Data (gitignored): `data/fingerspelling/raw/<dataset>/<letter>/*.jpg`

- [ ] **Step 1: Download Kaggle ASL Alphabet (primary, set A)**

Kaggle creds are in `~/.zshrc` (`KAGGLE_USERNAME`, `KAGGLE_KEY`). Run in foreground:
```bash
pip install -q kaggle
mkdir -p data/fingerspelling/raw
kaggle datasets download -d grassknoted/asl-alphabet -p data/fingerspelling/raw --unzip
```
Expected: `data/fingerspelling/raw/asl_alphabet_train/asl_alphabet_train/<A..Z>/` populated.

- [ ] **Step 2: Download a second RGB set (set B) for training diversity**

Pick one MediaPipe-extractable set, e.g.:
```bash
kaggle datasets download -d debashishsau/aslamerican-sign-language-aplhabet-dataset -p data/fingerspelling/raw --unzip
```
(If unavailable, substitute any RGB ASL-alphabet set; record the chosen slug in `docs/fingerspelling.md`.)

- [ ] **Step 3: Download a third RGB set (set C) — HELD OUT for cross-dataset eval**

Different signers/conditions, e.g. Massey or an ASL-alphabet test set:
```bash
kaggle datasets download -d danrasband/asl-alphabet-test -p data/fingerspelling/raw --unzip
```
Record the exact slug used in `docs/fingerspelling.md`. **Never train on set C.**

- [ ] **Step 4: Record provenance**

Write `docs/fingerspelling.md` with: the three dataset slugs, license notes, which is held out (C), and the rule "set C is eval-only." 

- [ ] **Step 5: Commit (docs only — data is gitignored)**

```bash
git add docs/fingerspelling.md
git commit -m "docs(fingerspelling): dataset provenance (A/B train, C held-out cross-dataset eval)"
```

---

### Task 4: `prepare_alphabet_dataset.py` — images → features `.npz` + yield report

**Files:**
- Create: `models/fingerspelling/prepare_alphabet_dataset.py`
- Test: add to `tests/test_fingerspelling.py`

- [ ] **Step 1: Write a failing test for the per-image conversion helper**

```python
# add to tests/test_fingerspelling.py
def test_letter_dirs_filters_to_24(tmp_path):
    from models.fingerspelling.prepare_alphabet_dataset import letter_dirs
    for name in ["A", "B", "J", "Z", "space", "C"]:
        (tmp_path / name).mkdir()
    got = sorted(letter_dirs(tmp_path).keys())
    assert got == ["a", "b", "c"]   # J,Z,space excluded; lowercased
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_fingerspelling.py::test_letter_dirs_filters_to_24 -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the prepare script**

```python
# models/fingerspelling/prepare_alphabet_dataset.py
#!/usr/bin/env python3
"""Public ASL-alphabet images -> MediaPipe hand landmarks -> handshape features -> .npz + yield report.

Usage:
  python models/fingerspelling/prepare_alphabet_dataset.py \
    --images-dir data/fingerspelling/raw/asl_alphabet_train/asl_alphabet_train \
    --dataset-name kaggle_aslalphabet \
    --out data/fingerspelling/kaggle_aslalphabet.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from landmark_baseline.extraction import extract_landmarks_from_frames
from models.fingerspelling.features import handshape_features, HAND_FEATURE_DIM
from models.fingerspelling.labels import LETTERS, LETTER_TO_INDEX

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def letter_dirs(images_dir: Path) -> dict[str, Path]:
    """Map letter -> subdir, keeping only the 24 static letters (case-insensitive)."""
    out = {}
    for sub in sorted(Path(images_dir).iterdir()):
        if sub.is_dir() and sub.name.lower() in LETTER_TO_INDEX:
            out[sub.name.lower()] = sub
    return out


def feature_from_image(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    landmarks = extract_landmarks_from_frames([img], frames_are_bgr=True)  # (1,543,3)
    return handshape_features(landmarks[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-letter", type=int, default=1500, help="Cap images/letter for speed.")
    args = p.parse_args()

    dirs = letter_dirs(Path(args.images_dir))
    if not dirs:
        raise SystemExit(f"No letter subdirs found under {args.images_dir}")

    X, y, yield_rows = [], [], {}
    for letter, sub in dirs.items():
        imgs = [f for f in sorted(sub.iterdir()) if f.suffix.lower() in IMG_SUFFIXES][: args.max_per_letter]
        kept = 0
        for img in imgs:
            feat = feature_from_image(img)
            if feat is None or feat.shape != (HAND_FEATURE_DIM,):
                continue
            X.append(feat); y.append(LETTER_TO_INDEX[letter]); kept += 1
        yield_rows[letter] = {"total": len(imgs), "kept": kept,
                              "yield": round(kept / len(imgs), 3) if imgs else 0.0}
        print(f"{letter}: {kept}/{len(imgs)} ({yield_rows[letter]['yield']})", flush=True)

    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int64)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, letters=np.array(LETTERS), dataset=args.dataset_name)
    report = {"dataset": args.dataset_name, "n": int(X.shape[0]),
              "feature_dim": HAND_FEATURE_DIM, "per_letter": yield_rows}
    out.with_suffix(".yield.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out} (N={X.shape[0]}) + yield report")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test (passes)**

Run: `python -m pytest tests/test_fingerspelling.py::test_letter_dirs_filters_to_24 -q`
Expected: PASS

- [ ] **Step 5: Run extraction on all three datasets (foreground; minutes each)**

```bash
python models/fingerspelling/prepare_alphabet_dataset.py --images-dir <A dir> --dataset-name set_a --out data/fingerspelling/set_a.npz
python models/fingerspelling/prepare_alphabet_dataset.py --images-dir <B dir> --dataset-name set_b --out data/fingerspelling/set_b.npz
python models/fingerspelling/prepare_alphabet_dataset.py --images-dir <C dir> --dataset-name set_c --out data/fingerspelling/set_c.npz
```

- [ ] **Step 6: F0 GATE — verify per-letter yield**

Run: `python -c "import json; r=json.load(open('data/fingerspelling/set_a.yield.json')); lo=min(v['kept'] for v in r['per_letter'].values()); print('min kept/letter =', lo, '| N =', r['n'])"`
Expected: **min kept/letter ≥ ~300** (target 500–1k). If a letter is far short → note it (likely a low-yield/occluded letter); flag for the confusable-cluster analysis in F1. Commit the yield reports:
```bash
git add data/fingerspelling/*.yield.json
git commit -m "chore(fingerspelling): F0 landmark-extraction yield reports (A/B/C)"
```

---

# PHASE F1 — train + cross-dataset eval

### Task 5: Eval harness + baseline floor (build eval first — goal-driven)

**Files:**
- Create: `models/fingerspelling/eval_fingerspelling.py`

- [ ] **Step 1: Implement eval harness (overall + per-letter + confusion matrix + abstention items)**

```python
# models/fingerspelling/eval_fingerspelling.py
#!/usr/bin/env python3
"""Evaluate a fingerspelling classifier: overall + per-letter accuracy, confusion matrix,
and per-sample top-k (for abstention calibration). Works for the cross-dataset gate.

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
    return {"overall_accuracy": round(overall, 4), "n": n,
            "per_letter": per_letter, "confusion": confusion.tolist(),
            "min_letter_accuracy": min((v["accuracy"] for v in per_letter.values()
                                        if v["accuracy"] is not None), default=None),
            "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = np.load(args.test, allow_pickle=True)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    import torch
    from models.fingerspelling.train_fingerspelling import MLP  # model class
    ckpt = torch.load(args.model, map_location="cpu")
    model = MLP(ckpt["input_dim"], ckpt["num_classes"]); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()

    report = evaluate(probs, y)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"overall={report['overall_accuracy']}  min_letter={report['min_letter_accuracy']}  n={report['n']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Baseline floor (sklearn) — sanity before the MLP**

Run (sets the "is there signal" floor; logistic on landmark features):
```bash
python -c "
import numpy as np; from sklearn.linear_model import LogisticRegression
a=np.load('data/fingerspelling/set_a.npz'); c=np.load('data/fingerspelling/set_c.npz')
m=LogisticRegression(max_iter=2000).fit(a['X'],a['y'])
print('cross-dataset (train A, test C) logistic acc =', round(m.score(c['X'],c['y']),4))
"
```
Expected: well above chance (1/24≈0.042). Record the number — it's the floor the MLP must beat.

- [ ] **Step 3: Commit**

```bash
git add models/fingerspelling/eval_fingerspelling.py
git commit -m "feat(fingerspelling): eval harness (per-letter + confusion matrix + abstention items) + baseline floor"
```

---

### Task 6: `train_fingerspelling.py` — small torch MLP (TDD on synthetic first)

**Files:**
- Create: `models/fingerspelling/train_fingerspelling.py`
- Test: add to `tests/test_fingerspelling.py`

- [ ] **Step 1: Write a failing test (MLP shape + overfits a tiny synthetic set)**

```python
# add to tests/test_fingerspelling.py
def test_mlp_forward_shape_and_learns():
    import torch, numpy as np
    from models.fingerspelling.train_fingerspelling import MLP, train_loop
    rng = np.random.default_rng(0)
    # 24 separable clusters in 63-D
    centers = rng.normal(size=(24, 63)).astype("float32") * 3
    X = np.repeat(centers, 8, axis=0) + rng.normal(size=(192, 63)).astype("float32") * 0.1
    y = np.repeat(np.arange(24), 8).astype("int64")
    model = MLP(63, 24)
    assert model(torch.from_numpy(X)).shape == (192, 24)
    acc = train_loop(model, X, y, X, y, epochs=80, lr=1e-2, batch_size=64)
    assert acc > 0.95   # must fit a trivially separable set
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_fingerspelling.py::test_mlp_forward_shape_and_learns -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the MLP + training**

```python
# models/fingerspelling/train_fingerspelling.py
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

    Xtr, ytr = _load_concat(args.train)
    Xval, yval = _load_concat([args.val])
    torch.manual_seed(42)
    model = MLP(Xtr.shape[1], len(LETTERS))
    acc = train_loop(model, Xtr, ytr, Xval, yval, epochs=args.epochs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": Xtr.shape[1],
                "num_classes": len(LETTERS), "letters": LETTERS}, out)
    print(f"cross-dataset val acc (train={args.train}, val={args.val}) = {round(acc, 4)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the synthetic test (passes)**

Run: `python -m pytest tests/test_fingerspelling.py::test_mlp_forward_shape_and_learns -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/fingerspelling/train_fingerspelling.py tests/test_fingerspelling.py
git commit -m "feat(fingerspelling): small MLP + training loop (TDD on synthetic separable set)"
```

---

### Task 7: F1 GATE — train on A+B, cross-dataset eval on C

**Files:** (runs only; produces reports)

- [ ] **Step 1: Train (foreground)**

```bash
python models/fingerspelling/train_fingerspelling.py \
  --train data/fingerspelling/set_a.npz data/fingerspelling/set_b.npz \
  --val data/fingerspelling/set_c.npz \
  --out models/fingerspelling/checkpoints/mlp.pt
```

- [ ] **Step 2: Cross-dataset eval (held-out C)**

```bash
python models/fingerspelling/eval_fingerspelling.py \
  --model models/fingerspelling/checkpoints/mlp.pt \
  --test data/fingerspelling/set_c.npz \
  --out data/fingerspelling/eval_on_C.json
```

- [ ] **Step 3: Read the gate signals**

Run: `python -c "import json;r=json.load(open('data/fingerspelling/eval_on_C.json'));print('overall',r['overall_accuracy'],'min_letter',r['min_letter_accuracy']); pl=r['per_letter']; print('worst:',sorted(((v['accuracy'],k) for k,v in pl.items() if v['accuracy'] is not None))[:6])"`

**Gate interpretation (decision, not pass/fail):**
- If overall ≥ baseline floor AND the confusable cluster (a/e/m/n/s/t) is the only weak spot → proceed to F2 (abstention will guard them).
- If overall **collapses vs in-dataset** (train-set acc high, C low) → **overfit alarm** → trigger user real-data recording (spec §4) before trusting.
- Record the confusion matrix; the off-diagonal mass on the folded cluster drives F2 thresholds.

- [ ] **Step 4: Commit the eval report**

```bash
git add data/fingerspelling/eval_on_C.json
git commit -m "chore(fingerspelling): F1 cross-dataset eval report (held-out set C)"
```

---

# PHASE F2 — abstention calibration (expand after F1 numbers exist)

> Exact thresholds depend on F1's confusion matrix, so this phase is specified by interface now and parameterized after F1 runs.

### Task 8: Calibrate per-letter / per-cluster abstention

**Files:** Create `models/fingerspelling/calibrate.py` (thin wrapper over `eval/abstention.py`).

- [ ] **Step 1:** Build abstention items from `eval_on_C.json` `results` (already `{expected, topk}` shaped for `eval/abstention.py`).
- [ ] **Step 2:** Use `abstention.risk_coverage_curve` + `recommend_threshold` for a global threshold; then compute **per-letter** thresholds for the confusable cluster (higher τ where confusion mass is high).
- [ ] **Step 3:** Report coverage / selective-accuracy / **confident-wrong rate**; target confident-wrong ≤ ~1–2% on the cluster (spec F-safety), record the coverage cost.
- [ ] **Step 4:** Save `models/fingerspelling/checkpoints/abstention.json` (per-letter thresholds).
- [ ] **Step 5:** Commit.

**F2 GATE:** a calibrated threshold set hitting the confident-wrong budget on held-out C, with coverage reported. (If coverage is unusably low → that's the signal the landmark-only features are insufficient on the cluster → consider the spec's targeted-hybrid escape hatch, or real data.)

---

# PHASE F3 — serve-time recognizer + routing

### Task 9: `FingerspellingRecognizer` (hold-window majority vote + abstention)

**Files:** Create `models/fingerspelling/recognizer.py`; tests in `tests/test_fingerspelling.py`.

- [ ] **Step 1:** Write tests: feeding N frames of one letter → that letter; mixed/low-confidence frames → abstain.
- [ ] **Step 2:** Implement `FingerspellingRecognizer.predict_frame(frame)->(letter|None)` using `handshape_features` → MLP → softmax, buffering a **hold-window** (e.g. last K frames) and emitting a letter only when the window agrees AND passes the per-letter abstention threshold.
- [ ] **Step 3:** Implement `should_fingerspell(word_backend_result)` — auto-fallback hook: returns True when the word backend abstained (reuse `eval/abstention.py`); plus an explicit `manual_spell_mode` flag.
- [ ] **Step 4:** Tests pass; commit.

**F3 GATE:** a CLI/demo that takes a short video of one held handshape and returns the letter or "not confident," and the routing hook returns the right mode on abstain vs confident word results.

---

# PHASE F4 — real-data trust gate

### Task 10: Real-data spot-check harness

**Files:** Create `models/fingerspelling/realdata_check.py`; recordings go in `data/fingerspelling/realdata/` (gitignored).

- [ ] **Step 1:** Define a tiny recording protocol in `docs/fingerspelling.md` (a few frames/letter, varied conditions — mirrors the G1 phone-clip discipline; measurement-only, never trained on).
- [ ] **Step 2:** Reuse `prepare_alphabet_dataset.py` to featurize the real recordings, then `eval_fingerspelling.py` to get per-letter accuracy + confusion matrix on real data.
- [ ] **Step 3:** Compare real-data accuracy to cross-dataset C. **F4 GATE (medical-trust):** real-data per-letter accuracy corroborates C within tolerance AND confident-wrong budget holds → only then is the tool presentable as reliable. If not → record more / trigger targeted hybrid.
- [ ] **Step 4:** Commit the comparison report.

---

## Self-review notes

- **Spec coverage:** interaction model (F3 isolated), data strategy + cross-dataset overfit alarm (F0/F1), xyz depth (Task 2), confusion-aware eval (Task 5), per-letter abstention (F2), real-data trust gate (F4), routing (Task 9), Karpathy smallest-model-first (baseline floor Task 5 before MLP Task 6) — all mapped.
- **Reuse:** `extract_landmarks_from_frames`, `constants`, `eval/abstention.py` reused; only `handshape_features` is new shared code (single train+serve fn — no skew).
- **Data-dependent phases:** F2–F4 are interface-specified now; their exact parameters (thresholds, tolerances) are set from F1's real eval numbers — deliberately not hardcoded before the data exists (goal-driven).
