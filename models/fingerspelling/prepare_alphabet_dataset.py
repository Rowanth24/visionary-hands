#!/usr/bin/env python3
"""Public ASL-alphabet images -> MediaPipe hand landmarks -> handshape features -> .npz + yield report.

Extraction is batched per letter (ONE MediaPipe Holistic context per letter, not per image —
per-image would reload the model thousands of times). Handles two real dir conventions:
plain letter dirs ("A/") and "<LETTER>-samples/" dirs.

Usage:
  python models/fingerspelling/prepare_alphabet_dataset.py \
    --images-dir data/fingerspelling/raw/asl_alphabet_train/asl_alphabet_train \
    --dataset-name set_a --out data/fingerspelling/set_a.npz --max-per-letter 250
"""
import argparse
import json
from pathlib import Path

import numpy as np

from models.fingerspelling.extraction import extract_hand_landmarks_from_frames
from models.fingerspelling.features import HAND_FEATURE_DIM, handshape_features
from models.fingerspelling.labels import LETTERS, LETTER_TO_INDEX

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _letter_key(name: str) -> str:
    """Leading letter token: 'A' -> 'a', 'M-samples' -> 'm', 'B_test' -> 'b'."""
    return name.strip().lower().replace("_", "-").split("-")[0]


def letter_dirs(images_dir) -> dict[str, Path]:
    """Map letter -> subdir, keeping only the 24 static letters (handles plain + '<L>-samples')."""
    out: dict[str, Path] = {}
    for sub in sorted(Path(images_dir).iterdir()):
        if not sub.is_dir():
            continue
        key = _letter_key(sub.name)
        if key in LETTER_TO_INDEX and key not in out:
            out[key] = sub
    return out


def extract_letter_features(image_paths, max_n):
    """Decode up to max_n images, run ONE Holistic pass over them, return (features list, n_decoded)."""
    import cv2

    frames = []
    for p in image_paths[:max_n]:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    if not frames:
        return [], 0
    landmarks = extract_hand_landmarks_from_frames(frames, frames_are_bgr=True)  # (N, 543, 3) via MediaPipe Hands
    feats = []
    for row in landmarks:
        f = handshape_features(row)
        if f is not None and f.shape == (HAND_FEATURE_DIM,):
            feats.append(f)
    return feats, len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-letter", type=int, default=250, help="Cap images/letter for speed.")
    args = p.parse_args()

    dirs = letter_dirs(Path(args.images_dir))
    if not dirs:
        raise SystemExit(f"No letter subdirs found under {args.images_dir}")

    X, y, yield_rows = [], [], {}
    for letter in LETTERS:
        sub = dirs.get(letter)
        if sub is None:
            yield_rows[letter] = {"total": 0, "kept": 0, "yield": 0.0}
            print(f"{letter}: MISSING dir", flush=True)
            continue
        imgs = [f for f in sorted(sub.iterdir()) if f.suffix.lower() in IMG_SUFFIXES]
        feats, n_decoded = extract_letter_features(imgs, args.max_per_letter)
        for f in feats:
            X.append(f); y.append(LETTER_TO_INDEX[letter])
        yield_rows[letter] = {"total": n_decoded, "kept": len(feats),
                              "yield": round(len(feats) / n_decoded, 3) if n_decoded else 0.0}
        print(f"{letter}: {len(feats)}/{n_decoded} ({yield_rows[letter]['yield']})", flush=True)

    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int64)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, letters=np.array(LETTERS), dataset=args.dataset_name)
    report = {"dataset": args.dataset_name, "n": int(X.shape[0]),
              "feature_dim": HAND_FEATURE_DIM, "per_letter": yield_rows}
    out.with_suffix(".yield.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out} (N={X.shape[0]}) + {out.with_suffix('.yield.json').name}")


if __name__ == "__main__":
    main()
