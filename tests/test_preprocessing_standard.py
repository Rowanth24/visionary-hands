#!/usr/bin/env python3
"""Gate G-C — train/serve preprocessing skew guard.

There must be exactly ONE preprocessing path for the landmark model:
landmark_baseline.preprocess.preprocess_landmark_sequence. Training
(pipeline/prepare_landmark_dataset.py) and serving
(landmark_baseline/inference.py, inference_onnx.py, and the harness
LandmarkBackend) all route through it. This test fails if the two paths ever
diverge — i.e. if the exact feature tensor a model sees at train time stops
matching what it sees at serve time.

Run (any venv with numpy+torch, e.g. .venv-oracle):
    source env.sh
    python -m pytest tests/test_preprocessing_standard.py -q
    # or:  python tests/test_preprocessing_standard.py
"""
from __future__ import annotations

import numpy as np

from landmark_baseline.inference import prepare_model_inputs
from landmark_baseline.preprocess import preprocess_landmark_sequence

TARGET_FRAMES = 64
LAYOUT = "dominant_both_hands"  # the deployed dual-hand model's layout


def _synthetic_sequence(num_frames: int = 30, seed: int = 0) -> np.ndarray:
    """A deterministic 543x3 MediaPipe-Holistic-shaped sequence (no NaNs => both hands present)."""
    rng = np.random.default_rng(seed)
    return rng.random((num_frames, 543, 3), dtype=np.float32)


def _train_side_feature(raw: np.ndarray) -> np.ndarray:
    """Reproduce exactly what prepare_landmark_dataset.py stores as the training feature."""
    processed = preprocess_landmark_sequence(raw, target_frames=TARGET_FRAMES, layout=LAYOUT)
    frames = processed.frames[..., :2]  # xy_only=True, matching the deployed config
    return frames.reshape(frames.shape[0], -1).astype(np.float32)


def _serve_side_feature(raw: np.ndarray) -> np.ndarray:
    """Reproduce exactly what inference.prepare_model_inputs feeds the model at serve time."""
    x, _mask, _meta = prepare_model_inputs(
        raw, target_frames=TARGET_FRAMES, xy_only=True, append_deltas=False, layout=LAYOUT
    )
    return x.numpy()[0].astype(np.float32)


def test_train_and_serve_features_are_identical():
    raw = _synthetic_sequence()
    train_feat = _train_side_feature(raw)
    serve_feat = _serve_side_feature(raw)
    assert train_feat.shape == serve_feat.shape, (
        f"train/serve shape skew: {train_feat.shape} vs {serve_feat.shape}"
    )
    np.testing.assert_array_equal(
        train_feat, serve_feat,
        err_msg="train/serve preprocessing diverged — both must use preprocess_landmark_sequence identically.",
    )


def test_preprocess_is_deterministic():
    raw = _synthetic_sequence(seed=7)
    a = preprocess_landmark_sequence(raw, target_frames=TARGET_FRAMES, layout=LAYOUT)
    b = preprocess_landmark_sequence(raw, target_frames=TARGET_FRAMES, layout=LAYOUT)
    np.testing.assert_array_equal(a.frames, b.frames)
    assert a.dominant_hand == b.dominant_hand and a.layout == LAYOUT


if __name__ == "__main__":
    test_train_and_serve_features_are_identical()
    test_preprocess_is_deterministic()
    print("OK — train/serve preprocessing is identical; preprocess_landmark_sequence is the single standard.")
