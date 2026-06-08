#!/usr/bin/env python3
"""Tests for the fingerspelling track. Pure stdlib + numpy. Run:
    python -m pytest tests/test_fingerspelling.py -q
"""
import numpy as np

from landmark_baseline.constants import LEFT_HAND_SLICE, RIGHT_HAND_SLICE
from models.fingerspelling.features import handshape_features, HAND_FEATURE_DIM


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
    arr2 = arr.copy()
    arr2[LEFT_HAND_SLICE] += np.array([0.1, -0.05, 0.02], dtype=np.float32)
    b = handshape_features(arr2)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_scale_invariant():
    arr = _frame_with_left_hand()
    a = handshape_features(arr)
    arr2 = arr.copy()
    wrist = arr2[LEFT_HAND_SLICE.start].copy()
    arr2[LEFT_HAND_SLICE] = wrist + (arr2[LEFT_HAND_SLICE] - wrist) * 2.0  # scale 2x about wrist
    b = handshape_features(arr2)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_handedness_canonical_left_equals_mirrored_right():
    left = _frame_with_left_hand(seed=1)
    right = np.full((543, 3), np.nan, dtype=np.float32)
    r = left[LEFT_HAND_SLICE].copy()
    r[:, 0] = 1.0 - r[:, 0]  # mirror x into a right hand
    right[RIGHT_HAND_SLICE] = r
    np.testing.assert_allclose(handshape_features(left), handshape_features(right), atol=1e-4)


def test_no_hand_returns_none():
    assert handshape_features(np.full((543, 3), np.nan, dtype=np.float32)) is None


def test_letter_dirs_filters_and_parses(tmp_path=None):
    import tempfile
    from pathlib import Path
    from models.fingerspelling.prepare_alphabet_dataset import letter_dirs
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name in ["A", "B", "J", "Z", "space", "C", "M-samples", "X_test", "nothing"]:
            (root / name).mkdir()
        got = sorted(letter_dirs(root).keys())
        assert got == ["a", "b", "c", "m", "x"]  # J,Z,space,nothing excluded; suffixes parsed


def test_geometric_and_augment_dims():
    from models.fingerspelling.features import (
        geometric_features, augment_features, GEOMETRIC_FEATURE_DIM, FULL_FEATURE_DIM,
    )
    f = handshape_features(_frame_with_left_hand(seed=3))
    g = geometric_features(f)
    assert g.shape == (GEOMETRIC_FEATURE_DIM,) == (15,)
    X = np.stack([f, handshape_features(_frame_with_left_hand(seed=4))])
    aug = augment_features(X)
    assert aug.shape == (2, FULL_FEATURE_DIM) == (2, 78)


def test_mlp_forward_shape_and_learns():
    import torch
    from models.fingerspelling.train_fingerspelling import MLP, train_loop
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(24, 63)).astype("float32") * 3   # 24 separable clusters in 63-D
    X = np.repeat(centers, 8, axis=0) + rng.normal(size=(192, 63)).astype("float32") * 0.1
    y = np.repeat(np.arange(24), 8).astype("int64")
    model = MLP(63, 24)
    assert model(torch.from_numpy(X)).shape == (192, 24)
    acc = train_loop(model, X, y, X, y, epochs=80, lr=1e-2, batch_size=64)
    assert acc > 0.95   # must fit a trivially separable set


def _stub_recognizer(per_letter=None, global_thr=0.5):
    from models.fingerspelling.recognizer import FingerspellingRecognizer
    from models.fingerspelling.labels import LETTERS as L
    r = object.__new__(FingerspellingRecognizer)  # no model load
    r.letters = L
    r.global_threshold = global_thr
    r.per_letter_threshold = per_letter or {}
    return r


def _onehot_probs(letter, conf, n=3):
    from models.fingerspelling.labels import LETTER_TO_INDEX
    rows = np.full((n, 24), (1 - conf) / 23, dtype=np.float32)
    rows[:, LETTER_TO_INDEX[letter]] = conf
    return rows


def test_decide_accepts_confident_reliable_letter():
    r = _stub_recognizer()
    letter, conf, reason = r.decide_from_probs(_onehot_probs("a", 0.9))
    assert letter == "a" and reason == "accept"


def test_decide_abstains_low_confidence():
    r = _stub_recognizer(global_thr=0.5)
    letter, conf, reason = r.decide_from_probs(_onehot_probs("a", 0.4))
    assert letter is None and reason.startswith("abstain")


def test_decide_abstains_capped_letter_even_if_confident():
    r = _stub_recognizer(per_letter={"n": 1.0})  # n is at the ceiling -> always abstain
    letter, conf, reason = r.decide_from_probs(_onehot_probs("n", 0.99))
    assert letter is None  # never emits a capped letter


def test_decide_no_hand_on_empty():
    r = _stub_recognizer()
    letter, conf, reason = r.decide_from_probs(np.empty((0, 24), dtype=np.float32))
    assert letter is None and reason == "no_hand"


def test_should_fingerspell_routing():
    from abstention import AbstentionPolicy
    from models.fingerspelling.recognizer import should_fingerspell
    policy = AbstentionPolicy(conf_threshold=0.5)
    assert should_fingerspell([("pain", 0.9), ("hurt", 0.05)], policy) is False   # word confident
    assert should_fingerspell([("pain", 0.4), ("hurt", 0.3)], policy) is True     # word abstains -> spell
    assert should_fingerspell([("pain", 0.9)], policy, manual_spell_mode=True) is True  # manual


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("OK — fingerspelling features pass.")
