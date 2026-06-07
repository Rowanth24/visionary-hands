"""Static handshape feature: dominant 21 hand landmarks, wrist-centred, scale-normalized,
handedness-canonical (mirror right->left), xyz. ONE function used at train and serve (no skew)."""
from __future__ import annotations

import numpy as np

from landmark_baseline.constants import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

HAND_POINTS = 21
HAND_FEATURE_DIM = HAND_POINTS * 3  # 63

# MediaPipe hand topology (landmark indices within the 21-point hand)
_FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
_TIPS = [4, 8, 12, 16, 20]
GEOMETRIC_FEATURE_DIM = 5 + 5 + 4 + 1  # curl(5) + tip-to-wrist(5) + adj inter-tip(4) + thumb-index(1) = 15
FULL_FEATURE_DIM = HAND_FEATURE_DIM + GEOMETRIC_FEATURE_DIM  # 78


def geometric_features(hand63: np.ndarray) -> np.ndarray:
    """Explicit finger-state geometry from the normalized 21x3 hand points (appearance-invariant).

    Targets the folded/spread confusions (m/n/s, u/v) that raw coords miss: per-finger curl
    (cos angle at PIP), tip-to-wrist extension, adjacent inter-fingertip spread, thumb-to-index.
    """
    p = np.asarray(hand63, dtype=np.float32).reshape(HAND_POINTS, 3)
    feats = []
    for idx in _FINGERS.values():
        mcp, pip, _dip, tip = (p[i] for i in idx)
        v1, v2 = pip - mcp, tip - pip
        cos = float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6))
        feats.append(cos)                              # curl
        feats.append(float(np.linalg.norm(tip - p[0])))  # tip-to-wrist extension
    for a, b in zip(_TIPS[:-1], _TIPS[1:]):
        feats.append(float(np.linalg.norm(p[a] - p[b])))  # adjacent inter-tip spread
    feats.append(float(np.linalg.norm(p[4] - p[5])))      # thumb-tip to index-mcp
    return np.array(feats, dtype=np.float32)


def augment_features(x63: np.ndarray) -> np.ndarray:
    """(N, 63) -> (N, 78): append geometric features to the normalized landmark coords."""
    x63 = np.asarray(x63, dtype=np.float32)
    geo = np.stack([geometric_features(row) for row in x63])
    return np.concatenate([x63, geo], axis=1)


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
