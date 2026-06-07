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
