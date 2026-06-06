"""Preprocessing utilities for landmark-based isolated sign models.

This ports the key SignSense ideas into a small NumPy module that is usable in
training pipelines and on-device inference wrappers without requiring TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import (
    LANDMARK_DIMS,
    LANDMARK_IDXS_LEFT_DOMINANT0,
    LANDMARK_IDXS_LEFT_DOMINANT_BOTH0,
    LANDMARK_IDXS_RIGHT_DOMINANT0,
    LANDMARK_IDXS_RIGHT_DOMINANT_BOTH0,
    LEFT_HAND_IDXS0,
    LIPS_COUNT,
    NON_LIP_X_OFFSET,
    PROCESSED_LANDMARK_COUNT,
    PROCESSED_LANDMARK_COUNT_BOTH,
    RIGHT_HAND_IDXS0,
    ROWS_PER_FRAME,
    TARGET_FRAME_COUNT,
)


@dataclass(frozen=True)
class PreprocessedSequence:
    frames: np.ndarray
    non_empty_frame_idxs: np.ndarray
    dominant_hand: str
    layout: str
    source_num_frames: int
    kept_num_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'frames': self.frames,
            'non_empty_frame_idxs': self.non_empty_frame_idxs,
            'dominant_hand': self.dominant_hand,
            'layout': self.layout,
            'source_num_frames': self.source_num_frames,
            'kept_num_frames': self.kept_num_frames,
        }


def _validate_sequence(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected sequence with shape [T, {ROWS_PER_FRAME}, {LANDMARK_DIMS}], got {arr.shape}')
    if arr.shape[1] != ROWS_PER_FRAME or arr.shape[2] != LANDMARK_DIMS:
        raise ValueError(f'Expected shape [T, {ROWS_PER_FRAME}, {LANDMARK_DIMS}], got {arr.shape}')
    return arr


def _frame_presence_count(sequence: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    hand = sequence[:, idxs, :]
    return np.sum(~np.isnan(hand), axis=(1, 2))


def infer_dominant_hand(sequence: np.ndarray) -> str:
    arr = _validate_sequence(sequence)
    left_present = int(np.sum(_frame_presence_count(arr, LEFT_HAND_IDXS0)))
    right_present = int(np.sum(_frame_presence_count(arr, RIGHT_HAND_IDXS0)))
    if left_present == 0 and right_present == 0:
        return 'unknown'
    return 'left' if left_present >= right_present else 'right'


def _pad_or_pool(
    frames: np.ndarray,
    non_empty_frame_idxs: np.ndarray,
    target_frames: int,
    landmark_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) == 0:
        return (
            np.zeros((target_frames, landmark_count, LANDMARK_DIMS), dtype=np.float32),
            np.full((target_frames,), -1.0, dtype=np.float32),
        )

    if len(frames) <= target_frames:
        out_frames = np.zeros((target_frames, frames.shape[1], frames.shape[2]), dtype=np.float32)
        out_indices = np.full((target_frames,), -1.0, dtype=np.float32)
        out_frames[: len(frames)] = np.nan_to_num(frames, nan=0.0)
        out_indices[: len(non_empty_frame_idxs)] = non_empty_frame_idxs.astype(np.float32)
        return out_frames, out_indices

    edges = np.linspace(0, len(frames), target_frames + 1, dtype=np.int32)
    pooled_frames = np.zeros((target_frames, frames.shape[1], frames.shape[2]), dtype=np.float32)
    pooled_indices = np.full((target_frames,), -1.0, dtype=np.float32)

    for i in range(target_frames):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            end = min(len(frames), start + 1)
            start = max(0, end - 1)
        window = frames[start:end]
        window_idxs = non_empty_frame_idxs[start:end]
        valid_counts = np.sum(~np.isnan(window), axis=0)
        pooled_frames[i] = np.divide(
            np.nansum(window, axis=0),
            valid_counts,
            out=np.zeros_like(pooled_frames[i]),
            where=valid_counts > 0,
        )
        if len(window_idxs):
            idx_valid = ~np.isnan(window_idxs)
            pooled_indices[i] = float(window_idxs[idx_valid].mean()) if np.any(idx_valid) else -1.0

    pooled_frames = np.nan_to_num(pooled_frames, nan=0.0)
    return pooled_frames.astype(np.float32), pooled_indices.astype(np.float32)


def _resolve_layout(layout: str) -> tuple[str, int]:
    name = str(layout).strip().lower()
    if name == 'dominant_hand':
        return name, PROCESSED_LANDMARK_COUNT
    if name == 'dominant_both_hands':
        return name, PROCESSED_LANDMARK_COUNT_BOTH
    raise ValueError(f'Unsupported landmark layout: {layout!r}')


def preprocess_landmark_sequence(
    sequence: np.ndarray,
    target_frames: int = TARGET_FRAME_COUNT,
    mirror_right_to_left: bool = True,
    layout: str = 'dominant_hand',
) -> PreprocessedSequence:
    arr = _validate_sequence(sequence)
    dominant = infer_dominant_hand(arr)
    layout_name, landmark_count = _resolve_layout(layout)

    if dominant == 'right':
        if layout_name == 'dominant_both_hands':
            selected = arr[:, LANDMARK_IDXS_RIGHT_DOMINANT_BOTH0, :].copy()
        else:
            selected = arr[:, LANDMARK_IDXS_RIGHT_DOMINANT0, :].copy()
        hand_presence = _frame_presence_count(arr, RIGHT_HAND_IDXS0)
        if mirror_right_to_left:
            selected[:, NON_LIP_X_OFFSET:, 0] = 1.0 - selected[:, NON_LIP_X_OFFSET:, 0]
    else:
        if layout_name == 'dominant_both_hands':
            selected = arr[:, LANDMARK_IDXS_LEFT_DOMINANT_BOTH0, :].copy()
        else:
            selected = arr[:, LANDMARK_IDXS_LEFT_DOMINANT0, :].copy()
        hand_presence = _frame_presence_count(arr, LEFT_HAND_IDXS0)
        dominant = 'left' if dominant == 'unknown' else dominant

    non_empty = np.flatnonzero(hand_presence > 0)
    if non_empty.size == 0:
        kept_frames = selected
        kept_indices = np.arange(selected.shape[0], dtype=np.float32)
    else:
        kept_frames = selected[non_empty]
        kept_indices = non_empty.astype(np.float32)

    frames, frame_idxs = _pad_or_pool(kept_frames, kept_indices, int(target_frames), landmark_count=landmark_count)
    return PreprocessedSequence(
        frames=frames,
        non_empty_frame_idxs=frame_idxs,
        dominant_hand=dominant,
        layout=layout_name,
        source_num_frames=int(arr.shape[0]),
        kept_num_frames=int(kept_frames.shape[0]),
    )
