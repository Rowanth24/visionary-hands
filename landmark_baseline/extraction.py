"""MediaPipe landmark extraction helpers for local video inference and dataset prep."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import FACE_SLICE, LEFT_HAND_SLICE, POSE_SLICE, RIGHT_HAND_SLICE


def _rows_from_holistic_result(results) -> np.ndarray:
    """Pack one MediaPipe Holistic result into the canonical 543x3 landmark grid."""
    rows = np.full((543, 3), np.nan, dtype=np.float32)
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            rows[FACE_SLICE.start + i] = (point.x, point.y, point.z)
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            rows[LEFT_HAND_SLICE.start + i] = (point.x, point.y, point.z)
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            rows[POSE_SLICE.start + i] = (point.x, point.y, point.z)
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            rows[RIGHT_HAND_SLICE.start + i] = (point.x, point.y, point.z)
    return rows


def _landmarks_from_rgb_frames(rgb_frames) -> list[np.ndarray]:
    """Run MediaPipe Holistic over an iterable of RGB frames."""
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    rows: list[np.ndarray] = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for rgb in rgb_frames:
            rows.append(_rows_from_holistic_result(holistic.process(rgb)))
    return rows


def extract_video_landmarks(video_path: Path, frame_stride: int = 1) -> np.ndarray:
    try:
        import cv2
        import mediapipe as mp  # noqa: F401  (validate availability before decoding)
    except ImportError as exc:
        raise RuntimeError(
            'MediaPipe landmark extraction requires OpenCV and MediaPipe in the active Python environment.'
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    rgb_frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame_stride > 1 and (frame_idx % frame_stride) != 0:
            frame_idx += 1
            continue
        rgb_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    capture.release()

    frames = _landmarks_from_rgb_frames(rgb_frames)
    if not frames:
        raise RuntimeError(f'No frames decoded from video: {video_path}')
    return np.stack(frames, axis=0).astype(np.float32)


def extract_landmarks_from_frames(frames, frame_stride: int = 1, frames_are_bgr: bool = True) -> np.ndarray:
    """Extract landmarks from in-memory frames (e.g. the benchmark harness's cv2 BGR frames).

    Mirrors extract_video_landmarks but skips video decoding so the same
    dual-hand preprocessing pipeline can run on frames the harness already read.
    """
    try:
        import cv2
        import mediapipe as mp  # noqa: F401  (validate availability before processing)
    except ImportError as exc:
        raise RuntimeError(
            'MediaPipe landmark extraction requires OpenCV and MediaPipe in the active Python environment.'
        ) from exc

    selected = list(frames)[:: max(1, frame_stride)] if frame_stride > 1 else list(frames)
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if frames_are_bgr else f for f in selected]
    rows = _landmarks_from_rgb_frames(rgb_frames)
    if not rows:
        raise RuntimeError('No frames provided for landmark extraction.')
    return np.stack(rows, axis=0).astype(np.float32)
