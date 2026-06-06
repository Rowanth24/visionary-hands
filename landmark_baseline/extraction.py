"""MediaPipe landmark extraction helpers for local video inference and dataset prep."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import FACE_SLICE, LEFT_HAND_SLICE, POSE_SLICE, RIGHT_HAND_SLICE


def extract_video_landmarks(video_path: Path, frame_stride: int = 1) -> np.ndarray:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'MediaPipe landmark extraction requires OpenCV and MediaPipe in the active Python environment.'
        ) from exc

    mp_holistic = mp.solutions.holistic
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    frames: list[np.ndarray] = []
    frame_idx = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_stride > 1 and (frame_idx % frame_stride) != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
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

            frames.append(rows)
            frame_idx += 1

    capture.release()
    if not frames:
        raise RuntimeError(f'No frames decoded from video: {video_path}')
    return np.stack(frames, axis=0).astype(np.float32)
