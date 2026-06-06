#!/usr/bin/env python3
"""Run offline webcam inference with the landmark TorchScript model."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from landmark_baseline.constants import FACE_SLICE, LEFT_HAND_SLICE, POSE_SLICE, RIGHT_HAND_SLICE, ROWS_PER_FRAME
from landmark_baseline.features import load_runtime_config
from landmark_baseline.inference import load_labels, load_torchscript_model, predict_from_landmarks, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help='Path to the exported TorchScript model.')
    parser.add_argument('--labels', type=Path, required=True, help='Path to labels.json.')
    parser.add_argument('--config', type=Path, default=None, help='Optional config.json. Defaults to model sibling config.json if present.')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, mps, or cuda')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--history-frames', type=int, default=64)
    parser.add_argument('--predict-every', type=int, default=4)
    parser.add_argument('--include-z', action='store_true', help='Use xyz landmarks instead of xy-only features.')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError('run_landmark_webcam.py requires OpenCV and MediaPipe in the active Python environment.') from exc

    args = parse_args()
    device = resolve_device(args.device)
    labels = load_labels(args.labels)
    model = load_torchscript_model(args.model, device)
    config_path = args.config or (args.model.parent / 'config.json')
    config = load_runtime_config(config_path)

    mp_holistic = mp.solutions.holistic
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open camera index {args.camera_index}')

    history: deque[np.ndarray] = deque(maxlen=max(1, args.history_frames))
    frame_count = 0
    last_label = 'warming_up'
    last_conf = 0.0

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                rows = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
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

                history.append(rows)
                frame_count += 1

                if len(history) >= 8 and (frame_count % max(1, args.predict_every) == 0):
                    sequence = np.stack(history, axis=0)
                    result = predict_from_landmarks(
                        model=model,
                        labels=labels,
                        landmarks=sequence,
                        device=device,
                        target_frames=max(1, args.history_frames),
                        xy_only=not args.include_z,
                        append_deltas=bool(config.get('append_deltas', False)),
                        layout=str(config.get('layout', 'dominant_hand')),
                        topk=3,
                    )
                    last_label = result.predicted_label
                    last_conf = result.confidence

                cv2.putText(
                    frame,
                    f'{last_label} {last_conf:.2f}',
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow('Landmark Sign Inference', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
