#!/usr/bin/env python3
"""Record a short webcam clip and score it with the ONNX landmark model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from landmark_baseline.features import load_runtime_config
from landmark_baseline.inference_onnx import load_labels, load_onnx_session, predict_video_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help='Path to the exported ONNX model.')
    parser.add_argument('--labels', type=Path, required=True, help='Path to labels.json.')
    parser.add_argument('--config', type=Path, default=None, help='Optional config.json. Defaults to model sibling config.json if present.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory to write the recorded clip and JSON result.')
    parser.add_argument('--expected-label', type=str, required=True, help='Expected sign label for this capture.')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--duration-seconds', type=float, default=3.0)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--target-frames', type=int, default=64)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--include-z', action='store_true', help='Use xyz landmarks instead of xy-only features.')
    return parser.parse_args()


def main() -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('validate_live_sign_capture_onnx.py requires OpenCV in the active Python environment.') from exc

    args = parse_args()
    labels = load_labels(args.labels)
    if args.expected_label not in labels:
        raise RuntimeError(f'Expected label {args.expected_label!r} is not present in {args.labels}')
    config_path = args.config or (args.model.parent / 'config.json')
    config = load_runtime_config(config_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    clip_path = args.output_dir / f'{timestamp}_{args.expected_label}.mp4'
    result_path = args.output_dir / f'{timestamp}_{args.expected_label}.json'

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open camera index {args.camera_index}')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(clip_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f'Failed to open writer for {clip_path}')

    started = time.time()
    try:
        while (time.time() - started) < max(0.5, args.duration_seconds):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            preview = frame.copy()
            cv2.putText(
                preview,
                f'Capture {args.expected_label}',
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('Live Sign Capture', preview)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break
    finally:
        capture.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    session = load_onnx_session(args.model)
    result = predict_video_onnx(
        session=session,
        labels=labels,
        video_path=clip_path,
        frame_stride=max(1, args.frame_stride),
        target_frames=args.target_frames,
        xy_only=not args.include_z,
        append_deltas=bool(config.get('append_deltas', False)),
        layout=str(config.get('layout', 'dominant_hand')),
        topk=max(1, args.topk),
    )

    payload = {
        'clip_path': str(clip_path),
        'model': str(args.model),
        'labels': str(args.labels),
        'config': str(config_path) if config else None,
        'provider': 'CPUExecutionProvider',
        'expected_label': args.expected_label,
        'correct': result.predicted_label == args.expected_label,
        **result.to_dict(),
    }
    result_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
