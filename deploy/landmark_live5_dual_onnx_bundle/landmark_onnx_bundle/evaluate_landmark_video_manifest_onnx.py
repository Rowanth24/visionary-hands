#!/usr/bin/env python3
"""Evaluate the offline ONNX video inference path against a manifest file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from landmark_baseline.features import load_runtime_config
from landmark_baseline.inference_onnx import load_labels, load_onnx_session, predict_video_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True, help='Path to manifest.jsonl with video_path and label fields.')
    parser.add_argument('--model', type=Path, required=True, help='Path to the exported ONNX model.')
    parser.add_argument('--labels', type=Path, required=True, help='Path to labels.json.')
    parser.add_argument('--config', type=Path, default=None, help='Optional config.json. Defaults to model sibling config.json if present.')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--target-frames', type=int, default=64)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--include-z', action='store_true', help='Use xyz landmarks instead of xy-only features.')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-json', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    session = load_onnx_session(args.model)
    config_path = args.config or (args.model.parent / 'config.json')
    config = load_runtime_config(config_path)

    rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    predictions: list[dict[str, object]] = []
    correct = 0
    for row in rows:
        video_path = Path(str(row['video_path']))
        truth = str(row['label'])
        result = predict_video_onnx(
            session=session,
            labels=labels,
            video_path=video_path,
            frame_stride=max(1, args.frame_stride),
            target_frames=args.target_frames,
            xy_only=not args.include_z,
            append_deltas=bool(config.get('append_deltas', False)),
            layout=str(config.get('layout', 'dominant_hand')),
            topk=max(1, args.topk),
        )
        hit = result.predicted_label == truth
        correct += int(hit)
        predictions.append(
            {
                'video_path': str(video_path),
                'true_label': truth,
                'correct': hit,
                **result.to_dict(),
            }
        )

    payload = {
        'manifest': str(args.manifest),
        'model': str(args.model),
        'labels': str(args.labels),
        'config': str(config_path) if config else None,
        'provider': 'CPUExecutionProvider',
        'samples': len(predictions),
        'accuracy': (correct / len(predictions)) if predictions else 0.0,
        'predictions': predictions,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
