#!/usr/bin/env python3
"""Run offline landmark inference on a local video using a TorchScript model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from landmark_baseline.features import load_runtime_config
from landmark_baseline.inference import load_labels, load_torchscript_model, predict_video, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True, help='Path to a local video clip.')
    parser.add_argument('--model', type=Path, required=True, help='Path to the exported TorchScript model.')
    parser.add_argument('--labels', type=Path, required=True, help='Path to labels.json.')
    parser.add_argument('--config', type=Path, default=None, help='Optional config.json. Defaults to model sibling config.json if present.')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, mps, or cuda')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--target-frames', type=int, default=64)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--include-z', action='store_true', help='Use xyz landmarks instead of xy-only features.')
    parser.add_argument('--output-json', type=Path, default=None, help='Optional path to write the prediction JSON.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    labels = load_labels(args.labels)
    model = load_torchscript_model(args.model, device)
    config_path = args.config or (args.model.parent / 'config.json')
    config = load_runtime_config(config_path)
    result = predict_video(
        model=model,
        labels=labels,
        video_path=args.video,
        device=device,
        frame_stride=max(1, args.frame_stride),
        target_frames=args.target_frames,
        xy_only=not args.include_z,
        append_deltas=bool(config.get('append_deltas', False)),
        layout=str(config.get('layout', 'dominant_hand')),
        topk=max(1, args.topk),
    )

    payload = {
        'video': str(args.video),
        'model': str(args.model),
        'labels': str(args.labels),
        'config': str(config_path) if config else None,
        'device': device.type,
        **result.to_dict(),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
