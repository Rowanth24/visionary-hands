#!/usr/bin/env python3
"""Export a landmark checkpoint to ONNX for edge-runtime comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landmark_baseline.features import append_frame_deltas_np
from landmark_baseline.model import build_landmark_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to best.pt')
    parser.add_argument('--data-root', type=Path, required=True, help='Prepared dataset root used for shape inference')
    parser.add_argument('--output', type=Path, required=True, help='Output path for model.onnx')
    parser.add_argument('--model-type', type=str, choices=('transformer', 'gru'), default=None)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gru-hidden-size', type=int, default=96)
    parser.add_argument('--gru-unidirectional', action='store_true')
    parser.add_argument('--opset-version', type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    labels = checkpoint.get('labels')
    if labels is None:
        labels = json.loads((args.data_root / 'labels.json').read_text(encoding='utf-8'))

    train_payload = np.load(args.data_root / 'train' / 'dataset.npz')
    feature_dim = int(train_payload['X'].shape[2] * train_payload['X'].shape[3])
    max_frames = int(train_payload['X'].shape[1])

    model_type = args.model_type or config.get('model_type', 'transformer')
    d_model = int(config.get('d_model', args.d_model))
    num_heads = int(config.get('num_heads', args.num_heads))
    num_layers = int(config.get('num_layers', args.num_layers))
    dropout = float(config.get('dropout', args.dropout))
    gru_hidden_size = int(config.get('gru_hidden_size', args.gru_hidden_size))
    gru_bidirectional = bool(config.get('gru_bidirectional', not args.gru_unidirectional))
    append_deltas = bool(config.get('append_deltas', False))
    if append_deltas:
        feature_dim *= 2

    model = build_landmark_model(
        model_type=model_type,
        feature_dim=feature_dim,
        num_classes=len(labels),
        max_frames=max_frames,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        gru_hidden_size=gru_hidden_size,
        gru_bidirectional=gru_bidirectional,
    ).cpu()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    example_x_np = train_payload['X'][:1].astype(np.float32).reshape(1, max_frames, -1)
    if append_deltas:
        example_x_np = append_frame_deltas_np(example_x_np)
    example_x = torch.from_numpy(example_x_np)
    example_mask = torch.from_numpy((train_payload['non_empty_frame_idxs'][:1].astype(np.bool_)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example_x, example_mask),
        args.output,
        input_names=['x', 'valid_mask'],
        output_names=['logits'],
        dynamic_axes={
            'x': {0: 'batch'},
            'valid_mask': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=args.opset_version,
        do_constant_folding=True,
    )
    print(
        json.dumps(
            {
                'checkpoint': str(args.checkpoint),
                'output': str(args.output),
                'model_type': model_type,
                'feature_dim': feature_dim,
                'max_frames': max_frames,
                'num_classes': len(labels),
                'append_deltas': append_deltas,
                'opset_version': args.opset_version,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
