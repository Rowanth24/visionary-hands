#!/usr/bin/env python3
"""Benchmark ONNX landmark models on prepared arrays."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help='Path to model.onnx')
    parser.add_argument('--dataset', type=Path, required=True, help='Path to prepared dataset.npz')
    parser.add_argument('--warmup-runs', type=int, default=10)
    parser.add_argument('--timed-runs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-json', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.dataset)
    x = payload['X'].astype(np.float32).reshape(payload['X'].shape[0], payload['X'].shape[1], -1)
    valid_mask = (payload['non_empty_frame_idxs'].astype(np.float32) >= 0)

    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    input_x = session.get_inputs()[0].name
    input_mask = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    batches = []
    for start in range(0, x.shape[0], max(1, args.batch_size)):
        end = min(x.shape[0], start + max(1, args.batch_size))
        batches.append((x[start:end], valid_mask[start:end]))
    if not batches:
        raise RuntimeError(f'No samples found in {args.dataset}')

    for _ in range(max(0, args.warmup_runs)):
        for xb, mb in batches:
            _ = session.run([output_name], {input_x: xb, input_mask: mb})

    start = time.perf_counter()
    for _ in range(max(1, args.timed_runs)):
        for xb, mb in batches:
            _ = session.run([output_name], {input_x: xb, input_mask: mb})
    elapsed = time.perf_counter() - start

    total_inferences = max(1, args.timed_runs) * sum(int(batch[0].shape[0]) for batch in batches)
    result = {
        'model': str(args.model),
        'dataset': str(args.dataset),
        'provider': 'CPUExecutionProvider',
        'samples': int(x.shape[0]),
        'batch_size': max(1, args.batch_size),
        'warmup_runs': max(0, args.warmup_runs),
        'timed_runs': max(1, args.timed_runs),
        'total_inferences': total_inferences,
        'total_seconds': elapsed,
        'seconds_per_inference': elapsed / total_inferences,
        'ms_per_inference': (elapsed * 1000.0) / total_inferences,
        'model_bytes': int(args.model.stat().st_size),
    }

    encoded = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
