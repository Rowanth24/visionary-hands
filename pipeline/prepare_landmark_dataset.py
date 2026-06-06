#!/usr/bin/env python3
"""Convert extracted raw landmark sequences into fixed-shape training arrays."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from landmark_baseline.constants import TARGET_FRAME_COUNT
from landmark_baseline.preprocess import preprocess_landmark_sequence

SPLITS = ('train', 'val', 'test')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True, help='Directory created by extract_wlasl_landmarks.py')
    parser.add_argument('--output-root', type=Path, required=True, help='Directory for processed split arrays')
    parser.add_argument('--target-frames', type=int, default=TARGET_FRAME_COUNT)
    parser.add_argument('--xy-only', action='store_true', help='Drop the z channel after preprocessing')
    parser.add_argument('--labels-json', type=Path, default=None, help='Optional explicit label list to include')
    parser.add_argument(
        '--layout',
        type=str,
        choices=('dominant_hand', 'dominant_both_hands'),
        default='dominant_hand',
        help='Canonical landmark layout to build into the prepared arrays.',
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise RuntimeError(f'Manifest not found: {path}')
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.raw_root / 'manifest.jsonl')
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest:
        by_split[str(row['split'])].append(row)

    available_train_labels = sorted({str(row['label']) for row in by_split['train']})
    if args.labels_json is not None:
        requested = [str(label) for label in json.loads(args.labels_json.read_text(encoding='utf-8'))]
        train_labels = [label for label in requested if label in available_train_labels]
    else:
        train_labels = available_train_labels
    if len(train_labels) < 2:
        raise RuntimeError('Need at least two labels in the train split to build a dataset.')
    label_to_idx = {label: idx for idx, label in enumerate(train_labels)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'labels.json').write_text(json.dumps(train_labels, indent=2), encoding='utf-8')

    summary: dict[str, object] = {
        'labels': train_labels,
        'target_frames': int(args.target_frames),
        'xy_only': bool(args.xy_only),
        'layout': str(args.layout),
        'splits': {},
    }

    for split in SPLITS:
        samples = by_split.get(split, [])
        frames_list = []
        frame_idx_list = []
        y_list = []
        meta_rows = []
        for row in samples:
            label = str(row['label'])
            if label not in label_to_idx:
                continue
            payload = np.load(str(row['landmark_path']))
            processed = preprocess_landmark_sequence(
                payload['landmarks'],
                target_frames=args.target_frames,
                layout=args.layout,
            )
            frames = processed.frames[..., :2] if args.xy_only else processed.frames
            frames_list.append(frames.astype(np.float32))
            frame_idx_list.append(processed.non_empty_frame_idxs.astype(np.float32))
            y_list.append(label_to_idx[label])
            meta_rows.append(
                {
                    'label': label,
                    'y': label_to_idx[label],
                    'video_path': row['video_path'],
                    'landmark_path': row['landmark_path'],
                    'dominant_hand': processed.dominant_hand,
                    'layout': processed.layout,
                    'source_num_frames': processed.source_num_frames,
                    'kept_num_frames': processed.kept_num_frames,
                }
            )

        if not frames_list:
            continue

        split_dir = args.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            split_dir / 'dataset.npz',
            X=np.stack(frames_list, axis=0).astype(np.float32),
            non_empty_frame_idxs=np.stack(frame_idx_list, axis=0).astype(np.float32),
            y=np.asarray(y_list, dtype=np.int64),
        )
        (split_dir / 'manifest.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in meta_rows) + ('\n' if meta_rows else ''),
            encoding='utf-8',
        )
        summary['splits'][split] = {
            'samples': len(frames_list),
            'labels': sorted({row['label'] for row in meta_rows}),
            'feature_shape': list(np.stack(frames_list[:1], axis=0).shape[1:]),
        }

    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
