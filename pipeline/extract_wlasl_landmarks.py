#!/usr/bin/env python3
"""Extract MediaPipe Holistic landmarks from WLASL split directories.

Outputs one `.npz` file per clip with shape `[T, 543, 3]` and a manifest file that
can be fed into `prepare_landmark_dataset.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from landmark_baseline.constants import LANDMARK_LAYOUT_NAME
from landmark_baseline.extraction import extract_video_landmarks

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
SPLITS = ('train', 'val', 'test')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True, help='Root containing train/val/test label directories')
    parser.add_argument('--output-root', type=Path, required=True, help='Directory to write extracted landmark sequences')
    parser.add_argument('--target-classes', type=int, default=8)
    parser.add_argument('--min-train-per-class', type=int, default=10)
    parser.add_argument('--min-val-per-class', type=int, default=2)
    parser.add_argument('--min-test-per-class', type=int, default=2)
    parser.add_argument('--labels-json', type=Path, default=None, help='Optional explicit label list to use')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--max-videos', type=int, default=0, help='Optional hard cap for quick smoke runs')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def iter_label_dirs(root: Path, split: str) -> Iterable[Path]:
    split_root = root / split
    if not split_root.exists():
        return []
    return sorted([p for p in split_root.iterdir() if p.is_dir()])


def list_videos(label_dir: Path) -> list[Path]:
    return sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def load_requested_labels(args: argparse.Namespace, counts: dict[str, dict[str, int]]) -> list[str]:
    if args.labels_json is not None:
        labels = json.loads(args.labels_json.read_text(encoding='utf-8'))
        return [str(label) for label in labels]

    candidates: list[tuple[int, str]] = []
    for label in sorted(set().union(*[set(split_counts) for split_counts in counts.values()])):
        train_n = counts['train'].get(label, 0)
        val_n = counts['val'].get(label, 0)
        test_n = counts['test'].get(label, 0)
        if train_n < args.min_train_per_class or val_n < args.min_val_per_class or test_n < args.min_test_per_class:
            continue
        candidates.append((train_n + val_n + test_n, label))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [label for _, label in candidates[: max(1, args.target_classes)]]


def build_counts(root: Path) -> dict[str, dict[str, int]]:
    counts = {split: {} for split in SPLITS}
    for split in SPLITS:
        for label_dir in iter_label_dirs(root, split):
            counts[split][label_dir.name] = len(list_videos(label_dir))
    return counts


def main() -> None:
    args = parse_args()
    counts = build_counts(args.dataset_root)
    labels = load_requested_labels(args, counts)
    if not labels:
        raise RuntimeError('No labels matched the requested dense-class constraints.')

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / 'manifest.jsonl'
    summary_path = args.output_root / 'summary.json'
    written = 0
    manifest_lines: list[str] = []

    for split in SPLITS:
        for label in labels:
            label_dir = args.dataset_root / split / label
            if not label_dir.exists():
                continue
            for video_path in list_videos(label_dir):
                rel = Path(split) / label / f'{video_path.stem}.npz'
                out_path = args.output_root / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not args.overwrite:
                    continue
                landmarks = extract_video_landmarks(video_path, frame_stride=max(1, args.frame_stride))
                np.savez_compressed(out_path, landmarks=landmarks)
                manifest_lines.append(
                    json.dumps(
                        {
                            'split': split,
                            'label': label,
                            'video_path': str(video_path),
                            'landmark_path': str(out_path),
                            'num_frames': int(landmarks.shape[0]),
                            'layout': LANDMARK_LAYOUT_NAME,
                        }
                    )
                )
                written += 1
                if args.max_videos > 0 and written >= args.max_videos:
                    break
            if args.max_videos > 0 and written >= args.max_videos:
                break
        if args.max_videos > 0 and written >= args.max_videos:
            break

    manifest_path.write_text('\n'.join(manifest_lines) + ('\n' if manifest_lines else ''), encoding='utf-8')
    summary_path.write_text(
        json.dumps(
            {
                'dataset_root': str(args.dataset_root),
                'output_root': str(args.output_root),
                'labels': labels,
                'counts': counts,
                'written_samples': written,
                'frame_stride': max(1, args.frame_stride),
                'layout': LANDMARK_LAYOUT_NAME,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(json.dumps({'labels': labels, 'written_samples': written, 'manifest': str(manifest_path)}, indent=2))


if __name__ == '__main__':
    main()
