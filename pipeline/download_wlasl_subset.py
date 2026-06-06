#!/usr/bin/env python3
"""Download a label-filtered WLASL subset from Kaggle while preserving split dirs."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

SPLITS = ('train', 'val', 'test')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='thtrnphc/wlasl1000')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--labels-json', type=Path, required=True)
    parser.add_argument('--page-size', type=int, default=500)
    parser.add_argument('--max-files-per-label-split', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_labels(path: Path) -> list[str]:
    labels = json.loads(path.read_text(encoding='utf-8'))
    return [str(label).strip() for label in labels if str(label).strip()]


def main() -> None:
    args = parse_args()
    labels = set(load_labels(args.labels_json))
    if not labels:
        raise RuntimeError('No labels provided.')

    api = KaggleApi()
    api.authenticate()

    counts = {split: {label: 0 for label in labels} for split in SPLITS}
    selected: list[str] = []
    page_token = None
    while True:
        resp = api.dataset_list_files(args.dataset, page_token=page_token, page_size=args.page_size)
        files = list(getattr(resp, 'dataset_files', []) or [])
        for entry in files:
            name = str(entry.name)
            parts = Path(name).parts
            if len(parts) < 4:
                continue
            _, split, label, filename = parts[0], parts[1], parts[2], parts[-1]
            if split not in SPLITS or label not in labels:
                continue
            if args.max_files_per_label_split > 0 and counts[split][label] >= args.max_files_per_label_split:
                continue
            dest_dir = args.output_root / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            if dest_path.exists() and not args.overwrite:
                counts[split][label] += 1
                continue
            api.dataset_download_file(args.dataset, name, path=str(dest_dir), force=True, quiet=False)
            flat_path = dest_dir / filename
            if not flat_path.exists():
                archive_path = dest_dir / f'{filename}.zip'
                if archive_path.exists():
                    with zipfile.ZipFile(archive_path) as zf:
                        zf.extractall(dest_dir)
                    archive_path.unlink()
                if not flat_path.exists():
                    raise RuntimeError(f'Expected downloaded file at {flat_path}')
            counts[split][label] += 1
            selected.append(name)
        page_token = getattr(resp, 'next_page_token', None)
        if not page_token:
            break

    summary = {
        'dataset': args.dataset,
        'labels': sorted(labels),
        'counts': counts,
        'downloaded_files': len(selected),
    }
    (args.output_root / 'download_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
