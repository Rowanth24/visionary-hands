#!/usr/bin/env python3
"""Assemble a self-contained ONNX landmark deployment bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


RUNTIME_FILES = [
    'predict_landmark_video_onnx.py',
    'evaluate_landmark_video_manifest_onnx.py',
    'run_landmark_webcam_onnx.py',
    'validate_live_sign_capture_onnx.py',
]

BASELINE_FILES = [
    '__init__.py',
    'constants.py',
    'extraction.py',
    'features.py',
    'preprocess.py',
    'prediction.py',
    'inference_onnx.py',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True, help='Directory containing model.onnx, labels.json, config.json, metrics.json')
    parser.add_argument('--output-dir', type=Path, required=True, help='Destination directory for the deployment bundle')
    parser.add_argument('--bundle-name', type=str, default='landmark_onnx_bundle')
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_requirements(path: Path) -> None:
    content = '\n'.join([
        'numpy',
        'onnxruntime',
        'opencv-contrib-python',
        '# MediaPipe wheel availability depends on the target OS/arch',
        '# Install the appropriate mediapipe package for the target device',
    ]) + '\n'
    path.write_text(content, encoding='utf-8')


def write_readme(path: Path, bundle_name: str, labels: list[str]) -> None:
    content = f"""# {bundle_name}

## Contents

- `model.onnx`
- `labels.json`
- `config.json`
- `metrics.json`
- ONNX runtime scripts
- `landmark_baseline/` support package

## Labels

{', '.join(labels)}

## Quick start

Single video:

```bash
python predict_landmark_video_onnx.py --model model.onnx --labels labels.json --video /path/to/clip.mp4
```

Webcam:

```bash
python run_landmark_webcam_onnx.py --model model.onnx --labels labels.json
```

Live validation capture:

```bash
python validate_live_sign_capture_onnx.py --model model.onnx --labels labels.json --output-dir captures --expected-label {labels[0]}
```
"""
    path.write_text(content, encoding='utf-8')


def make_zip(bundle_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_root.rglob('*')):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(bundle_root))


def main() -> None:
    args = parse_args()
    bundle_root = args.output_dir / args.bundle_name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    required = ['model.onnx', 'labels.json', 'config.json', 'metrics.json']
    for name in required:
        src = args.run_dir / name
        if not src.exists():
            raise RuntimeError(f'Missing required file in run dir: {src}')
        copy_file(src, bundle_root / name)

    data_sidecar = args.run_dir / 'model.onnx.data'
    if data_sidecar.exists():
        copy_file(data_sidecar, bundle_root / 'model.onnx.data')

    repo_root = Path(__file__).resolve().parent
    for name in RUNTIME_FILES:
        copy_file(repo_root / name, bundle_root / name)

    baseline_dst = bundle_root / 'landmark_baseline'
    for name in BASELINE_FILES:
        copy_file(repo_root / 'landmark_baseline' / name, baseline_dst / name)

    docs_src = repo_root / 'docs' / 'landmark_progress.md'
    if docs_src.exists():
        copy_file(docs_src, bundle_root / 'docs' / 'landmark_progress.md')

    labels = json.loads((args.run_dir / 'labels.json').read_text(encoding='utf-8'))
    write_requirements(bundle_root / 'requirements-onnx.txt')
    write_readme(bundle_root / 'README.md', args.bundle_name, [str(label) for label in labels])

    manifest = {
        'bundle_name': args.bundle_name,
        'source_run_dir': str(args.run_dir),
        'labels': labels,
        'files': sorted(str(path.relative_to(bundle_root)) for path in bundle_root.rglob('*') if path.is_file()),
    }
    (bundle_root / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    zip_path = args.output_dir / f'{args.bundle_name}.zip'
    make_zip(bundle_root, zip_path)
    print(
        json.dumps(
            {
                'bundle_root': str(bundle_root),
                'zip_path': str(zip_path),
                'labels': labels,
                'file_count': len(manifest['files']),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
