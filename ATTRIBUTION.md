# Attribution & Provenance

"Visionary Hands" merges two prior bodies of work. This file records what came
from where, so nothing is mis-claimed and licensing can be sorted before any
public release.

> ⚠️ **Licensing caveat:** the upstream clone published **no LICENSE file** as of
> the vendoring date (2026-06). The vendored files below are included for
> internal research use only. Do **not** relicense, open-source, or redistribute
> them as original work without confirming permission with the upstream author.

---

## 1. Landmark pipeline — yours (from `~/EPICS_Project`)

Author: Rowanth Selvakumaran (+ team). MediaPipe landmark extraction, the
dual-hand transformer / GRU classifier, training, ONNX / TorchScript export,
and Kaggle run tooling.

- `landmark_baseline/` — the importable library (model, preprocess, inference, labels)
- `pipeline/` — data prep (extract, prepare, download, analyze)
- `models/landmark/` — train / export / inference / eval / webcam scripts + checkpoint
- `deploy/landmark_live5_dual_onnx_bundle/` — exported live bundle

## 2. Vanilla I3D oracle — UPSTREAM (vendored)

Source: **github.com/sumedhsp/Sign-Language-Recognition** (Sumedh S. Patkar).
Clearly upstream files, vendored under `eval/` (kept with the harness so their
relative paths/imports resolve):

- `eval/pytorch_i3d.py` — `InceptionI3d` (Kinetics I3D), the oracle model
- `eval/videotransforms.py`
- `eval/datasets/`, `eval/preprocess/`, `eval/configfiles/`
- `eval/train_vanilla_v1_*.py`, `eval/test_vanilla_v1_*.py` — upstream train/test scripts
- `oracle/checkpoints/best_model_original_backup.pth` — trained I3D weights

## 3. Benchmark harness — yours, built on the upstream base (MIXED provenance)

Developed/modified during your own sessions (incl. Codex) on top of the upstream
repo. Treat as **your additions, but verify authorship before any relicensing**:

- `eval/asl_backends.py` — backend abstraction + candidate shortlisting
- `eval/benchmark_head_to_head.py`, `eval/benchmark_models.py`,
  `eval/benchmark_vanilla_manifest.py`, `eval/benchmark_landmark_closed_set.py`
- `eval/prepare_cross_source_finetune_data.py`, `eval/train_vanilla_cross_source.py`
- `eval/infer_videos.py`, `eval/live_demo.py`, `eval/landmark_utils.py`
- `eval/configs.py`, `eval/custom_models.py` (adapted wrappers around the I3D)
- `eval/benchmark/` — manifests, result JSONs, sample clips (your testing record)

## 4. Explicitly NOT included (other authors' artifacts)

Left in the upstream clone on purpose — not ours to claim:

- `Computer_Vision_Project.pdf`, `diagram.png`, `showcase/`, `index.md`,
  upstream `README.md`, `start_kit/`

---

*If/when this repo is open-sourced, resolve §2 and §3 licensing first.*
