# Landmark Branch for Offline Raspberry Pi Deployment

## Goal

Move the project toward an offline-friendly sign classification branch that can run on Raspberry Pi OS without any cloud service dependency.

## What this branch changes

- Training stays off-device on a workstation or Kaggle.
- Deployment target is Raspberry Pi OS.
- Inference path becomes:
  1. camera frame capture
  2. MediaPipe Holistic landmarks
  3. fixed-length landmark preprocessing
  4. lightweight classifier
  5. local label map lookup
- No Gemini or network API dependency is part of the runtime path.

## Why landmark-first helps on Raspberry Pi

The current RGB video model is too heavy for an offline Raspberry Pi target. A landmark-based model is materially lighter because it only needs a sequence of pose and hand coordinates after extraction.

## Current deliverables

- `extract_wlasl_landmarks.py`
  - extracts `T x 543 x 3` MediaPipe landmark sequences from split-dir WLASL data
  - writes one `.npz` per sample plus a manifest
- `prepare_landmark_dataset.py`
  - converts raw extracted sequences into fixed-shape processed arrays
  - applies canonical handedness normalization and 64-frame padding/pooling
  - supports both dominant-hand-only and dual-hand layouts
- `landmark_baseline/preprocess.py`
  - pure NumPy preprocessing logic suitable for training and later on-device inference wrappers
- `train_landmark_transformer.py`
  - trains a lightweight PyTorch transformer on prepared landmark arrays
  - exports `best.pt`, `metrics.json`, and optional TorchScript `model.ts`
- `predict_landmark_video.py`
  - runs offline inference on a local video clip using `model.ts` and `labels.json`
- `evaluate_landmark_video_manifest.py`
  - validates the full video-to-prediction path against a manifest of labeled clips
- `run_landmark_webcam.py`
  - runs a local offline webcam loop on Mac using the exported TorchScript model
- `export_landmark_torchscript.py`
  - exports a CPU-portable TorchScript model from `best.pt`
- `benchmark_landmark_model.py`
  - measures TorchScript inference latency on prepared landmark arrays
- `export_landmark_onnx.py`
  - exports a checkpoint to ONNX for CPU edge-runtime comparison
- `benchmark_landmark_onnx.py`
  - measures ONNX Runtime CPU inference latency on prepared landmark arrays

## Deployment constraints

- Runtime must stay offline.
- Model export target should be `TFLite` or another edge-friendly format.
- The Raspberry Pi runtime should depend only on local assets:
  - model file
  - labels JSON
  - local Python packages or wheels
- The LLM sentence-construction step is out of scope for deployment.

## Immediate next steps

1. Export the best landmark model to the final edge runtime format.
2. Build a local webcam inference loop for Mac first, then adapt it to Raspberry Pi OS.
3. Compare TorchScript vs TFLite latency on the actual Raspberry Pi target.
4. Expand beyond the current `5`-class live subset once the offline path is stable.

## Mac-first local commands

Use separate local environments:

- `.venv-landmarks` for MediaPipe/OpenCV extraction and live inference
- `.venv-onnx` for ONNX export/runtime tools, to avoid the protobuf conflict with MediaPipe

Train on prepared arrays:

```bash
.venv-landmarks/bin/python train_landmark_transformer.py \
  --data-root data/wlasl_live_5_dual_prepared \
  --output-root runs/landmark_live5_transformer_compact_dual_v1 \
  --epochs 25 \
  --batch-size 16 \
  --model-type transformer \
  --d-model 64 \
  --num-heads 4 \
  --num-layers 1 \
  --dropout 0.2 \
  --export-torchscript
```

Run offline inference on one clip:

```bash
.venv-landmarks/bin/python predict_landmark_video.py \
  --model runs/landmark_live5_transformer_compact_dual_v1/model.ts \
  --labels runs/landmark_live5_transformer_compact_dual_v1/labels.json \
  --video data/wlasl_distinct_8_full/test/bowling/07383.mp4 \
  --device cpu
```

Evaluate the full video inference path:

```bash
.venv-landmarks/bin/python evaluate_landmark_video_manifest.py \
  --manifest data/wlasl_live_5_prepared/test/manifest.jsonl \
  --model runs/landmark_live5_transformer_compact_dual_v1/model.ts \
  --labels runs/landmark_live5_transformer_compact_dual_v1/labels.json \
  --device cpu \
  --output-json runs/landmark_live5_transformer_compact_dual_v1/test_video_inference.json
```

Run the local webcam loop:

```bash
.venv-landmarks/bin/python run_landmark_webcam.py \
  --model runs/landmark_live5_transformer_compact_dual_v1/model.ts \
  --labels runs/landmark_live5_transformer_compact_dual_v1/labels.json
```

Export a CPU-portable TorchScript model from a checkpoint:

```bash
.venv-landmarks/bin/python export_landmark_torchscript.py \
  --checkpoint runs/landmark_live5_transformer_compact_dual_v1/best.pt \
  --data-root data/wlasl_live_5_dual_prepared \
  --output runs/landmark_live5_transformer_compact_dual_v1/model.ts
```

Benchmark a TorchScript model on prepared test data:

```bash
.venv-landmarks/bin/python benchmark_landmark_model.py \
  --model runs/landmark_live5_transformer_compact_dual_v1/model.ts \
  --dataset data/wlasl_live_5_dual_prepared/test/dataset.npz \
  --device cpu \
  --output-json runs/landmark_live5_transformer_compact_dual_v1/benchmark_cpu.json
```

Export ONNX from the compact transformer checkpoint:

```bash
.venv-onnx/bin/python export_landmark_onnx.py \
  --checkpoint runs/landmark_live5_transformer_compact_dual_v1/best.pt \
  --data-root data/wlasl_live_5_dual_prepared \
  --output runs/landmark_live5_transformer_compact_dual_v1/model.onnx
```

Benchmark ONNX Runtime on CPU:

```bash
.venv-onnx/bin/python benchmark_landmark_onnx.py \
  --model runs/landmark_live5_transformer_compact_dual_v1/model.onnx \
  --dataset data/wlasl_live_5_dual_prepared/test/dataset.npz \
  --output-json runs/landmark_live5_transformer_compact_dual_v1/benchmark_onnx_cpu.json
```

## Risk to validate early

MediaPipe availability on Raspberry Pi OS should be validated early on the actual target image. If Python wheels are not available for the target image, the fallback is to build the wheel ahead of time and vendor it with the deployment bundle.
