# Landmark Branch Progress

## Current capability level

### Level 0: Dataset preparation

Implemented and working.

- Download subset: `download_wlasl_subset.py`
- Extract landmarks: `extract_wlasl_landmarks.py`
- Build fixed-shape arrays: `prepare_landmark_dataset.py`

Current prepared dataset:

- `8` classes
- `98` train / `24` val / `20` test
- source summary: `data/wlasl_distinct_8_full_prepared/summary.json`

### Level 1: Offline batch evaluation on prepared arrays

Implemented and working.

- Compact transformer checkpoint: `runs/landmark_wlasl8_transformer_compact_v1/best.pt`
- Compact transformer metrics: `runs/landmark_wlasl8_transformer_compact_v1/metrics.json`

Current result:

- best val: `0.375`
- test: `0.35`

Current better live-oriented subset:

- `5` classes: `bowling`, `computer`, `deaf`, `drink`, `yes`
- compact transformer checkpoint: `runs/landmark_live5_transformer_compact_v1/best.pt`
- best val: `0.4706`
- test: `0.4167`

### Level 2: Offline raw-video inference

Implemented and working for both runtimes.

- TorchScript single-video inference: `predict_landmark_video.py`
- ONNX single-video inference: `predict_landmark_video_onnx.py`

Full raw-video path has been validated against the test manifest:

- TorchScript raw-video accuracy: `0.35`
  - `runs/landmark_wlasl8_transformer_compact_v1/test_video_inference.json`
- ONNX raw-video accuracy: `0.35`
  - `runs/landmark_wlasl8_transformer_compact_v1/test_video_inference_onnx.json`

Live-oriented `5`-class raw-video path:

- ONNX raw-video accuracy: `0.4167`
  - `runs/landmark_live5_transformer_compact_v1/test_video_inference_onnx.json`

### Level 3: Live sign capture from webcam

Implemented, not yet fully field-validated.

- TorchScript webcam loop: `run_landmark_webcam.py`
- ONNX webcam loop: `run_landmark_webcam_onnx.py`

This means the system can now:

- capture camera frames
- extract MediaPipe landmarks locally
- classify a live sign sequence locally
- display a predicted label and confidence

What is not validated yet:

- sustained live-use quality across many signs
- camera positioning / lighting robustness
- Raspberry Pi performance

### Level 4: Deployable ONNX bundle

Implemented and smoke-validated.

- bundle root: `deployment/landmark_live5_onnx_bundle`
- zip archive: `deployment/landmark_live5_onnx_bundle.zip`

The bundle includes:

- `model.onnx`
- `labels.json`
- ONNX inference scripts
- `landmark_baseline/` runtime helpers
- `requirements-onnx.txt`
- progress/readme docs

## How to check progress

### 1. Fastest regression check

Use the labeled raw-video manifest. This is the main repeatable test.

TorchScript:

```bash
.venv-landmarks/bin/python evaluate_landmark_video_manifest.py \
  --manifest data/wlasl_distinct_8_full_prepared/test/manifest.jsonl \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.ts \
  --labels runs/landmark_wlasl8_transformer_compact_v1/labels.json \
  --device cpu \
  --output-json runs/landmark_wlasl8_transformer_compact_v1/test_video_inference.json
```

ONNX:

```bash
.venv-landmarks/bin/python evaluate_landmark_video_manifest_onnx.py \
  --manifest data/wlasl_distinct_8_full_prepared/test/manifest.jsonl \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.onnx \
  --labels runs/landmark_wlasl8_transformer_compact_v1/labels.json \
  --output-json runs/landmark_wlasl8_transformer_compact_v1/test_video_inference_onnx.json
```

What to watch:

- overall accuracy
- which labels fail repeatedly
- whether raw-video accuracy stays aligned with prepared-array accuracy

### 2. Single known-clip spot check

This is useful after any model/export change.

```bash
.venv-landmarks/bin/python predict_landmark_video_onnx.py \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.onnx \
  --labels runs/landmark_wlasl8_transformer_compact_v1/labels.json \
  --video data/wlasl_distinct_8_full/test/bowling/07383.mp4
```

Good known sample:

- `data/wlasl_distinct_8_full/test/bowling/07383.mp4`

### 3. Live webcam check

This verifies real capture, not just canned clips.

TorchScript:

```bash
.venv-landmarks/bin/python run_landmark_webcam.py \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.ts \
  --labels runs/landmark_wlasl8_transformer_compact_v1/labels.json
```

ONNX:

```bash
.venv-landmarks/bin/python run_landmark_webcam_onnx.py \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.onnx \
  --labels runs/landmark_wlasl8_transformer_compact_v1/labels.json
```

What counts as progress here:

- the label updates in real time
- confidence is stable instead of jumping randomly
- repeated performances of the same sign converge to the same prediction

### 3b. Live capture validation run

This is the cleanest single test case for a real sign you perform in front of the camera.

```bash
.venv-landmarks/bin/python validate_live_sign_capture_onnx.py \
  --model runs/landmark_live5_transformer_compact_v1/model.onnx \
  --labels runs/landmark_live5_transformer_compact_v1/labels.json \
  --output-dir captures/live5 \
  --expected-label bowling
```

What it does:

- records a short webcam clip
- saves the clip to disk
- runs ONNX inference on that recorded clip
- writes a JSON result with `expected_label`, `predicted_label`, `confidence`, and `correct`

### 4. Runtime/edge check

This is the deployment-oriented benchmark.

TorchScript benchmark:

```bash
.venv-landmarks/bin/python benchmark_landmark_model.py \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.ts \
  --dataset data/wlasl_distinct_8_full_prepared/test/dataset.npz \
  --device cpu \
  --output-json runs/landmark_wlasl8_transformer_compact_v1/benchmark_cpu.json
```

ONNX benchmark:

```bash
.venv-onnx/bin/python benchmark_landmark_onnx.py \
  --model runs/landmark_wlasl8_transformer_compact_v1/model.onnx \
  --dataset data/wlasl_distinct_8_full_prepared/test/dataset.npz \
  --output-json runs/landmark_wlasl8_transformer_compact_v1/benchmark_onnx_cpu.json
```

Current benchmark snapshot:

- TorchScript: `0.328 ms` / inference, `214 KB`
- ONNX: `0.362 ms` / inference, `15 KB`

## Current recommendation

- Default runtime for Mac development: TorchScript or ONNX both work
- Preferred deployment target for Raspberry Pi: ONNX
- Best current `8`-class model: `runs/landmark_wlasl8_transformer_compact_v1`
- Best current live-oriented model: `runs/landmark_live5_transformer_compact_v1`
- Best current deployment bundle: `deployment/landmark_live5_onnx_bundle`
