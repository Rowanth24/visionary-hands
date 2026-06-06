# landmark_onnx_bundle

## Contents

- `model.onnx`
- `labels.json`
- `config.json`
- `metrics.json`
- ONNX runtime scripts
- `landmark_baseline/` support package

## Labels

bowling, computer, deaf, drink, yes

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
python validate_live_sign_capture_onnx.py --model model.onnx --labels labels.json --output-dir captures --expected-label bowling
```
