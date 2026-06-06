# Visionary Hands

ASL → text for **closed-vocabulary, offline, two-way medical communication** —
built to be a stopgap when a qualified interpreter or working VRI isn't
available, **augmenting** interpreters, never replacing them.

> North star: **model accuracy / real-world generalization** (phone-camera, not
> just in-distribution WLASL). Hardware (Jetson Orin Nano) is a *tunable*
> constraint, not a gate. Full rationale + roadmap:
> [`docs/VisionaryHands_Program_Plan_Merge.md`](docs/VisionaryHands_Program_Plan_Merge.md).

This repo merges two prior efforts into one (see [`ATTRIBUTION.md`](ATTRIBUTION.md)):

- a **landmark pipeline** (MediaPipe → dual-hand transformer) — appearance-invariant,
  the edge/student path;
- a **vanilla I3D oracle** + **benchmark harness** — the strong in-distribution
  teacher and the single judge both models are scored by.

## Layout

```
visionary-hands/
├── landmark_baseline/   # landmark library (model, preprocess, inference, labels) — importable package
├── pipeline/            # data prep: extract / prepare / download / analyze
├── models/landmark/     # train / export / inference / eval / webcam scripts (+ checkpoints/, gitignored)
├── eval/                # unified benchmark harness + vendored I3D modules + benchmark/ data
├── oracle/              # thin I3D entry point (run_i3d.py) + weights (checkpoints/, gitignored)
├── deploy/              # edge runtime + TTS (stub; Phase 5) + the live ONNX bundle
├── docs/                # program plan, analysis, progress notes
├── env.sh               # sets PYTHONPATH (repo root + eval/)
├── requirements-landmarks.txt   # training + MediaPipe extraction + export
├── requirements-onnx.txt        # live ONNX inference (separate venv — protobuf conflict)
└── requirements-oracle.txt      # I3D oracle + harness
```

Weights, videos, and large caches are **gitignored** — they sit on disk so things
run, but aren't version-controlled. The big `cross_source_*` training caches
(~1.4 GB) were intentionally left in the original clone, not copied here.

## Quick start

```bash
cd ~/visionary-hands
source env.sh                       # PYTHONPATH = repo root + eval/
```

Three Python 3.11 venvs mirror the original split (MediaPipe and onnxruntime
disagree on protobuf, so they stay separate):

```bash
# oracle + harness
python3.11 -m venv .venv-oracle && . .venv-oracle/bin/activate \
  && pip install -r requirements-oracle.txt && deactivate
# landmark training / extraction / export
python3.11 -m venv .venv-landmarks && . .venv-landmarks/bin/activate \
  && pip install -r requirements-landmarks.txt && deactivate
# live ONNX inference
python3.11 -m venv .venv-onnx && . .venv-onnx/bin/activate \
  && pip install -r requirements-onnx.txt && deactivate
```

## Gate G0 — one command per model, from one repo

**Oracle (vanilla I3D):**
```bash
. .venv-oracle/bin/activate && source env.sh
python oracle/run_i3d.py eval/benchmark/multisource_videos/05_wlasl_secretary_50041.mp4
```

**Landmark model (ONNX bundle):**
```bash
. .venv-onnx/bin/activate && source env.sh
python models/landmark/predict_landmark_video_onnx.py \
  --bundle deploy/landmark_live5_dual_onnx_bundle \
  --video eval/benchmark/multisource_videos/05_wlasl_secretary_50041.mp4
```
*(exact landmark flags confirmed at gate-run; see `models/landmark/predict_landmark_video_onnx.py --help`)*

## Status

- ✅ **G0 scaffold** — repo unified; both codebases present; oracle wrapper + harness in place.
- ⏳ **G0 gate verification** — pending the three venvs (heavy torch/mediapipe installs).
- ⬜ **G-B** (next) — run the EPICS dual-hand landmark transformer through *this*
  harness head-to-head vs the I3D. The prior "landmark loses (0.45)" used a
  weaker encoder; that verdict is invalid until redone here.

See the program plan in `docs/` for the full phased, evidence-gated roadmap.
