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

Three Python 3.11 venvs mirror the original split. The split keeps heavy
training deps out of the lean live-inference env; it is **not** forced by a
protobuf conflict — verified at G0, `mediapipe==0.10.9` and `onnxruntime`
coexist in `.venv-onnx` (protobuf settles at 3.20.3). For the G0 gate you only
need `.venv-oracle` and `.venv-onnx`; `.venv-landmarks` is for training/export.

```bash
# oracle + harness (I3D)
python3.11 -m venv .venv-oracle && . .venv-oracle/bin/activate \
  && pip install -r requirements-oracle.txt && deactivate
# live ONNX inference (landmark model — extraction + inference in one process)
python3.11 -m venv .venv-onnx && . .venv-onnx/bin/activate \
  && pip install -r requirements-onnx.txt && deactivate
# landmark training / extraction / export (not needed for the G0 gate)
python3.11 -m venv .venv-landmarks && . .venv-landmarks/bin/activate \
  && pip install -r requirements-landmarks.txt && deactivate
```

> `mediapipe` is pinned to **0.10.9** in both landmark envs: it still ships the
> legacy `mp.solutions.holistic` API used by extraction; `>=0.10.15` removed it.

## Gate G0 — one command per model, from one repo

**Oracle (vanilla I3D):**
```bash
. .venv-oracle/bin/activate && source env.sh
python oracle/run_i3d.py eval/benchmark/multisource_videos/05_wlasl_secretary_50041.mp4
```

**Landmark model (ONNX bundle):**
```bash
. .venv-onnx/bin/activate && source env.sh
BUNDLE=deploy/landmark_live5_dual_onnx_bundle/landmark_onnx_bundle
python models/landmark/predict_landmark_video_onnx.py \
  --video eval/benchmark/multisource_videos/05_wlasl_secretary_50041.mp4 \
  --model  "$BUNDLE/model.onnx" \
  --labels "$BUNDLE/labels.json"
```
The script takes `--video/--model/--labels` (config auto-loads from the model's
sibling `config.json`); run with `--help` for the rest. Bundle vocab is 5 classes
(`bowling, computer, deaf, drink, yes`), so "secretary" is out-of-vocab — it
returns a confident-wrong label, which is exactly what the later abstention layer
is for. G0 only asks that it *runs*.

## Status

- ✅ **G0 scaffold** — repo unified; both codebases present; oracle wrapper + harness in place.
- ✅ **G0 gate** — both run, one command each. I3D → `secretary 0.467` (in-vocab,
  correct); landmark ONNX runs end-to-end (MediaPipe extraction → ONNX). Verified
  on `.venv-oracle` + `.venv-onnx`, Python 3.11.9.
- ⬜ **G-B** (next) — run the EPICS dual-hand landmark transformer through *this*
  harness head-to-head vs the I3D. The prior "landmark loses (0.45)" used a
  weaker encoder; that verdict is invalid until redone here.

See the program plan in `docs/` for the full phased, evidence-gated roadmap.
