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
- ✅ **G-B** — EPICS dual-hand landmark transformer vs vanilla I3D through *one*
  harness (new `LandmarkBackend` in `eval/asl_backends.py`; runner
  `eval/benchmark_landmark_vs_vanilla.py`). On n=12 held-out WLASL (5 shared
  words): I3D closed-set **1.00**, landmark **0.58** (= its training acc), I3D
  open-set 0.33. The invalid "landmark loses (0.45)" verdict (weaker
  `HolisticSequenceEncoder`) is **retired**. ⚠️ n=12 is *not* decision-grade
  (plan Gate G1) and this is in-distribution WLASL — the landmark model's claimed
  cross-source/phone edge is untested here. Real verdict: G1/G3 on the grown
  benchmark. Results: `eval/benchmark/gB_landmark_vs_vanilla.json`.
- ✅ **G-C** — preprocessing standard locked. Train (`prepare_landmark_dataset.py`)
  and serve (`inference.py`/`inference_onnx.py`/`LandmarkBackend`) all use the one
  canonical `preprocess_landmark_sequence` (dual-hand, handedness-canonical).
  `tests/test_preprocessing_standard.py` asserts train/serve features are
  byte-identical; legacy `eval/landmark_utils.py` + `benchmark_landmark_closed_set.py`
  marked DEPRECATED. See `docs/preprocessing_standard.md`.
- ✅ **G-D (audit + fixes)** — cross-source code reviewed and two gaps closed:
  layout-skewed default paths **fixed** (benchmark/preprocess now resolve under `eval/`),
  and **signer-disjoint splitting implemented** (`--signer-disjoint` + `--wlasl-annotations`
  join WLASL v0.3 `signer_id`; invariant tested). Still open: HF download go/no-go,
  landmark-side adaptation. Findings: `docs/g_d_cross_source_audit.md`.
- 🔁 **G1** — directional run DONE (2026-07-17): 32 clips, 2 sources, both models,
  per-source CIs (`eval/benchmark/g1/g1_directional_report.json`; landmark 0.32
  wlasl / **0.0 aslense** — the cross-domain gap is now measured, not suspected).
  Still ⬜ decision-grade: needs ≥150 clips, ≥3 sources (phone clips + ASL Citizen;
  see `docs/g2_data_status.md` for the exact acquisition steps).
- ✅ **Abstention wired live** (2026-07-17) — word model calibrated on the mixed
  directional benchmark: only conf ≥ 0.95 keeps confident-wrong ≤ 5%, so the live
  path abstains → falls back to fingerspelling by default (fingerspelling-first,
  enforced by data). Thresholds: `models/landmark/checkpoints/abstention.json`
  (marked directional), read by `deploy/run_live.py`. See `docs/abstention.md`.
- ✅ **F4 fingerspelling real-data gate PASS** (2026-07-17) — held-out set_c
  (n=684): selective precision **0.984** at 0.632 coverage; thresholds unchanged;
  confusable cluster is c/m/n/o/p/r/u. Report:
  `models/fingerspelling/checkpoints/f4_gate_report.json`, doc: `docs/fingerspelling_f4.md`.
- ✅ **Live pipeline (W7, laptop)** (2026-07-17) — `deploy/run_live.py`: capture
  (webcam/clip) → word model → abstention → fingerspelling fallback → TTS
  (`pipeline/tts.py`: say → pyttsx3 → null, debounced). Jetson remains Phase 5.
- 🔁 **Multi-source data (W1/W2)** — MS-ASL adapter flowing (annotations local,
  C-UDA license; yt-dlp probe: ~58% dead links); ASL Citizen documented as a
  one-time 42.8 GB human step. Adapters + signer-disjoint splits tested offline.
  Roadmap + numbers: `docs/g2_data_status.md`.

See the program plan in `docs/` for the full phased, evidence-gated roadmap; the
current execution plan is `docs/superpowers/plans/2026-07-17-fall2026-execution.md`.
