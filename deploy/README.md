# deploy/ — runnable demos + edge runtime (Jetson still Phase 5)

`run_live.py` (streaming) and `run_demo.py` (clip-based) run today on a laptop.
The Jetson/TensorRT story remains **Phase 5**, not now.

Target device: **Jetson Orin Nano** (Ampere GPU, ~40 TOPS, 8 GB) — treated as a
*tunable* constraint, not a gate. Per the accuracy-first objective hierarchy
(see `../docs/VisionaryHands_Program_Plan_Merge.md`, Addendum A), we adapt the
runtime to the most accurate model, not the other way around.

Planned contents (later):
- ONNX / TensorRT export + quantization recipes for the chosen model
- MediaPipe extraction runtime (the live bottleneck) tuning

The runnable ONNX bundle that will seed the Jetson work lives at
`landmark_live5_dual_onnx_bundle/` (copied here in G0).

## `run_live.py` — live demo (capture → word model → abstain → fingerspelling → TTS)

Streaming version of the full path over a sliding window (default 48 frames,
stride 24), speaking recognized words/letters via `pipeline/tts.py`
(macOS `say` → `pyttsx3` → silent, auto-selected; identical text debounced 3 s).
Uses the same canonical `LandmarkBackend` path as the benchmark harness — no
separate live preprocessing. Runs in `.venv-onnx` or any env with
mediapipe+cv2+torch.

```bash
source env.sh
# clip mode (headless, what tests/test_run_live_clip.py exercises)
.venv-onnx/bin/python deploy/run_live.py --source eval/benchmark/multisource_videos/<clip>.mp4 --no-tts
# webcam, speaking
.venv-onnx/bin/python deploy/run_live.py --source webcam
```

Decision lines: `[t=  1.60s] WORD drink (0.87)` / `LETTER A (0.93)` /
`ABSTAIN (<reason>)`. Word-model abstention thresholds are read from
`models/landmark/checkpoints/abstention.json` when present (see the fall 2026
execution plan, Task E); until then a `--word-conf-threshold` default applies —
expect overconfident WORD lines on out-of-vocab signs until calibrated.

## `run_demo.py` — end-to-end demo (word model + fingerspelling fallback)

A clip-based orchestration of the full path, in one process (`.venv-oracle`):
closed-vocab word model (`LandmarkBackend`) → abstention (`eval/abstention.py`) →
on abstain or `--manual-spell`, fingerspelling (`FingerspellingRecognizer`). Holistic
(word) and Hands (fingerspelling) MediaPipe models coexist in one process.

```bash
. .venv-oracle/bin/activate && source env.sh
# confident word -> WORD
python deploy/run_demo.py --word-clip eval/benchmark/g1/videos/wlasl/computer/12306.mp4
# low-confidence word -> AUTO fallback to spelling
python deploy/run_demo.py --word-clip <low-conf clip> --spell-input <letter images / video>
# force spelling
python deploy/run_demo.py --word-clip <clip> --manual-spell --spell-input <letter images / video>
```

Verified routing: confident→WORD, word-abstains→LETTER (auto), manual→LETTER. This is the
clip-based precursor to the Phase-5 live `run_live.py` (capture → … → TTS).
