# deploy/ — edge runtime (stub)

Placeholder for the on-device deployment story. Populated in **Phase 5**, not now.

Target device: **Jetson Orin Nano** (Ampere GPU, ~40 TOPS, 8 GB) — treated as a
*tunable* constraint, not a gate. Per the accuracy-first objective hierarchy
(see `../docs/VisionaryHands_Program_Plan_Merge.md`, Addendum A), we adapt the
runtime to the most accurate model, not the other way around.

Planned contents (later):
- ONNX / TensorRT export + quantization recipes for the chosen model
- MediaPipe extraction runtime (the live bottleneck) tuning
- Text-to-speech (TTS) output for the two-way medical exchange
- A single `run_live.py` that ties capture → model → shortlist → abstention → TTS

For now this is empty by design. The current runnable ONNX bundle that will seed
this work lives at `landmark_live5_dual_onnx_bundle/` (copied here in G0).

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
