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
