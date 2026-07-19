# Fall 2026 Execution Plan — Close the Gaps Without Phone Clips

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Advance Visionary Hands from "harness + fingerspelling done, everything else scaffolded" to a working live pipeline, a closed fingerspelling gate, and a real multi-source evaluation path — using only data obtainable today (no phone recordings).

**Architecture:** Keep the existing structure: `landmark_baseline/` is the shared preprocessing/inference library, `eval/` is the single judge (per-source manifests + abstention), `models/` holds the two model tracks (landmark words, fingerspelling letters), `pipeline/` holds data prep, `deploy/` holds runnable artifacts. New work plugs into those seams — no restructuring.

**Tech Stack:** Python 3.11 (`.venv-oracle` for eval/training, `.venv-onnx` for live inference), mediapipe==0.10.9, onnxruntime, PyTorch, pytest. TTS via macOS `say` now, abstracted for espeak/piper on Jetson later.

**Branch:** `fall2026-progress` (user's `main` untouched). No remote exists yet — commits are local.

**Baseline (audited 2026-07-17):** 22 commits (Jun 5–7). DONE: G0 scaffold, G-B/G-C/G-D (harness seam `eval/asl_backends.py:LandmarkBackend`, train/serve preprocessing locked + byte-identical test, signer-disjoint splitting), fingerspelling F0–F3 (MLP 0.85 cross-dataset, calibrated abstention at target precision 0.95). Tests: 24 pass / 1 fixture-skip. NOT DONE: multi-source data (W1/W2 — nothing on disk), live pipeline + TTS (W7), word-model abstention calibration (W4), fingerspelling F4 real-data gate (W3), repo has no remote/backup.

**Constraint honored:** Real-world phone recordings do not exist yet. ASL Citizen (crowd-sourced webcam) is the stand-in for the phone domain; when phone clips arrive they become one more `source` in the same G1 manifest — zero rework.

---

## Task A — Hygiene: testable env + delete deprecated skew path

**Files:**
- Modify: `.venv-oracle` (env only: install `pytest`, `scikit-learn`)
- Delete: `eval/landmark_utils.py`, `eval/benchmark_landmark_closed_set.py` (both marked DEPRECATED; skew risk)
- Modify: any file that still imports the deleted modules (grep first)

**Steps:**
- [ ] `"$HOME/visionary-hands/.venv-oracle/bin/pip" install pytest scikit-learn`
- [ ] Run `cd ~/visionary-hands && source env.sh && .venv-oracle/bin/pytest tests/ -v` → expect 24+ pass, 0 fail (baseline)
- [ ] `grep -rn "landmark_utils\|benchmark_landmark_closed_set" --include="*.py" .` → update/remove stale imports (docs references may stay, annotated)
- [ ] Delete the two deprecated files
- [ ] Re-run pytest → same green
- [ ] Commit: `chore: make pytest runnable in .venv-oracle; remove deprecated skew-risk eval path`

## Task B — W7: live pipeline `deploy/run_live.py` + TTS abstraction

**Files:**
- Create: `pipeline/tts.py` — `BaseTTS.speak(text)`, `SayTTS` (macOS `say`, subprocess), `Pyttsx3TTS` (optional import), `NullTTS`; `create_tts(prefer=None)` picks by availability. Debounce: never re-speak identical text within N seconds.
- Create: `deploy/run_live.py` — args: `--source webcam|<video path>`, `--bundle deploy/landmark_live5_dual_onnx_bundle/landmark_onnx_bundle`, `--tts/--no-tts`, `--window`/`--stride`. Loop: frames → MediaPipe → `landmark_baseline` canonical preprocessing (same path as `LandmarkBackend` — NO new preprocessing code) → ONNX word model → sliding-window majority/temporal vote → abstention gate (reuse `eval/abstention.py` logic + thresholds file) → on abstain: route to fingerspelling recognizer (`models/fingerspelling/recognizer.py` routing hook from F3) → emit text (stdout + optional TTS).
- Test: `tests/test_tts.py` (selection + debounce, subprocess mocked), `tests/test_run_live_clip.py` (clip mode on an existing `eval/benchmark/` video: runs headless, produces ≥1 predict-or-abstain decision, exit 0).

**Steps:**
- [ ] Write `tests/test_tts.py` (failing) → implement `pipeline/tts.py` → green
- [ ] Write `tests/test_run_live_clip.py` (failing) → implement `deploy/run_live.py` in clip mode → green (use `.venv-onnx` deps; guard heavy imports so pytest collection works in `.venv-oracle`)
- [ ] Webcam path: same loop with `cv2.VideoCapture(0)`; cannot be exercised headless — verified later by user; clip mode is the tested path
- [ ] Update `deploy/README.md`: how to run live demo, TTS notes, Jetson still Phase 5
- [ ] Commit: `feat(deploy): run_live — capture → word model → abstain → fingerspelling fallback → TTS`

## Task C — W3 F4: fingerspelling real-data gate

**Files:**
- Create: `models/fingerspelling/f4_gate.py` — evaluate `checkpoints/mlp.pt` + `checkpoints/abstention.json` on held-out `data/fingerspelling/set_c.npz`: per-letter precision/coverage at current thresholds; recalibrate on set_a/b only, validate on set_c; analyze the c/r/o confusable cluster (thresholds pinned at 1.0 = letters never fire).
- Create: `docs/fingerspelling_f4.md` — results table, c/r/o decision, gate verdict.
- Modify (only if validated better): `models/fingerspelling/checkpoints/abstention.json`
- Test: `tests/test_f4_gate.py` — held-out selective precision ≥ 0.90 at whatever coverage the report claims; asserts the gate artifact exists and is internally consistent.

**Steps:**
- [ ] Implement `f4_gate.py` (reuse `eval/abstention.py` selective-metrics code — do not duplicate)
- [ ] Run on set_c; write `docs/fingerspelling_f4.md` with real numbers
- [ ] c/r/o: report per-pair confusion; if merging into an "abstain cluster" (spell-again prompt) beats per-letter suppression, encode that in `abstention.json` policy; otherwise document why current handling stands
- [ ] Add `tests/test_f4_gate.py` → green
- [ ] Commit: `feat(fingerspelling): F4 real-data gate — held-out calibration + confusable-cluster decision`

## Task D — W1/W2: multi-source data groundwork (ASL Citizen + MS-ASL)

**Files:**
- Create: `pipeline/sources/asl_citizen.py`, `pipeline/sources/msasl.py` — download/index for the locked vocab only (5 demo words from `models/landmark/checkpoints/labels.json` + candidate medical seed vocab from `docs/vocab-policy.md`). MS-ASL = yt-dlp per-clip with dead-link tolerance; ASL Citizen = whatever access its license portal allows (research actual terms first — if click-through-gated, emit exact manual instructions instead of scraping around the gate).
- Create: `pipeline/build_g1_manifest.py` (or extend `eval/build_g1_manifest.py`) — emit real G1 manifest with per-source entries (`wlasl`, `msasl`, `asl_citizen`, later `phone`), signer-disjoint via existing split code.
- Create: `docs/g2_data_status.md` — what was obtained, what is gated on a human step, exact next commands.
- Test: `tests/test_g1_manifest.py` — manifest schema, signer-disjointness, per-source counts.

**Steps:**
- [ ] Research (web): ASL Citizen current download terms; MS-ASL annotation availability. Record findings + URLs in `docs/g2_data_status.md`
- [ ] Implement source modules + manifest builder with tests (manifest logic testable with synthetic entries — no network in tests)
- [ ] Attempt small legal download for locked vocab; if blocked by license click-through, STOP at scripts + instructions (do not bypass)
- [ ] If ≥2 sources materialize: run `eval/benchmark_g1.py` per-source with LandmarkBackend + oracle; save results JSON + summarize in `docs/g2_data_status.md`
- [ ] Commit: `feat(pipeline): ASL Citizen + MS-ASL source modules, real G1 manifest builder`

## Task E — W4: word-model abstention wiring (follows B; small)

- [ ] Run `eval/calibrate_abstention.py` for the word model against existing `eval/benchmark/multisource20_manifest.json` clips → `models/landmark/checkpoints/abstention.json` (marked "directional until G1" inside the JSON)
- [ ] `deploy/run_live.py` loads it (Task B already reads a thresholds file — this just supplies the real one)
- [ ] Commit: `feat(eval): directional word-model abstention thresholds wired into live path`

## Task F — Backup/remote (report-only)

- [ ] Check `gh auth status`. Do NOT create a remote without the user's word. Report recommendation: private GitHub repo + push both branches.

---

## Sequencing

A first (baseline green). Then B, C, D dispatched in parallel (disjoint files; agents do NOT run git — supervisor reviews, tests, commits sequentially). E after B. F is a one-liner check. Final: full pytest, update `README.md` status table, summary commit.

## Self-review notes

- Spec coverage: W1→D, W2→D (execution gated on data access), W3→C, W4→E(+B), W5/W6 already done (verified by audit + passing tests), W7→B. Hygiene→A. Backup→F.
- Known honest gaps: G2 pooled *training* needs GPU (Kaggle) once data exists — out of scope for this session, scripts land in D. Webcam+TTS end-user check needs a human at the machine.
