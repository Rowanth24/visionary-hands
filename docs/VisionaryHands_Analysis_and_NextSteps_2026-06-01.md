# Visionary Hands — Issue Analysis & Next Steps
**ASL → Text (and basic 2-way) communication tool · EPICS Spring 2026**
Prepared 2026-06-01 from: `EPICS_Project/` (Kaggle run outputs + local runs), the Spring 2026 Design Doc, and the team's own `docs/landmark_progress.md` + `docs/landmark_branch_raspberry_pi.md`.

> Scope of this document: **analysis + recommended next steps only. No code was changed.**

---

## 1. Executive summary

The project is in better shape than the raw numbers suggest — but the team is **spending almost all of its compute, disk, and attention on the one pipeline that cannot work for this project, while the pipeline that *can* work is starved of the one thing it needs: data.**

Three findings drive everything below:

1. **The I3D video pipeline is a dead end.** Across **8 Kaggle runs (v13–v23)** on 20 classes it never beat ~16% validation / ~16% test accuracy — barely above the 5% random-chance floor — and it fails the project's own 70% quality gate every single time. The training curves show classic catastrophic overfitting (train → 88%, val stuck ~10%). It also can never run on the team's stated edge target (an offline Raspberry Pi). **Recommend: stop running it.**

2. **The landmark pipeline is the right architecture and is the one that works.** The best model in the entire repo is `landmark_live5_transformer_compact_dual_v1`: **70.6% val / 58.3% test on 5 words**, lightweight, exported to ONNX, CPU-ready. Switching from single-hand (66 pts) to dual-hand (92 pts) features jumped test accuracy from 41.7% → 58.3% — a concrete, reproducible win.

3. **Data quantity is THE bottleneck, not architecture or hyperparameters.** Every model is trained on **64–98 clips across 5–8 classes (~13 clips/class)**. That is far too little for any sequence model to generalize. The team has run ~17 training variations total; tuning has plateaued. The next 10x improvement comes from **data**, not another LR schedule.

There is also a **reporting gap to close before the 5/2 instructor sign-off**: the design doc states "≈75–85% accuracy," but no saved evaluation anywhere in the repo exceeds **58.3% (on 5 words)**. The 75–85% reads as an informal live-demo impression, not a measured held-out number. This needs to be reconciled honestly.

---

## 2. What the system actually is

Two parallel pipelines were built. They are **not** two versions of the same thing — they are different bets:

| | **I3D video pipeline** | **Landmark pipeline** |
|---|---|---|
| Input | Raw RGB video clips (50 frames) | MediaPipe landmarks (T×543×3 → reduced to 66/92 pts, xy-only, 64 frames) |
| Model | I3D ResNet-50 backbone (~28M params, 164 MB `.pt`) | Compact Transformer / GRU (tiny, ONNX-exportable) |
| Driver | `vl-signlanguage-paper-dual-gpu.py` (166 KB Kaggle kernel) | `train_landmark_transformer.py` + `landmark_baseline/` |
| Output artifacts | 640–700 MB per run (`best.pt`, `last.pt`, 300 MB `signvision_paper_artifacts.zip`) | A few MB per run |
| Edge-deployable? | **No** (won't fit/run on Pi) | **Yes** (ONNX/TFLite, CPU, offline) |
| Best result | ~16% (20 cls) | **58.3% (5 cls dual-hand)** |
| Status | Failing the 70% gate on all 8 runs | Mature: data-prep → eval → raw-video → live webcam → ONNX bundle (Levels 0–4 done) |

The landmark pipeline already has the full toolchain the project needs: extraction, dataset prep (with handedness normalization + dual-hand layout), training, TorchScript + ONNX export, single-clip inference, manifest evaluation, live webcam loop, CPU latency benchmarking, and a packaged deployment bundle. **The engineering is done; the data and the consolidation decision are not.**

---

## 3. The evidence

### 3a. I3D / WLASL Kaggle runs (20 classes, gate = 70%)

| Run | best_val_acc | test_acc | Epochs | Gate | Train clips |
|-----|-------------|----------|--------|------|-------------|
| v13 | 16.1% | 10.2% | 10 | ❌ | 243 |
| v14 | 26.8% | 14.3% | 5 | ❌ | 243 |
| v18 | 23.2% | 16.3% | 3 | ❌ | 243 |
| v19 | 21.4% | 12.2% | 8 | ❌ | 243 |
| v20 | 19.6% | 12.2% | 4 | ❌ | 243 |
| v21 | 15.5% | 8.0% | 8 | ❌ | 248 |
| v22 | 15.5% | 10.0% | 10 | ❌ | 248 |
| v23 | 19.6% | 12.2% | 4 | ❌ | 243 |

Random chance on 20 classes ≈ 5%. The model is essentially **not learning a generalizable signal.** v22 epoch-by-epoch shows the cause unmistakably:

```
epoch  train_acc  val_acc   val_loss
  1      12.1%      6.9%      3.57
  6      48.8%     13.8%      4.12
  8      79.0%     15.5%      3.82   ← memorizing train set
 10      87.9%     12.1%      3.82   ← val flat/declining, val_loss rising
```

This is **textbook data starvation + overfitting**, not a tuning problem. More epochs, unfreezing, balanced samplers, and LR schedules were all tried across the 8 runs and none moved val accuracy.

### 3b. Landmark runs (the deployable path)

| Run | Classes | val_acc | test_acc | Train clips |
|-----|---------|---------|----------|-------------|
| **live5 transformer — dual-hand** | 5 | **70.6%** | **58.3%** | 64 |
| live5 transformer — single-hand | 5 | 47.1% | 41.7% | 64 |
| live5 transformer — delta | 5 | 47.1% | 41.7% | 64 |
| wlasl8 transformer compact | 8 | 37.5% | 35.0% | 98 |
| wlasl8 v1 | 8 | 37.5% | 30.0% | 98 |
| wlasl8 transformer aug | 8 | 33.3% | 25.0% | 98 |
| wlasl8 transformer xyz | 8 | 33.3% | 25.0% | 98 |
| wlasl8 gru | 8 | 29.2% | 25.0% | 98 |

The dual-hand curve is **healthy** — train and val rise together (train 31%→73%, val 29%→~65%), small generalization gap — confirming the architecture is sound and would respond to more data. (Note: adding the z/depth axis — `xyz` run — did **not** help at this data scale; revisit only after data grows.)

### 3c. Vocabulary

All work is word-level on a handful of WLASL glosses: **`bowling, candy, computer, cousin, deaf, drink, pizza, yes`** (8-class) and **`bowling, computer, deaf, drink, yes`** (5-class "live"). There is **no fingerspelling/alphabet model anywhere in the repo**, even though the design doc lists "recognize the entire ASL alphabet" as in-scope.

---

## 4. Issue analysis (ranked)

### 🔴 CRITICAL

**C1 — Effort is concentrated on the non-viable pipeline.**
The I3D pipeline consumes ~5 GB of disk, all the GPU runs (v13–v23), and the 166 KB paper kernel — and it fails the gate every time *and* can't deploy to the Pi target. Continuing it burns the team's scarcest resources (Kaggle GPU hours, time before sign-off) for zero deliverable value.
→ **Park/retire I3D. Reallocate 100% to the landmark pipeline.**

**C2 — Data starvation is the root cause of low accuracy.**
~13 clips/class makes ≥80% accuracy mathematically unlikely for *any* model. This single issue explains the I3D failure, the landmark ceiling, and the gap to the 80% requirement.
→ **A concrete data-collection plan is the highest-leverage next step (Section 5, Phase 1).**

**C3 — Reported accuracy (75–85%) is not supported by any saved metric (max measured = 58.3% on 5 words).**
This is a validity/credibility risk for the 5/2 instructor & mentor sign-off and for the design doc's verification table.
→ **Reconcile before sign-off:** either (a) produce a measured number via a fixed back-test set, or (b) reword the claim as "live-demo impression on 5 words, formal held-out accuracy = X%."

### 🟠 HIGH

**H1 — Vocabulary far below the stated scope.** 5–8 words vs. "alphabet + common words." The gap between demo and deliverable is large and unacknowledged in the roadmap.

**H2 — No alphabet/fingerspelling model exists**, despite being an in-scope deliverable. This is also the *easiest* high-accuracy win available (static hand-shape classification).

**H3 — Two pipelines split focus and there's no written decision to consolidate.** The team docs still hedge ("TorchScript or ONNX both work," two "best" models). Ambiguity slows everyone.

**H4 — Edge/latency requirement (≤1.5 s end-to-end) is asserted but not measured on a real target.** No Jetson/Pi on hand; CPU benchmarks exist but the *full* capture→landmark→infer→TTS budget on the actual device is unverified.

### 🟡 MEDIUM

**M1 — Repository hygiene: 7.5 GB / 70,052 files, no git, no .gitignore.**
Two virtualenvs committed (`.venv-landmarks` 1.1 GB, `.venv-onnx` 680 MB), 15k `.pyc`, `.DS_Store`, and ~10 near-duplicate `kaggle_outputs_v*` snapshots each repeating a 164 MB `best.pt` + `last.pt` + 300 MB artifacts zip (~5 GB of duplicates). Reclaimable: **~7 GB**.

**M2 — Experiment tracking is ad-hoc.** Good per-run `run_config.json` + `metrics.json` exist, but there's no single index to see the trajectory, so the same hyperparameter regions risk being re-explored.

**M3 — Reproducibility cost on Kaggle.** Each run re-emits a 300 MB artifacts zip and full checkpoints; switching to Kaggle dataset versioning for inputs and emitting only the best checkpoint would cut run cost and storage sharply.

**M4 — Validation gap (acknowledged in doc).** No formal testing with Deaf-community / ASL users yet; accuracy is measured on WLASL clips, not on the actual end users or signing-style diversity.

### 🟢 LOW / hygiene
- `embeddedviewer.js` (2.7 MB) and saved `kaggle_*.html` pages look like vendored junk — verify and remove.
- `xy_only=true` discards depth; fine for now (see 3b) but worth a note in the model card.
- Motion letters (J, Z) will need special handling whenever the alphabet model lands.

---

## 5. Recommended next steps (phased)

The throughline: **kill the dead weight → fix the data → consolidate → prove the deliverables → clean up.** Ordered by leverage.

### Phase 0 — Stop the bleeding (this week, ~1 day)
- [ ] **Freeze the I3D/paper pipeline.** No more `v*` GPU runs. Keep one best snapshot for the record; archive the rest.
- [ ] **`git init` + `.gitignore`** (`.venv-*`, `__pycache__`, `*.pyc`, `.DS_Store`, `*.pt`, `*.zip`, large `data/`). Commit the source (~a few MB) so the actual work is finally version-controlled.
- [ ] **Dedupe `kaggle_outputs_v*`:** keep the single best run + a small metrics manifest; move/delete the rest. Reclaims ~7 GB. Move any model you must keep to external storage / Git LFS / a Kaggle dataset.

### Phase 1 — Attack the data bottleneck (highest leverage, ~1–2 weeks)
This is the step that actually raises accuracy. Pick a **fixed demo vocabulary** and collect real data for it.
- [ ] **Lock a demo vocabulary** (recommend ~10–15 high-value words the community partner cares about + the alphabet as a separate mode). Stop drifting between 5 and 8.
- [ ] **Collect 50–100+ clips per class** via the existing webcam path (`run_landmark_webcam.py` / `validate_live_sign_capture_onnx.py` already capture live). Vary **signer, lighting, distance, background, hand** to fight the overfitting seen in §3.
- [ ] **Supplement with all available WLASL instances** for those exact glosses (you already have `download_wlasl_subset.py`), plus **temporal/spatial augmentation** on landmarks (time-warp, jitter, mirror, frame-drop) — cheap and effective for sequence models.
- [ ] **Standardize on the dual-hand layout** (`feature_shape [64, 92, 2]`) — it already gave +16 pts.
- [ ] Re-train the compact transformer; expect a meaningful jump purely from data.

### Phase 2 — Land the easy deliverable: alphabet/fingerspelling (~1 week, parallelizable)
- [ ] Build a **static hand-shape classifier** on MediaPipe **hand** landmarks (single-frame MLP). This is a well-trodden, high-accuracy task and directly fulfills an in-scope deliverable. Public ASL-alphabet datasets exist; combine with a few self-captured samples.
- [ ] Exclude or special-case **J and Z** (motion letters). Expose alphabet as a distinct "spelling mode" in the demo.
- [ ] This likely hits ≥80% far faster than the word model and de-risks the semester deliverable.

### Phase 3 — Consolidate & prove the requirements (~1 week)
- [ ] **Write a one-paragraph decision record:** "Landmark + ONNX is the production pipeline; I3D is archived." Remove the hedging in the docs.
- [ ] **Build a fixed back-test manifest** (the design doc's "10–50 videos not in WLASL") and report **top-1 and top-3 accuracy + a confusion matrix.** (Top-3 is genuinely useful for an assistive UX with a confirm step.)
- [ ] **Reconcile the 75–85% claim** with the measured number before 5/2.
- [ ] **Measure the ≤1.5 s end-to-end budget** (capture → MediaPipe → infer → TTS) on a real device. Buy/borrow a Raspberry Pi (or Jetson) now; the landmark+ONNX path almost certainly meets it, but it must be *shown*, not asserted. Compare TorchScript vs TFLite per the team's own plan.

### Phase 4 — Close the 2-way loop & validation (toward Fall 2026)
- [ ] Wire **TTS** (model output → speech) and **speech-to-text** (non-signer → text) into the demo — design-doc deliverable, mostly integration glue.
- [ ] **Plan Deaf-community / ASL-user testing** for Fall (the doc's stated validation gap). Even 2–3 ASL users testing the locked vocabulary would massively strengthen validation and surface signing-style robustness issues.
- [ ] Add a small **experiment index** (one CSV/MD appended per run) so the accuracy trajectory is visible at a glance.

---

## 6. Strategic decisions for you (the real forks)

These aren't engineering details — they change the roadmap and only you/the team can choose:

1. **Narrow-and-excellent vs. broad-and-mediocre vocabulary.**
   Given the ≥80% requirement and a *demo*, a small locked vocabulary (10–15 words) + alphabet, done well, beats 50 shaky words. **Recommendation: narrow-and-excellent.**

2. **Self-collected data vs. WLASL-only.**
   WLASL clips don't match your webcam domain (distance, framing, signer). Self-collected data for the locked vocab is the fastest path to real accuracy *and* matches deployment conditions. **Recommendation: hybrid, lean self-collected.**

3. **Retire I3D permanently, or keep it as a "research" track?**
   It can't run on the Pi and isn't learning. **Recommendation: archive it; revisit only if you ever get thousands of clips/class and a GPU deployment target — neither is true now.**

4. **Continuous translation (the `seq2seq_bart` / How2Sign config in the kernel) — in or out?**
   It's far beyond current data/scope and the team docs already say LLM sentence-construction is out of scope for deployment. **Recommendation: explicitly out of scope; remove the dangling config so it stops implying a commitment.**

---

## 7. What I could not access (and how to unblock)

- **Live kaggle.com runs:** I analyzed the **locally synced outputs** (`kaggle_outputs_v8`–`v23`, the `runs/` dir, and the `vl-signlanguage-paper-dual-gpu` kernel), which *are* the saved results of those runs. I did not query kaggle.com itself (I was scoped to `EPICS_Project/` and the design doc, and have no authenticated Kaggle session here).
  → If you want a live pull (e.g., compare the two Kaggle accounts "Rowanth Kaggle" / "Laksya Kaggle," check kernel run history/GPU quota), share the kernel slugs/URLs or allow `~/.kaggle/kaggle.json` access and I can fetch run metadata directly.

---

## Appendix — key file map (landmark pipeline)

| Stage | Script |
|-------|--------|
| Extract MediaPipe landmarks (T×543×3) | `extract_wlasl_landmarks.py` |
| Prepare fixed arrays (handedness norm, 64-frame, dual-hand) | `prepare_landmark_dataset.py` |
| Pure-NumPy preprocessing (train + on-device) | `landmark_baseline/preprocess.py` |
| Train compact transformer/GRU | `train_landmark_transformer.py` |
| Export | `export_landmark_torchscript.py`, `export_landmark_onnx.py` |
| Single-clip inference | `predict_landmark_video.py`, `predict_landmark_video_onnx.py` |
| Manifest evaluation (main regression test) | `evaluate_landmark_video_manifest.py` (+`_onnx`) |
| Live webcam | `run_landmark_webcam.py` (+`_onnx`), `validate_live_sign_capture_onnx.py` |
| Latency benchmark (CPU/edge) | `benchmark_landmark_model.py`, `benchmark_landmark_onnx.py` |
| Package deployable bundle | `package_landmark_bundle.py` |
| Best model today | `runs/landmark_live5_transformer_compact_dual_v1/` → `deployment/landmark_live5_dual_onnx_bundle/` |
