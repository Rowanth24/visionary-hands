# Gate G-D — cross-source code audit (reuse, don't rebuild)

Audit of the two existing cross-source scripts the plan says G2/G4 should build on:
`eval/prepare_cross_source_finetune_data.py` and `eval/train_vanilla_cross_source.py`.

**Verdict: reusable foundation, with fixable gaps.** Both compile in `.venv-oracle`.
The data-prep + adaptation loop is sound and source-aware. Before a real run, fix
the layout-skewed default paths and decide on dataset downloads; for G2 add
signer-disjoint splitting (not implemented); for the landmark model a parallel
adaptation path still has to be written (these scripts are vanilla/I3D-only).

---

## 1. `prepare_cross_source_finetune_data.py` (341 lines)

Builds a mixed-source fine-tune set and writes `train_manifest.json` / `val_manifest.json`.

- **Sources (both HuggingFace, `urllib` — NOT yt-dlp, NOT YouTube scraping):**
  - WLASL — `huggingface.co/datasets/Voxel51/WLASL`
  - ASLense — `huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset`
  - `build_wlasl_index` reads local WLASL metadata; `build_aslense_index` lists the HF
    dataset tree via the HF API and caches `aslense_index.json`; `download_video`
    fetches ASLense clips over HTTPS into `--output-dir`.
- **Leakage-aware:** excludes held-out labels/ids (from `--heldout-manifest`) so the
  fine-tune set can't contaminate the benchmark eval set. Good.
- **Outputs:** `train_manifest.json`, `val_manifest.json`, `aslense_index.json`,
  a summary, and downloaded videos — all under `--output-dir`.
- **Manifest format** matches `MixedSignDataset` (samples with `local_video` + `label`).

⚠️ **Network access:** running this downloads from HuggingFace. Clean (no yt-dlp), but
still external fetches — needs a go decision before running.

## 2. `train_vanilla_cross_source.py` (357 lines)

Fine-tunes the vanilla model on the mixed-source manifest, source-aware.

- **Model:** the "oracle" is I3D features → `SignLanguageTransformer` (transformer
  head) → `classifier`. The I3D backbone is **frozen**; `--train-scope` adapts either
  just `model.transformer.classifier` (`classifier`, default) or the whole
  `model.transformer` (`transformer`). Attribute paths verified to exist. ✓
- **Anti-forgetting regularizer — actually L2-SP, not true EWC:**
  `anchor_regularization = mean((param - anchor)^2)`, `loss = ce + anchor_l2 * anchor`
  (`--anchor-l2` default 0.1). The plan called it "EWC-style"; precisely it is
  **L2-toward-starting-point (L2-SP)** — no Fisher-information weighting. Cheaper than
  EWC; upgrade to diagonal-Fisher EWC only if L2-SP underperforms.
- **Source-aware eval:** `evaluate()` reports per-source accuracy (totals keyed by
  source) — directly useful for G1-style per-source reporting.
- **Inputs:** `--train-manifest --val-manifest --video-dir --base-checkpoint
  --output-dir` + hyperparams. **Outputs:** best/last checkpoints + history.
- **Dataset:** `eval/datasets/mixed_sign_dataset.py :: MixedSignDataset` present;
  reads `local_video` + label, needs a labels index file.

## 3. Existing trained outputs (in the original clone, NOT copied here)

```
~/Vanilla Model ASL/Sign-Language-Recognition/benchmark/
  cross_source_validated_12/run_001/{best,last}_cross_source_model.pth
  cross_source_adaptation/run_001/last_cross_source_model.pth
  cross_source_validated/run_001/{best,last}_cross_source_model.pth
```

Prior adapted checkpoints from earlier runs — reusable as warm starts or for
evaluation, but they live in the clone (copy in if needed).

## 4. Findings to fix before a real run

1. **Layout-skewed default paths (breakage).** Defaults assume the OLD repo layout
   where `benchmark/` sat at repo root; here it's `eval/benchmark/`:
   - `--heldout-manifest` defaults to `repo_root/benchmark/multisource20_manifest.json`
     → should be `eval/benchmark/multisource20_manifest.json` (pass explicitly or fix).
   - `--wlasl` defaults to `~/wlasl_samples.json` → **does not exist** (WLASL metadata
     isn't in the repo); must be supplied.
   - Same root cause as the `MEDIAPIPE_HOLISTIC_MODEL_PATH` (`parents[1]/benchmark/...`)
     in `asl_backends.py:54`. A small follow-up could repoint all three.
2. **Network downloads** (HF) — needs a go decision (not yt-dlp, but still external).
3. **Signer-disjoint NOT implemented** — `prepare` splits by sample/id, not by signer.
   Gate G2 explicitly requires **signer-disjoint** splits to kill leakage that inflates
   in-distribution numbers. This logic must be added for G2.
4. **Vanilla/I3D-only.** Both scripts adapt the I3D-transformer oracle. The **landmark**
   model's cross-source adaptation path does not exist yet — for a landmark G4 we'd
   reuse the landmark training (`models/landmark/train_landmark_transformer.py`) on the
   mixed-source landmark features, with an analogous anchor regularizer.

## 5. How G2 builds on this

- Use `prepare_cross_source_finetune_data.py` to assemble the WLASL+ASLense (+MS-ASL
  later) video pool with leakage exclusion — **after** repointing default paths and
  approving downloads.
- **Add signer-disjoint splitting** (new code) to meet the G2 bar (≥50 clips/word,
  signer-disjoint per-source splits).
- For the landmark track, run the existing landmark extraction
  (`pipeline/extract_wlasl_landmarks.py` → `prepare_landmark_dataset.py`) over the
  pooled videos so both models train on the same source pool.

## 6. How G4 builds on this

- `train_vanilla_cross_source.py` is the vanilla adaptation experiment for G4 ("beat
  vanilla+shortlist"): fine-tune the head with L2-SP anchoring, evaluate per-source on
  the G1 harness.
- Mirror it for the landmark model (anchor-regularized landmark fine-tune).
- Compare both against the frozen-base baselines through `benchmark_g1.py` (per-source,
  CIs) — decision by data, not opinion.

## 7. Concrete next steps (when G2/G4 start)

1. Repoint the three layout-skewed default paths (small PR) — independent, do anytime.
2. Decide on HF dataset downloads (go/no-go).
3. Add signer-disjoint split support to `prepare_cross_source_finetune_data.py`.
4. Write the landmark-side cross-source adaptation (reuse landmark trainer + anchor).
5. Optionally evaluate the existing clone checkpoints through `benchmark_g1.py` to see
   what prior adaptation already bought.

**Nothing here needs rebuilding.** The pieces exist; G2/G4 are mostly wiring +
signer-disjoint splits + a landmark-side mirror of the adaptation loop.
