# Fingerspelling-first track — design spec

**Status:** approved direction (brainstorm + council debate), pending spec review → writing-plans.
**Date:** 2026-06-06

## 1. Purpose & scope

A fingerspelling recognizer that serves as the **OOV / names / meds fallback** for the
closed-vocab medical communication system — when a word isn't in the closed vocabulary
(a patient's name, a drug), the user spells it.

**In scope (Phase 1 — isolated letter-at-a-time):**
- The **24 static ASL letters** (A–Z excluding **J** and **Z**, which are dynamic).
- One handshape at a time: user forms a letter, system classifies it (or abstains),
  user moves to the next letter to build a word.
- Offline, on **Jetson Orin Nano**.
- Abstention-guarded (it must decline rather than guess a drug-name letter wrong).

**Non-goals (deferred to Phase 2):**
- Continuous / co-articulated fingerspelling (segmenting a fluid stream).
- Dynamic letters J and Z (require motion modeling).
- Word-level autocorrect / language-model assistance over spelled letters.

## 2. Decisions (locked with the user)

| Decision | Choice |
|---|---|
| Interaction model | Phased: **isolated letter-at-a-time now**, continuous later |
| Frame source | **Public datasets only** for now; team records real video **iff** eval reveals overfit |
| Routing | **Auto-fallback on abstention** (low-confidence word → suggest spell mode) **+ manual toggle** |
| Architecture | **Per-frame landmark MLP** (Approach A) + serve-time hold-window majority vote + abstention |

## 3. Architecture (Approach A + debate modifications)

```
camera frame
  → MediaPipe Holistic (existing extraction)         # 21 hand landmarks/hand
  → preprocess_landmark_sequence(1 frame,            # REUSE the G-C standard:
       layout="dominant_hand", xyz)                  # dominant-hand, handedness-canonical, normalized
  → feature vector (xyz)                             # depth INCLUDED (debate mod #1)
  → small MLP  →  softmax over 24 letters
  → serve-time hold-window majority vote             # temporal stability w/o temporal training
  → AbstentionPolicy (per-letter / per-cluster thresholds)   # REUSE eval/abstention.py (mod #3)
  → letter OR "not confident, repeat"
```

**Why landmark MLP (not CNN):** appearance-invariant (mitigates the public-data overfit
risk the user flagged), tiny/fast on Jetson, and reuses the entire landmark + abstention +
preprocessing stack. The image-CNN alternative was rejected — it is the *worst* choice for
the overfit risk and breaks the project's landmark-first philosophy.

**Debate modifications baked in:**
1. **xyz, not xy** — include MediaPipe depth to recover thumb/finger separation on the
   folded-letter cluster (A/E/M/N/S/T).
2. **Confusion-aware eval is the gate** — per-letter accuracy + a full confusion matrix,
   not overall accuracy (which would hide the confusable cluster failing).
3. **Per-letter / per-cluster abstention thresholds** — abstain harder on the confusable
   cluster; target a strict confident-wrong budget, accept lower coverage, let the user repeat.
4. **Mandatory real-data spot-check gate before any medical-trust claim** — cross-dataset
   eval is the early *alarm*; it is NOT the green light. A small real recording is required
   to *trust* the numbers (not necessarily to train) before the tool is presented as reliable.
5. **Targeted hybrid escape hatch** — if the confusable cluster stays bad after xyz +
   abstention, add image detail *only for those letters* (not a full CNN), preserving
   appearance-invariance everywhere else.

## 4. Data plan (the crux: enough frames for 24 letters)

Static handshape on ~21 landmarks (≈63 features xyz) needs only **hundreds–low-thousands of
clean frames/letter** for 24 well-separated classes — public data far exceeds this.

- **Sign Language MNIST** — 24 classes, **letter-only, J & Z already excluded** (≈1k/letter).
  Near-perfect match; primary starter set.
- **Kaggle "ASL Alphabet"** (Akash) — ≈3k images/letter; second source.
- **A third public set** (Massey / ASL-FS / similar) held out **for cross-dataset eval**.

**Pipeline:** public images → MediaPipe Holistic → 21 hand landmarks →
`preprocess_landmark_sequence` (1 frame, dominant_hand, xyz) → feature `.npz` per split.

**Frame-budget target:** ≥ **500–1,000 usable landmark frames/letter** after extraction
(MediaPipe fails to detect a hand on some images — track per-letter **yield**), balanced
across the 24 classes.

**Overfit detector:** train on sets A+B, **test on held-out set C** (cross-dataset). If
per-letter accuracy on C collapses vs A/B → overfit alarm → user records real ASL-alphabet
video. (Public sets share biases — single-signer, right-hand, clean background — so
cross-dataset is the *alarm*, the real-data spot-check is the *trust gate*.)

## 5. Integration

- **Routing:** the closed-vocab word backend abstains (low confidence) → system suggests
  spell mode (auto-fallback); clinician can also flip a manual **spell-mode toggle**.
  Reuses `eval/abstention.py` for the abstain signal.
- **Reuse, don't rebuild:** MediaPipe extraction, `preprocess_landmark_sequence` (G-C
  standard — no new preprocessing, no train/serve skew), and the abstention layer are all
  reused. Only the classifier + its data prep + eval are new.

## 6. Repo layout (minimal, Karpathy-lean)

```
models/fingerspelling/
  labels.json                  # the 24 letters
  prepare_alphabet_dataset.py  # public images -> landmarks -> npz  (reuses landmark_baseline)
  train_fingerspelling.py      # small MLP on landmark features
  eval_fingerspelling.py       # cross-dataset acc + per-letter confusion matrix + abstention curve
  checkpoints/                 # gitignored (weights)
tests/test_fingerspelling.py   # preprocessing-reuse + classifier I/O + abstention wiring
docs/fingerspelling.md         # usage + data sourcing + the medical-trust gate
```
Reused unchanged: `landmark_baseline/` (extraction, preprocess, features), `eval/abstention.py`.

## 7. Coding approach (Karpathy guidelines)

- **Think before coding:** assumptions stated here; the one real unknown is the confusable
  cluster — addressed by xyz + confusion-aware eval before any model-size escalation.
- **Simplicity first:** start with the *smallest* model that works — a shallow MLP (or even a
  kNN/logistic baseline first to set the floor). No temporal model, no CNN, no augmentation
  framework until the data proves they're needed.
- **Surgical / reuse:** no new preprocessing (reuse the G-C standard), no new abstention
  (reuse `eval/abstention.py`), no changes to the word-model code. New code is confined to
  `models/fingerspelling/`.
- **Goal-driven, verifiable:** every phase has a measurable gate (below). Build the eval
  (confusion matrix + abstention curve) *before/with* the model, then iterate to pass it.

## 8. Success criteria (verifiable gates)

- **F-baseline:** a kNN/logistic floor on cross-dataset eval, to know what "learning nothing"
  vs "real signal" looks like.
- **F-accuracy:** per-letter accuracy + confusion matrix on the **held-out dataset C**;
  overall is reported but the **per-letter minimum** and the confusable-cluster numbers are
  the gate.
- **F-safety (abstention):** with per-letter thresholds, reach a **strict confident-wrong
  budget on a drug-name-relevant letter set** (target ≤ ~1–2% confident-wrong), reporting the
  coverage that buys. Calibrated with `calibrate_abstention.py`-style sweeps.
- **F-trust gate:** **before any "this works" medical claim**, a small real-data spot-check
  (user-recorded) must corroborate the cross-dataset numbers. Cross-dataset alone is not
  sufficient.

## 9. Phased milestones

- **F0 — scaffold + data:** `models/fingerspelling/` skeleton, labels, `prepare_alphabet_dataset.py`;
  acquire 2–3 public sets (download decision required); extract landmarks; report per-letter yield.
- **F1 — train + cross-dataset eval:** baseline floor, then MLP; per-letter + confusion-matrix
  report on held-out set C. **Overfit alarm check here.**
- **F2 — abstention calibration:** per-letter / per-cluster thresholds; confident-wrong budget
  vs coverage curve; confusable-cluster handling.
- **F3 — integration:** auto-fallback-on-abstention routing + manual toggle; live hold-window
  majority vote.
- **F4 — trust gate + Phase 2 seeds:** real-data spot-check (user records iff needed);
  groundwork for continuous fingerspelling + J/Z.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Folded-letter confusability (A/E/M/N/S/T) | xyz depth; confusion-aware gate; hard abstention on cluster; targeted hybrid hatch |
| Public-data overfit invisible to cross-public eval | cross-dataset eval as alarm + mandatory real-data trust gate |
| Confident-wrong on a drug name = safety incident | strict per-letter confident-wrong budget; abstain-and-repeat UX |
| MediaPipe hand-detection yield on some images | per-letter yield tracking; drop/augment low-yield letters; real data if needed |
| Scope creep into continuous/J-Z | explicitly Phase 2; Phase 1 ships the isolated fallback first |

## 11. Open items (carried, not blocking the spec)

- **Download go/no-go** for the public datasets (direct downloads, not yt-dlp) — needed at F0.
- Exact third dataset choice for the held-out cross-dataset split.
