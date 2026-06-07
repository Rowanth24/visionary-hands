# Fingerspelling track — usage, data, gates

Phase-1 isolated 24-letter recognizer. Spec: `docs/superpowers/specs/2026-06-06-fingerspelling-design.md`.
Plan: `docs/superpowers/plans/2026-06-06-fingerspelling.md`.

## Extractor: MediaPipe **Hands** (not Holistic)

The word model uses MediaPipe **Holistic** (face+pose+hands) — right for full-body sign
videos. But Holistic locates hands via the body ROI and **fails on hand-crop alphabet
images**. Measured on the held-out set's confusable letters: **Holistic 15% vs Hands 90%**
hand-detection. So fingerspelling extraction uses `models/fingerspelling/extraction.py`
(`extract_hand_landmarks_from_frames`, MediaPipe Hands), placing the detected hand into the
same 543-row layout so `handshape_features` is unchanged. *(Supersedes the spec's earlier
mention of `extract_landmarks_from_frames`.)*

## Datasets (downloads done; raw lives under gitignored `data/fingerspelling/`)

| Role | Kaggle slug | Notes |
|---|---|---|
| **A — train** | `grassknoted/asl-alphabet` | 87k imgs, 3000/letter, 200×200 RGB (largely single-signer) |
| **B — train (diversity)** | `dorukdemirci/asl-alphabet-dataset` | ~100/letter, independent source |
| **C — held-out eval** | `danrasband/asl-alphabet-test` | independent photos; **eval-only, NEVER train** |

Extraction caps: A at 350/letter (foreground time), B/C uncapped. License note: grassknoted is
GPL-2.0 — for internal research/eval use; revisit before any redistribution (see ATTRIBUTION.md).

## F0 extraction yield (handshape features kept)

- **set_a:** N=7,390 (most letters 350; low-yield: **n=51, m=164** — folded-thumb letters)
- **set_b:** N=2,390 (~96–102/letter, balanced)
- **set_c (held-out):** N=684 (~20–30/letter)

The low-yield confusable letters (n, m) are flagged for the F1 confusion-matrix analysis and
the F2 per-letter abstention thresholds — exactly the cluster the design expects to guard.

## Pipeline

```bash
. .venv-oracle/bin/activate && source env.sh
# images -> Hands landmarks -> handshape features -> npz + yield report
python models/fingerspelling/prepare_alphabet_dataset.py \
  --images-dir <letter-subfolders> --dataset-name set_x --out data/fingerspelling/set_x.npz
```
Accepts plain letter dirs (`A/`) and `<LETTER>-samples/` dirs. J and Z are excluded (dynamic).

## F1 results — cross-dataset eval (held-out set C)

- **Baseline floor** (logistic, train A+B → C): **0.87** cross-dataset (chance 0.042).
- **MLP** (train A+B → held-out C): **0.85** overall; `min_letter_accuracy = 0.07`.
- **Overfit alarm NOT triggered** — landmark features transfer across datasets; even a linear
  model generalizes. The landmark-first bet is validated at this gate.
- **Failures are concentrated in the predicted confusable cluster.** Top confusions
  (true→pred, count): n→m (24), t→l (9), v→u (9), s→m (8), u→r (8), p→q (7), v→k (6), g→r (5).
  Folded-thumb / 2-finger / pointing look-alikes — exactly what F2 abstention must guard.

**F1 GATE: PASS → F2.** No overfit collapse; the confusion matrix gives F2 its per-letter
abstention targets (n, m, s, p, q, v, u, t). (Eval JSON: `data/fingerspelling/eval_on_C.json`,
gitignored.)

## F2 results — abstention calibration (held-out C)

- **A single global confidence threshold fails:** confident-wrong ≤ 2% requires τ≈1.0 →
  ~0 coverage. The model is *confidently* wrong on the folded cluster (n→m), so confidence
  alone can't separate errors. (τ=0.90 → coverage 0.83, confident-wrong 0.073.)
- **Per-letter precision calibration (target 0.95) salvages 18/24 letters:** a,c,d,f,g,h,s,t,
  v,x,y need no gate; b,e,k,o,p,q,w need a higher threshold.
- **6 letters can't reach precision at any confidence → always-abstain on this data:
  i, l, m, n, r, u** (the confusion sinks). Escape hatches triggered for these: F4 real-data
  and/or targeted hybrid (image detail for the folded cluster only).
- **Caveat:** held-out n is tiny (predicted-`n` = 4; ~20–30/letter), so the always-abstain set
  is partly small-n artifact. Robust signal = n→m (24×). Final set firms up at F4.

**F2 GATE: PASS-WITH-CAVEAT → F3.** Per-letter thresholds in `models/fingerspelling/
checkpoints/abstention.json` (gitignored). Recognizer abstains on the 6 unreliable letters
until the F4 real-data gate / targeted hybrid resolves them.

## Medical-trust gate (carried)

Cross-dataset eval (train A+B, test C) is the **overfit alarm**, not the green light. Before any
"this works" medical claim, a small **real-data spot-check** (user-recorded) must corroborate
the cross-dataset numbers (spec F4). Public sets share biases; cross-public eval alone is
insufficient for clinical trust.
