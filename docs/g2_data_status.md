# G2 multi-source data — status & exact next steps (2026-07-17)

Where the W1/W2 workstreams stand: the *code* path (source adapters → shared
entry shape → signer-disjoint splits → G1 manifest → benchmark → abstention
calibration) is built and tested end-to-end. The *data* now has one scripted
source flowing (MS-ASL) and one documented human step (ASL Citizen).

## Source-by-source

### MS-ASL — SCRIPTED ✅ (annotations local, videos via yt-dlp)
- Annotations: [Microsoft Download Center id=100121](https://www.microsoft.com/en-us/download/details.aspx?id=100121)
  — MS-ASL.zip (1.9 MB), extracted at `data/multisource/msasl/MS-ASL/`.
  License: **C-UDA-0.1** (Computational Use of Data Agreement; the PDF ships in
  the archive). 1,000 classes, 16k+ train entries, **real signer ids**.
- Locked-vocab coverage (train split): deaf 46 · yes 46 · drink 43 ·
  computer 36 · bowling 9 — all five words present (val/test add more).
- Videos are YouTube links; a fraction are dead. `pipeline/sources/msasl.py`
  `download_subset()` fetches per-clip sections via yt-dlp, tolerates dead
  links, and reports downloaded/dead counts. Raw media lands in
  `data/multisource/msasl/videos/` (gitignored).

### ASL Citizen — ONE-TIME HUMAN STEP ⚠️ (the phone-domain eval proxy)
- Official: [Download Center id=105253](https://www.microsoft.com/en-us/download/details.aspx?id=105253)
  — single `ASL_Citizen.zip` (**42.8 GB**, v1.0, no registration) + Demographics.zip.
  No per-gloss remote subsetting exists, so one person downloads once to a
  machine with ~90 GB free, extracts to `data/multisource/asl_citizen/`.
  License/details: [project page](https://www.microsoft.com/en-us/research/project/asl-citizen/).
- Kaggle route (fits the team's training workflow): community mirror
  [`abd0kamel/asl-citizen`](https://www.kaggle.com/datasets/abd0kamel/asl-citizen)
  can be attached to the Kaggle notebooks directly — **verify its license field
  against the official terms first; it is a third-party re-upload.**
- `pipeline/sources/asl_citizen.py` normalizes the split CSVs once the archive
  exists; until then every call fails loudly with these instructions.

### Phone recordings — NOT YET (by design)
Declared as an empty `phone` source in the G1 manifest shape; when the team's
recordings arrive they drop into `eval/benchmark/g1/videos/phone/` and join the
same benchmark with zero code changes.

## What already ran today (directional, n=32)

`pipeline/build_g1_directional_manifest.py` assembled the 32 on-disk clips
(12 in-vocab WLASL + 20 mixed WLASL/ASLense incl. OOV) and
`eval/benchmark_g1.py` scored both models (`eval/benchmark/g1/g1_directional_report.json`):

| view | overall top-1 | wlasl | aslense |
|---|---|---|---|
| vanilla_open | 0.406 | 0.545 | 0.10 |
| vanilla_closed | 0.375 | 0.545 | 0.0 |
| landmark (5-cls toy) | 0.219 | 0.318 | **0.0** |

`eval/calibrate_abstention.py --write-json` then set the word model's live
thresholds (`models/landmark/checkpoints/abstention.json`): **conf ≥ 0.95** is
the only operating point with confident-wrong ≤ 5% on mixed-domain data — i.e.
the current word model must almost always abstain outside its comfort zone and
the fingerspelling track (F4: 0.984 selective precision) carries the demo.
That is the fingerspelling-first strategy, now enforced by calibration rather
than judgment.

## Exact next steps (in order)

1. **Human:** run the ASL Citizen bulk download (or attach the Kaggle mirror to
   the notebooks after license check). ~1 command + disk space.
2. **Scripted:** `msasl.download_subset()` for the full locked vocab (+ the
   medical seed vocab from `docs/vocab-policy.md`). Measured probe (2026-07-17,
   bowling, n=12 attempts): **5 downloaded / 7 dead (~58% dead-link rate)** —
   plan clip budgets at roughly 2× the target count.
3. Build the real G1 manifest over wlasl + msasl + asl_citizen with
   ≥150 clips, run `benchmark_g1.py`, recalibrate abstention on it
   (drop `decision_grade=false`).
4. **G2 pooling proof (Kaggle GPU):** train WLASL+MS-ASL for the locked vocab
   with the signer-disjoint splitter, test held-out ASL Citizen, per-source
   metrics. Only after this, scale vocabulary.
