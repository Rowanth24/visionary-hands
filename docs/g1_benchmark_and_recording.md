# Gate G1 — benchmark + phone-camera recording guide

G1 is the first **decision-grade** benchmark. G-B (n=12, in-distribution WLASL)
could only prove the harness works; it could not decide landmark-vs-vanilla because
it never tested the landmark model's actual claim: **cross-source / real-world
(phone) generalization.** G1 fixes that.

**G1 passes when** the benchmark has **≥150–200 clips**, **≥3 sources including
phone**, and we report **per-source top-1 accuracy with 95% confidence intervals**.
`benchmark_g1.py` enforces and prints this; it labels a run `NOT decision-grade`
until n≥150 and sources≥3.

---

## 1. Vocabulary (what to sign)

The deployed model's live vocab — record exactly these 5 glosses:

```
bowling, computer, deaf, drink, yes
```

> If/when the vocab grows, update `models/landmark/checkpoints/labels.json`,
> retrain, and re-record. For now G1 is scoped to these 5.

## 2. How many clips — targets

We need ≥150 total across ≥3 sources. Two sources are datasets we already have or
can pull (WLASL + one more); **phone is the one you record.**

| Source | Who | Target / word | Total | Notes |
|---|---|---|---|---|
| **phone** | **you** | **8–10** | **40–50** | the real-world generalization test |
| wlasl | existing | ~10 | ~50 | `~/EPICS_Project/data/wlasl_distinct_8_full` |
| 3rd (MS-ASL / ASL Citizen) | to acquire | ~10 | ~50 | download step — will ask first |

**Phone, concretely: 8–10 clips per word minimum.** Why not fewer: at n=10 per
word, one wrong clip = 10% — the Wilson CI is already wide. At n=2–3 (G-B) a single
miss swings accuracy by 30–50% and the CI is useless. 8–10 keeps per-word CIs
interpretable; **more is better** if recording is cheap.

Spread those 8–10 across **2–3 signers** if you can (see §4). E.g. 10 clips/word =
~3–4 per signer × 3 signers, or 5 each × 2 signers.

## 3. Variants of the same sign

ASL has lexical/regional variants for many words. Handle them explicitly so they
inform rather than pollute the number:

- **Primary variant (`v1`)** = the one that matches the **training data** (WLASL).
  Most phone clips should be `v1`. This is the fair test of the deployed model.
- **Alternate variants (`v2`, `v3`, …)** = other accepted signs for the same gloss.
  Record a few, tagged separately. The runner buckets by `variant_id`, so we can
  report "accuracy on the variant we trained on" vs "accuracy on unseen variants"
  without one masking the other.
- If unsure which variant WLASL used, sign the most common ACDHH/standard form as
  `v1` and note alternates as `v2`. Don't silently mix variants under one tag.

**Rule of thumb:** keep the variant consistent *within* a `variant_id`; never relabel
a different sign as the same variant.

## 4. Signers

- Record **≥2 signers** if possible (tag `s1`, `s2`, …). Signer diversity is what
  makes the result generalize beyond one person's hands. The runner reports
  per-signer accuracy.
- One signer is acceptable for a first pass (tag `s1`), but flag it — single-signer
  numbers overstate real-world accuracy.
- Keep a private note of who each `sN` is; don't put names in filenames/manifest.

## 5. Recording conditions — vary them on purpose

The point of the phone source is to break the in-distribution illusion, so **vary
the nuisance factors** across your clips (don't shoot 10 identical takes):

- **Lighting:** bright, dim, side-lit, backlit (mix).
- **Background:** plain wall, cluttered room, patterned — a few of each.
- **Distance:** ~0.5 m, ~1 m, ~1.5 m.
- **Angle:** mostly frontal (the real use case), a couple slightly off-axis.
- **Clothing:** vary sleeve length / color vs. skin contrast at least once.
- **Speed:** natural pace; include one slower and one faster take per word.
- **Camera:** matches real use — **seated, face-to-face, camera roughly chest/face
  height** (this mirrors the advisor form-factor constraint: a front-worn/handheld
  camera only supports seated face-to-face exchange). If you'll deploy front-worn,
  record some clips from that vantage.

Each axis you vary widens the realism of the test. If every clip is identical, G1
just re-measures in-distribution accuracy with extra steps.

## 6. Technical specs (per clip)

- **One isolated sign per clip** (WLASL style). Start and end on a neutral rest
  pose; sign once in the middle.
- **Length ~1–3 s** (roughly 30–90 frames). The intake warns if <8 or >240 frames.
- Hands **fully in frame** for the whole sign; both hands visible for two-handed signs.
- **mp4**, 720p+ is plenty, 30 fps typical. No filters/beautify, no heavy compression.
- Hold the phone steady (or prop it). Avoid motion blur.

## 7. Naming + where to put files

Drop clips here (this dir is gitignored — clips never get committed, only the manifest):

```
eval/benchmark/g1/videos/phone/
```

Use either layout:

- **Flat (recommended for phone):**
  `phone_<word>_<signer>_<variant>_<take>.mp4`
  e.g. `phone_drink_s1_v1_01.mp4`, `phone_drink_s1_v1_02.mp4`, `phone_yes_s2_v2_01.mp4`
- **Subfolders:** `phone/<word>/<anything>.mp4` (signer/variant parsed from filename
  if present, else defaults `unknown`/`v1`).

`<signer>` = `s1`, `s2`, …  `<variant>` = `v1`, `v2`, …  `<take>` = `01`, `02`, …

## 8. Build the manifest (after recording)

```bash
. .venv-oracle/bin/activate && source env.sh
python eval/build_g1_manifest.py \
  --video-dir eval/benchmark/g1/videos \
  --source phone \
  --vocab bowling,computer,deaf,drink,yes \
  --out-manifest eval/benchmark/g1/g1_manifest.json \
  --merge --check-hands
```

`--check-hands` runs MediaPipe on a few frames and warns on clips with no detectable
hand (re-shoot those). `--merge` lets you add sources incrementally (run again with
`--source wlasl`, then `--source msasl`). It prints a word×source coverage table so
you can see where you're short.

## 9. Run the benchmark

```bash
python eval/benchmark_g1.py \
  --manifest eval/benchmark/g1/g1_manifest.json \
  --video-dir eval/benchmark/g1/videos \
  --vanilla-checkpoint oracle/checkpoints/best_model_original_backup.pth \
  --landmark-model models/landmark/checkpoints/model.ts \
  --landmark-labels models/landmark/checkpoints/labels.json \
  --output-json eval/benchmark/g1/g1_report.json
```

Reports overall + **per-source** + per-signer + per-word top-1 with **Wilson 95%
CIs**, for three views: `landmark`, `vanilla_closed` (I3D restricted to the 5
words — fair), `vanilla_open` (I3D over full WLASL — real difficulty).

## 10. How to read the result (what "win" means)

- The decision question is **cross-source generalization**, especially the **phone**
  row. In-distribution (wlasl) the I3D will likely lead (it did at G-B). The
  landmark model earns its place if it **holds up on phone** where appearance-based
  models degrade.
- Compare **CIs, not point estimates.** If two models' 95% intervals overlap, the
  difference is not real at this n — grow the set, don't decide.
- A model "wins" a source only when its CI clears the other's on that source.
- This feeds Gate G3's 2-axis scoreboard (accuracy × deployability).

## 11. What's NOT decided here

G1 measures; it doesn't retrain. Signer-disjoint splits and ≥50 clips/word are
**G2**. The Ollama re-rank / shortlist experiments are **G4**. Keep phone clips for
**measurement only** — never train on them (that would destroy them as an
unbiased test set).
