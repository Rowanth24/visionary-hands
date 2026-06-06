# Abstention — "not confident" instead of confident-wrong

A shipping precondition (locked objective): in a medical exchange, emitting a
confident-wrong gloss is worse than emitting nothing. The system must be able to
decline. This is the abstention layer.

## Mechanism (`eval/abstention.py`)

`AbstentionPolicy(conf_threshold, margin_threshold)` decides ANSWER vs ABSTAIN on a
ranked candidate list (works for any backend view — landmark, vanilla_closed,
vanilla_open):

- **confidence:** abstain if top-1 probability < `conf_threshold`
- **margin:** abstain if `top1 - top2 < margin_threshold` (ambiguity guard; `0` disables)

It's an analysis/decision layer over backend outputs — not yet wired into live
inference (see "Status" below).

## How to score it (precision/coverage, not accuracy)

Plain accuracy hides abstention. The three numbers that matter:

| metric | meaning |
|---|---|
| **coverage** | fraction of clips answered (not abstained) |
| **selective_accuracy** | accuracy **among answered** clips |
| **confident_wrong_rate** | fraction of **all** clips that were answered **and wrong** — the "worse than nothing" number we bound |

Goal: **maximize coverage subject to confident_wrong_rate ≤ budget** (default 5%).

## Calibrating the threshold (`eval/calibrate_abstention.py`)

Thresholds must be calibrated on **decision-grade** data (G1, n≥150, real sources
incl. phone). On a small/dry report the tool runs but labels itself illustrative.

```bash
python eval/calibrate_abstention.py \
  --report eval/benchmark/g1/g1_report.json \
  --model landmark \
  --target-risk 0.05 --by-source
```

It prints the risk/coverage curve over the threshold sweep and recommends the
lowest threshold (max coverage) whose confident-wrong rate ≤ target. `benchmark_g1.py`
records per-clip top-k scores so its reports are calibration-ready.

Example shape (dry run, n=12 WLASL — illustrative only):

```
conf_thr  coverage  selective_acc  confident_wrong
  0.30     1.000     0.583          0.417
  0.50     0.417     0.800          0.083
  0.90     0.167     1.000          0.000
```

Read it as: stricter threshold → fewer answers, but safer. Pick the point where the
confident-wrong rate is acceptable for a clinical setting and coverage is still useful.

## Status

- ✅ policy + selective metrics + calibration tool + tests (`tests/test_abstention.py`)
- ✅ `benchmark_g1.py` emits per-clip top-k (calibration-ready)
- ⏳ **calibrate on real G1 data** once phone clips land (n=12 is not decision-grade)
- ⏳ **wire into live inference** (`deploy/run_live.py`, Phase 5) using the calibrated
  threshold — capture → model → abstention → TTS only when confident

## Design notes

- Per-source thresholds may differ (phone vs in-distribution); `--by-source` reports
  per-source coverage/risk at the chosen threshold so this is visible.
- Calibrate the **deployed** view (landmark, or vanilla_closed if that ships) — not
  vanilla_open, which exists mainly to show real-world difficulty.
- A clinical setting likely wants a **low** confident-wrong budget (≤5%, maybe lower)
  even at the cost of coverage; that's a product/clinical call, made on G1 numbers.
