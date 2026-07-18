# F4 — Fingerspelling real-data trust gate (2026-07-17)

**Verdict: PASS** — status-quo abstention holds the precision bar on truly held-out data.

F2 calibrated the per-letter thresholds *on* set_c (small n, self-described
"directional"). F4 closes that loop honestly: score the current thresholds on
set_c, recalibrate on set_a+set_b only, and evaluate a confusable-cluster
alternative — all through `models/fingerspelling/f4_gate.py`, artifact at
`models/fingerspelling/checkpoints/f4_gate_report.json`.

## Results on held-out set_c (n = 684)

| Policy | Coverage | Selective precision | Confident-wrong rate |
|---|---|---|---|
| **status quo (kept)** | 0.632 | **0.984** | 0.010 |
| recalibrated on a+b | 1.000 | 0.842 | 0.158 |
| cluster policy | 0.959 | 0.939 | 0.059 |

Bar: precision ≥ 0.95 target / ≥ 0.90 floor. Only the status quo clears the
target → **`abstention.json` unchanged** (do-no-harm default in `choose_policy`).

- *Recalibrated* fails exactly as theory predicts: set_a+set_b are the MLP's
  training data, so thresholds fit there are overconfident (there is no clean
  third split — noted as a data gap for the next capture session).
- *Cluster policy* (pinned letters emit "confusable — please spell again
  slowly" at a shared threshold) buys +33 pts coverage at 0.939 precision —
  below target, so not adopted, but it is the best candidate once more
  calibration data exists. The report JSON carries its full numbers.

## The confusable cluster is bigger than expected

Pinned letters (threshold ≥ 0.99, can never fire): **c, m, n, o, p, r, u** —
not just the c/r/o trio flagged earlier. Held-out confusion confirms genuine
ASL handshape neighborhoods, not model noise:

- `n` → predicted `m` in 27/28 cases (closed-fist neighbors)
- `u` → predicted `r` in 9/28 cases (crossed vs. parallel fingers)
- `c` ↔ `o` (open-curve pair); within-cluster rates 0.68–1.0

**Decision:** keep per-letter suppression (status quo) for the shipping path;
revisit the cluster prompt when a dedicated calibration split exists (letters
above are recoverable in live use via the existing "spell again" abstain flow).

## Reproduce

```bash
source env.sh
.venv-oracle/bin/python models/fingerspelling/f4_gate.py --no-write
```

Guarded by `tests/test_f4_gate.py` (report exists, schema, precision floor,
non-trivial coverage).
