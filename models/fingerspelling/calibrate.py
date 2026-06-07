#!/usr/bin/env python3
"""Calibrate fingerspelling abstention from a cross-dataset eval report (Gate F2).

Two views:
  - GLOBAL: reuse eval/abstention.py to sweep one confidence threshold (risk/coverage curve)
    and recommend the lowest threshold whose confident-wrong rate <= budget.
  - PER-LETTER: for each predicted letter, find the smallest top-1 score threshold that makes
    accepted predictions-of-that-letter reach a precision target. The confusable cluster
    (n/m/s/p/q/v...) ends up with high thresholds (abstain more), which is the point.

NOTE: the held-out set is small (~20-30/letter), so these thresholds are DIRECTIONAL. The
binding confident-wrong budget is locked at the F4 real-data trust gate.

Usage:
  python models/fingerspelling/calibrate.py \
    --report data/fingerspelling/eval_on_C.json \
    --target-risk 0.02 --target-precision 0.95 \
    --out models/fingerspelling/checkpoints/abstention.json
"""
import argparse
import json
from pathlib import Path

# eval/ is on PYTHONPATH via env.sh
from abstention import recommend_threshold, risk_coverage_curve

THRESHOLDS = [round(0.02 * i, 2) for i in range(0, 51)]  # 0.00..1.00 step 0.02


def items_from_report(results):
    return [{"candidates": [(c["label"], c["score"]) for c in r["topk"]], "expected": r["expected"]}
            for r in results if r.get("topk")]


def per_letter_thresholds(results, target_precision):
    """For each predicted letter, smallest top-1 threshold giving accept-precision >= target."""
    by_pred = {}
    for r in results:
        if not r.get("topk"):
            continue
        top = r["topk"][0]
        by_pred.setdefault(top["label"], []).append((top["score"], top["label"] == r["expected"]))

    thresholds, diagnostics = {}, {}
    for letter, rows in by_pred.items():
        chosen = 1.0  # default: always abstain unless a threshold reaches target precision
        best = None
        for tau in THRESHOLDS:
            accepted = [correct for score, correct in rows if score >= tau]
            if not accepted:
                continue
            precision = sum(accepted) / len(accepted)
            coverage = len(accepted) / len(rows)
            if precision >= target_precision:
                chosen = tau
                best = {"threshold": tau, "precision": round(precision, 3),
                        "coverage_of_predictions": round(coverage, 3), "n_pred": len(rows)}
                break
        thresholds[letter] = chosen
        diagnostics[letter] = best or {"threshold": 1.0, "note": "no threshold met precision target",
                                       "n_pred": len(rows)}
    return thresholds, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--target-risk", type=float, default=0.02, help="Max global confident-wrong rate.")
    ap.add_argument("--target-precision", type=float, default=0.95, help="Per-letter accept precision.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = report["results"]
    items = items_from_report(results)

    # Global view (reuse abstention.py)
    print("GLOBAL risk/coverage (conf_thr  coverage  selective_acc  confident_wrong):")
    for row in risk_coverage_curve(items, thresholds=THRESHOLDS):
        if abs((row["conf_threshold"] * 100) % 10) < 1e-6:  # print every 0.10 for brevity
            sa = "n/a" if row["selective_accuracy"] is None else f"{row['selective_accuracy']:.3f}"
            print(f"  {row['conf_threshold']:.2f}   cov={row['coverage']:.3f}  sel_acc={sa}  cw={row['confident_wrong_rate']:.3f}")
    rec = recommend_threshold(items, target_confident_wrong_rate=args.target_risk, thresholds=THRESHOLDS)
    print(f"\nGLOBAL recommended (confident-wrong <= {args.target_risk}): conf_threshold={rec['conf_threshold']} "
          f"coverage={rec['coverage']} selective_acc={rec['selective_accuracy']} cw={rec['confident_wrong_rate']}")

    # Per-letter view
    pl, diag = per_letter_thresholds(results, args.target_precision)
    abstain_always = sorted(k for k, v in pl.items() if v >= 1.0)
    print(f"\nPER-LETTER thresholds (precision target {args.target_precision}):")
    for letter in sorted(pl):
        print(f"  {letter}: tau={pl[letter]:.2f}  {diag[letter]}")
    if abstain_always:
        print(f"\n⚠️  letters with NO threshold meeting precision (always-abstain on this data): {abstain_always}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "global_threshold": rec["conf_threshold"], "margin_threshold": 0.0,
        "target_risk": args.target_risk, "target_precision": args.target_precision,
        "per_letter_threshold": pl, "diagnostics": diag,
        "note": "directional — small held-out set; finalize at F4 real-data gate",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
