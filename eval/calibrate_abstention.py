#!/usr/bin/env python3
"""Calibrate the abstention threshold on a benchmark report (Gate G1 deliverable).

Reads a report written by benchmark_g1.py (per-clip top-k scores per model view),
sweeps the confidence threshold, prints the risk/coverage curve, and recommends a
threshold that keeps the confident-wrong rate within a budget while maximizing
coverage. Run per model view (the shippable ones are landmark and vanilla_closed).

CALIBRATE ON DECISION-GRADE DATA. On a tiny dry-run report the recommendation is
not meaningful — it just exercises the tool.

Usage:
  python eval/calibrate_abstention.py \
    --report eval/benchmark/g1/g1_report.json \
    --model landmark \
    --target-risk 0.05
"""
import argparse
import json
from pathlib import Path

from abstention import recommend_threshold, risk_coverage_curve


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate abstention threshold from a G1 report.")
    p.add_argument("--report", required=True, help="g1_report.json from benchmark_g1.py.")
    p.add_argument("--model", default="landmark",
                   choices=("landmark", "vanilla_closed", "vanilla_open"),
                   help="Which model view to calibrate.")
    p.add_argument("--target-risk", type=float, default=0.05,
                   help="Max acceptable confident-wrong rate (fraction of all clips).")
    p.add_argument("--margin-threshold", type=float, default=0.0,
                   help="Optional ambiguity-margin guard (top1 - top2).")
    p.add_argument("--by-source", action="store_true",
                   help="Also print per-source risk/coverage at the recommended threshold.")
    p.add_argument("--write-json", default=None,
                   help="Write the recommended thresholds to this JSON path "
                        "(schema: conf_threshold/margin_threshold, read by deploy/run_live.py).")
    return p.parse_args()


def items_for_view(records, view):
    """Build abstention items [{candidates, expected}] from a report's per-clip top-k."""
    items = []
    for r in records:
        topk = r.get(view, {}).get("topk")
        if not topk:
            continue
        items.append({"candidates": [(c["label"], c["score"]) for c in topk], "expected": r["label"]})
    return items


def main():
    args = parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    records = report.get("results", [])
    items = items_for_view(records, args.model)
    if not items:
        raise SystemExit(f"No per-clip top-k for view '{args.model}' in {args.report}. "
                         "Re-run benchmark_g1.py (it records top-k).")

    n = report.get("n_clips", len(records))
    decision_grade = report.get("decision_grade", False)
    print(f"Report: {args.report}  view={args.model}  n={len(items)}  "
          f"({'DECISION-GRADE' if decision_grade else 'NOT decision-grade — calibration is illustrative only'})\n")

    print("conf_thr  coverage  selective_acc  confident_wrong")
    for row in risk_coverage_curve(items, margin_threshold=args.margin_threshold):
        sa = "n/a  " if row["selective_accuracy"] is None else f"{row['selective_accuracy']:.3f}"
        print(f"  {row['conf_threshold']:.2f}     {row['coverage']:.3f}     {sa}          {row['confident_wrong_rate']:.3f}")

    rec = recommend_threshold(items, target_confident_wrong_rate=args.target_risk,
                              margin_threshold=args.margin_threshold)
    print(f"\nRecommended for confident-wrong ≤ {args.target_risk}:")
    print(f"  conf_threshold={rec['conf_threshold']}  margin={rec['margin_threshold']}  "
          f"coverage={rec['coverage']}  selective_accuracy={rec['selective_accuracy']}  "
          f"confident_wrong_rate={rec['confident_wrong_rate']}")
    if rec["confident_wrong_rate"] is not None and rec["confident_wrong_rate"] > args.target_risk:
        print(f"  ⚠️  no threshold meets the target on this data; this is the lowest-risk option.")

    if args.write_json:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "conf_threshold": rec["conf_threshold"],
            "margin_threshold": rec["margin_threshold"],
            "target_risk": args.target_risk,
            "coverage": rec["coverage"],
            "selective_accuracy": rec["selective_accuracy"],
            "confident_wrong_rate": rec["confident_wrong_rate"],
            "calibrated_on": {"report": args.report, "view": args.model,
                              "decision_grade": decision_grade},
            "note": "directional until decision-grade G1 data exists" if not decision_grade else "",
        }, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if args.by_source:
        from abstention import AbstentionPolicy, selective_metrics
        policy = AbstentionPolicy(rec["conf_threshold"], args.margin_threshold)
        sources = sorted({r["source"] for r in records})
        print(f"\nPer-source at conf_threshold={rec['conf_threshold']}:")
        for src in sources:
            src_items = items_for_view([r for r in records if r["source"] == src], args.model)
            m = selective_metrics(src_items, policy)
            print(f"  {src:<8} coverage={m['coverage']} selective_acc={m['selective_accuracy']} "
                  f"confident_wrong={m['confident_wrong_rate']} (n={m['n']})")


if __name__ == "__main__":
    main()
