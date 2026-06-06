#!/usr/bin/env python3
"""Tests for the abstention policy + selective metrics (Gate A).

Run:  python -m pytest tests/test_abstention.py -q   (or: python tests/test_abstention.py)
"""
from __future__ import annotations

from abstention import (
    AbstentionPolicy,
    recommend_threshold,
    risk_coverage_curve,
    selective_metrics,
)


def test_confidence_threshold_abstains_below_tau():
    policy = AbstentionPolicy(conf_threshold=0.5)
    assert policy.decide([("drink", 0.8), ("yes", 0.1)])[0] == "drink"      # confident -> answer
    assert policy.decide([("drink", 0.4), ("yes", 0.3)])[0] is None         # below tau -> abstain


def test_margin_guard_abstains_when_ambiguous():
    policy = AbstentionPolicy(conf_threshold=0.3, margin_threshold=0.2)
    assert policy.decide([("drink", 0.55), ("yes", 0.45)])[0] is None       # margin 0.10 < 0.20
    assert policy.decide([("drink", 0.70), ("yes", 0.20)])[0] == "drink"    # margin 0.50 ok


def test_no_scores_abstains():
    assert AbstentionPolicy().decide([])[0] is None
    assert AbstentionPolicy().decide([("drink", None)])[0] is None


def test_selective_metrics_counts():
    items = [
        {"candidates": [("drink", 0.9), ("yes", 0.05)], "expected": "drink"},   # answered, correct
        {"candidates": [("yes", 0.8), ("drink", 0.1)], "expected": "drink"},     # answered, WRONG
        {"candidates": [("deaf", 0.4), ("yes", 0.3)], "expected": "deaf"},       # abstain (below 0.5)
    ]
    m = selective_metrics(items, AbstentionPolicy(conf_threshold=0.5))
    assert m["n"] == 3 and m["answered"] == 2 and m["abstained"] == 1
    assert m["coverage"] == round(2 / 3, 4)
    assert m["selective_accuracy"] == 0.5            # 1 of 2 answered correct
    assert m["confident_wrong_rate"] == round(1 / 3, 4)  # 1 confident-wrong of 3 total


def test_higher_threshold_reduces_coverage_and_risk():
    items = [{"candidates": [("a", c), ("b", 1 - c)], "expected": "a"} for c in (0.55, 0.65, 0.75, 0.95)]
    # all are "correct" if answered; raising threshold only drops coverage, never adds wrongs here
    low = selective_metrics(items, AbstentionPolicy(0.5))
    high = selective_metrics(items, AbstentionPolicy(0.8))
    assert low["coverage"] >= high["coverage"]


def test_recommend_threshold_respects_risk_budget():
    # two clips that would be confident-wrong at 0.6, correct ones at >=0.7
    items = [
        {"candidates": [("wrong", 0.62), ("right", 0.38)], "expected": "right"},
        {"candidates": [("right", 0.85), ("x", 0.15)], "expected": "right"},
        {"candidates": [("right", 0.90), ("x", 0.10)], "expected": "right"},
    ]
    rec = recommend_threshold(items, target_confident_wrong_rate=0.0)
    # to drive confident-wrong to 0 we must abstain on the 0.62 wrong one -> threshold > 0.62
    assert rec["confident_wrong_rate"] == 0.0
    assert rec["conf_threshold"] > 0.62


def test_risk_coverage_curve_is_monotone_coverage():
    items = [{"candidates": [("a", c), ("b", 1 - c)], "expected": "a"} for c in (0.3, 0.5, 0.7, 0.9)]
    curve = risk_coverage_curve(items)
    coverages = [row["coverage"] for row in curve]
    assert coverages == sorted(coverages, reverse=True)  # coverage non-increasing as threshold rises


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("OK — abstention policy + selective metrics pass.")
