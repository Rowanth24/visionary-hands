"""Abstention — say 'not confident' instead of risking a confident-wrong answer.

In a medical exchange a confident-wrong gloss is worse than no answer, so the
system must be allowed to decline. This module is backend-agnostic: it scores any
ranked candidate list (landmark, vanilla_closed, vanilla_open) and decides whether
to ANSWER or ABSTAIN, using two interpretable signals:

  - confidence: abstain if top-1 probability < conf_threshold
  - margin:     abstain if (top1 - top2) < margin_threshold  (ambiguity guard)

Thresholds are NOT hardcoded wisdom — calibrate them on a decision-grade G1 report
(see calibrate_abstention.py). The defaults here are placeholders until then.

Scoring an abstaining system is a precision/coverage tradeoff, not plain accuracy:
  - coverage              = fraction of clips answered (not abstained)
  - selective_accuracy    = accuracy among ANSWERED clips
  - confident_wrong_rate  = fraction of ALL clips that were answered AND wrong
                            (the 'worse than nothing' number we bound)
"""
from __future__ import annotations

from dataclasses import dataclass


def _normalize(text) -> str:
    return str(text).strip().lower()


def _label_score_pairs(candidates):
    """Accept BackendResult-style candidates, dicts, or (label, score) tuples -> [(label, score)]."""
    pairs = []
    for c in candidates:
        if hasattr(c, "text"):
            pairs.append((c.text, c.score))
        elif isinstance(c, dict):
            pairs.append((c.get("label"), c.get("score")))
        else:
            pairs.append((c[0], c[1]))
    return pairs


@dataclass(frozen=True)
class AbstentionPolicy:
    conf_threshold: float = 0.5   # placeholder — calibrate on G1
    margin_threshold: float = 0.0  # 0.0 disables the margin guard

    def decide(self, candidates):
        """Return (predicted_label_or_None, reason). None => ABSTAIN."""
        pairs = _label_score_pairs(candidates)
        if not pairs or pairs[0][1] is None:
            return None, "no_scores"
        top_label, top1 = pairs[0]
        top2 = pairs[1][1] if len(pairs) > 1 and pairs[1][1] is not None else 0.0
        if top1 < self.conf_threshold:
            return None, f"low_confidence({top1:.3f}<{self.conf_threshold})"
        if (top1 - top2) < self.margin_threshold:
            return None, f"low_margin({top1 - top2:.3f}<{self.margin_threshold})"
        return top_label, "accept"


def selective_metrics(items, policy):
    """items: iterable of {"candidates": [...], "expected": label}. Returns selective metrics."""
    items = list(items)
    n = len(items)
    if n == 0:
        return {"n": 0, "coverage": None, "selective_accuracy": None, "confident_wrong_rate": None, "abstained": 0}
    answered = correct = confident_wrong = 0
    for it in items:
        label, _reason = policy.decide(it["candidates"])
        if label is None:
            continue
        answered += 1
        if _normalize(label) == _normalize(it["expected"]):
            correct += 1
        else:
            confident_wrong += 1
    return {
        "n": n,
        "answered": answered,
        "abstained": n - answered,
        "coverage": round(answered / n, 4),
        "selective_accuracy": round(correct / answered, 4) if answered else None,
        "confident_wrong_rate": round(confident_wrong / n, 4),
        "conf_threshold": policy.conf_threshold,
        "margin_threshold": policy.margin_threshold,
    }


def risk_coverage_curve(items, thresholds=None, margin_threshold=0.0):
    """Sweep conf_threshold; return one selective_metrics row per threshold (ascending)."""
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(0, 21)]  # 0.00 .. 1.00
    return [selective_metrics(items, AbstentionPolicy(t, margin_threshold)) for t in sorted(thresholds)]


def recommend_threshold(items, target_confident_wrong_rate=0.05, margin_threshold=0.0, thresholds=None):
    """Pick the lowest conf_threshold (=> max coverage) whose confident_wrong_rate <= target.

    Returns the chosen metrics row, or the strictest row if none meet the target.
    """
    curve = risk_coverage_curve(items, thresholds=thresholds, margin_threshold=margin_threshold)
    feasible = [row for row in curve if row["confident_wrong_rate"] is not None
                and row["confident_wrong_rate"] <= target_confident_wrong_rate]
    if feasible:
        # lowest threshold among feasible = highest coverage
        return min(feasible, key=lambda r: r["conf_threshold"])
    # none feasible: return the row with the lowest risk (strictest useful threshold)
    return min(curve, key=lambda r: (r["confident_wrong_rate"] if r["confident_wrong_rate"] is not None else 1.0))
