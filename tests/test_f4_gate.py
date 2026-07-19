#!/usr/bin/env python3
"""F4 gate artifact tests — the committed report must back the doc's claims.

Validates models/fingerspelling/checkpoints/f4_gate_report.json (no model run:
regenerating is `python models/fingerspelling/f4_gate.py --no-write`).

Run:  .venv-oracle/bin/pytest tests/test_f4_gate.py -q   (after `source env.sh`)
"""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "models" / "fingerspelling" / "checkpoints" / "f4_gate_report.json"

REQUIRED_KEYS = {"gate", "n_heldout", "policies", "chosen_policy", "verdict", "cluster",
                 "target_precision", "floor_precision", "confusion_analysis"}
POLICIES = {"status_quo", "recalibrated", "cluster_policy"}


def _report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_report_exists_and_has_schema():
    # Arrange / Act
    report = _report()

    # Assert
    assert REQUIRED_KEYS <= set(report)
    assert report["gate"] == "F4"
    assert POLICIES <= set(report["policies"])
    for name in POLICIES:
        overall = report["policies"][name]["overall"]
        assert {"coverage", "selective_accuracy", "n"} <= set(overall)


def test_chosen_policy_meets_precision_floor_with_real_coverage():
    # Arrange
    report = _report()
    chosen = report["policies"][report["chosen_policy"]]["overall"]

    # Assert: precision floor met, and the system is not trivially abstaining on everything
    assert chosen["selective_accuracy"] is not None
    assert chosen["selective_accuracy"] >= report["floor_precision"]
    assert chosen["coverage"] > 0.25
    assert report["verdict"] in {"pass", "conditional"}


def test_heldout_set_is_nontrivial():
    # Arrange
    report = _report()

    # Assert: gate was judged on a real held-out set, not a handful of frames
    assert report["n_heldout"] >= 300
    assert report["heldout_data"].endswith("set_c.npz")
