#!/usr/bin/env python3
"""Clip-mode smoke test for deploy/run_live.py (Gate W7, headless).

run_live.py needs onnxruntime + mediapipe + cv2 — normally only in .venv-onnx.
The test runs run_live.py as a subprocess with the first interpreter that has
those deps (this venv, else the repo's .venv-onnx); skips if neither does, so
pytest collection stays green in .venv-oracle.

Run:  .venv-oracle/bin/pytest tests/test_run_live_clip.py -q   (after `source env.sh`)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIRS = (
    REPO_ROOT / "eval" / "benchmark" / "g1" / "videos",
    REPO_ROOT / "eval" / "benchmark" / "multisource_videos",
)
DECISION_RE = re.compile(r"^\[t=\s*[\d.]+s\]\s+(WORD|LETTER|ABSTAIN)", re.MULTILINE)


def _runtime_interpreter():
    """First python that can import the live-runtime deps (current venv, else .venv-onnx)."""
    for python in (Path(sys.executable), REPO_ROOT / ".venv-onnx" / "bin" / "python"):
        if not python.exists():
            continue
        probe = subprocess.run(
            [str(python), "-c", "import onnxruntime, mediapipe, cv2"],
            capture_output=True, timeout=60,
        )
        if probe.returncode == 0:
            return python
    return None


def _smallest_clip():
    clips = [p for d in VIDEO_DIRS if d.exists() for p in d.rglob("*.mp4")]
    return min(clips, key=lambda p: p.stat().st_size) if clips else None


def test_run_live_clip_mode_exits_zero_and_emits_a_decision():
    python = _runtime_interpreter()
    if python is None:
        pytest.skip("no interpreter with onnxruntime+mediapipe+cv2 (set up .venv-onnx)")
    clip = _smallest_clip()
    if clip is None:
        pytest.skip("no benchmark .mp4 found under eval/benchmark/")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'eval'}"
    proc = subprocess.run(
        [str(python), str(REPO_ROOT / "deploy" / "run_live.py"),
         "--source", str(clip), "--no-tts"],
        capture_output=True, text=True, timeout=55, env=env, cwd=str(REPO_ROOT),
    )

    assert proc.returncode == 0, f"run_live.py failed.\nstderr:\n{proc.stderr[-2000:]}"
    assert DECISION_RE.search(proc.stdout), (
        f"no decision line (WORD/LETTER/ABSTAIN) in stdout:\n{proc.stdout}"
    )
