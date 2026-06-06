#!/usr/bin/env python3
"""Gate G0 — thin oracle entry point.

Runs the vendored vanilla I3D (InceptionI3d) on one or more video clips by
reusing the unified eval harness (../eval/infer_videos.py). This file is
deliberately thin: it owns no model code. The actual model lives in
eval/pytorch_i3d.py (vendored from github.com/sumedhsp/Sign-Language-Recognition
— see ../ATTRIBUTION.md); the trained weights live in oracle/checkpoints/.

Usage (from repo root, after `source env.sh`):
    python oracle/run_i3d.py eval/benchmark/multisource_videos/05_wlasl_secretary_50041.mp4

Any infer_videos.py flag passes straight through, e.g. --top-k, --clip-length,
--stride, --labels-file. --backend defaults to "vanilla" and --checkpoint
defaults to the vendored I3D weights if you don't override them.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DEFAULT_CKPT = ROOT / "oracle" / "checkpoints" / "best_model_original_backup.pth"

# Make the harness + vendored upstream modules importable.
sys.path.insert(0, str(EVAL_DIR))


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        sys.exit(
            "usage: python oracle/run_i3d.py <video> [<video> ...] [infer_videos flags]\n"
            f"default checkpoint: {DEFAULT_CKPT}"
        )
    if "--backend" not in argv:
        argv = ["--backend", "vanilla", *argv]
    if "--checkpoint" not in argv:
        if not DEFAULT_CKPT.exists():
            sys.exit(f"[run_i3d] default checkpoint missing: {DEFAULT_CKPT}\n"
                     "Pass --checkpoint <path.pth> explicitly.")
        argv = ["--checkpoint", str(DEFAULT_CKPT), *argv]

    sys.argv = ["infer_videos.py", *argv]
    import infer_videos  # noqa: E402  (path set above)
    infer_videos.main()


if __name__ == "__main__":
    main()
