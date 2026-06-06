# oracle/ — the vanilla I3D, as a thin external dependency

This directory is intentionally small. It does **not** contain model code.

- `run_i3d.py` — thin entry point. Reuses `../eval/infer_videos.py` to run the
  vanilla I3D on one or more clips. One command, from the repo root.
- `checkpoints/best_model_original_backup.pth` — the trained I3D weights
  (gitignored; lives on disk, not in version control).

The model definition (`InceptionI3d`) is **vendored** into `../eval/pytorch_i3d.py`
from the upstream WLASL starter `github.com/sumedhsp/Sign-Language-Recognition`.
See `../ATTRIBUTION.md` for provenance and the licensing caveat (no upstream
LICENSE was published — do not relicense or redistribute as original work).

## Run (from repo root)

```bash
source env.sh
python oracle/run_i3d.py eval/benchmark/multisource_videos/05_wlasl_secretary_50041.mp4
```

Requires the oracle/eval env: `pip install -r ../requirements-oracle.txt`.
