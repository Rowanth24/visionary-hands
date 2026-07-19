"""ASL Citizen source adapter — the phone/webcam-domain proxy for held-out eval.

83,399 webcam clips, 2,731 glosses, 52 consenting signers (Microsoft Research).
Distribution is a single bulk archive — there is NO per-gloss remote subsetting,
so acquisition is a one-time human step (see NeedsManualStep message below /
docs/g2_data_status.md). Once ASL_Citizen.zip is extracted locally (or attached
as a Kaggle dataset in the team's notebooks), load_index() normalizes the split
CSVs to the shared entry shape and everything downstream just works.

Expected extracted layout (v1.0):
    <root>/splits/{train,val,test}.csv   columns include Participant ID,
                                         Video file, Gloss
    <root>/videos/<Video file>.mp4
"""
from __future__ import annotations

import csv
from pathlib import Path

from pipeline.sources import NeedsManualStep

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {"Participant ID", "Video file", "Gloss"}

MANUAL_STEP = """\
ASL Citizen cannot be subset-downloaded by script — one person does this once:
  1. Official (preferred, license-clean): download ASL_Citizen.zip (42.8 GB) from
     https://www.microsoft.com/en-us/download/details.aspx?id=105253
     (license details: https://www.microsoft.com/en-us/research/project/asl-citizen/)
     and extract to data/multisource/asl_citizen/ on a machine with ~90 GB free.
  2. Kaggle alternative (for the team's existing Kaggle training notebooks):
     attach the community mirror kaggle.com/datasets/abd0kamel/asl-citizen —
     VERIFY its license field matches the official terms before relying on it;
     it is a third-party re-upload.
Then re-run this pipeline pointing --asl-citizen-root at the extracted folder."""


def load_index(root, splits=SPLITS):
    """Normalized entries from the extracted archive; raises NeedsManualStep if absent."""
    root = Path(root)
    split_dir = root / "splits"
    if not split_dir.is_dir():
        raise NeedsManualStep(f"ASL Citizen not found at {root}.\n{MANUAL_STEP}")
    entries = []
    for split in splits:
        path = split_dir / f"{split}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing expected columns {sorted(missing)} "
                                 f"(got {reader.fieldnames})")
            for row in reader:
                video_file = row["Video file"]
                entries.append({
                    "label": row["Gloss"].strip().lower(),
                    "video_id": Path(video_file).stem,
                    "filepath": str(Path("videos") / video_file),
                    "signer_id": f"aslcitizen_{row['Participant ID']}",
                    "source": "asl_citizen",
                    "split": split,
                })
    return entries


def filter_to_vocab(entries, vocab):
    wanted = {w.lower() for w in vocab}
    return [e for e in entries if e["label"] in wanted]


def download_subset(entries, dest, max_bytes=None):
    """Bulk-only distribution: always a human step."""
    raise NeedsManualStep(MANUAL_STEP)
