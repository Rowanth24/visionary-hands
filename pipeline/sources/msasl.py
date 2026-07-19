"""MS-ASL source adapter — annotations are local JSON, videos are YouTube links.

Annotations: https://www.microsoft.com/en-us/download/details.aspx?id=100121
(MS-ASL.zip, 1.9 MB, license C-UDA-0.1 — computational use permitted; the
archive ships the agreement PDF). Videos must be fetched per-clip from YouTube
(yt-dlp), and a meaningful fraction of links are dead — download_subset
tolerates that and reports dead/alive counts.

All 5 locked vocab words exist (train counts: deaf 46, yes 46, drink 43,
computer 36, bowling 9) and entries carry real signer ids -> signer-disjoint
splits work via the existing splitter.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SPLITS = ("train", "val", "test")


def load_index(msasl_dir, splits=SPLITS):
    """Normalized entries from MSASL_<split>.json files (see pipeline.sources docstring)."""
    entries = []
    for split in splits:
        path = Path(msasl_dir) / f"MSASL_{split}.json"
        for i, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            video_id = f"{split}_{i}_{row['url'].split('v=')[-1]}"
            entries.append({
                "label": row["clean_text"].lower(),
                "video_id": video_id,
                "filepath": f"{video_id}.mp4",
                "signer_id": f"msasl_{row['signer_id']}",
                "source": "msasl",
                "url": row["url"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "box": row.get("box"),
                "split": split,
            })
    return entries


def filter_to_vocab(entries, vocab):
    wanted = {w.lower() for w in vocab}
    return [e for e in entries if e["label"] in wanted]


def download_subset(entries, dest, max_bytes=2_000_000_000, ytdlp="yt-dlp", timeout=120):
    """Fetch each entry's clip section via yt-dlp; tolerate dead links.

    Returns {"downloaded": [...], "dead": [...], "skipped_budget": [...]}.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    summary = {"downloaded": [], "dead": [], "skipped_budget": []}
    spent = 0
    for e in entries:
        out = dest / e["filepath"]
        if out.exists():
            summary["downloaded"].append(e["video_id"])
            continue
        if spent >= max_bytes:
            summary["skipped_budget"].append(e["video_id"])
            continue
        section = f"*{e['start_time']}-{e['end_time']}"
        proc = subprocess.run(
            [ytdlp, "--quiet", "--no-warnings", "--download-sections", section,
             "-f", "mp4[height<=480]/best[height<=480]/best", "-o", str(out), e["url"]],
            capture_output=True, timeout=timeout,
        )
        if proc.returncode == 0 and out.exists():
            spent += out.stat().st_size
            summary["downloaded"].append(e["video_id"])
        else:
            summary["dead"].append(e["video_id"])
    return summary
