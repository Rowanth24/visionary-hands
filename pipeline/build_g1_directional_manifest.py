#!/usr/bin/env python3
"""Build the G1-*directional* manifest from clips already on disk.

NOT decision-grade G1 (no phone source, signer_ids are video-id proxies, n=32):
this exists so the G1 report/calibration machinery runs end-to-end today and
word-model abstention gets *directional* thresholds (plan Task E). When real
sources land (asl_citizen, phone), rebuild with the true G1 builder instead.

Combines:
  eval/benchmark/g1/videos/wlasl/<label>/<id>.mp4      (in-vocab, source=wlasl)
  eval/benchmark/multisource_videos/ via multisource20_manifest (incl. OOV labels
    — closed-vocab models can never be right on them, exactly the confident-wrong
    events abstention must suppress)

Usage (from repo root):
  python pipeline/build_g1_directional_manifest.py \
    --out eval/benchmark/g1/g1_directional_manifest.json
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_VIDEOS = REPO_ROOT / "eval" / "benchmark" / "g1" / "videos"
MULTISOURCE_MANIFEST = REPO_ROOT / "eval" / "benchmark" / "multisource20_manifest.json"
VOCAB = ["bowling", "computer", "deaf", "drink", "yes"]


def wlasl_samples(start_index):
    samples = []
    for i, clip in enumerate(sorted(G1_VIDEOS.glob("wlasl/*/*.mp4")), start=start_index):
        samples.append({
            "sample_index": i,
            "label": clip.parent.name,
            "source": "wlasl",
            "signer_id": f"wlasl_{clip.stem}",   # proxy: WLASL clip ids ~ distinct signers
            "variant_id": "v1",
            "local_video": str(clip.relative_to(G1_VIDEOS)),
        })
    return samples


def multisource_samples(start_index):
    manifest = json.loads(MULTISOURCE_MANIFEST.read_text(encoding="utf-8"))
    samples = []
    for i, row in enumerate(manifest["samples"], start=start_index):
        samples.append({
            "sample_index": i,
            "label": row["pair_label"],
            "source": row["source"],
            "signer_id": f"{row['source']}_{row['video_id']}",
            "variant_id": "v1",
            # relative to the g1 videos dir the runner is pointed at
            "local_video": f"../../multisource_videos/{row['local_video']}",
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/benchmark/g1/g1_directional_manifest.json")
    args = ap.parse_args()

    wlasl = wlasl_samples(0)
    multi = multisource_samples(len(wlasl))
    manifest = {
        "name": "g1_directional",
        "decision_grade": False,
        "note": "directional only — no phone source, signer_ids are proxies; see module docstring",
        "vocab": VOCAB,
        "samples": wlasl + multi,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    in_vocab = sum(1 for s in manifest["samples"] if s["label"] in VOCAB)
    print(f"wrote {out}: {len(manifest['samples'])} clips "
          f"({in_vocab} in-vocab, {len(manifest['samples']) - in_vocab} OOV), "
          f"sources={sorted({s['source'] for s in manifest['samples']})}")


if __name__ == "__main__":
    main()
