#!/usr/bin/env python3
"""Gate G1 — decision-grade benchmark: landmark vs vanilla I3D, per source, with CIs.

Unlike Gate G-B (n=12, in-distribution WLASL only, plumbing check), G1 is the first
benchmark allowed to inform a model decision. It requires:
  - >= 150-200 clips,
  - >= 3 sources INCLUDING phone-camera,
  - per-source top-1 accuracy with 95% confidence intervals.

This runner reads a manifest (see docs/g1_benchmark_and_recording.md), scores both
models through the shared harness backends, and reports overall + per-source +
per-signer accuracy with Wilson 95% intervals. It reuses the Gate G-B scoring
helpers so both gates judge models identically.

Manifest shape (eval/benchmark/g1/g1_manifest.example.json):
  {
    "name": "g1_benchmark",
    "vocab": ["bowling","computer","deaf","drink","yes"],
    "samples": [
      {"sample_index":0,"label":"drink","source":"phone","signer_id":"s1",
       "variant_id":"v1","local_video":"phone/phone_drink_s1_v1_01.mp4"}
    ]
  }

Usage (from repo root, .venv-oracle, after `source env.sh`):
  python eval/benchmark_g1.py \
    --manifest eval/benchmark/g1/g1_manifest.json \
    --video-dir eval/benchmark/g1/videos \
    --vanilla-checkpoint oracle/checkpoints/best_model_original_backup.pth \
    --landmark-model models/landmark/checkpoints/model.ts \
    --landmark-labels models/landmark/checkpoints/labels.json \
    --output-json eval/benchmark/g1/g1_report.json
"""
import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from asl_backends import LandmarkBackend, VanillaASLBackend, normalize_prediction_text
from benchmark_landmark_vs_vanilla import closed_set_index_map, vanilla_mean_probs
from infer_videos import read_video_frames

MIN_DECISION_GRADE_CLIPS = 150


def build_parser():
    code_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Gate G1: per-source landmark vs vanilla benchmark with CIs.")
    p.add_argument("--manifest", required=True, help="G1 manifest JSON.")
    p.add_argument("--video-dir", required=True, help="Base dir for manifest local_video paths.")
    p.add_argument("--vanilla-checkpoint", required=True)
    p.add_argument("--vanilla-labels-file", default=str(code_dir / "preprocess" / "wlasl_class_list.txt"))
    p.add_argument("--landmark-model", required=True)
    p.add_argument("--landmark-labels", required=True)
    p.add_argument("--landmark-config", default=None)
    p.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    p.add_argument("--clip-length", type=int, default=64)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-json", required=True)
    return p


def wilson_ci(correct, total, z=1.96):
    """Wilson score 95% interval for a binomial proportion (robust at small n)."""
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def summarize(records, key, group=None):
    """Top-1 accuracy + Wilson CI over records, optionally one group bucket."""
    rows = records if group is None else [r for r in records if r.get(group[0]) == group[1]]
    n = len(rows)
    correct = sum(1 for r in rows if r[key]["correct"])
    return {
        "n": n,
        "correct": correct,
        "top1_accuracy": round(correct / n, 4) if n else None,
        "ci95_wilson": wilson_ci(correct, n),
    }


def main():
    args = build_parser().parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = manifest["samples"]
    vocab = manifest.get("vocab")
    video_dir = Path(args.video_dir)

    if not samples:
        raise SystemExit("Manifest has no samples.")
    if vocab is None:
        vocab = sorted({s["label"] for s in samples})

    vanilla = VanillaASLBackend(
        checkpoint_path=args.vanilla_checkpoint,
        labels_path=args.vanilla_labels_file,
        device_name=args.device,
    )
    landmark = LandmarkBackend(
        model_path=args.landmark_model,
        labels_path=args.landmark_labels,
        device_name=args.device,
        config_path=args.landmark_config,
    )

    closed_map, missing = closed_set_index_map(vanilla.labels, vocab)
    if missing:
        print(f"WARNING: vocab labels not in vanilla vocabulary (closed-set skips them): {missing}")
    closed_labels = [lbl for lbl in vocab if lbl in closed_map]
    closed_index_tensor = torch.tensor([closed_map[lbl] for lbl in closed_labels], dtype=torch.long)

    n = len(samples)
    print(f"Vanilla: {vanilla.describe()}  |  Landmark: {landmark.describe()}")
    print(f"Manifest: {n} clips, vocab={vocab}")
    if n < MIN_DECISION_GRADE_CLIPS:
        print(f"⚠️  n={n} < {MIN_DECISION_GRADE_CLIPS}: below decision-grade. Treat as a dry run until the set grows.")
    sources = sorted({s.get("source", "unknown") for s in samples})
    print(f"Sources: {sources}\n")

    records = []
    for s in samples:
        expected = s["label"]
        expected_norm = normalize_prediction_text(expected)
        clip = video_dir / s["local_video"]
        if not clip.exists():
            print(f"SKIP missing: {clip}")
            continue
        frames = read_video_frames(clip)

        t0 = time.perf_counter()
        probs_mean, windows = vanilla_mean_probs(vanilla, frames, args.clip_length, args.stride, args.top_k)
        v_ms = (time.perf_counter() - t0) * 1000.0
        open_top1 = vanilla.labels[int(torch.argmax(probs_mean))]
        closed_top1 = closed_labels[int(torch.argmax(probs_mean[closed_index_tensor]))] if closed_labels else None

        t0 = time.perf_counter()
        lm = landmark.infer_clip(frames, args.top_k)
        l_ms = (time.perf_counter() - t0) * 1000.0

        records.append({
            "sample_index": s.get("sample_index"),
            "label": expected,
            "source": s.get("source", "unknown"),
            "signer_id": s.get("signer_id"),
            "variant_id": s.get("variant_id"),
            "clip": s["local_video"],
            "vanilla_open": {"top1": open_top1, "correct": normalize_prediction_text(open_top1) == expected_norm,
                             "latency_ms": v_ms, "windows": windows},
            "vanilla_closed": {"top1": closed_top1,
                               "correct": closed_top1 is not None and normalize_prediction_text(closed_top1) == expected_norm},
            "landmark": {"top1": lm.text, "correct": normalize_prediction_text(lm.text) == expected_norm,
                         "confidence": lm.candidates[0].score, "latency_ms": l_ms},
        })
        print(f"{s.get('sample_index'):>3} [{s.get('source','?'):<6}] {expected:<10} "
              f"open={open_top1:<14} closed={str(closed_top1):<10} landmark={lm.text:<10}({lm.candidates[0].score:.2f})",
              flush=True)

    if not records:
        raise SystemExit("No clips scored (all missing?).")

    models = ("vanilla_open", "vanilla_closed", "landmark")
    overall = {m: summarize(records, m) for m in models}
    per_source = {src: {m: summarize(records, m, ("source", src)) for m in models} for src in sources}
    signers = sorted({r["signer_id"] for r in records if r.get("signer_id")})
    per_signer = {sg: {m: summarize(records, m, ("signer_id", sg)) for m in models} for sg in signers}
    per_word = {w: {m: summarize(records, m, ("label", w)) for m in models} for w in vocab}

    report = {
        "gate": "G1",
        "decision_grade": len(records) >= MIN_DECISION_GRADE_CLIPS and len(sources) >= 3,
        "n_clips": len(records),
        "n_sources": len(sources),
        "sources": sources,
        "vocab": vocab,
        "config": {"vanilla_checkpoint": args.vanilla_checkpoint, "landmark_model": args.landmark_model,
                   "device": args.device, "clip_length": args.clip_length, "stride": args.stride},
        "overall": overall,
        "per_source": per_source,
        "per_signer": per_signer,
        "per_word": per_word,
        "mean_latency_ms": {
            "vanilla": round(statistics.mean(r["vanilla_open"]["latency_ms"] for r in records), 1),
            "landmark": round(statistics.mean(r["landmark"]["latency_ms"] for r in records), 1),
        },
        "results": records,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Gate G1 ({'DECISION-GRADE' if report['decision_grade'] else 'NOT decision-grade yet'}: "
          f"n={len(records)}, sources={len(sources)}) ===")
    for m in models:
        o = overall[m]
        print(f"  {m:<15} {o['top1_accuracy']}  CI95={o['ci95_wilson']}  (n={o['n']})")
    print("\nPer-source top-1 (landmark | vanilla_closed | vanilla_open):")
    for src in sources:
        ps = per_source[src]
        print(f"  {src:<8} L={ps['landmark']['top1_accuracy']} {ps['landmark']['ci95_wilson']} | "
              f"Vc={ps['vanilla_closed']['top1_accuracy']} | Vo={ps['vanilla_open']['top1_accuracy']} "
              f"(n={ps['landmark']['n']})")
    print(f"\nwrote {out.resolve()}")


if __name__ == "__main__":
    main()
