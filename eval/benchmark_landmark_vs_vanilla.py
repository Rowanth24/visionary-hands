#!/usr/bin/env python3
"""Gate G-B — landmark dual-hand transformer vs vanilla I3D, one harness, one judge.

Runs the EPICS dual-hand landmark transformer (LandmarkBackend, TorchScript) and
the vanilla InceptionI3d (VanillaASLBackend) over the SAME held-out clips and
scores them identically. This replaces the invalid prior "landmark loses (0.45)"
result, which never ran this model (it used landmark_utils.HolisticSequenceEncoder).

Three columns are reported per clip:
  - vanilla_open    : I3D top-1 over its full 1000-class vocabulary (real-world).
  - vanilla_closed  : I3D top-1 restricted to the landmark model's vocabulary
                      (apples-to-apples with the closed-set landmark model).
  - landmark        : dual-hand transformer top-1 (closed-set by construction).

METHODOLOGY CAVEAT: this is a tiny set (n is typically ~12). Per the program plan
(Gate G1: "no model decisions on n=20 again") this is a PLUMBING / methodology
experiment — it proves the harness judges both models fairly and kills the invalid
0.45 verdict. It is NOT a decision-grade accuracy verdict. That comes at G1/G3 on
the grown benchmark.

Test layout expected (class-subfolder):  <test-dir>/<label>/<clip>.mp4

Usage (from repo root, after `source env.sh`, in .venv-oracle):
  python eval/benchmark_landmark_vs_vanilla.py \
    --test-dir ~/EPICS_Project/data/wlasl_distinct_8_full/test \
    --labels bowling,computer,deaf,drink,yes \
    --vanilla-checkpoint oracle/checkpoints/best_model_original_backup.pth \
    --landmark-model models/landmark/checkpoints/model.ts \
    --landmark-labels models/landmark/checkpoints/labels.json \
    --output-json eval/benchmark/gB_landmark_vs_vanilla.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from asl_backends import (
    LandmarkBackend,
    VanillaASLBackend,
    make_windows,
    normalize_prediction_text,
)
from infer_videos import read_video_frames

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def build_parser():
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Gate G-B: landmark transformer vs vanilla I3D, one judge.")
    parser.add_argument("--test-dir", required=True, help="Directory of <label>/<clip>.mp4 subfolders.")
    parser.add_argument("--labels", required=True,
                        help="Comma-separated landmark vocabulary (the shared closed set), or a path to a labels file.")
    parser.add_argument("--vanilla-checkpoint", required=True, help="Path to the I3D checkpoint (.pth).")
    parser.add_argument("--vanilla-labels-file", default=str(code_dir / "preprocess" / "wlasl_class_list.txt"),
                        help="WLASL 1000-class label list for the vanilla backend.")
    parser.add_argument("--landmark-model", required=True, help="TorchScript landmark model (.ts).")
    parser.add_argument("--landmark-labels", required=True, help="Landmark labels.json.")
    parser.add_argument("--landmark-config", default=None, help="Optional landmark config.json (defaults to model sibling).")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--clip-length", type=int, default=64, help="Vanilla clip length.")
    parser.add_argument("--stride", type=int, default=8, help="Vanilla window stride.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", required=True, help="Where to write the report JSON.")
    return parser


def resolve_labels(labels_arg):
    path = Path(labels_arg)
    if path.exists():
        if path.suffix == ".json":
            return [str(x) for x in json.loads(path.read_text(encoding="utf-8"))]
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [tok.strip() for tok in labels_arg.split(",") if tok.strip()]


def discover_clips(test_dir, labels):
    """Return [(label, path)] for every clip under <test-dir>/<label>/."""
    clips = []
    for label in labels:
        sub = Path(test_dir).expanduser() / label
        if not sub.is_dir():
            continue
        for clip in sorted(sub.iterdir()):
            if clip.suffix.lower() in VIDEO_SUFFIXES:
                clips.append((label, clip))
    return clips


def closed_set_index_map(vanilla_labels, shared_labels):
    """Map each shared label to its index in the vanilla 1000-class list (normalized match)."""
    normalized = {normalize_prediction_text(lbl): idx for idx, lbl in enumerate(vanilla_labels)}
    mapping = {}
    missing = []
    for label in shared_labels:
        idx = normalized.get(normalize_prediction_text(label))
        if idx is None:
            missing.append(label)
        else:
            mapping[label] = idx
    return mapping, missing


def vanilla_mean_probs(backend, frames, clip_length, stride, top_k):
    """Window-aggregate the vanilla backend to a single full-vocabulary mean-probability vector."""
    prepared = backend.prepare_frames(frames)
    windows = make_windows(prepared, clip_length, stride)
    probs_sum = None
    for window in windows:
        result = backend.infer_prepared_clip(window, top_k)
        probs = result.raw["candidate_probabilities"]
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(windows), len(windows)


def main():
    args = build_parser().parse_args()
    shared_labels = resolve_labels(args.labels)
    clips = discover_clips(args.test_dir, shared_labels)
    if not clips:
        raise SystemExit(f"No clips found under {args.test_dir} for labels {shared_labels}")

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

    closed_map, missing = closed_set_index_map(vanilla.labels, shared_labels)
    if missing:
        print(f"WARNING: shared labels not in vanilla vocabulary (closed-set view will skip them): {missing}")
    closed_labels = [lbl for lbl in shared_labels if lbl in closed_map]
    closed_index_tensor = torch.tensor([closed_map[lbl] for lbl in closed_labels], dtype=torch.long)

    print(f"Vanilla: {vanilla.describe()}  |  Landmark: {landmark.describe()}")
    print(f"Clips: {len(clips)}  Shared vocab: {shared_labels}\n")

    records = []
    for idx, (expected, clip_path) in enumerate(clips):
        expected_norm = normalize_prediction_text(expected)
        frames = read_video_frames(clip_path)

        t0 = time.perf_counter()
        probs_mean, window_count = vanilla_mean_probs(vanilla, frames, args.clip_length, args.stride, args.top_k)
        vanilla_ms = (time.perf_counter() - t0) * 1000.0
        open_top1 = vanilla.labels[int(torch.argmax(probs_mean))]
        closed_top1 = closed_labels[int(torch.argmax(probs_mean[closed_index_tensor]))] if closed_labels else None

        t0 = time.perf_counter()
        lm_result = landmark.infer_clip(frames, args.top_k)
        landmark_ms = (time.perf_counter() - t0) * 1000.0
        landmark_top1 = lm_result.text

        record = {
            "index": idx,
            "expected": expected,
            "clip": str(clip_path),
            "vanilla_open": {"top1": open_top1, "correct": normalize_prediction_text(open_top1) == expected_norm,
                             "latency_ms": vanilla_ms, "windows": window_count},
            "vanilla_closed": {"top1": closed_top1,
                               "correct": closed_top1 is not None and normalize_prediction_text(closed_top1) == expected_norm},
            "landmark": {"top1": landmark_top1, "correct": normalize_prediction_text(landmark_top1) == expected_norm,
                         "confidence": lm_result.candidates[0].score, "latency_ms": landmark_ms},
        }
        records.append(record)
        print(f"{idx:02d} {expected:<10} | open={open_top1:<14} closed={str(closed_top1):<10} "
              f"landmark={landmark_top1:<10}({lm_result.candidates[0].score:.2f}) "
              f"| V {vanilla_ms:5.0f}ms  L {landmark_ms:5.0f}ms", flush=True)

    def acc(key):
        return sum(1 for r in records if r[key]["correct"]) / len(records)

    def per_word(key):
        words = {}
        for r in records:
            w = r["expected"]
            words.setdefault(w, {"n": 0, "correct": 0})
            words[w]["n"] += 1
            words[w]["correct"] += int(r[key]["correct"])
        return {w: {"n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / v["n"]} for w, v in words.items()}

    summary = {
        "n_clips": len(records),
        "shared_vocab": shared_labels,
        "top1_accuracy": {
            "vanilla_open": acc("vanilla_open"),
            "vanilla_closed": acc("vanilla_closed"),
            "landmark": acc("landmark"),
        },
        "mean_latency_ms": {
            "vanilla": statistics.mean(r["vanilla_open"]["latency_ms"] for r in records),
            "landmark": statistics.mean(r["landmark"]["latency_ms"] for r in records),
        },
        "per_word_accuracy": {
            "vanilla_open": per_word("vanilla_open"),
            "vanilla_closed": per_word("vanilla_closed"),
            "landmark": per_word("landmark"),
        },
    }

    report = {
        "gate": "G-B",
        "purpose": "Plumbing/methodology experiment — one harness judges both models. NOT decision-grade (small n; see Gate G1).",
        "config": {
            "test_dir": str(Path(args.test_dir).expanduser()),
            "vanilla_checkpoint": args.vanilla_checkpoint,
            "landmark_model": args.landmark_model,
            "device": args.device, "clip_length": args.clip_length, "stride": args.stride, "top_k": args.top_k,
        },
        "vanilla_describe": vanilla.describe(),
        "landmark_describe": landmark.describe(),
        "summary": summary,
        "results": records,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Gate G-B summary (NOT decision-grade — n={}) ===".format(len(records)))
    print(json.dumps(summary["top1_accuracy"], indent=2))
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
