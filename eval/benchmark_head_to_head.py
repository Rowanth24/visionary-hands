import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from asl_backends import (
    OllamaASLBackend,
    OllamaHybridASLBackend,
    VanillaASLBackend,
    aggregate_vanilla_candidates,
    normalize_prediction_text,
)
from infer_videos import read_video_frames


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a source-aware head-to-head benchmark for vanilla vs ollama_hybrid."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a benchmark manifest JSON with sample metadata.",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing downloaded local video files referenced by the manifest.",
    )
    parser.add_argument(
        "--vanilla-checkpoint",
        required=True,
        help="Path to the promoted vanilla checkpoint.",
    )
    parser.add_argument(
        "--labels-file",
        default="code/preprocess/wlasl_class_list.txt",
        help="Path to the WLASL class label file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Execution device for the vanilla model.",
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:e2b",
        help="Local Ollama model name.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL of the local Ollama daemon.",
    )
    parser.add_argument(
        "--ollama-sample-count",
        type=int,
        default=8,
        help="Number of frames sampled for Ollama.",
    )
    parser.add_argument(
        "--ollama-frame-size",
        type=int,
        default=384,
        help="Maximum frame side length sent to Ollama.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=180,
        help="Ollama timeout in seconds.",
    )
    parser.add_argument(
        "--ollama-retries",
        type=int,
        default=2,
        help="Ollama request retries.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=64,
        help="Vanilla clip length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Vanilla window stride.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k candidates to retain.",
    )
    parser.add_argument(
        "--ollama-proposal-top-k",
        type=int,
        default=5,
        help="How many vanilla proposals to expose to Ollama.",
    )
    parser.add_argument(
        "--ollama-rerank-margin",
        type=float,
        default=0.05,
        help="Maximum proposal-score gap that still allows hybrid reranking.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Where to write the benchmark report JSON.",
    )
    return parser


def load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def expected_label_for_sample(sample):
    return sample.get("label") or sample.get("pair_label") or sample.get("expected")


def extract_candidate_metrics(candidates, expected_text):
    rank = None
    score = None
    normalized_candidates = []

    for idx, candidate in enumerate(candidates, start=1):
        text = normalize_prediction_text(candidate.text)
        normalized_candidates.append({"text": candidate.text, "score": candidate.score})
        if text == expected_text and rank is None:
            rank = idx
            score = candidate.score

    return normalized_candidates, rank, score


def summarize_records(records, key):
    correct = sum(1 for record in records if record[key]["correct"])
    top5 = sum(1 for record in records if record[key]["rank_of_expected"] is not None and record[key]["rank_of_expected"] <= 5)
    reciprocal_ranks = []
    expected_scores = []
    latencies = []

    for record in records:
        rank = record[key]["rank_of_expected"]
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        score = record[key]["score_of_expected"]
        if score is not None:
            expected_scores.append(score)
        latencies.append(record[key]["latency_ms"])

    summary = {
        "top1_accuracy": correct / len(records),
        "top5_hit_rate": top5 / len(records),
        "mean_reciprocal_rank_top5": sum(reciprocal_ranks) / len(records),
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "max_latency_ms": max(latencies),
        "mean_score_of_expected_when_in_top5": statistics.mean(expected_scores) if expected_scores else None,
    }
    return summary


def summarize_pairwise(records):
    summary = {
        "both_correct": 0,
        "vanilla_only_correct": 0,
        "hybrid_only_correct": 0,
        "both_wrong": 0,
        "hybrid_changed_top1": 0,
        "hybrid_reranked": 0,
        "both_wrong_vanilla_better_rank": 0,
        "both_wrong_hybrid_better_rank": 0,
        "both_wrong_same_rank": 0,
    }

    for record in records:
        vanilla_correct = record["vanilla"]["correct"]
        hybrid_correct = record["hybrid"]["correct"]

        if record["vanilla"]["predicted"] != record["hybrid"]["predicted"]:
            summary["hybrid_changed_top1"] += 1
        if record["hybrid"].get("mode") == "reranked":
            summary["hybrid_reranked"] += 1

        if vanilla_correct and hybrid_correct:
            summary["both_correct"] += 1
        elif vanilla_correct and not hybrid_correct:
            summary["vanilla_only_correct"] += 1
        elif hybrid_correct and not vanilla_correct:
            summary["hybrid_only_correct"] += 1
        else:
            summary["both_wrong"] += 1
            vanilla_rank = record["vanilla"]["rank_of_expected"] or math.inf
            hybrid_rank = record["hybrid"]["rank_of_expected"] or math.inf
            if vanilla_rank < hybrid_rank:
                summary["both_wrong_vanilla_better_rank"] += 1
            elif hybrid_rank < vanilla_rank:
                summary["both_wrong_hybrid_better_rank"] += 1
            else:
                summary["both_wrong_same_rank"] += 1

    return summary


def main():
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    samples = manifest["samples"]
    video_dir = Path(args.video_dir)

    vanilla = VanillaASLBackend(
        checkpoint_path=args.vanilla_checkpoint,
        labels_path=args.labels_file,
        device_name=args.device,
    )
    ollama = OllamaASLBackend(
        model_name=args.ollama_model,
        host=args.ollama_host,
        sample_count=args.ollama_sample_count,
        frame_size=args.ollama_frame_size,
        timeout_seconds=args.ollama_timeout,
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        keep_alive="15m",
        request_retries=args.ollama_retries,
    )
    hybrid = OllamaHybridASLBackend(
        ollama_backend=ollama,
        proposal_backend=vanilla,
        proposal_top_k=args.ollama_proposal_top_k,
        clip_length=args.clip_length,
        stride=args.stride,
        rerank_margin=args.ollama_rerank_margin,
    )

    warm_sample = samples[0]
    warm_frames = read_video_frames(video_dir / warm_sample["local_video"])
    aggregate_vanilla_candidates(vanilla, warm_frames, args.clip_length, args.stride, args.top_k)
    hybrid.infer_clip(warm_frames, args.top_k)

    records = []
    for sample in samples:
        expected_text = normalize_prediction_text(expected_label_for_sample(sample))
        video_path = video_dir / sample["local_video"]
        frames = read_video_frames(video_path)

        started = time.perf_counter()
        vanilla_candidates, window_count = aggregate_vanilla_candidates(
            backend=vanilla,
            frames=frames,
            clip_length=args.clip_length,
            stride=args.stride,
            top_k=args.top_k,
        )
        vanilla_latency_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        hybrid_result = hybrid.infer_clip(frames, args.top_k)
        hybrid_latency_ms = (time.perf_counter() - started) * 1000.0

        vanilla_texts, vanilla_rank, vanilla_score = extract_candidate_metrics(vanilla_candidates, expected_text)
        hybrid_texts, hybrid_rank, hybrid_score = extract_candidate_metrics(hybrid_result.candidates, expected_text)

        record = {
            "sample_index": sample["sample_index"],
            "source": sample.get("source"),
            "expected": expected_label_for_sample(sample),
            "video": sample["local_video"],
            "vanilla": {
                "predicted": vanilla_candidates[0].text,
                "correct": normalize_prediction_text(vanilla_candidates[0].text) == expected_text,
                "latency_ms": vanilla_latency_ms,
                "window_count": window_count,
                "rank_of_expected": vanilla_rank,
                "score_of_expected": vanilla_score,
                "top5": vanilla_texts,
            },
            "hybrid": {
                "predicted": hybrid_result.text,
                "correct": normalize_prediction_text(hybrid_result.text) == expected_text,
                "latency_ms": hybrid_latency_ms,
                "rank_of_expected": hybrid_rank,
                "score_of_expected": hybrid_score,
                "mode": hybrid_result.raw.get("mode") if hybrid_result.raw else None,
                "ollama_choice": hybrid_result.raw.get("ollama_choice") if hybrid_result.raw else None,
                "proposal_window_count": hybrid_result.raw.get("proposal_window_count") if hybrid_result.raw else None,
                "top5": hybrid_texts,
            },
        }
        records.append(record)
        print(
            f"{sample['sample_index']:02d} {sample.get('source', 'unknown')} {expected_label_for_sample(sample)}: "
            f"vanilla={record['vanilla']['predicted']} ({vanilla_latency_ms:.0f} ms) | "
            f"hybrid={record['hybrid']['predicted']} [{record['hybrid']['mode']}] ({hybrid_latency_ms:.0f} ms)",
            flush=True,
        )

    source_breakdown = {}
    for source in sorted({record["source"] for record in records}):
        source_records = [record for record in records if record["source"] == source]
        source_breakdown[source] = {
            "vanilla": summarize_records(source_records, "vanilla"),
            "hybrid": summarize_records(source_records, "hybrid"),
            "pairwise": summarize_pairwise(source_records),
        }

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "video_dir": str(video_dir.resolve()),
        "seed": manifest.get("seed"),
        "paired_labels": manifest.get("paired_labels"),
        "models": {
            "vanilla": summarize_records(records, "vanilla"),
            "hybrid": summarize_records(records, "hybrid"),
        },
        "pairwise": summarize_pairwise(records),
        "source_breakdown": source_breakdown,
        "hybrid_mode_counts": dict(Counter(record["hybrid"].get("mode") or "none" for record in records)),
        "results": records,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["models"], indent=2))
    print(json.dumps(report["pairwise"], indent=2))
    print(json.dumps(report["source_breakdown"], indent=2))
    print(f"wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
