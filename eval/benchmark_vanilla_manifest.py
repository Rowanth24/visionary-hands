import argparse
import json
import math
import statistics
from pathlib import Path

from asl_backends import VanillaASLBackend, aggregate_vanilla_candidates, normalize_prediction_text
from infer_videos import read_video_frames


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a vanilla checkpoint on a manifest of local sign-language videos."
    )
    parser.add_argument("--manifest", required=True, help="Manifest JSON with sample metadata.")
    parser.add_argument("--video-dir", required=True, help="Directory containing local videos.")
    parser.add_argument("--checkpoint", required=True, help="Path to the vanilla checkpoint.")
    parser.add_argument("--labels-file", default="code/preprocess/wlasl_class_list.txt", help="Path to label file.")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument(
        "--candidate-labels",
        help="Optional comma-separated shortlist of allowed labels for vanilla inference.",
    )
    parser.add_argument(
        "--candidate-labels-file",
        help="Optional file containing allowed labels, one per line or comma-separated.",
    )
    parser.add_argument(
        "--candidate-labels-from-manifest",
        action="store_true",
        help="Restrict inference to the unique expected labels present in the manifest.",
    )
    parser.add_argument(
        "--frame-normalization",
        choices=("none", "motion", "mediapipe"),
        default="none",
        help="Optional per-clip spatial normalization before vanilla inference.",
    )
    parser.add_argument(
        "--mediapipe-sample-count",
        type=int,
        default=12,
        help="How many frames to probe when estimating a MediaPipe-based crop.",
    )
    parser.add_argument("--clip-length", type=int, default=64, help="Clip length.")
    parser.add_argument("--stride", type=int, default=8, help="Window stride.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k candidates.")
    parser.add_argument("--output-json", required=True, help="Where to write the evaluation report.")
    return parser


def expected_label(sample):
    return sample.get("label") or sample.get("pair_label") or sample.get("expected")


def resolve_video_name(sample):
    local_name = sample.get("local_video")
    if local_name:
        return local_name
    label = expected_label(sample)
    if label and sample.get("filepath"):
        return f"{sample['sample_index']:02d}_{label}_{Path(sample['filepath']).name}"
    raise KeyError("Manifest samples must include either 'local_video' or both 'filepath' and an expected label.")


def main():
    args = build_parser().parse_args()
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = payload["samples"]
    video_dir = Path(args.video_dir)

    candidate_labels = args.candidate_labels
    if args.candidate_labels_from_manifest:
        manifest_labels = sorted({expected_label(sample) for sample in samples if expected_label(sample)})
        candidate_labels = ",".join(manifest_labels)

    backend = VanillaASLBackend(
        args.checkpoint,
        args.labels_file,
        args.device,
        frame_normalization=args.frame_normalization,
        mediapipe_sample_count=args.mediapipe_sample_count,
        candidate_labels=candidate_labels,
        candidate_labels_file=args.candidate_labels_file,
    )
    results = []

    for sample in samples:
        local_name = resolve_video_name(sample)

        frames = read_video_frames(video_dir / local_name)
        candidates, window_count = aggregate_vanilla_candidates(
            backend=backend,
            frames=frames,
            clip_length=args.clip_length,
            stride=args.stride,
            top_k=args.top_k,
        )
        normalized_expected = normalize_prediction_text(expected_label(sample))
        rank = next(
            (
                index
                for index, candidate in enumerate(candidates, start=1)
                if normalize_prediction_text(candidate.text) == normalized_expected
            ),
            None,
        )
        expected_score = None if rank is None else candidates[rank - 1].score

        results.append(
            {
                "sample_index": sample["sample_index"],
                "source": sample.get("source"),
                "video": local_name,
                "expected": expected_label(sample),
                "predicted": candidates[0].text,
                "correct": normalize_prediction_text(candidates[0].text) == normalized_expected,
                "window_count": window_count,
                "rank_of_expected": rank,
                "score_of_expected": expected_score,
                "top5": [{"text": candidate.text, "score": candidate.score} for candidate in candidates],
            }
        )
        print(
            f"{sample['sample_index']:02d} {sample.get('source', 'unknown')} {expected_label(sample)}: "
            f"{candidates[0].text} rank={rank}",
            flush=True,
        )

    def summarize(records):
        lateless = {
            "top1_accuracy": sum(1 for record in records if record["correct"]) / len(records),
            "top5_hit_rate": sum(1 for record in records if record["rank_of_expected"] is not None and record["rank_of_expected"] <= 5) / len(records),
            "mean_reciprocal_rank_top5": sum(0.0 if record["rank_of_expected"] is None else 1.0 / record["rank_of_expected"] for record in records) / len(records),
        }
        scores = [record["score_of_expected"] for record in records if record["score_of_expected"] is not None]
        if scores:
            lateless["mean_score_of_expected_when_in_top5"] = statistics.mean(scores)
        else:
            lateless["mean_score_of_expected_when_in_top5"] = None
        return lateless

    source_breakdown = {}
    for source in sorted({record.get("source") for record in results if record.get("source")}):
        source_records = [record for record in results if record.get("source") == source]
        source_breakdown[source] = summarize(source_records)

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "video_dir": str(video_dir.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backend": backend.describe(),
        "frame_normalization": args.frame_normalization,
        "candidate_label_count": backend.candidate_label_count,
        "summary": summarize(results),
        "source_breakdown": source_breakdown,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["source_breakdown"], indent=2))
    print(f"wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
