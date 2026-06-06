"""DEPRECATED (Gate G-C) — superseded by benchmark_landmark_vs_vanilla.py.

This nearest-neighbor closed-set eval uses the legacy landmark_utils encoder
(HolisticSequenceEncoder / normalize_landmark_points), NOT the trained dual-hand
transformer's canonical preprocessing. Its results are the source of the retired
"landmark loses (0.45)"-style verdict. For landmark-vs-vanilla, use
benchmark_landmark_vs_vanilla.py (Gate G-B). Kept for historical reference only.
See docs/preprocessing_standard.md.
"""

import argparse
import json
import math
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from asl_backends import VanillaASLBackend, aggregate_vanilla_candidates, normalize_prediction_text
from infer_videos import read_video_frames
from landmark_utils import HolisticSequenceEncoder, landmark_distance


WLASL_BASE_URL = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/"


def build_parser():
    code_dir = Path(__file__).resolve().parent  # eval/ — benchmark/ and preprocess/ live here post-merge
    parser = argparse.ArgumentParser(
        description="Benchmark landmark-only and landmark-fused closed-set inference on a held-out manifest."
    )
    parser.add_argument("--manifest", required=True, help="Held-out benchmark manifest JSON.")
    parser.add_argument("--video-dir", required=True, help="Directory containing the query videos.")
    parser.add_argument(
        "--wlasl-metadata",
        default=str(code_dir / "benchmark" / "wlasl_samples.json"),
        help="Local WLASL metadata JSON used to build support sets.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Vanilla checkpoint used for the closed-set vanilla baseline.",
    )
    parser.add_argument(
        "--labels-file",
        default=str(code_dir / "preprocess" / "wlasl_class_list.txt"),
        help="Path to the WLASL class label file.",
    )
    parser.add_argument(
        "--support-per-label",
        type=int,
        default=2,
        help="How many WLASL support videos to use per label.",
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
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Vanilla inference device.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="How many landmark frames to sample per clip.",
    )
    parser.add_argument(
        "--fusion-weight",
        type=float,
        default=0.5,
        help="Weight assigned to vanilla closed-set probabilities in the fusion score.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(code_dir / "benchmark" / "landmark_closed_set"),
        help="Directory for support videos and cached landmarks.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Where to write the benchmark report JSON.",
    )
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
    raise KeyError("Manifest sample is missing video naming information.")


def load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload["samples"]
    labels = sorted({normalize_prediction_text(expected_label(sample)) for sample in samples})
    return payload, samples, labels


def load_wlasl_index(metadata_path, allowed_labels, exclude_video_ids):
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    items = payload["samples"] if isinstance(payload, dict) else payload
    by_label = {label: [] for label in allowed_labels}

    for item in items:
        label = normalize_prediction_text(item["gloss"]["label"])
        if label not in by_label:
            continue
        video_id = Path(item["filepath"]).stem
        if video_id in exclude_video_ids:
            continue
        by_label[label].append(
            {
                "label": label,
                "video_id": video_id,
                "filepath": item["filepath"],
                "frame_count": item["metadata"].get("total_frame_count") or 0,
            }
        )

    for label in by_label:
        by_label[label].sort(key=lambda item: (item["frame_count"], item["video_id"]), reverse=True)
    return by_label


def ensure_support_video(entry, support_dir):
    support_dir = Path(support_dir)
    support_dir.mkdir(parents=True, exist_ok=True)
    target = support_dir / f"{entry['label'].replace(' ', '_')}_{entry['video_id']}.mp4"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = WLASL_BASE_URL + entry["filepath"]
    temp_path = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        with open(temp_path, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    temp_path.replace(target)
    return target


def aggregate_vanilla_probabilities(backend, frames, clip_length, stride):
    prepared_frames = backend.prepare_frames(frames)
    windows = []
    if len(prepared_frames) < clip_length:
        windows = [prepared_frames + [prepared_frames[-1]] * (clip_length - len(prepared_frames))]
    else:
        starts = list(range(0, len(prepared_frames) - clip_length + 1, stride))
        last_start = len(prepared_frames) - clip_length
        if starts[-1] != last_start:
            starts.append(last_start)
        windows = [prepared_frames[start:start + clip_length] for start in starts]

    probs_sum = None
    for window in windows:
        result = backend.infer_prepared_clip(window, top_k=min(5, backend.num_classes))
        probs = result.raw["probabilities"]
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(windows)


def summarize(results, key):
    correct = sum(1 for result in results if result[key]["correct"])
    top5 = sum(1 for result in results if result[key]["rank_of_expected"] is not None and result[key]["rank_of_expected"] <= 5)
    reciprocal_rank = sum(0.0 if result[key]["rank_of_expected"] is None else 1.0 / result[key]["rank_of_expected"] for result in results) / len(results)
    return {
        "top1_accuracy": correct / len(results),
        "top5_hit_rate": top5 / len(results),
        "mean_reciprocal_rank_top5": reciprocal_rank,
    }


def summarize_by_source(results, key):
    output = {}
    for source in sorted({result["source"] for result in results if result["source"]}):
        source_results = [result for result in results if result["source"] == source]
        output[source] = summarize(source_results, key)
    return output


def rank_for_label(ordered_labels, expected):
    normalized = normalize_prediction_text(expected)
    for index, item in enumerate(ordered_labels, start=1):
        if normalize_prediction_text(item["label"]) == normalized:
            return index
    return None


def build_report_section(ordered_labels, expected):
    rank = rank_for_label(ordered_labels, expected)
    top_label = ordered_labels[0]["label"]
    return {
        "predicted": top_label,
        "correct": normalize_prediction_text(top_label) == normalize_prediction_text(expected),
        "rank_of_expected": rank,
        "top5": ordered_labels[:5],
    }


def main():
    args = build_parser().parse_args()
    payload, samples, closed_set_labels = load_manifest(args.manifest)
    query_video_dir = Path(args.video_dir)
    work_dir = Path(args.work_dir)
    support_video_dir = work_dir / "support_videos" / Path(args.manifest).stem
    support_cache_dir = work_dir / "support_landmarks" / Path(args.manifest).stem
    query_cache_dir = work_dir / "query_landmarks" / Path(args.manifest).stem

    exclude_video_ids = {sample["video_id"] for sample in samples if sample.get("video_id")}
    support_index = load_wlasl_index(args.wlasl_metadata, closed_set_labels, exclude_video_ids)

    support_videos = {}
    for label in closed_set_labels:
        selected = support_index[label][: args.support_per_label]
        if not selected:
            raise RuntimeError(f"No WLASL support videos were found for label '{label}'.")
        support_videos[label] = [ensure_support_video(entry, support_video_dir) for entry in selected]

    encoder = HolisticSequenceEncoder(sequence_length=args.sequence_length)
    support_descriptors = {
        label: [encoder.encode_with_cache(path, support_cache_dir) for path in paths]
        for label, paths in support_videos.items()
    }

    vanilla = VanillaASLBackend(
        checkpoint_path=args.checkpoint,
        labels_path=args.labels_file,
        device_name=args.device,
    )
    label_to_index = {normalize_prediction_text(label): index for index, label in enumerate(vanilla.labels)}
    closed_set_indices = {label: label_to_index[label] for label in closed_set_labels}

    results = []

    for sample in samples:
        expected = expected_label(sample)
        video_name = resolve_video_name(sample)
        video_path = query_video_dir / video_name
        frames = read_video_frames(video_path)
        query_descriptor = encoder.encode_with_cache(video_path, query_cache_dir)

        landmark_ranking = []
        for label in closed_set_labels:
            distances = [landmark_distance(query_descriptor, descriptor) for descriptor in support_descriptors[label]]
            landmark_ranking.append(
                {
                    "label": label,
                    "score": float(math.exp(-min(distances))),
                    "distance": float(min(distances)),
                }
            )
        landmark_ranking.sort(key=lambda item: item["distance"])

        probs = aggregate_vanilla_probabilities(vanilla, frames, args.clip_length, args.stride)
        vanilla_ranking = [
            {
                "label": label,
                "score": float(probs[index].item()),
            }
            for label, index in closed_set_indices.items()
        ]
        vanilla_ranking.sort(key=lambda item: item["score"], reverse=True)

        vanilla_total = sum(item["score"] for item in vanilla_ranking) or 1.0
        landmark_total = sum(item["score"] for item in landmark_ranking) or 1.0
        landmark_scores = {item["label"]: item["score"] / landmark_total for item in landmark_ranking}
        fusion_ranking = []
        for item in vanilla_ranking:
            label = item["label"]
            vanilla_score = item["score"] / vanilla_total
            landmark_score = landmark_scores[label]
            fusion_score = (args.fusion_weight * vanilla_score) + ((1.0 - args.fusion_weight) * landmark_score)
            fusion_ranking.append(
                {
                    "label": label,
                    "score": fusion_score,
                    "vanilla_score": vanilla_score,
                    "landmark_score": landmark_score,
                }
            )
        fusion_ranking.sort(key=lambda item: item["score"], reverse=True)

        record = {
            "sample_index": sample["sample_index"],
            "source": sample.get("source"),
            "video": video_name,
            "expected": expected,
            "vanilla_closed": build_report_section(vanilla_ranking, expected),
            "landmark_only": build_report_section(landmark_ranking, expected),
            "fusion": build_report_section(fusion_ranking, expected),
        }
        results.append(record)
        print(
            f"{sample['sample_index']:02d} {sample.get('source', 'unknown')} {expected}: "
            f"vanilla={record['vanilla_closed']['predicted']} "
            f"landmark={record['landmark_only']['predicted']} "
            f"fusion={record['fusion']['predicted']}",
            flush=True,
        )

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "video_dir": str(query_video_dir.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "support_per_label": args.support_per_label,
        "closed_set_labels": closed_set_labels,
        "support_videos": {label: [str(path.resolve()) for path in paths] for label, paths in support_videos.items()},
        "vanilla_closed_summary": summarize(results, "vanilla_closed"),
        "landmark_only_summary": summarize(results, "landmark_only"),
        "fusion_summary": summarize(results, "fusion"),
        "vanilla_closed_by_source": summarize_by_source(results, "vanilla_closed"),
        "landmark_only_by_source": summarize_by_source(results, "landmark_only"),
        "fusion_by_source": summarize_by_source(results, "fusion"),
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["vanilla_closed_summary"], indent=2))
    print(json.dumps(report["landmark_only_summary"], indent=2))
    print(json.dumps(report["fusion_summary"], indent=2))
    print(f"wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
