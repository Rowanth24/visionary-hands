import argparse
import json
from pathlib import Path

import cv2
import torch

from asl_backends import (
    VanillaASLBackend,
    add_backend_args,
    create_backend_from_args,
    load_labels,
    normalize_prediction_text,
)


def build_parser():
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run whole-video inference on one or more ASL videos."
    )
    add_backend_args(parser)
    parser.add_argument(
        "videos",
        nargs="+",
        help="Video paths to evaluate.",
    )
    parser.add_argument(
        "--split-json",
        default=str(code_dir / "preprocess" / "nslt_1000.json"),
        help="Optional split file used to look up expected labels by video stem.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=64,
        help="Number of frames per vanilla inference window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Window stride for vanilla whole-video voting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many predictions to print.",
    )
    return parser


def load_expected_labels(split_json_path):
    split_json = Path(split_json_path)
    if not split_json.exists():
        return {}

    with split_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {video_id: item["action"][0] for video_id, item in data.items()}


def read_video_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise RuntimeError(f"No frames were read from: {video_path}")
    return frames


def make_windows(frames, clip_length, stride):
    if len(frames) < clip_length:
        return [frames + [frames[-1]] * (clip_length - len(frames))]

    starts = list(range(0, len(frames) - clip_length + 1, stride))
    last_start = len(frames) - clip_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return [frames[start:start + clip_length] for start in starts]


def infer_with_vanilla_windows(backend, frames, clip_length, stride, top_k):
    prepared_frames = backend.prepare_frames(frames)
    windows = make_windows(prepared_frames, clip_length, stride)
    probs_sum = None

    for window in windows:
        result = backend.infer_prepared_clip(window, top_k)
        probs = result.raw["candidate_probabilities"]
        probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_mean = probs_sum / len(windows)
    candidates = [
        (candidate.text, candidate.score)
        for candidate in backend.candidates_from_restricted_probabilities(probs_mean, top_k)
    ]
    return candidates, len(windows)


def load_label_lookup(labels_path):
    lines = Path(labels_path).read_text(encoding="utf-8").splitlines()
    return load_labels(labels_path, len(lines))


def main():
    args = build_parser().parse_args()
    backend = create_backend_from_args(args)
    expected_labels = load_expected_labels(args.split_json)
    label_lookup = load_label_lookup(args.labels_file)

    correct = 0
    evaluated = 0

    print(f"Running backend: {backend.describe()}")
    if args.backend == "vanilla":
        print(f"Loaded checkpoint: {args.checkpoint}")
        if backend.candidate_label_count is not None:
            print(f"Candidate labels: {backend.candidate_label_count}")
            if backend.missing_candidate_labels:
                print(f"Ignored unknown labels: {', '.join(backend.missing_candidate_labels[:10])}")
    elif args.backend == "ollama_hybrid":
        print(f"Ollama model: {args.ollama_model}")
        print(f"Proposal checkpoint: {args.ollama_proposal_checkpoint}")
    else:
        print(f"Ollama model: {args.ollama_model}")

    for video in args.videos:
        video_path = Path(video).expanduser().resolve()
        frames = read_video_frames(video_path)
        print()
        print(video_path)
        print(f"frames={len(frames)}")

        if isinstance(backend, VanillaASLBackend):
            candidates, unit_count = infer_with_vanilla_windows(
                backend=backend,
                frames=frames,
                clip_length=args.clip_length,
                stride=args.stride,
                top_k=args.top_k,
            )
            predicted_text = candidates[0][0]
            print(f"windows={unit_count}")
        else:
            result = backend.infer_clip(frames, args.top_k)
            candidates = [(candidate.text, candidate.score) for candidate in result.candidates]
            predicted_text = result.text
            print(f"sampled_frames={min(len(frames), args.ollama_sample_count)}")

        expected_idx = expected_labels.get(video_path.stem)
        if expected_idx is not None and expected_idx < len(label_lookup):
            expected_text = label_lookup[expected_idx]
            is_correct = normalize_prediction_text(predicted_text) == normalize_prediction_text(expected_text)
            evaluated += 1
            correct += int(is_correct)
            status = "correct" if is_correct else "wrong"
            print(f"expected={expected_idx} ({expected_text}) status={status}")

        for rank, (text, score) in enumerate(candidates, start=1):
            normalized = normalize_prediction_text(text)
            marker = ""
            if expected_idx is not None and expected_idx < len(label_lookup):
                marker = "*" if normalized == normalize_prediction_text(label_lookup[expected_idx]) else " "
            if score is None:
                print(f"{marker} {rank}. {text}")
            else:
                print(f"{marker} {rank}. {text}\t{score:.4f}")

    if evaluated:
        print()
        print(f"exact_match_accuracy={correct / evaluated:.4f} ({correct}/{evaluated})")


if __name__ == "__main__":
    main()
