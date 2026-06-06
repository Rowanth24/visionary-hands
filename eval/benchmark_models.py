import argparse
import json
from pathlib import Path

from asl_backends import OllamaASLBackend, OllamaHybridASLBackend, VanillaASLBackend, normalize_prediction_text
from infer_videos import infer_with_vanilla_windows, read_video_frames


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark vanilla and Ollama ASL backends on a manifest of public videos."
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/wlasl20_manifest.json",
        help="Path to the benchmark manifest JSON.",
    )
    parser.add_argument(
        "--video-dir",
        default="benchmark/videos",
        help="Directory containing the downloaded benchmark videos.",
    )
    parser.add_argument(
        "--current-vanilla",
        required=True,
        help="Path to the current vanilla checkpoint.",
    )
    parser.add_argument(
        "--candidate-vanilla",
        help="Optional path to a replacement vanilla checkpoint to compare.",
    )
    parser.add_argument(
        "--labels-file",
        default="code/preprocess/wlasl_class_list.txt",
        help="Path to the class label file.",
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:e2b",
        help="Ollama model name to benchmark.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip the Ollama benchmark.",
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
        help="Number of candidates to retain in reports.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Vanilla device.",
    )
    parser.add_argument(
        "--ollama-sample-count",
        type=int,
        default=6,
        help="Number of sampled frames for Ollama.",
    )
    parser.add_argument(
        "--ollama-frame-size",
        type=int,
        default=448,
        help="Maximum frame side length for Ollama.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=180,
        help="Ollama request timeout in seconds.",
    )
    parser.add_argument(
        "--ollama-retries",
        type=int,
        default=2,
        help="How many times to retry a failed Ollama request.",
    )
    parser.add_argument(
        "--ollama-proposal-checkpoint",
        help="Optional vanilla checkpoint used to run an ollama_hybrid benchmark.",
    )
    parser.add_argument(
        "--ollama-proposal-top-k",
        type=int,
        default=5,
        help="How many vanilla proposal labels to expose to ollama_hybrid.",
    )
    parser.add_argument(
        "--ollama-rerank-margin",
        type=float,
        default=0.05,
        help="Score gap threshold that allows ollama_hybrid to override the top vanilla proposal.",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark/benchmark_results.json",
        help="Path to write the benchmark results JSON.",
    )
    return parser


def load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["samples"]


def evaluate_vanilla(name, checkpoint_path, labels_path, device, samples, video_dir, clip_length, stride, top_k):
    backend = VanillaASLBackend(checkpoint_path=checkpoint_path, labels_path=labels_path, device_name=device)
    return evaluate_backend(name, backend, samples, video_dir, clip_length, stride, top_k)


def evaluate_ollama(name, model_name, samples, video_dir, top_k, sample_count, frame_size, timeout_seconds, retries):
    backend = OllamaASLBackend(
        model_name=model_name,
        host="http://localhost:11434",
        sample_count=sample_count,
        frame_size=frame_size,
        timeout_seconds=timeout_seconds,
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        keep_alive="15m",
        request_retries=retries,
    )
    return evaluate_backend(name, backend, samples, video_dir, clip_length=64, stride=8, top_k=top_k)


def evaluate_ollama_hybrid(
    name,
    model_name,
    proposal_checkpoint,
    labels_path,
    device,
    samples,
    video_dir,
    top_k,
    sample_count,
    frame_size,
    timeout_seconds,
    retries,
    proposal_top_k,
    clip_length,
    stride,
    rerank_margin,
):
    ollama_backend = OllamaASLBackend(
        model_name=model_name,
        host="http://localhost:11434",
        sample_count=sample_count,
        frame_size=frame_size,
        timeout_seconds=timeout_seconds,
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        keep_alive="15m",
        request_retries=retries,
    )
    proposal_backend = VanillaASLBackend(
        checkpoint_path=proposal_checkpoint,
        labels_path=labels_path,
        device_name=device,
    )
    backend = OllamaHybridASLBackend(
        ollama_backend=ollama_backend,
        proposal_backend=proposal_backend,
        proposal_top_k=proposal_top_k,
        clip_length=clip_length,
        stride=stride,
        rerank_margin=rerank_margin,
    )
    return evaluate_backend(name, backend, samples, video_dir, clip_length=clip_length, stride=stride, top_k=top_k)


def evaluate_backend(name, backend, samples, video_dir, clip_length, stride, top_k):
    results = []
    correct = 0

    for sample in samples:
        video_name = f"{sample['sample_index']:02d}_{sample['label']}_{Path(sample['filepath']).name}"
        video_path = Path(video_dir) / video_name
        try:
            frames = read_video_frames(video_path)

            if isinstance(backend, VanillaASLBackend):
                candidates, window_count = infer_with_vanilla_windows(
                    backend=backend,
                    frames=frames,
                    clip_length=clip_length,
                    stride=stride,
                    top_k=top_k,
                )
                predicted_text = candidates[0][0]
                extra = {"window_count": window_count}
                normalized_candidates = [{"text": text, "score": score} for text, score in candidates]
            else:
                result = backend.infer_clip(frames, top_k)
                predicted_text = result.text
                extra = {
                    "sampled_frames": min(len(frames), getattr(getattr(backend, "ollama_backend", backend), "sample_count", len(frames))),
                    "variant": result.raw.get("variant") if result.raw else None,
                    "mode": result.raw.get("mode") if result.raw else None,
                    "ollama_choice": result.raw.get("ollama_choice") if result.raw else None,
                }
                normalized_candidates = [{"text": candidate.text, "score": candidate.score} for candidate in result.candidates]

            error_text = None
        except Exception as exc:
            predicted_text = "__error__"
            extra = {"error": str(exc)}
            normalized_candidates = []
            error_text = str(exc)

        is_correct = normalize_prediction_text(predicted_text) == normalize_prediction_text(sample["label"])
        correct += int(is_correct)
        results.append(
            {
                "sample_index": sample["sample_index"],
                "video": video_name,
                "expected": sample["label"],
                "predicted": predicted_text,
                "correct": is_correct,
                "candidates": normalized_candidates,
                **extra,
            }
        )
        if error_text is not None:
            print(f"{name} error on {video_name}: {error_text}")

    return {
        "name": name,
        "accuracy": correct / len(samples),
        "correct": correct,
        "total": len(samples),
        "results": results,
    }


def maybe_promote_candidate(results, current_path, candidate_path):
    if not candidate_path:
        return None
    current_acc = results["current_vanilla"]["accuracy"]
    candidate_acc = results["candidate_vanilla"]["accuracy"]
    if candidate_acc > current_acc:
        return {
            "promote": True,
            "from": current_path,
            "to": candidate_path,
            "reason": f"candidate accuracy {candidate_acc:.3f} > current accuracy {current_acc:.3f}",
        }
    return {
        "promote": False,
        "from": current_path,
        "to": candidate_path,
        "reason": f"candidate accuracy {candidate_acc:.3f} <= current accuracy {current_acc:.3f}",
    }


def main():
    args = build_parser().parse_args()
    samples = load_manifest(args.manifest)
    video_dir = Path(args.video_dir)

    results = {
        "manifest": str(Path(args.manifest).resolve()),
        "video_dir": str(video_dir.resolve()),
        "models": {},
    }

    current = evaluate_vanilla(
        name="current_vanilla",
        checkpoint_path=args.current_vanilla,
        labels_path=args.labels_file,
        device=args.device,
        samples=samples,
        video_dir=video_dir,
        clip_length=args.clip_length,
        stride=args.stride,
        top_k=args.top_k,
    )
    results["models"]["current_vanilla"] = current

    if args.candidate_vanilla:
        candidate = evaluate_vanilla(
            name="candidate_vanilla",
            checkpoint_path=args.candidate_vanilla,
            labels_path=args.labels_file,
            device=args.device,
            samples=samples,
            video_dir=video_dir,
            clip_length=args.clip_length,
            stride=args.stride,
            top_k=args.top_k,
        )
        results["models"]["candidate_vanilla"] = candidate
        results["promotion_decision"] = maybe_promote_candidate(
            results={"current_vanilla": current, "candidate_vanilla": candidate},
            current_path=args.current_vanilla,
            candidate_path=args.candidate_vanilla,
        )

    if not args.skip_ollama:
        ollama = evaluate_ollama(
            name="ollama",
            model_name=args.ollama_model,
            samples=samples,
            video_dir=video_dir,
            top_k=args.top_k,
            sample_count=args.ollama_sample_count,
            frame_size=args.ollama_frame_size,
            timeout_seconds=args.ollama_timeout,
            retries=args.ollama_retries,
        )
        results["models"]["ollama"] = ollama

        if args.ollama_proposal_checkpoint:
            ollama_hybrid = evaluate_ollama_hybrid(
                name="ollama_hybrid",
                model_name=args.ollama_model,
                proposal_checkpoint=args.ollama_proposal_checkpoint,
                labels_path=args.labels_file,
                device=args.device,
                samples=samples,
                video_dir=video_dir,
                top_k=args.top_k,
                sample_count=args.ollama_sample_count,
                frame_size=args.ollama_frame_size,
                timeout_seconds=args.ollama_timeout,
                retries=args.ollama_retries,
                proposal_top_k=args.ollama_proposal_top_k,
                clip_length=args.clip_length,
                stride=args.stride,
                rerank_margin=args.ollama_rerank_margin,
            )
            results["models"]["ollama_hybrid"] = ollama_hybrid

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, result in results["models"].items():
        print(f"{name}: {result['correct']}/{result['total']} = {result['accuracy']:.3f}")
    if "promotion_decision" in results:
        decision = results["promotion_decision"]
        print(decision["reason"])
    print(f"wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
