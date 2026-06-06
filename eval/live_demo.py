import argparse
import sys
import time
from collections import deque

import cv2
import torch

from asl_backends import BackendResult, Candidate, VanillaASLBackend, add_backend_args, create_backend_from_args


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run live sign-language inference from a webcam."
    )
    add_backend_args(parser)
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=64,
        help="Number of frames kept in the rolling clip buffer.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=8,
        help="Run inference every N captured frames once the buffer is full.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many predictions to display.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        help="Exponential smoothing factor for vanilla probability outputs.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested webcam width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested webcam height.",
    )
    parser.add_argument(
        "--window-name",
        default="ASL Live Demo",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--mirror-display",
        action="store_true",
        help="Mirror only the display output for a more natural webcam preview.",
    )
    return parser


def smooth_vanilla_result(result, backend, top_k, smoothing, previous_probs):
    probs = result.raw["candidate_probabilities"]
    smoothed_probs = probs if previous_probs is None else (smoothing * previous_probs) + ((1.0 - smoothing) * probs)
    smoothed_result = BackendResult(
        text=backend.candidates_from_restricted_probabilities(smoothed_probs, 1)[0].text,
        candidates=backend.candidates_from_restricted_probabilities(smoothed_probs, top_k),
        raw=result.raw,
    )
    return smoothed_result, smoothed_probs


def draw_overlay(frame, backend_name, clip_fill, clip_length, latency_ms, result):
    overlay_lines = [
        f"Backend: {backend_name}",
        f"Clip: {clip_fill}/{clip_length} frames",
        "Controls: q or esc to quit",
    ]
    if latency_ms is not None:
        overlay_lines.append(f"Inference: {latency_ms:.0f} ms")

    if result is None:
        overlay_lines.append("Waiting for a full clip...")
    else:
        overlay_lines.append("")
        for rank, candidate in enumerate(result.candidates, start=1):
            if candidate.score is None:
                overlay_lines.append(f"{rank}. {candidate.text}")
            else:
                overlay_lines.append(f"{rank}. {candidate.text} ({candidate.score * 100:.1f}%)")

    y = 32
    for line in overlay_lines:
        if not line:
            y += 18
            continue
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (64, 255, 64),
            2,
            cv2.LINE_AA,
        )
        y += 28


def main():
    args = build_parser().parse_args()
    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("--smoothing must be in the range [0.0, 1.0).")

    backend = create_backend_from_args(args)
    backend_name = backend.describe()

    backend_api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera_index, backend_api)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_buffer = deque(maxlen=args.clip_length)
    last_result = None
    last_latency_ms = None
    frame_index = 0
    smoothed_probs = None

    print(f"Running backend: {backend_name}")
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
    print("Opening webcam window. Press q or esc to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            frame_buffer.append(frame.copy())

            if len(frame_buffer) == args.clip_length and frame_index % args.infer_every == 0:
                started = time.perf_counter()
                result = backend.infer_clip(list(frame_buffer), args.top_k)
                last_latency_ms = (time.perf_counter() - started) * 1000.0

                if isinstance(backend, VanillaASLBackend):
                    result, smoothed_probs = smooth_vanilla_result(
                        result=result,
                        backend=backend,
                        top_k=args.top_k,
                        smoothing=args.smoothing,
                        previous_probs=smoothed_probs,
                    )

                last_result = result

            display_frame = cv2.flip(frame, 1) if args.mirror_display else frame.copy()
            draw_overlay(
                display_frame,
                backend_name=backend_name,
                clip_fill=len(frame_buffer),
                clip_length=args.clip_length,
                latency_ms=last_latency_ms,
                result=last_result,
            )
            cv2.imshow(args.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
