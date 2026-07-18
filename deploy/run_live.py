#!/usr/bin/env python3
"""Live pipeline: frame stream -> word model -> abstain -> fingerspelling -> text/TTS.

The streaming counterpart of run_demo.py: same models, same canonical
preprocessing (LandmarkBackend), same routing hook — but over a sliding window
on a webcam or video-file stream, with optional speech output (pipeline/tts.py).

Decision lines (one per evaluated window) are the machine-readable contract
tested by tests/test_run_live_clip.py:
  [t=  1.60s] WORD drink (0.87)
  [t=  3.20s] LETTER A (0.93)
  [t=  4.80s] ABSTAIN (low confidence)

Usage (from repo root, after `source env.sh`; needs mediapipe+cv2+torch):
  python deploy/run_live.py --source webcam            # live camera, speaks
  python deploy/run_live.py --source path/to/clip.mp4 --no-tts
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Live ASL: word model + abstention + fingerspelling fallback.")
    p.add_argument("--source", required=True, help="'webcam' (device 0), a device index, or a video path.")
    p.add_argument("--model", default="models/landmark/checkpoints/model.ts")
    p.add_argument("--labels", default="models/landmark/checkpoints/labels.json")
    p.add_argument("--config", default=None, help="Runtime config JSON (default: config.json next to model).")
    p.add_argument("--abstention", default="models/landmark/checkpoints/abstention.json",
                   help="Word-model abstention thresholds JSON (falls back to CLI thresholds if missing).")
    p.add_argument("--word-conf-threshold", type=float, default=0.5)
    p.add_argument("--word-margin", type=float, default=0.0)
    p.add_argument("--window", type=int, default=48, help="Frames per evaluated window.")
    p.add_argument("--stride", type=int, default=24, help="Frames between evaluations.")
    p.add_argument("--fs-model", default="models/fingerspelling/checkpoints/mlp.pt")
    p.add_argument("--fs-abstention", default="models/fingerspelling/checkpoints/abstention.json")
    p.add_argument("--tts", dest="tts", action="store_true", default=True)
    p.add_argument("--no-tts", dest="tts", action="store_false")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_word_thresholds(args):
    """Thresholds from the abstention JSON when present, else CLI values (with a warning)."""
    path = REPO_ROOT / args.abstention
    if path.exists():
        data = json.loads(path.read_text())
        return (float(data.get("conf_threshold", args.word_conf_threshold)),
                float(data.get("margin_threshold", args.word_margin)))
    print(f"[warn] no word abstention file at {args.abstention}; "
          f"using --word-conf-threshold={args.word_conf_threshold}", file=sys.stderr)
    return args.word_conf_threshold, args.word_margin


def open_capture(source):
    import cv2
    if source == "webcam" or source.isdigit():
        cap = cv2.VideoCapture(0 if source == "webcam" else int(source))
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    return cap


def frame_stream(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def evaluate_window(frames, t, word, word_policy, fs, tts):
    """One decision for one window: WORD, LETTER, or ABSTAIN. Returns the printed line."""
    from models.fingerspelling.recognizer import should_fingerspell

    result = word.infer_clip(frames, top_k=5)
    if not should_fingerspell(result.candidates, word_policy, manual_spell_mode=False):
        top = result.candidates[0]
        line = f"[t={t:6.2f}s] WORD {top.text} ({top.score:.2f})"
        print(line, flush=True)
        if tts:
            tts.speak(top.text)
        return line

    letter, conf, info = fs.predict_clip(frames, static_image_mode=False)
    if letter is not None:
        line = f"[t={t:6.2f}s] LETTER {letter} ({conf:.2f})"
        print(line, flush=True)
        if tts:
            tts.speak(letter)
    else:
        line = f"[t={t:6.2f}s] ABSTAIN ({info.get('reason', 'not confident')})"
        print(line, flush=True)
    return line


def main():
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "eval"))
    from asl_backends import LandmarkBackend
    from abstention import AbstentionPolicy
    from models.fingerspelling.recognizer import FingerspellingRecognizer
    from pipeline.tts import create_tts

    conf_threshold, margin = load_word_thresholds(args)
    word = LandmarkBackend(model_path=str(REPO_ROOT / args.model), labels_path=str(REPO_ROOT / args.labels),
                           device_name=args.device, config_path=args.config)
    word_policy = AbstentionPolicy(conf_threshold=conf_threshold, margin_threshold=margin)
    fs = FingerspellingRecognizer(str(REPO_ROOT / args.fs_model), str(REPO_ROOT / args.fs_abstention),
                                  device=args.device)
    tts = create_tts() if args.tts else None
    print(f"[run_live] model={word.describe()} conf>={conf_threshold} margin>={margin} "
          f"window={args.window} stride={args.stride} tts={tts.name if tts else 'off'}", file=sys.stderr)

    cap = open_capture(args.source)
    import cv2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if fps > 1 else 30.0

    buffer, since_eval, frame_idx, decisions = [], 0, 0, 0
    try:
        for frame in frame_stream(cap):
            frame_idx += 1
            buffer.append(frame)
            buffer = buffer[-args.window:]
            since_eval += 1
            if len(buffer) < min(args.window, 8) or since_eval < args.stride:
                continue
            since_eval = 0
            evaluate_window(list(buffer), frame_idx / fps, word, word_policy, fs, tts)
            decisions += 1
    except KeyboardInterrupt:
        print("[run_live] interrupted", file=sys.stderr)
    finally:
        cap.release()

    # Short clips can end before a full stride elapses — evaluate what we have.
    if decisions == 0 and buffer:
        evaluate_window(list(buffer), frame_idx / fps, word, word_policy, fs, tts)
        decisions += 1

    if decisions == 0:
        print("[run_live] no frames read from source", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
