#!/usr/bin/env python3
"""End-to-end demo: closed-vocab word model -> abstain -> fingerspelling fallback.

One runnable path that ties the two models together via the routing hook:
  1. Run the closed-vocab word model (LandmarkBackend) on a word clip.
  2. If it's confident -> output the word.
  3. If it abstains (likely OOV: a name/med) OR --manual-spell -> fingerspelling mode:
     run FingerspellingRecognizer on the spelling input -> letter(s) or "not confident".

This exercises LandmarkBackend + eval/abstention.AbstentionPolicy + should_fingerspell +
FingerspellingRecognizer in a single process. Foreground; .venv-oracle (torch + mediapipe 0.10.9).

Usage (from repo root, after `source env.sh`):
  python deploy/run_demo.py \
    --word-clip eval/benchmark/g1/videos/wlasl/drink/17730.mp4 \
    --landmark-model models/landmark/checkpoints/model.ts \
    --landmark-labels models/landmark/checkpoints/labels.json
  # force spelling on a folder of letter images (or a video):
  python deploy/run_demo.py --word-clip <clip> --manual-spell \
    --spell-input data/fingerspelling/raw/asl-alphabet-test/A \
    --fs-model models/fingerspelling/checkpoints/mlp.pt \
    --fs-abstention models/fingerspelling/checkpoints/abstention.json
"""
import argparse
from pathlib import Path

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="Word model + fingerspelling fallback demo.")
    p.add_argument("--word-clip", required=True, help="Video clip of a (closed-vocab) sign.")
    p.add_argument("--landmark-model", default="models/landmark/checkpoints/model.ts")
    p.add_argument("--landmark-labels", default="models/landmark/checkpoints/labels.json")
    p.add_argument("--landmark-config", default=None)
    p.add_argument("--word-conf-threshold", type=float, default=0.5, help="Word-model abstain threshold.")
    p.add_argument("--word-margin", type=float, default=0.0)
    p.add_argument("--manual-spell", action="store_true", help="Force fingerspelling mode.")
    p.add_argument("--spell-input", default=None, help="Folder of letter images OR a video for spelling.")
    p.add_argument("--fs-model", default="models/fingerspelling/checkpoints/mlp.pt")
    p.add_argument("--fs-abstention", default="models/fingerspelling/checkpoints/abstention.json")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_spell_frames(spell_input):
    """A folder of images -> list of frames (static images); a video -> decoded frames."""
    import cv2
    path = Path(spell_input)
    if path.is_dir():
        imgs = [cv2.imread(str(f)) for f in sorted(path.iterdir()) if f.suffix.lower() in IMG_SUFFIXES][:12]
        return [i for i in imgs if i is not None], True   # static_image_mode=True
    from infer_videos import read_video_frames
    return read_video_frames(path), False


def main():
    args = parse_args()
    from asl_backends import LandmarkBackend
    from abstention import AbstentionPolicy
    from infer_videos import read_video_frames
    from models.fingerspelling.recognizer import FingerspellingRecognizer, should_fingerspell

    # 1. Closed-vocab word model
    word = LandmarkBackend(model_path=args.landmark_model, labels_path=args.landmark_labels,
                           device_name=args.device, config_path=args.landmark_config)
    frames = read_video_frames(Path(args.word_clip))
    result = word.infer_clip(frames, top_k=5)
    word_policy = AbstentionPolicy(conf_threshold=args.word_conf_threshold, margin_threshold=args.word_margin)

    print(f"[word model] top: {result.candidates[0].text} ({result.candidates[0].score:.2f})  "
          f"vocab={word.describe()}")

    # 2/3. Route
    spell = should_fingerspell(result.candidates, word_policy, manual_spell_mode=args.manual_spell)
    if not spell:
        decided, _reason = word_policy.decide(result.candidates)
        print(f"\n==> WORD: {decided}  (confident; no spelling needed)")
        return

    reason = "manual spell-mode" if args.manual_spell else "word model not confident (likely a name/med)"
    print(f"\n[router] {reason} -> FINGERSPELLING mode")
    if not args.spell_input:
        print("==> please fingerspell the word  (re-run with --spell-input <frames/video>)")
        return

    fs = FingerspellingRecognizer(args.fs_model, args.fs_abstention, device=args.device)
    spell_frames, static = load_spell_frames(args.spell_input)
    letter, conf, info = fs.predict_clip(spell_frames, static_image_mode=static)
    if letter is None:
        print(f"==> LETTER: not confident, please repeat  ({info['reason']}; {info['frames_with_hand']} frames w/ hand)")
    else:
        print(f"==> LETTER: {letter}  ({conf:.2f})")


if __name__ == "__main__":
    main()
