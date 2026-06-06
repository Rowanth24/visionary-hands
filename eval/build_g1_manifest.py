#!/usr/bin/env python3
"""Gate G1 — build/append the benchmark manifest from recorded clips, with validation.

Scans a source subdirectory of clips (e.g. your phone recordings), parses
label/signer/variant, validates each clip (openable, sane length, optional hand
check), and writes or merges into the G1 manifest consumed by benchmark_g1.py.

Two accepted layouts under <video-dir>/<source>/ :
  (a) class subfolders:  <source>/<word>/<anything>.mp4
  (b) flat filenames:    <source>/<source>_<word>_<signer>_<variant>_<take>.mp4
                         e.g. phone_drink_s1_v1_01.mp4   (signer 'sN', variant 'vN')

Usage:
  python eval/build_g1_manifest.py \
    --video-dir eval/benchmark/g1/videos \
    --source phone \
    --vocab bowling,computer,deaf,drink,yes \
    --out-manifest eval/benchmark/g1/g1_manifest.json \
    --merge --check-hands
"""
import argparse
import json
import re
from pathlib import Path

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
SIGNER_RE = re.compile(r"^s\d+$", re.IGNORECASE)
VARIANT_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Build/append the G1 benchmark manifest from recorded clips.")
    p.add_argument("--video-dir", required=True, help="Base dir; local_video paths are relative to this.")
    p.add_argument("--source", required=True, help="Source name AND subdir under --video-dir to scan (e.g. phone, wlasl).")
    p.add_argument("--vocab", required=True, help="Comma-separated words or a path to a labels file.")
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--merge", action="store_true", help="Append to an existing manifest (dedupe by local_video).")
    p.add_argument("--signer-default", default="unknown")
    p.add_argument("--variant-default", default="v1")
    p.add_argument("--min-frames", type=int, default=8, help="Warn if a clip has fewer frames.")
    p.add_argument("--max-frames", type=int, default=240, help="Warn if a clip has more frames (~8s @30fps).")
    p.add_argument("--check-hands", action="store_true",
                   help="Run MediaPipe on sampled frames; warn if no hand is detected (needs mediapipe==0.10.9).")
    return p.parse_args()


def resolve_vocab(arg):
    path = Path(arg)
    if path.exists():
        if path.suffix == ".json":
            return [str(x) for x in json.loads(path.read_text(encoding="utf-8"))]
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [t.strip() for t in arg.split(",") if t.strip()]


def parse_flat_name(stem, vocab_norm):
    """Return (label, signer, variant) from a flat filename, or (None,..) if no vocab word found."""
    tokens = stem.split("_")
    label = signer = variant = None
    for tok in tokens:
        if tok.lower() in vocab_norm and label is None:
            label = vocab_norm[tok.lower()]
        elif SIGNER_RE.match(tok) and signer is None:
            signer = tok.lower()
        elif VARIANT_RE.match(tok) and variant is None:
            variant = tok.lower()
    return label, signer, variant


def validate_clip(path, min_frames, max_frames, check_hands):
    """Return (ok, frame_count, warnings[])."""
    import cv2

    warnings = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, ["cannot open"]
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    n = len(frames)
    if n == 0:
        return False, 0, ["no decodable frames"]
    if n < min_frames:
        warnings.append(f"short ({n} frames)")
    if n > max_frames:
        warnings.append(f"long ({n} frames)")

    if check_hands:
        try:
            import mediapipe as mp
            idxs = [0, n // 2, n - 1]
            holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5)
            hand_seen = False
            for i in idxs:
                rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                if res.left_hand_landmarks or res.right_hand_landmarks:
                    hand_seen = True
                    break
            holistic.close()
            if not hand_seen:
                warnings.append("NO HAND detected in sampled frames")
        except Exception as exc:  # noqa: BLE001 — validation is best-effort
            warnings.append(f"hand-check skipped ({exc})")
    return True, n, warnings


def main():
    args = parse_args()
    vocab = resolve_vocab(args.vocab)
    vocab_norm = {w.lower(): w for w in vocab}
    video_dir = Path(args.video_dir)
    scan_dir = video_dir / args.source
    if not scan_dir.is_dir():
        raise SystemExit(f"Source dir not found: {scan_dir}")

    clips = sorted(p for p in scan_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        raise SystemExit(f"No video clips under {scan_dir}")

    new_samples = []
    skipped = []
    for clip in clips:
        rel = clip.relative_to(video_dir).as_posix()
        # Layout (a): parent dir is a vocab word.
        if clip.parent.name.lower() in vocab_norm:
            label = vocab_norm[clip.parent.name.lower()]
            _, signer, variant = parse_flat_name(clip.stem, vocab_norm)
        else:  # Layout (b): parse from filename.
            label, signer, variant = parse_flat_name(clip.stem, vocab_norm)
        if label is None:
            skipped.append((rel, "no vocab word in path/name"))
            continue

        ok, n_frames, warns = validate_clip(clip, args.min_frames, args.max_frames, args.check_hands)
        if not ok:
            skipped.append((rel, "; ".join(warns)))
            continue
        if warns:
            print(f"  WARN {rel}: {'; '.join(warns)}")

        new_samples.append({
            "label": label,
            "source": args.source,
            "signer_id": signer or args.signer_default,
            "variant_id": variant or args.variant_default,
            "local_video": rel,
            "frame_count": n_frames,
        })

    # Merge / write.
    out = Path(args.out_manifest)
    existing = []
    vocab_out = vocab
    if args.merge and out.exists():
        prior = json.loads(out.read_text(encoding="utf-8"))
        existing = prior.get("samples", [])
        vocab_out = prior.get("vocab", vocab)
    seen = {s["local_video"] for s in existing}
    merged = list(existing) + [s for s in new_samples if s["local_video"] not in seen]
    for i, s in enumerate(merged):
        s["sample_index"] = i

    manifest = {
        "name": "g1_benchmark",
        "vocab": vocab_out,
        "sources": sorted({s["source"] for s in merged}),
        "samples": merged,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Coverage report.
    by_word_source = {}
    for s in merged:
        by_word_source.setdefault((s["label"], s["source"]), 0)
        by_word_source[(s["label"], s["source"])] += 1
    print(f"\nManifest: {len(merged)} clips across sources {manifest['sources']} -> {out}")
    print("Coverage (word x source):")
    for w in vocab_out:
        row = "  ".join(f"{src}={by_word_source.get((w, src), 0)}" for src in manifest["sources"])
        print(f"  {w:<12} {row}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for rel, why in skipped[:20]:
            print(f"  - {rel}: {why}")


if __name__ == "__main__":
    main()
