import argparse
import json
import random
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


ASLENSE_PARTS = [f"part_{index}" for index in range(1, 12)]
WLASL_BASE_URL = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/"
ASLENSE_BASE_URL = "https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset/resolve/main/"


def build_parser():
    workspace_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a mixed-source fine-tuning set using overlapping WLASL and Aslense labels."
    )
    parser.add_argument(
        "--wlasl-metadata",
        default=str(workspace_root / "wlasl_samples.json"),
        help="Path to the local WLASL metadata JSON.",
    )
    parser.add_argument(
        "--labels-file",
        default=str(repo_root / "code" / "preprocess" / "wlasl_class_list.txt"),
        help="Path to the 1000-class label file.",
    )
    parser.add_argument(
        "--heldout-manifest",
        default=str(repo_root / "benchmark" / "multisource20_manifest.json"),
        help="Manifest of paired benchmark clips to exclude from adaptation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "benchmark" / "cross_source_adaptation"),
        help="Directory to write manifests, cache files, and downloaded videos.",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=20,
        help="How many overlapping labels to include in the adaptation set.",
    )
    parser.add_argument(
        "--wlasl-train-per-label",
        type=int,
        default=2,
        help="How many WLASL clips per label to place in train.",
    )
    parser.add_argument(
        "--wlasl-val-per-label",
        type=int,
        default=1,
        help="How many WLASL clips per label to place in val.",
    )
    parser.add_argument(
        "--aslense-train-per-label",
        type=int,
        default=2,
        help="How many Aslense clips per label to place in train.",
    )
    parser.add_argument(
        "--aslense-val-per-label",
        type=int,
        default=1,
        help="How many Aslense clips per label to place in val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260421,
        help="Random seed for deterministic selection.",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Refresh the cached Aslense index instead of reusing it.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds for dataset downloads.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="How many times to retry a failed dataset download.",
    )
    return parser


def load_allowed_labels(labels_file):
    labels = set()
    with open(labels_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_text, label = line.split(maxsplit=1)
            if int(idx_text) < 1000:
                labels.add(label.lower())
    return labels


def load_heldout_info(manifest_path):
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    heldout_labels = set(payload.get("paired_labels", []))
    exclude_wlasl_ids = {sample["video_id"] for sample in payload["samples"] if sample.get("source") == "wlasl"}
    exclude_aslense_ids = {sample["video_id"] for sample in payload["samples"] if sample.get("source") == "aslense"}
    return heldout_labels, exclude_wlasl_ids, exclude_aslense_ids


def build_wlasl_index(metadata_path, allowed_labels, heldout_labels, exclude_ids):
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    items = payload["samples"] if isinstance(payload, dict) else payload
    by_label = {}

    for item in items:
        label = item["gloss"]["label"].lower()
        video_id = Path(item["filepath"]).stem
        if label not in allowed_labels or label in heldout_labels or video_id in exclude_ids:
            continue
        by_label.setdefault(label, []).append(
            {
                "source": "wlasl",
                "label": label,
                "filepath": item["filepath"],
                "video_id": video_id,
                "frame_count": item["metadata"].get("total_frame_count"),
                "width": item["metadata"].get("frame_width"),
                "height": item["metadata"].get("frame_height"),
                "duration": item["metadata"].get("duration"),
            }
        )
    return by_label


def build_aslense_index(cache_path, allowed_labels, heldout_labels, exclude_ids, refresh):
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload

    by_label = {}
    for part in ASLENSE_PARTS:
        url = f"https://huggingface.co/api/datasets/ZahidYasinMittha/American-Sign-Language-Dataset/tree/main/{part}?recursive=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        for item in data:
            path = item.get("path", "")
            if not path.lower().endswith(".mp4"):
                continue

            video_id = Path(path).stem
            label = video_id.split("-", 1)[-1].replace("_", " ").lower()
            if label not in allowed_labels or label in heldout_labels or video_id in exclude_ids:
                continue
            by_label.setdefault(label, []).append(
                {
                    "source": "aslense",
                    "label": label,
                    "filepath": path,
                    "video_id": video_id,
                }
            )

    cache_path.write_text(json.dumps(by_label, indent=2), encoding="utf-8")
    return by_label


def choose_labels(wlasl_index, aslense_index, num_labels, counts_needed):
    qualified = []
    for label in sorted(set(wlasl_index) & set(aslense_index)):
        if len(wlasl_index[label]) < counts_needed["wlasl"]:
            continue
        if len(aslense_index[label]) < counts_needed["aslense"]:
            continue
        qualified.append(label)

    qualified.sort(key=lambda label: (len(aslense_index[label]), len(wlasl_index[label]), label), reverse=True)
    return qualified[:num_labels]


def choose_samples(entries, count, seed):
    ordered = list(entries)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:count]


def local_video_name(sample_index, source, label, video_id, suffix):
    safe_label = label.replace(" ", "_")
    return f"{sample_index:03d}_{source}_{safe_label}_{video_id}{suffix}"


def download_video(sample, target, timeout_seconds, retries):
    if target.exists() and target.stat().st_size > 0:
        return

    if sample["source"] == "wlasl":
        url = WLASL_BASE_URL + sample["filepath"]
    else:
        url = ASLENSE_BASE_URL + urllib.parse.quote(sample["filepath"])

    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    temp_path = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                with open(temp_path, "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            temp_path.replace(target)
            return
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            print(f"download retry {attempt}/{retries} failed for {target.name}: {exc}", flush=True)

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_error}")


def build_split_samples(selected_labels, wlasl_index, aslense_index, args):
    train_samples = []
    val_samples = []
    train_count = {"wlasl": args.wlasl_train_per_label, "aslense": args.aslense_train_per_label}
    val_count = {"wlasl": args.wlasl_val_per_label, "aslense": args.aslense_val_per_label}

    train_index = 1
    val_index = 1

    for label in selected_labels:
        wlasl_entries = choose_samples(
            wlasl_index[label],
            train_count["wlasl"] + val_count["wlasl"],
            seed=f"{args.seed}:wlasl:{label}",
        )
        aslense_entries = choose_samples(
            aslense_index[label],
            train_count["aslense"] + val_count["aslense"],
            seed=f"{args.seed}:aslense:{label}",
        )

        for source_name, entries, train_n, val_n in (
            ("wlasl", wlasl_entries, train_count["wlasl"], val_count["wlasl"]),
            ("aslense", aslense_entries, train_count["aslense"], val_count["aslense"]),
        ):
            train_pick = entries[:train_n]
            val_pick = entries[train_n:train_n + val_n]

            for sample in train_pick:
                suffix = Path(sample["filepath"]).suffix or ".mp4"
                enriched = dict(sample)
                enriched["sample_index"] = train_index
                enriched["local_video"] = local_video_name(train_index, source_name, label, sample["video_id"], suffix)
                train_samples.append(enriched)
                train_index += 1

            for sample in val_pick:
                suffix = Path(sample["filepath"]).suffix or ".mp4"
                enriched = dict(sample)
                enriched["sample_index"] = val_index
                enriched["local_video"] = local_video_name(val_index, source_name, label, sample["video_id"], suffix)
                val_samples.append(enriched)
                val_index += 1

    return train_samples, val_samples


def write_manifest(path, seed, labels, samples):
    payload = {
        "seed": seed,
        "count": len(samples),
        "labels": labels,
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = load_allowed_labels(args.labels_file)
    heldout_labels, exclude_wlasl_ids, exclude_aslense_ids = load_heldout_info(args.heldout_manifest)
    wlasl_index = build_wlasl_index(args.wlasl_metadata, allowed_labels, heldout_labels, exclude_wlasl_ids)
    aslense_index = build_aslense_index(
        cache_path=output_dir / "aslense_index.json",
        allowed_labels=allowed_labels,
        heldout_labels=heldout_labels,
        exclude_ids=exclude_aslense_ids,
        refresh=args.refresh_index,
    )

    selected_labels = choose_labels(
        wlasl_index=wlasl_index,
        aslense_index=aslense_index,
        num_labels=args.num_labels,
        counts_needed={
            "wlasl": args.wlasl_train_per_label + args.wlasl_val_per_label,
            "aslense": args.aslense_train_per_label + args.aslense_val_per_label,
        },
    )
    if len(selected_labels) < args.num_labels:
        raise RuntimeError(f"Only found {len(selected_labels)} qualified overlapping labels; requested {args.num_labels}.")

    train_samples, val_samples = build_split_samples(selected_labels, wlasl_index, aslense_index, args)

    for sample in train_samples + val_samples:
        download_video(
            sample,
            videos_dir / sample["local_video"],
            timeout_seconds=args.download_timeout,
            retries=args.download_retries,
        )

    write_manifest(output_dir / "train_manifest.json", args.seed, selected_labels, train_samples)
    write_manifest(output_dir / "val_manifest.json", args.seed, selected_labels, val_samples)

    summary = {
        "selected_labels": selected_labels,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "videos_dir": str(videos_dir.resolve()),
        "heldout_manifest": str(Path(args.heldout_manifest).resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
