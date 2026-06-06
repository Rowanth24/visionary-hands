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
    code_dir = Path(__file__).resolve().parent  # eval/ — benchmark/ and preprocess/ live here post-merge
    parser = argparse.ArgumentParser(
        description="Build a mixed-source fine-tuning set using overlapping WLASL and Aslense labels."
    )
    parser.add_argument(
        "--wlasl-metadata",
        default=str(code_dir / "benchmark" / "wlasl_samples.json"),
        help="Path to the local WLASL metadata JSON (supply it here; not shipped in-repo).",
    )
    parser.add_argument(
        "--labels-file",
        default=str(code_dir / "preprocess" / "wlasl_class_list.txt"),
        help="Path to the 1000-class label file.",
    )
    parser.add_argument(
        "--heldout-manifest",
        default=str(code_dir / "benchmark" / "multisource20_manifest.json"),
        help="Manifest of paired benchmark clips to exclude from adaptation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(code_dir / "benchmark" / "cross_source_adaptation"),
        help="Directory to write manifests, cache files, and downloaded videos.",
    )
    parser.add_argument(
        "--wlasl-annotations",
        default=None,
        help="Path to canonical WLASL v0.3 annotations JSON (gloss->instances with signer_id). "
             "Required for --signer-disjoint; the FiftyOne wlasl_samples.json lacks signer ids.",
    )
    parser.add_argument(
        "--signer-disjoint",
        action="store_true",
        help="Split WLASL so no signer appears in both train and val (Gate G2). "
             "ASLense has no signer metadata, so it falls back to clip-disjoint only.",
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


def build_signer_map(annotations_path):
    """video_id (str) -> signer_id, from canonical WLASL v0.3 annotations (gloss->instances)."""
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    signer_map = {}
    for gloss in data:
        for inst in gloss.get("instances", []):
            video_id = str(inst.get("video_id"))
            signer = inst.get("signer_id")
            if video_id and signer is not None:
                signer_map[video_id] = signer
    return signer_map


def attach_signers(wlasl_index, signer_map):
    """Annotate each WLASL index entry with its signer_id (None if unknown)."""
    for entries in wlasl_index.values():
        for entry in entries:
            entry["signer_id"] = signer_map.get(str(entry["video_id"]))
    return wlasl_index


def signer_disjoint_wlasl_split(selected_labels, wlasl_index, train_per_label, val_per_label, seed):
    """Globally partition signers into train vs val (disjoint), then pick per-label quotas.

    Returns (train_picks, val_picks, info) where *_picks are {label: [entries]} and info
    reports the val-signer set, train/val signer counts, and any per-label shortfalls.
    A signer is wholly in train or wholly in val, so no signer crosses the split for ANY
    label. Entries with no signer_id can't be guaranteed disjoint -> placed in train only.
    """
    signers = {}          # signer_id -> [entries across selected labels]
    missing = []          # entries with no signer_id
    for label in selected_labels:
        for entry in wlasl_index.get(label, []):
            sid = entry.get("signer_id")
            if sid is None:
                missing.append(entry)
            else:
                signers.setdefault(sid, []).append(entry)

    signer_ids = list(signers.keys())
    random.Random(f"{seed}:signer-split").shuffle(signer_ids)

    # Greedily designate val signers until each label's val quota is (approximately) met.
    val_need = {label: val_per_label for label in selected_labels}
    val_signers = set()
    for sid in signer_ids:
        covered = {e["label"] for e in signers[sid]}
        if any(val_need[label] > 0 for label in covered):
            val_signers.add(sid)
            for label in covered:
                contributed = sum(1 for e in signers[sid] if e["label"] == label)
                val_need[label] = max(0, val_need[label] - contributed)
        if all(v == 0 for v in val_need.values()):
            break

    train_by_label = {label: [] for label in selected_labels}
    val_by_label = {label: [] for label in selected_labels}
    for sid, entries in signers.items():
        bucket = val_by_label if sid in val_signers else train_by_label
        for entry in entries:
            bucket[entry["label"]].append(entry)
    for entry in missing:  # unknown signer -> train only (cannot guarantee disjointness)
        train_by_label[entry["label"]].append(entry)

    def trim(by_label, n, tag):
        out = {}
        for label, entries in by_label.items():
            ordered = list(entries)
            random.Random(f"{seed}:{tag}:{label}").shuffle(ordered)
            out[label] = ordered[:n]
        return out

    train_picks = trim(train_by_label, train_per_label, "train")
    val_picks = trim(val_by_label, val_per_label, "val")

    shortfalls = {
        label: {"train": len(train_picks[label]), "val": len(val_picks[label])}
        for label in selected_labels
        if len(train_picks[label]) < train_per_label or len(val_picks[label]) < val_per_label
    }
    info = {
        "val_signers": sorted(val_signers),
        "train_signer_count": len(signers) - len(val_signers),
        "val_signer_count": len(val_signers),
        "missing_signer_entries": len(missing),
        "shortfalls": shortfalls,
    }
    return train_picks, val_picks, info


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


def build_split_samples(selected_labels, wlasl_index, aslense_index, args, signer_disjoint_picks=None):
    train_samples = []
    val_samples = []
    train_count = {"wlasl": args.wlasl_train_per_label, "aslense": args.aslense_train_per_label}
    val_count = {"wlasl": args.wlasl_val_per_label, "aslense": args.aslense_val_per_label}

    train_index = 1
    val_index = 1

    for label in selected_labels:
        if signer_disjoint_picks is not None:
            wlasl_train = signer_disjoint_picks["train"].get(label, [])
            wlasl_val = signer_disjoint_picks["val"].get(label, [])
        else:
            wlasl_entries = choose_samples(
                wlasl_index[label],
                train_count["wlasl"] + val_count["wlasl"],
                seed=f"{args.seed}:wlasl:{label}",
            )
            wlasl_train = wlasl_entries[:train_count["wlasl"]]
            wlasl_val = wlasl_entries[train_count["wlasl"]:train_count["wlasl"] + val_count["wlasl"]]

        aslense_entries = choose_samples(
            aslense_index[label],
            train_count["aslense"] + val_count["aslense"],
            seed=f"{args.seed}:aslense:{label}",
        )
        aslense_train = aslense_entries[:train_count["aslense"]]
        aslense_val = aslense_entries[train_count["aslense"]:train_count["aslense"] + val_count["aslense"]]

        for source_name, train_pick, val_pick in (
            ("wlasl", wlasl_train, wlasl_val),
            ("aslense", aslense_train, aslense_val),
        ):
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

    signer_map = None
    if args.signer_disjoint:
        if not args.wlasl_annotations:
            raise RuntimeError("--signer-disjoint requires --wlasl-annotations (canonical WLASL v0.3 JSON with signer_id).")
        signer_map = build_signer_map(args.wlasl_annotations)
        attach_signers(wlasl_index, signer_map)
        print(f"Loaded {len(signer_map)} video_id->signer_id mappings for signer-disjoint splitting.")

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

    signer_disjoint_picks = None
    signer_split_info = None
    if args.signer_disjoint:
        train_picks, val_picks, signer_split_info = signer_disjoint_wlasl_split(
            selected_labels, wlasl_index, args.wlasl_train_per_label, args.wlasl_val_per_label, args.seed,
        )
        # Invariant: no signer in both train and val (verify via video_id -> signer).
        train_signers = {signer_map.get(str(e["video_id"])) for es in train_picks.values() for e in es}
        val_signers = {signer_map.get(str(e["video_id"])) for es in val_picks.values() for e in es}
        overlap = (train_signers & val_signers) - {None}
        if overlap:
            raise RuntimeError(f"Signer-disjoint invariant violated: signers in both splits: {sorted(overlap)}")
        signer_disjoint_picks = {"train": train_picks, "val": val_picks}
        print(f"Signer-disjoint WLASL split: {signer_split_info['train_signer_count']} train signers, "
              f"{signer_split_info['val_signer_count']} val signers; "
              f"{signer_split_info['missing_signer_entries']} entries had no signer_id (placed in train).")
        if signer_split_info["shortfalls"]:
            print(f"⚠️  per-label shortfalls (fewer than requested due to signer-disjoint constraint): "
                  f"{signer_split_info['shortfalls']}")
        print("⚠️  ASLense has no signer metadata — it is clip-disjoint only, not signer-disjoint.")

    train_samples, val_samples = build_split_samples(
        selected_labels, wlasl_index, aslense_index, args, signer_disjoint_picks=signer_disjoint_picks,
    )

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
        "signer_disjoint": bool(args.signer_disjoint),
        "signer_split": signer_split_info,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
