#!/usr/bin/env python3
"""Audit the official WLASL annotation file.

This keeps subset decisions tied to the official `WLASL_v0.3.json` metadata
instead of ad hoc split directories or secondary Kaggle mirrors.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help="Path to WLASL_v0.3.json")
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=None,
        help="Optional label list to inspect inside the official metadata.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[100, 300, 1000, 2000],
        help="Subset sizes to summarize using the official top-K gloss ranking.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=15,
        help="How many top glosses to print in the text summary.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the audit JSON.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected list payload in {path}")
    return rows


def build_class_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    class_rows: list[dict[str, object]] = []
    for entry in rows:
        gloss = str(entry["gloss"])
        instances = list(entry.get("instances", []))
        split_counts = Counter(str(inst.get("split", "")) for inst in instances)
        signer_ids = {int(inst["signer_id"]) for inst in instances if inst.get("signer_id") is not None}
        source_counts = Counter(str(inst.get("source", "")) for inst in instances)
        class_rows.append(
            {
                "gloss": gloss,
                "clips": len(instances),
                "train": int(split_counts.get("train", 0)),
                "val": int(split_counts.get("val", 0)),
                "test": int(split_counts.get("test", 0)),
                "unique_signers": len(signer_ids),
                "top_sources": source_counts.most_common(5),
            }
        )
    class_rows.sort(key=lambda row: (-int(row["clips"]), str(row["gloss"])))
    return class_rows


def summarize_subset(class_rows: list[dict[str, object]], k: int) -> dict[str, object]:
    subset = class_rows[: max(0, min(int(k), len(class_rows)))]
    train_counts = [int(row["train"]) for row in subset]
    val_counts = [int(row["val"]) for row in subset]
    test_counts = [int(row["test"]) for row in subset]
    signer_counts = [int(row["unique_signers"]) for row in subset]
    return {
        "classes": len(subset),
        "clips": sum(int(row["clips"]) for row in subset),
        "train": sum(train_counts),
        "val": sum(val_counts),
        "test": sum(test_counts),
        "median_train_per_class": statistics.median(train_counts) if train_counts else 0,
        "median_val_per_class": statistics.median(val_counts) if val_counts else 0,
        "median_test_per_class": statistics.median(test_counts) if test_counts else 0,
        "median_signers_per_class": statistics.median(signer_counts) if signer_counts else 0,
    }


def inspect_labels(class_rows: list[dict[str, object]], labels: list[str]) -> list[dict[str, object]]:
    by_gloss = {str(row["gloss"]): row for row in class_rows}
    out: list[dict[str, object]] = []
    for label in labels:
        row = by_gloss.get(str(label))
        if row is None:
            out.append({"gloss": str(label), "missing": True})
            continue
        out.append(
            {
                "gloss": str(row["gloss"]),
                "clips": int(row["clips"]),
                "train": int(row["train"]),
                "val": int(row["val"]),
                "test": int(row["test"]),
                "unique_signers": int(row["unique_signers"]),
                "top_sources": row["top_sources"],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args.annotations)
    class_rows = build_class_rows(rows)

    global_split_counts = Counter()
    global_source_counts = Counter()
    global_signer_counts = Counter()
    for entry in rows:
        for inst in entry.get("instances", []):
            global_split_counts[str(inst.get("split", ""))] += 1
            global_source_counts[str(inst.get("source", ""))] += 1
            if inst.get("signer_id") is not None:
                global_signer_counts[int(inst["signer_id"])] += 1

    selected_labels: list[str] = []
    if args.labels_json is not None:
        selected_labels = [str(label) for label in json.loads(args.labels_json.read_text(encoding="utf-8"))]

    report = {
        "annotations": str(args.annotations),
        "num_glosses": len(class_rows),
        "num_instances": sum(int(row["clips"]) for row in class_rows),
        "global_splits": dict(global_split_counts),
        "num_signers": len(global_signer_counts),
        "top_sources": global_source_counts.most_common(15),
        "top_glosses": class_rows[: max(1, args.top_classes)],
        "subsets": {str(k): summarize_subset(class_rows, k) for k in args.topk},
        "selected_labels": inspect_labels(class_rows, selected_labels) if selected_labels else [],
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
