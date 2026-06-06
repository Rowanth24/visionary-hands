#!/usr/bin/env python3
"""Tests for signer-disjoint WLASL splitting (Gate G-D / G2 enabler).

Pure stdlib — runs in any venv.  python tests/test_signer_disjoint.py
"""
from __future__ import annotations

from prepare_cross_source_finetune_data import (
    attach_signers,
    build_signer_map,
    signer_disjoint_wlasl_split,
)


def _entry(label, video_id, signer_id=None):
    e = {"label": label, "video_id": str(video_id), "filepath": f"{video_id}.mp4"}
    if signer_id is not None:
        e["signer_id"] = signer_id
    return e


def _signers_in(picks, video_to_signer):
    return {video_to_signer.get(e["video_id"]) for es in picks.values() for e in es} - {None}


def test_no_signer_crosses_train_val():
    # 2 labels, several signers; each signer signs both labels (worst case for disjointness).
    video_to_signer = {}
    wlasl_index = {"a": [], "b": []}
    vid = 0
    for signer in range(1, 9):           # 8 signers
        for label in ("a", "b"):
            for _ in range(2):           # 2 clips per (signer,label)
                wlasl_index[label].append(_entry(label, vid, signer))
                video_to_signer[str(vid)] = signer
                vid += 1

    train, val, info = signer_disjoint_wlasl_split(
        ["a", "b"], wlasl_index, train_per_label=3, val_per_label=2, seed=42,
    )
    train_signers = _signers_in(train, video_to_signer)
    val_signers = _signers_in(val, video_to_signer)
    assert not (train_signers & val_signers), f"signer leak across splits: {train_signers & val_signers}"
    # quotas respected (never exceed requested)
    for label in ("a", "b"):
        assert len(train[label]) <= 3 and len(val[label]) <= 2
    assert info["val_signer_count"] >= 1 and info["train_signer_count"] >= 1


def test_unknown_signer_entries_go_to_train_only():
    wlasl_index = {"a": [_entry("a", 1, None), _entry("a", 2, 5), _entry("a", 3, 6)]}
    video_to_signer = {"2": 5, "3": 6}  # video 1 has no signer
    train, val, info = signer_disjoint_wlasl_split(
        ["a"], wlasl_index, train_per_label=3, val_per_label=1, seed=1,
    )
    # the unknown-signer clip (video_id "1") must never appear in val
    val_ids = {e["video_id"] for es in val.values() for e in es}
    assert "1" not in val_ids
    assert info["missing_signer_entries"] == 1


def test_build_signer_map_and_attach():
    annotations = [
        {"gloss": "drink", "instances": [
            {"video_id": "17730", "signer_id": 12},
            {"video_id": "17713", "signer_id": 7},
        ]},
    ]
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ann.json"
        p.write_text(json.dumps(annotations))
        m = build_signer_map(p)
    assert m == {"17730": 12, "17713": 7}

    wlasl_index = {"drink": [_entry("drink", "17730"), _entry("drink", "99999")]}
    attach_signers(wlasl_index, m)
    assert wlasl_index["drink"][0]["signer_id"] == 12
    assert wlasl_index["drink"][1]["signer_id"] is None  # not in map


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("OK — signer-disjoint splitting holds the invariant.")
