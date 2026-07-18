#!/usr/bin/env python3
"""Tests for pipeline/sources adapters (plan Task D) — fully offline, synthetic data.

The contract under test: every source normalizes to the entry shape the existing
signer-disjoint splitter consumes, and gated sources fail loudly with human
instructions instead of half-working.

Run:  .venv-oracle/bin/pytest tests/test_multisource_sources.py -q  (after `source env.sh`)
"""
from __future__ import annotations

import json

import pytest

from pipeline.sources import NeedsManualStep
from pipeline.sources import asl_citizen, msasl

ENTRY_KEYS = {"label", "video_id", "filepath", "signer_id", "source"}


def _write_msasl_split(tmp_path, split, rows):
    (tmp_path / f"MSASL_{split}.json").write_text(json.dumps(rows), encoding="utf-8")


def _msasl_row(text, signer, url="https://www.youtube.com/watch?v=abc123"):
    return {"clean_text": text, "signer_id": signer, "url": url,
            "start_time": 0.0, "end_time": 2.5, "box": [0, 0, 1, 1]}


def test_msasl_load_index_normalizes_to_shared_entry_shape(tmp_path):
    # Arrange
    _write_msasl_split(tmp_path, "train", [_msasl_row("Drink", 7)])

    # Act
    entries = msasl.load_index(tmp_path, splits=("train",))

    # Assert
    assert len(entries) == 1
    e = entries[0]
    assert ENTRY_KEYS <= set(e)
    assert e["label"] == "drink"                    # lowercased
    assert e["signer_id"] == "msasl_7"              # source-prefixed, split-safe
    assert e["source"] == "msasl"
    assert e["filepath"].endswith(".mp4")


def test_msasl_filter_to_vocab_is_case_insensitive(tmp_path):
    # Arrange
    _write_msasl_split(tmp_path, "train",
                       [_msasl_row("drink", 1), _msasl_row("balloon", 2)])
    entries = msasl.load_index(tmp_path, splits=("train",))

    # Act
    kept = msasl.filter_to_vocab(entries, ["DRINK", "yes"])

    # Assert
    assert [e["label"] for e in kept] == ["drink"]


def test_asl_citizen_load_index_reads_split_csvs(tmp_path):
    # Arrange
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.csv").write_text(
        "Participant ID,Video file,Gloss\nP1,P1_DRINK_001.mp4,DRINK\n", encoding="utf-8")

    # Act
    entries = asl_citizen.load_index(tmp_path, splits=("train",))

    # Assert
    e = entries[0]
    assert ENTRY_KEYS <= set(e)
    assert e["label"] == "drink"
    assert e["signer_id"] == "aslcitizen_P1"
    assert e["filepath"] == "videos/P1_DRINK_001.mp4"


def test_asl_citizen_missing_root_raises_manual_step_with_instructions(tmp_path):
    # Act / Assert: message must carry the human instruction sheet
    with pytest.raises(NeedsManualStep) as exc:
        asl_citizen.load_index(tmp_path / "nope")
    assert "42.8 GB" in str(exc.value)
    assert "microsoft.com" in str(exc.value)


def test_asl_citizen_download_is_always_a_manual_step(tmp_path):
    with pytest.raises(NeedsManualStep):
        asl_citizen.download_subset([], tmp_path)


def test_asl_citizen_unexpected_csv_schema_fails_loudly(tmp_path):
    # Arrange: column drift in a future archive version must not silently mis-parse
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.csv").write_text("participant,file,word\nP1,x.mp4,drink\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError):
        asl_citizen.load_index(tmp_path, splits=("train",))


def test_normalized_entries_feed_existing_signer_disjoint_splitter(tmp_path):
    # Arrange: two sources' entries, multiple signers per label
    from prepare_cross_source_finetune_data import signer_disjoint_wlasl_split

    _write_msasl_split(tmp_path, "train",
                       [_msasl_row("drink", s) for s in range(6)])
    entries = msasl.load_index(tmp_path, splits=("train",))
    by_label = {"drink": entries}                   # the splitter's index shape

    # Act
    train_picks, val_picks, _info = signer_disjoint_wlasl_split(
        ["drink"], by_label, train_per_label=3, val_per_label=1, seed=7)

    # Assert: no signer appears in both train and val
    train_signers = {e["signer_id"] for rows in train_picks.values() for e in rows}
    val_signers = {e["signer_id"] for rows in val_picks.values() for e in rows}
    assert train_signers and val_signers
    assert not (train_signers & val_signers)
