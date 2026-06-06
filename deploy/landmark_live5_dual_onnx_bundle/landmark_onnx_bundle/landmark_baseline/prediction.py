"""Shared prediction result types for landmark inference backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str
    confidence: float
    topk: list[dict[str, float | str]]
    dominant_hand: str
    source_num_frames: int
    kept_num_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            'predicted_label': self.predicted_label,
            'confidence': self.confidence,
            'topk': self.topk,
            'dominant_hand': self.dominant_hand,
            'source_num_frames': self.source_num_frames,
            'kept_num_frames': self.kept_num_frames,
        }
