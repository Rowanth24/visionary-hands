"""Offline inference helpers for landmark-based isolated-sign models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .extraction import extract_video_landmarks
from .features import append_frame_deltas_np
from .prediction import PredictionResult
from .preprocess import preprocess_landmark_sequence


def resolve_device(name: str = 'auto') -> torch.device:
    if name == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(name)


def load_labels(labels_path: Path) -> list[str]:
    return [str(label) for label in json.loads(labels_path.read_text(encoding='utf-8'))]


def prepare_model_inputs(
    landmarks: np.ndarray,
    target_frames: int = 64,
    xy_only: bool = True,
    append_deltas: bool = False,
    layout: str = 'dominant_hand',
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    processed = preprocess_landmark_sequence(landmarks, target_frames=target_frames, layout=layout)
    frames = processed.frames[..., :2] if xy_only else processed.frames
    flat = frames.reshape(1, frames.shape[0], -1).astype(np.float32)
    if append_deltas:
        flat = append_frame_deltas_np(flat)
    x = torch.from_numpy(flat)
    valid_mask = torch.from_numpy((processed.non_empty_frame_idxs >= 0).reshape(1, -1))
    meta = {
        'dominant_hand': processed.dominant_hand,
        'layout': processed.layout,
        'source_num_frames': processed.source_num_frames,
        'kept_num_frames': processed.kept_num_frames,
    }
    return x, valid_mask, meta


def load_torchscript_model(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def predict_from_landmarks(
    model: torch.jit.ScriptModule,
    labels: list[str],
    landmarks: np.ndarray,
    device: torch.device,
    target_frames: int = 64,
    xy_only: bool = True,
    append_deltas: bool = False,
    layout: str = 'dominant_hand',
    topk: int = 3,
) -> PredictionResult:
    x, valid_mask, meta = prepare_model_inputs(
        landmarks,
        target_frames=target_frames,
        xy_only=xy_only,
        append_deltas=append_deltas,
        layout=layout,
    )
    x = x.to(device)
    valid_mask = valid_mask.to(device)
    with torch.no_grad():
        logits = model(x, valid_mask)
        probs = torch.softmax(logits[0], dim=0)
        values, indices = torch.topk(probs, k=min(int(topk), probs.numel()))
    topk_rows = [
        {'label': labels[int(idx)], 'confidence': float(value)}
        for value, idx in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]
    return PredictionResult(
        predicted_label=topk_rows[0]['label'],
        confidence=float(topk_rows[0]['confidence']),
        topk=topk_rows,
        dominant_hand=str(meta['dominant_hand']),
        source_num_frames=int(meta['source_num_frames']),
        kept_num_frames=int(meta['kept_num_frames']),
    )


def predict_video(
    model: torch.jit.ScriptModule,
    labels: list[str],
    video_path: Path,
    device: torch.device,
    frame_stride: int = 1,
    target_frames: int = 64,
    xy_only: bool = True,
    append_deltas: bool = False,
    layout: str = 'dominant_hand',
    topk: int = 3,
) -> PredictionResult:
    landmarks = extract_video_landmarks(video_path, frame_stride=max(1, frame_stride))
    return predict_from_landmarks(
        model=model,
        labels=labels,
        landmarks=landmarks,
        device=device,
        target_frames=target_frames,
        xy_only=xy_only,
        append_deltas=append_deltas,
        layout=layout,
        topk=topk,
    )
