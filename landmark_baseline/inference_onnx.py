"""ONNX Runtime inference helpers for landmark-based isolated-sign models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .extraction import extract_video_landmarks
from .features import append_frame_deltas_np
from .prediction import PredictionResult
from .preprocess import preprocess_landmark_sequence


def load_labels(labels_path: Path) -> list[str]:
    return [str(label) for label in json.loads(labels_path.read_text(encoding='utf-8'))]


def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])


def prepare_onnx_inputs(
    landmarks: np.ndarray,
    target_frames: int = 64,
    xy_only: bool = True,
    append_deltas: bool = False,
    layout: str = 'dominant_hand',
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    processed = preprocess_landmark_sequence(landmarks, target_frames=target_frames, layout=layout)
    frames = processed.frames[..., :2] if xy_only else processed.frames
    x = frames.reshape(1, frames.shape[0], -1).astype(np.float32)
    if append_deltas:
        x = append_frame_deltas_np(x)
    valid_mask = (processed.non_empty_frame_idxs >= 0).reshape(1, -1)
    meta = {
        'dominant_hand': processed.dominant_hand,
        'layout': processed.layout,
        'source_num_frames': processed.source_num_frames,
        'kept_num_frames': processed.kept_num_frames,
    }
    return x, valid_mask, meta


def predict_from_landmarks_onnx(
    session: ort.InferenceSession,
    labels: list[str],
    landmarks: np.ndarray,
    target_frames: int = 64,
    xy_only: bool = True,
    append_deltas: bool = False,
    layout: str = 'dominant_hand',
    topk: int = 3,
) -> PredictionResult:
    x, valid_mask, meta = prepare_onnx_inputs(
        landmarks,
        target_frames=target_frames,
        xy_only=xy_only,
        append_deltas=append_deltas,
        layout=layout,
    )
    input_x = session.get_inputs()[0].name
    input_mask = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_x: x, input_mask: valid_mask})[0][0]
    logits = logits.astype(np.float64)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    top_indices = np.argsort(-probs)[: max(1, min(int(topk), len(labels)))]
    topk_rows = [
        {'label': labels[int(idx)], 'confidence': float(probs[int(idx)])}
        for idx in top_indices.tolist()
    ]
    return PredictionResult(
        predicted_label=topk_rows[0]['label'],
        confidence=float(topk_rows[0]['confidence']),
        topk=topk_rows,
        dominant_hand=str(meta['dominant_hand']),
        source_num_frames=int(meta['source_num_frames']),
        kept_num_frames=int(meta['kept_num_frames']),
    )


def predict_video_onnx(
    session: ort.InferenceSession,
    labels: list[str],
    video_path: Path,
    frame_stride: int = 1,
    target_frames: int = 64,
    xy_only: bool = True,
    append_deltas: bool = False,
    layout: str = 'dominant_hand',
    topk: int = 3,
) -> PredictionResult:
    landmarks = extract_video_landmarks(video_path, frame_stride=max(1, frame_stride))
    return predict_from_landmarks_onnx(
        session=session,
        labels=labels,
        landmarks=landmarks,
        target_frames=target_frames,
        xy_only=xy_only,
        append_deltas=append_deltas,
        layout=layout,
        topk=topk,
    )
