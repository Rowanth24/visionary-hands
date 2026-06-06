"""Feature helpers for landmark model training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def append_frame_deltas_np(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError(f'Expected [N,T,F] or [T,F], got shape {arr.shape}')
    deltas = np.diff(arr, axis=1, prepend=arr[:, :1])
    out = np.concatenate([arr, deltas.astype(np.float32)], axis=-1)
    return out[0] if squeeze else out


def append_frame_deltas_torch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f'Expected [N,T,F], got shape {tuple(x.shape)}')
    deltas = torch.diff(x, dim=1, prepend=x[:, :1])
    return torch.cat([x, deltas], dim=-1)


def load_runtime_config(config_path: Path | None) -> dict[str, object]:
    if config_path is None or not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding='utf-8'))
