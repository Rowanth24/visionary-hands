#!/usr/bin/env python3
"""Serve-time fingerspelling recognizer + routing hook (Gate F3).

A held handshape -> MediaPipe Hands -> handshape_features (+geometry, standardized) -> MLP
-> aggregate over the held frames -> per-letter abstention -> a letter or "not confident".

Per-letter thresholds (from F2 calibration) encode the ceiling: letters that couldn't reach
precision are set to always-abstain, so the recognizer declines rather than guess them.

Routing: should_fingerspell() returns True when the closed-vocab word model abstains
(reusing eval/abstention.py) or when the clinician flips manual spell-mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.fingerspelling.extraction import extract_hand_landmarks_from_frames
from models.fingerspelling.features import augment_features, handshape_features
from models.fingerspelling.labels import LETTERS

DEFAULT_GLOBAL_THRESHOLD = 0.5


class FingerspellingRecognizer:
    def __init__(self, model_path, abstention_path=None, device="cpu"):
        import torch
        from models.fingerspelling.train_fingerspelling import MLP

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self._torch = torch
        self.letters = ckpt.get("letters", LETTERS)
        self.geometric = bool(ckpt.get("geometric"))
        self.mean = ckpt.get("mean")
        self.std = ckpt.get("std")
        self.model = MLP(ckpt["input_dim"], ckpt["num_classes"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.global_threshold = DEFAULT_GLOBAL_THRESHOLD
        self.per_letter_threshold: dict[str, float] = {}
        if abstention_path and Path(abstention_path).exists():
            cfg = json.loads(Path(abstention_path).read_text(encoding="utf-8"))
            self.global_threshold = float(cfg.get("global_threshold", DEFAULT_GLOBAL_THRESHOLD))
            self.per_letter_threshold = {k: float(v) for k, v in cfg.get("per_letter_threshold", {}).items()}

    # --- feature/model plumbing ---
    def _vectorize(self, landmarks_frame: np.ndarray):
        feat = handshape_features(landmarks_frame)
        if feat is None:
            return None
        x = feat[None, :]
        if self.geometric:
            x = augment_features(x)
            if self.mean is not None:
                x = (x - self.mean) / self.std
        return x.astype(np.float32)

    def _probs(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            logits = self.model(self._torch.from_numpy(x))
            return self._torch.softmax(logits, dim=1).numpy()

    def _threshold_for(self, letter: str) -> float:
        return self.per_letter_threshold.get(letter, self.global_threshold)

    # --- the testable decision core (no MediaPipe) ---
    def decide_from_probs(self, probs: np.ndarray):
        """probs: (n_frames, 24) softmax rows. Returns (letter|None, confidence, reason)."""
        if probs.shape[0] == 0:
            return None, 0.0, "no_hand"
        mean_probs = probs.mean(axis=0)
        idx = int(mean_probs.argmax())
        letter, conf = self.letters[idx], float(mean_probs[idx])
        thr = self._threshold_for(letter)
        if conf >= thr and thr < 1.0:
            return letter, conf, "accept"
        return None, conf, f"abstain(conf={conf:.2f}<thr={thr:.2f})"

    # --- clip-level entry point (isolated letter-at-a-time) ---
    def predict_clip(self, frames, frames_are_bgr=True, static_image_mode=False):
        """frames: list of camera frames of ONE held handshape. Returns (letter|None, confidence, info)."""
        landmarks = extract_hand_landmarks_from_frames(
            frames, frames_are_bgr=frames_are_bgr, static_image_mode=static_image_mode)
        rows = [self._vectorize(lm) for lm in landmarks]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None, 0.0, {"reason": "no_hand", "frames_with_hand": 0}
        probs = np.concatenate([self._probs(r) for r in rows], axis=0)
        letter, conf, reason = self.decide_from_probs(probs)
        return letter, conf, {"reason": reason, "frames_with_hand": len(rows), "frames_total": len(frames)}

    def reliable_letters(self):
        """Letters the recognizer will ever emit (threshold < 1.0) — documents the ceiling."""
        return [l for l in self.letters if self._threshold_for(l) < 1.0]

    def abstain_only_letters(self):
        return [l for l in self.letters if self._threshold_for(l) >= 1.0]


def should_fingerspell(word_candidates, word_policy, manual_spell_mode=False) -> bool:
    """Route to fingerspelling when the word model abstains (likely OOV) or manual toggle is on.

    word_candidates: the closed-vocab backend's ranked candidates (label, score) / dicts.
    word_policy: an eval/abstention.AbstentionPolicy for the word model.
    """
    if manual_spell_mode:
        return True
    label, _reason = word_policy.decide(word_candidates)
    return label is None  # word model declined -> try spelling
