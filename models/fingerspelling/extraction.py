"""MediaPipe Hands extraction for fingerspelling.

The word model uses MediaPipe Holistic (face+pose+hands) because it works on full-body
sign videos. But Holistic locates hands via the body ROI, so it fails on hand-crop
alphabet images (~15% hand-detection). MediaPipe Hands detects the hand directly
(~90% on the same images), so fingerspelling uses Hands. Output is placed into the same
543-row Holistic layout (hand in its LEFT/RIGHT slice) so handshape_features is unchanged.
"""
from __future__ import annotations

import numpy as np

from landmark_baseline.constants import LEFT_HAND_SLICE, RIGHT_HAND_SLICE, ROWS_PER_FRAME


def _empty_frame() -> np.ndarray:
    return np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)


def _placed_543(hand_landmarks, handedness_label: str) -> np.ndarray:
    rows = _empty_frame()
    sl = RIGHT_HAND_SLICE if handedness_label == "Right" else LEFT_HAND_SLICE
    for i, lm in enumerate(hand_landmarks.landmark):
        rows[sl.start + i] = (lm.x, lm.y, lm.z)
    return rows


def extract_hand_landmarks_from_frames(frames, frames_are_bgr: bool = True,
                                       static_image_mode: bool = True) -> np.ndarray:
    """frames -> (N, 543, 3) with the detected hand in its LEFT/RIGHT slice (NaN if no hand).

    static_image_mode=True for independent images (dataset prep); False for video/live (tracking).
    """
    import cv2
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=static_image_mode, max_num_hands=1, min_detection_confidence=0.5,
    )
    out = []
    try:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frames_are_bgr else frame
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                label = "Right"
                if res.multi_handedness:
                    label = res.multi_handedness[0].classification[0].label
                out.append(_placed_543(res.multi_hand_landmarks[0], label))
            else:
                out.append(_empty_frame())
    finally:
        hands.close()
    return np.stack(out, axis=0) if out else np.empty((0, ROWS_PER_FRAME, 3), dtype=np.float32)
