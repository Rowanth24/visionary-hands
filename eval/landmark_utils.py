import os
from pathlib import Path

import cv2
import numpy as np

from asl_backends import ensure_mediapipe_holistic_model


POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)
POINT_COUNT = len(POSE_INDICES) + 21 + 21


def sample_to_fixed(frames, count):
    if not frames:
        raise RuntimeError("Cannot sample landmarks from an empty frame list.")
    indices = np.linspace(0, len(frames) - 1, num=count, dtype=int)
    return [frames[int(index)] for index in indices]


def normalize_landmark_points(points, masks):
    coords = np.zeros((POINT_COUNT, 2), dtype=np.float32)
    valid = np.zeros((POINT_COUNT,), dtype=np.float32)

    left_shoulder = points.get("pose_11")
    right_shoulder = points.get("pose_12")
    center = np.array([0.5, 0.5], dtype=np.float32)
    scale = 0.35
    angle = 0.0

    if left_shoulder is not None and right_shoulder is not None:
        left = np.asarray(left_shoulder, dtype=np.float32)
        right = np.asarray(right_shoulder, dtype=np.float32)
        center = (left + right) * 0.5
        shoulder_dist = max(1e-3, float(np.linalg.norm(right - left)))
        scale = max(0.25, shoulder_dist * 2.25)
        angle = float(np.arctan2(right[1] - left[1], right[0] - left[0]))
    else:
        hand_points = [np.asarray(points[key], dtype=np.float32) for key in points if key.startswith("left_") or key.startswith("right_")]
        if hand_points:
            hand_array = np.stack(hand_points)
            center = hand_array.mean(axis=0)
            hand_extent = np.max(hand_array, axis=0) - np.min(hand_array, axis=0)
            scale = max(0.22, float(max(hand_extent[0], hand_extent[1])) * 2.8)
        else:
            pose_points = [np.asarray(points[key], dtype=np.float32) for key in points if key.startswith("pose_")]
            if pose_points:
                pose_array = np.stack(pose_points)
                center = pose_array.mean(axis=0)
                pose_extent = np.max(pose_array, axis=0) - np.min(pose_array, axis=0)
                scale = max(0.3, float(max(pose_extent[0], pose_extent[1])) * 1.5)

    cos_theta = float(np.cos(-angle))
    sin_theta = float(np.sin(-angle))

    def project(raw_point):
        dx, dy = np.asarray(raw_point, dtype=np.float32) - center
        rotated_x = (cos_theta * dx) - (sin_theta * dy)
        rotated_y = (sin_theta * dx) + (cos_theta * dy)
        return np.array([rotated_x / scale, rotated_y / scale], dtype=np.float32)

    point_order = [("pose", index) for index in POSE_INDICES]
    point_order.extend(("left", index) for index in range(21))
    point_order.extend(("right", index) for index in range(21))

    for output_index, (group, point_index) in enumerate(point_order):
        key = f"{group}_{point_index}"
        if not masks.get(key, False):
            continue
        coords[output_index] = np.clip(project(points[key]), -3.0, 3.0)
        valid[output_index] = 1.0

    return coords, valid


def landmark_distance(descriptor_a, descriptor_b):
    coords_a = descriptor_a["coords"]
    coords_b = descriptor_b["coords"]
    mask_a = descriptor_a["mask"]
    mask_b = descriptor_b["mask"]

    valid = (mask_a > 0.5) & (mask_b > 0.5)
    if np.any(valid):
        coord_delta = np.abs(coords_a - coords_b).sum(axis=-1)
        coord_distance = float(coord_delta[valid].mean())
    else:
        coord_distance = 3.0

    velocity_a = np.diff(coords_a, axis=0)
    velocity_b = np.diff(coords_b, axis=0)
    velocity_mask = ((mask_a[1:] > 0.5) & (mask_a[:-1] > 0.5) & (mask_b[1:] > 0.5) & (mask_b[:-1] > 0.5))
    if np.any(velocity_mask):
        velocity_delta = np.abs(velocity_a - velocity_b).sum(axis=-1)
        motion_distance = float(velocity_delta[velocity_mask].mean())
    else:
        motion_distance = coord_distance

    presence_distance = float(np.abs(mask_a - mask_b).mean())
    return coord_distance + (0.5 * motion_distance) + (0.25 * presence_distance)


class HolisticSequenceEncoder:
    def __init__(self, sequence_length=16, input_size=512):
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe import tasks
        from mediapipe.tasks.python import vision

        self.sequence_length = sequence_length
        self.input_size = input_size
        self.mp = mp
        model_path = ensure_mediapipe_holistic_model()
        self.landmarker = vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
            )
        )

    def _detect_frame(self, frame):
        interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > self.input_size else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (self.input_size, self.input_size), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)

        points = {}
        masks = {}

        if result.pose_landmarks:
            for index in POSE_INDICES:
                landmark = result.pose_landmarks[index]
                visibility = getattr(landmark, "visibility", 1.0)
                presence = getattr(landmark, "presence", 1.0)
                if visibility >= 0.35 and presence >= 0.35:
                    points[f"pose_{index}"] = (landmark.x, landmark.y)
                    masks[f"pose_{index}"] = True

        for hand_name, landmarks in (("left", result.left_hand_landmarks), ("right", result.right_hand_landmarks)):
            if not landmarks:
                continue
            for index, landmark in enumerate(landmarks):
                points[f"{hand_name}_{index}"] = (landmark.x, landmark.y)
                masks[f"{hand_name}_{index}"] = True

        coords, mask = normalize_landmark_points(points, masks)
        return coords, mask

    def encode_frames(self, frames):
        sampled = sample_to_fixed(frames, self.sequence_length)
        coords = np.zeros((self.sequence_length, POINT_COUNT, 2), dtype=np.float32)
        mask = np.zeros((self.sequence_length, POINT_COUNT), dtype=np.float32)

        for index, frame in enumerate(sampled):
            try:
                frame_coords, frame_mask = self._detect_frame(frame)
            except RuntimeError:
                frame_coords = np.zeros((POINT_COUNT, 2), dtype=np.float32)
                frame_mask = np.zeros((POINT_COUNT,), dtype=np.float32)
            coords[index] = frame_coords
            mask[index] = frame_mask

        return {
            "coords": coords,
            "mask": mask,
            "sequence_length": self.sequence_length,
        }

    def encode_video(self, video_path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for landmarks: {video_path}")

        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()

        if not frames:
            raise RuntimeError(f"No frames were read for landmarks: {video_path}")
        return self.encode_frames(frames)

    def encode_with_cache(self, video_path, cache_dir):
        video_path = Path(video_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{video_path.stem}.npz"

        if cache_path.exists():
            payload = np.load(cache_path)
            return {
                "coords": payload["coords"],
                "mask": payload["mask"],
                "sequence_length": int(payload["sequence_length"].item()),
            }

        descriptor = self.encode_video(video_path)
        np.savez_compressed(
            cache_path,
            coords=descriptor["coords"],
            mask=descriptor["mask"],
            sequence_length=np.array([descriptor["sequence_length"]], dtype=np.int32),
        )
        return descriptor
