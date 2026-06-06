import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_labels(labels_path):
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_text, label = line.split(maxsplit=1)
            labels[label.lower()] = int(idx_text)
    return labels


def load_rgb_frames_from_video_path(video_path, start, num):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))

    for _ in range(min(num, max(0, total_frames - start))):
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        height, width = frame.shape[:2]
        if min(height, width) < 226:
            scale = 226.0 / min(height, width)
            frame = cv2.resize(
                frame,
                (math.ceil(width * scale), math.ceil(height * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        if frame.shape[0] != 256 or frame.shape[1] != 256:
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = (frame / 255.0) * 2.0 - 1.0
        frames.append(frame)

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames were read from: {video_path}")
    return np.asarray(frames, dtype=np.float32), total_frames


class MixedSignDataset(data_utl.Dataset):
    def __init__(self, manifest_path, video_dir, labels_path, split, transforms=None, clip_length=64):
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.samples = payload["samples"]
        self.video_dir = Path(video_dir)
        self.split = split
        self.transforms = transforms
        self.clip_length = clip_length
        self.label_to_index = load_labels(labels_path)
        self.num_classes = len(self.label_to_index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label_text = (sample.get("label") or sample.get("pair_label") or sample.get("expected") or "").lower()
        if label_text not in self.label_to_index:
            raise KeyError(f"Label '{label_text}' is not present in the provided labels file.")

        video_path = self.video_dir / sample["local_video"]
        total_frames = int(sample.get("frame_count") or 0)
        if total_frames > self.clip_length:
            if self.split == "train":
                start_frame = random.randint(0, total_frames - self.clip_length)
            else:
                start_frame = (total_frames - self.clip_length) // 2
        else:
            start_frame = 0

        frames, total_frames = load_rgb_frames_from_video_path(video_path, start_frame, self.clip_length)
        frames = self.pad_frames(frames, self.clip_length, training=self.split == "train")
        if self.transforms is not None:
            frames = self.transforms(frames)

        return (
            video_to_tensor(frames),
            torch.tensor(self.label_to_index[label_text], dtype=torch.long),
            sample["local_video"],
            sample.get("source", "unknown"),
        )

    @staticmethod
    def pad_frames(frames, total_frames, training):
        if frames.shape[0] >= total_frames:
            return frames

        num_padding = total_frames - frames.shape[0]
        if training and random.random() > 0.5:
            pad_img = frames[0]
            pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
            return np.concatenate([pad, frames], axis=0)

        pad_img = frames[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
        return np.concatenate([frames, pad], axis=0)
