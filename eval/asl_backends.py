import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import RemoteDisconnected
from pathlib import Path

import cv2
import numpy as np
import torch

import videotransforms
from custom_models import I3DFeatureExtractor, SignLanguageRecognitionModel
from pytorch_i3d import InceptionI3d


OLLAMA_ASL_SYSTEM_PROMPT = """You translate short American Sign Language clips into concise English glosses.
The user will provide ordered frames from earliest to latest.
Focus on the meaning of the sign, not a description of the visible hand motion.
Always provide the single best lexical guess.
Return lowercase text.
If the clip looks like a single WLASL-style isolated sign, prefer a single English word.
Do not explain your reasoning.
Only return "unknown" when the frames are unusable or the signer is not visible enough to form any reasonable guess."""

OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "confidence": {"type": "number"},
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "confidence"],
                "additionalProperties": False,
            },
            "maxItems": 3,
        },
    },
    "required": ["translation", "confidence", "alternatives"],
    "additionalProperties": False,
}

DEFAULT_VANILLA_CHECKPOINT = Path.home() / "Downloads" / "best_model.pth"
MEDIAPIPE_HOLISTIC_MODEL_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "models" / "mediapipe" / "holistic_landmarker.task"
MEDIAPIPE_HOLISTIC_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"


@dataclass
class Candidate:
    text: str
    score: float | None = None


@dataclass
class BackendResult:
    text: str
    candidates: list[Candidate]
    raw: dict | None = None


class BaseASLBackend:
    name = "base"
    supports_window_voting = False

    def prepare_frames(self, frames):
        return list(frames)

    def infer_clip(self, frames, top_k):
        raise NotImplementedError

    def describe(self):
        return self.name


def default_vanilla_checkpoint_path():
    return DEFAULT_VANILLA_CHECKPOINT if DEFAULT_VANILLA_CHECKPOINT.exists() else None


def choose_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    unexpected = []
    mismatched = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value

    missing = sorted(set(model_state.keys()) - set(filtered.keys()))
    return filtered, missing, unexpected, mismatched


def infer_num_classes(state_dict):
    for key in ("transformer.classifier.weight", "module.transformer.classifier.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise RuntimeError("Could not infer the number of classes from the checkpoint.")


def load_labels(labels_path, num_classes):
    labels = [f"class_{i}" for i in range(num_classes)]
    with open(labels_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                idx_text, label = line.split("\t", 1)
            else:
                idx_text, label = line.split(maxsplit=1)
            idx = int(idx_text)
            if 0 <= idx < num_classes:
                labels[idx] = label
    return labels


def parse_candidate_label_strings(candidate_labels=None, candidate_labels_file=None):
    values = []
    if candidate_labels:
        values.extend(candidate_labels.split(","))
    if candidate_labels_file:
        with open(candidate_labels_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if "," in stripped:
                    values.extend(part.strip() for part in stripped.split(","))
                else:
                    values.append(stripped)
    return [normalize_prediction_text(value) for value in values if normalize_prediction_text(value)]


def candidate_indices_for_labels(all_labels, candidate_labels):
    if not candidate_labels:
        return None, []

    normalized_to_index = {}
    for index, label in enumerate(all_labels):
        normalized_to_index.setdefault(normalize_prediction_text(label), index)

    indices = []
    missing = []
    seen = set()
    for label in candidate_labels:
        if label in seen:
            continue
        seen.add(label)
        index = normalized_to_index.get(label)
        if index is None:
            missing.append(label)
            continue
        indices.append(index)
    return indices, missing


def preprocess_vanilla_frame(frame):
    height, width = frame.shape[:2]
    min_side = min(height, width)
    if min_side < 226:
        scale = 226.0 / min_side
        frame = cv2.resize(
            frame,
            (math.ceil(width * scale), math.ceil(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    if frame.shape[0] != 256 or frame.shape[1] != 256:
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
    return ((frame.astype(np.float32) / 255.0) * 2.0) - 1.0


def clip_to_tensor(frames, crop_transform):
    clip = np.asarray(frames, dtype=np.float32)
    clip = crop_transform(clip)
    clip = np.ascontiguousarray(clip.transpose(3, 0, 1, 2))
    return torch.from_numpy(clip).unsqueeze(0)


def normalize_prediction_text(text):
    keep = []
    for char in text.lower().strip():
        if char.isalnum() or char.isspace():
            keep.append(char)
    return " ".join("".join(keep).split())


class VanillaASLBackend(BaseASLBackend):
    name = "vanilla"
    supports_window_voting = True

    def __init__(
        self,
        checkpoint_path,
        labels_path,
        device_name,
        frame_normalization="none",
        mediapipe_sample_count=12,
        candidate_labels=None,
        candidate_labels_file=None,
    ):
        self.device = choose_device(device_name)
        self.model, self.num_classes = self._build_model(checkpoint_path)
        self.labels = load_labels(labels_path, self.num_classes)
        self.crop_transform = videotransforms.CenterCrop(224)
        self.frame_normalization = frame_normalization
        self.mediapipe_sample_count = mediapipe_sample_count
        self._mediapipe_estimator = None
        parsed_candidates = parse_candidate_label_strings(candidate_labels, candidate_labels_file)
        self.candidate_indices, self.missing_candidate_labels = candidate_indices_for_labels(self.labels, parsed_candidates)
        self.candidate_label_count = None if self.candidate_indices is None else len(self.candidate_indices)
        if self.candidate_indices is not None and not self.candidate_indices:
            raise ValueError("The requested candidate label set does not match any known class labels.")
        self._candidate_index_tensor = None if self.candidate_indices is None else torch.tensor(self.candidate_indices, dtype=torch.long)
        self._candidate_labels = self.labels if self.candidate_indices is None else [self.labels[index] for index in self.candidate_indices]

    def _build_model(self, checkpoint_path):
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint_obj)
        num_classes = infer_num_classes(state_dict)

        i3d = InceptionI3d(num_classes=1000, in_channels=3)
        model = SignLanguageRecognitionModel(I3DFeatureExtractor(i3d), num_classes)
        normalized_state = strip_module_prefix(state_dict)
        filtered_state, missing, unexpected, mismatched = filter_state_dict_for_model(model, normalized_state)
        if mismatched:
            raise RuntimeError(f"Checkpoint has mismatched tensor shapes: {mismatched[:5]}")
        if missing:
            raise RuntimeError(f"Checkpoint is missing required tensors: {missing[:10]}")
        if unexpected:
            print(f"Ignoring {len(unexpected)} unexpected checkpoint tensors")
        model.load_state_dict(filtered_state, strict=True)
        model.to(self.device)
        model.eval()
        return model, num_classes

    def prepare_frames(self, frames):
        return normalize_frames(
            frames,
            mode=self.frame_normalization,
            mediapipe_estimator=self._get_mediapipe_estimator(),
            mediapipe_sample_count=self.mediapipe_sample_count,
        )

    def _get_mediapipe_estimator(self):
        if self.frame_normalization != "mediapipe":
            return None
        if self._mediapipe_estimator is None:
            self._mediapipe_estimator = MediaPipeRegionEstimator()
        return self._mediapipe_estimator

    @torch.inference_mode()
    def infer_prepared_clip(self, frames, top_k):
        processed_frames = [preprocess_vanilla_frame(frame) for frame in frames]
        clip_tensor = clip_to_tensor(processed_frames, self.crop_transform).to(self.device)
        logits = self.model(clip_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        candidate_view = self.restrict_probabilities(probs)
        candidates = self.candidates_from_restricted_probabilities(candidate_view, top_k)
        return BackendResult(
            text=candidates[0].text,
            candidates=candidates,
            raw={
                "probabilities": probs,
                "candidate_probabilities": candidate_view,
                "candidate_indices": self.candidate_indices,
            },
        )

    def infer_clip(self, frames, top_k):
        return self.infer_prepared_clip(self.prepare_frames(frames), top_k)

    def restrict_probabilities(self, probs):
        if self._candidate_index_tensor is None:
            return probs
        candidate_view = probs[self._candidate_index_tensor]
        total = float(candidate_view.sum().item())
        if total > 0.0:
            candidate_view = candidate_view / total
        return candidate_view

    def candidates_from_restricted_probabilities(self, candidate_view, top_k):
        top_scores, top_indices = torch.topk(candidate_view, k=min(top_k, candidate_view.shape[0]))
        return [
            Candidate(text=self._candidate_labels[idx], score=float(score))
            for idx, score in zip(top_indices.tolist(), top_scores.tolist())
        ]

    def describe(self):
        suffix = ""
        if self.candidate_label_count is not None:
            suffix = f":candidates={self.candidate_label_count}"
        return f"{self.name}:{self.device.type}:{self.frame_normalization}{suffix}"


def make_windows(frames, clip_length, stride):
    if len(frames) < clip_length:
        return [frames + [frames[-1]] * (clip_length - len(frames))]

    starts = list(range(0, len(frames) - clip_length + 1, stride))
    last_start = len(frames) - clip_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return [frames[start:start + clip_length] for start in starts]


def aggregate_vanilla_candidates(backend, frames, clip_length, stride, top_k):
    prepared_frames = backend.prepare_frames(frames)
    windows = make_windows(prepared_frames, clip_length, stride)
    probs_sum = None

    for window in windows:
        result = backend.infer_prepared_clip(window, top_k)
        probs = result.raw["candidate_probabilities"]
        probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_mean = probs_sum / len(windows)
    candidates = backend.candidates_from_restricted_probabilities(probs_mean, top_k)
    return candidates, len(windows)


class OllamaASLBackend(BaseASLBackend):
    name = "ollama"
    supports_window_voting = False

    def __init__(
        self,
        model_name,
        host,
        sample_count,
        frame_size,
        timeout_seconds,
        temperature,
        top_p,
        top_k,
        keep_alive,
        request_retries=2,
        system_prompt=None,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.sample_count = sample_count
        self.frame_size = frame_size
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.keep_alive = keep_alive
        self.request_retries = request_retries
        self.system_prompt = system_prompt or OLLAMA_ASL_SYSTEM_PROMPT

    def describe(self):
        return f"{self.name}:{self.model_name}"

    def infer_clip(self, frames, top_k):
        focus_frames = focus_crop_frames(frames)
        sampled = sample_evenly(focus_frames, min(self.sample_count, 6))
        encoded_images = [encode_frame_for_ollama(frame, self.frame_size) for frame in sampled]
        prompt = (
            f"You are given {len(sampled)} ASL frames ordered from earliest to latest.\n"
            "Translate the sign into its English meaning, not a description of the visible motion.\n"
            "Return one short lowercase gloss whenever possible."
        )
        content = self.run_prompt(prompt, encoded_images)
        parsed = {
            "translation": extract_first_text_line(content),
            "confidence": None,
            "alternatives": [],
            "variant": "raw_frames",
            "sample_count": len(sampled),
        }
        return build_backend_result(parsed, top_k)

    def run_prompt(self, prompt, encoded_images):
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_predict": 128,
            },
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                    "images": encoded_images,
                },
            ],
        }
        response = post_ollama_chat(
            self.host,
            payload,
            self.timeout_seconds,
            retries=self.request_retries,
        )
        return response["message"]["content"]

    def choose_from_candidates(self, frames, candidates):
        focus_frames = focus_crop_frames(frames)
        sampled = sample_evenly(focus_frames, min(self.sample_count, 6))
        encoded_images = [encode_frame_for_ollama(frame, self.frame_size) for frame in sampled]
        label_text = ", ".join(candidate.text for candidate in candidates)
        prompt = (
            f"You are given {len(sampled)} ASL frames ordered from earliest to latest.\n"
            f"Choose the best matching English gloss from this candidate list only: {label_text}.\n"
            "Return exactly one candidate from the list and nothing else.\n"
            "Choose the meaning of the sign, not a description of the visible motion."
        )
        return extract_first_text_line(self.run_prompt(prompt, encoded_images))


class OllamaHybridASLBackend(BaseASLBackend):
    name = "ollama_hybrid"
    supports_window_voting = False

    def __init__(
        self,
        ollama_backend,
        proposal_backend,
        proposal_top_k,
        clip_length,
        stride,
        rerank_margin,
    ):
        self.ollama_backend = ollama_backend
        self.proposal_backend = proposal_backend
        self.proposal_top_k = proposal_top_k
        self.clip_length = clip_length
        self.stride = stride
        self.rerank_margin = rerank_margin

    def describe(self):
        return f"{self.name}:{self.ollama_backend.model_name}+{self.proposal_backend.device.type}"

    def infer_clip(self, frames, top_k):
        proposal_candidates, window_count = aggregate_vanilla_candidates(
            backend=self.proposal_backend,
            frames=frames,
            clip_length=self.clip_length,
            stride=self.stride,
            top_k=max(top_k, self.proposal_top_k),
        )
        if not proposal_candidates:
            raise RuntimeError("The proposal backend did not return any candidates.")

        choice_text = self.ollama_backend.choose_from_candidates(frames, proposal_candidates[: self.proposal_top_k])
        matched_candidate = match_candidate(choice_text, proposal_candidates[: self.proposal_top_k])
        primary = proposal_candidates[0]
        reranked = False

        if matched_candidate is not None and matched_candidate.text != primary.text:
            primary_score = primary.score or 0.0
            matched_score = matched_candidate.score or 0.0
            if primary_score - matched_score <= self.rerank_margin:
                primary = matched_candidate
                reranked = True

        ordered_candidates = [primary]
        seen = {normalize_prediction_text(primary.text)}
        for candidate in proposal_candidates:
            normalized = normalize_prediction_text(candidate.text)
            if normalized in seen:
                continue
            ordered_candidates.append(candidate)
            seen.add(normalized)
            if len(ordered_candidates) >= top_k:
                break

        return BackendResult(
            text=primary.text,
            candidates=ordered_candidates,
            raw={
                "mode": "reranked" if reranked else ("agreed" if matched_candidate is not None else "fallback"),
                "ollama_choice": choice_text,
                "proposal_window_count": window_count,
                "proposal_candidates": [
                    {"text": candidate.text, "score": candidate.score}
                    for candidate in proposal_candidates[: self.proposal_top_k]
                ],
            },
        )


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ollama_json(content):
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {
        "translation": content.strip() or "unknown",
        "confidence": None,
        "alternatives": [],
    }


def extract_first_text_line(content):
    text = (content or "").strip()
    if not text:
        return "unknown"
    line = text.splitlines()[0].strip().strip("\"'")
    return normalize_prediction_text(line) or "unknown"


def build_backend_result(parsed, top_k):
    primary_text = normalize_prediction_text(parsed.get("translation", "unknown")) or "unknown"
    candidates = [Candidate(text=primary_text, score=safe_float(parsed.get("confidence")))]
    for item in parsed.get("alternatives", [])[:max(0, top_k - 1)]:
        alt_text = normalize_prediction_text(item.get("text", ""))
        if not alt_text or alt_text == primary_text:
            continue
        candidates.append(Candidate(text=alt_text, score=safe_float(item.get("confidence"))))

    return BackendResult(text=primary_text, candidates=candidates, raw=parsed)


def match_candidate(choice_text, candidates):
    normalized_choice = normalize_prediction_text(choice_text)
    if not normalized_choice:
        return None

    by_text = {normalize_prediction_text(candidate.text): candidate for candidate in candidates}
    if normalized_choice in by_text:
        return by_text[normalized_choice]

    for normalized_text, candidate in by_text.items():
        if normalized_choice in normalized_text or normalized_text in normalized_choice:
            return candidate
    return None


def sample_evenly(frames, count):
    if not frames:
        raise RuntimeError("No frames were provided for inference.")
    if len(frames) <= count:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, num=count, dtype=int)
    return [frames[int(index)] for index in indices]


class MediaPipeRegionEstimator:
    def __init__(self):
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe import tasks
        from mediapipe.tasks.python import vision

        model_path = ensure_mediapipe_holistic_model()
        self.mp = mp
        self.landmarker = vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
            )
        )

    def estimate_bbox(self, frames, sample_count):
        sampled = sample_evenly(frames, min(len(frames), max(2, sample_count)))
        frame_h, frame_w = frames[0].shape[:2]
        boxes = []

        for frame in sampled:
            interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > 512 else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (512, 512), interpolation=interpolation)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            try:
                results = self.landmarker.detect(image)
            except RuntimeError:
                continue
            bbox = extract_mediapipe_bbox(results, resized.shape[1], resized.shape[0])
            if bbox is None:
                continue

            scale_x = frame_w / resized.shape[1]
            scale_y = frame_h / resized.shape[0]
            x1, y1, x2, y2 = bbox
            boxes.append(
                (
                    max(0, int(round(x1 * scale_x))),
                    max(0, int(round(y1 * scale_y))),
                    min(frame_w, int(round(x2 * scale_x))),
                    min(frame_h, int(round(y2 * scale_y))),
                )
            )

        if not boxes:
            return None
        return merge_landmark_boxes(boxes, frame_w, frame_h)


def ensure_mediapipe_holistic_model():
    if MEDIAPIPE_HOLISTIC_MODEL_PATH.exists() and MEDIAPIPE_HOLISTIC_MODEL_PATH.stat().st_size > 0:
        return MEDIAPIPE_HOLISTIC_MODEL_PATH

    MEDIAPIPE_HOLISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MEDIAPIPE_HOLISTIC_MODEL_URL, MEDIAPIPE_HOLISTIC_MODEL_PATH)
    return MEDIAPIPE_HOLISTIC_MODEL_PATH


def extract_mediapipe_bbox(results, frame_w, frame_h):
    points = []

    def extend_hand(hand_landmarks):
        if not hand_landmarks:
            return False
        for landmark in hand_landmarks:
            points.append((landmark.x * frame_w, landmark.y * frame_h))
        return True

    hand_found = extend_hand(results.left_hand_landmarks)
    hand_found = extend_hand(results.right_hand_landmarks) or hand_found

    pose = results.pose_landmarks if results.pose_landmarks else None
    if pose is not None:
        pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        for index in pose_indices:
            landmark = pose[index]
            visibility = getattr(landmark, "visibility", 1.0)
            presence = getattr(landmark, "presence", 1.0)
            if visibility < 0.35 or presence < 0.35:
                continue
            points.append((landmark.x * frame_w, landmark.y * frame_h))

        if not hand_found:
            shoulders = [pose[index] for index in (11, 12) if pose[index].visibility >= 0.35]
            hips = [pose[index] for index in (23, 24) if pose[index].visibility >= 0.35]
            if shoulders:
                shoulder_x = np.mean([landmark.x for landmark in shoulders]) * frame_w
                shoulder_y = np.mean([landmark.y for landmark in shoulders]) * frame_h
                points.append((shoulder_x, max(0.0, shoulder_y - 0.30 * frame_h)))
            if shoulders and hips:
                shoulder_width = abs(shoulders[0].x - shoulders[-1].x) * frame_w if len(shoulders) == 2 else frame_w * 0.18
                chest_center_x = np.mean([landmark.x for landmark in shoulders]) * frame_w
                chest_center_y = (np.mean([landmark.y for landmark in shoulders]) * 0.55 + np.mean([landmark.y for landmark in hips]) * 0.45) * frame_h
                points.extend(
                    [
                        (chest_center_x - 1.1 * shoulder_width, chest_center_y),
                        (chest_center_x + 1.1 * shoulder_width, chest_center_y),
                    ]
                )

    if not points:
        return None

    xs = np.array([point[0] for point in points], dtype=np.float32)
    ys = np.array([point[1] for point in points], dtype=np.float32)
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(width * 1.45, frame_w * 0.34)
    height = max(height * 1.55, frame_h * 0.50)
    cy -= 0.08 * height

    return clamp_bbox(
        (
            int(round(cx - width / 2.0)),
            int(round(cy - height / 2.0)),
            int(round(cx + width / 2.0)),
            int(round(cy + height / 2.0)),
        ),
        frame_w,
        frame_h,
    )


def merge_landmark_boxes(boxes, frame_w, frame_h):
    x1s = np.array([bbox[0] for bbox in boxes], dtype=np.float32)
    y1s = np.array([bbox[1] for bbox in boxes], dtype=np.float32)
    x2s = np.array([bbox[2] for bbox in boxes], dtype=np.float32)
    y2s = np.array([bbox[3] for bbox in boxes], dtype=np.float32)

    x1 = float(np.percentile(x1s, 15))
    y1 = float(np.percentile(y1s, 10))
    x2 = float(np.percentile(x2s, 85))
    y2 = float(np.percentile(y2s, 90))

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(width * 1.12, frame_w * 0.36)
    height = max(height * 1.12, frame_h * 0.54)
    cy -= 0.05 * height

    return clamp_bbox(
        (
            int(round(cx - width / 2.0)),
            int(round(cy - height / 2.0)),
            int(round(cx + width / 2.0)),
            int(round(cy + height / 2.0)),
        ),
        frame_w,
        frame_h,
    )


def clamp_bbox(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(frame_w - 1, x1)),
        max(0, min(frame_h - 1, y1)),
        max(1, min(frame_w, x2)),
        max(1, min(frame_h, y2)),
    )


def normalize_frames(frames, mode, mediapipe_estimator=None, mediapipe_sample_count=12):
    if mode == "none":
        return list(frames)
    if mode == "motion":
        return focus_crop_frames(frames)
    if mode != "mediapipe":
        raise ValueError(f"Unsupported frame normalization mode: {mode}")

    if mediapipe_estimator is None:
        return focus_crop_frames(frames)

    bbox = mediapipe_estimator.estimate_bbox(frames, mediapipe_sample_count)
    if bbox is None:
        fallback = estimate_motion_bbox(frames)
        if fallback is None:
            return list(frames)
        bbox = fallback
    return [crop_to_bbox(frame, bbox) for frame in frames]


def focus_crop_frames(frames):
    bbox = estimate_motion_bbox(frames)
    if bbox is None:
        return list(frames)
    return [crop_to_bbox(frame, bbox) for frame in frames]


def estimate_motion_bbox(frames):
    sampled = sample_evenly(frames, min(len(frames), 8))
    reference = resize_frame_for_ollama(sampled[0], 256)
    gray_frames = [cv2.cvtColor(resize_frame_for_ollama(frame, 256), cv2.COLOR_BGR2GRAY) for frame in sampled]
    motion = np.zeros_like(gray_frames[0], dtype=np.float32)

    for previous, current in zip(gray_frames, gray_frames[1:]):
        motion += cv2.absdiff(current, previous).astype(np.float32)

    if float(motion.max()) <= 0.0:
        return None

    motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=2.0)
    threshold_value = max(12.0, float(motion.mean() + motion.std() * 0.75))
    _, thresh = cv2.threshold(motion, threshold_value, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)

    points = cv2.findNonZero(thresh)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    ref_h, ref_w = reference.shape[:2]
    min_w = max(1, int(ref_w * 0.35))
    min_h = max(1, int(ref_h * 0.5))
    cx = x + w // 2
    cy = y + h // 2
    w = max(w, min_w)
    h = max(h, min_h)
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)
    x1 = max(0, cx - w // 2 - pad_w)
    y1 = max(0, cy - h // 2 - pad_h)
    x2 = min(ref_w, cx + w // 2 + pad_w)
    y2 = min(ref_h, cy + h // 2 + pad_h)

    orig_h, orig_w = frames[0].shape[:2]
    scale_x = orig_w / ref_w
    scale_y = orig_h / ref_h
    return (
        max(0, int(round(x1 * scale_x))),
        max(0, int(round(y1 * scale_y))),
        min(orig_w, int(round(x2 * scale_x))),
        min(orig_h, int(round(y2 * scale_y))),
    )


def crop_to_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def make_contact_sheet(frames):
    if not frames:
        raise RuntimeError("Cannot build a contact sheet from zero frames.")

    prepared = [resize_frame_for_ollama(frame, 196) for frame in frames]
    cell_h = max(frame.shape[0] for frame in prepared)
    cell_w = max(frame.shape[1] for frame in prepared)
    columns = min(3, len(prepared))
    rows = int(math.ceil(len(prepared) / columns))
    header_h = 44
    footer_h = 18
    gutter = 8
    canvas_h = header_h + rows * cell_h + (rows + 1) * gutter + footer_h
    canvas_w = columns * cell_w + (columns + 1) * gutter
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    cv2.putText(
        canvas,
        "asl sequence: earliest -> latest",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    for index, frame in enumerate(prepared):
        row = index // columns
        column = index % columns
        y = header_h + gutter + row * (cell_h + gutter)
        x = gutter + column * (cell_w + gutter)
        h, w = frame.shape[:2]
        canvas[y:y + h, x:x + w] = frame
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            str(index + 1),
            (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def make_motion_heatmap(frames):
    if not frames:
        raise RuntimeError("Cannot build a motion heatmap from zero frames.")

    prepared = [resize_frame_for_ollama(frame, 256) for frame in frames]
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in prepared]
    motion = np.zeros_like(gray_frames[0], dtype=np.float32)
    for previous, current in zip(gray_frames, gray_frames[1:]):
        motion += cv2.absdiff(current, previous).astype(np.float32)

    if float(motion.max()) <= 0.0:
        motion = np.ones_like(motion, dtype=np.float32)

    motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=2.0)
    motion = motion / motion.max()
    heatmap = cv2.applyColorMap(np.uint8(np.clip(motion * 255.0, 0, 255)), cv2.COLORMAP_TURBO)
    anchor = prepared[len(prepared) // 2]
    overlay = cv2.addWeighted(anchor, 0.55, heatmap, 0.45, 0.0)
    canvas = np.zeros((overlay.shape[0] + 40, overlay.shape[1], 3), dtype=np.uint8)
    canvas[40:, :, :] = overlay
    cv2.putText(
        canvas,
        "motion emphasis",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return canvas


def encode_visual_summary_images(frames, max_side):
    return [
        encode_frame_for_ollama(make_contact_sheet(frames), max_side),
        encode_frame_for_ollama(make_motion_heatmap(frames), max_side),
    ]


def resize_frame_for_ollama(frame, max_side):
    height, width = frame.shape[:2]
    current_max = max(height, width)
    if current_max <= max_side:
        return frame
    scale = max_side / current_max
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def encode_frame_for_ollama(frame, max_side):
    resized = resize_frame_for_ollama(frame, max_side)
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode a frame for Ollama.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def post_ollama_chat(host, payload, timeout_seconds, retries=0):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{host}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if attempt >= retries:
                raise RuntimeError(f"Ollama HTTP error {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, RemoteDisconnected, TimeoutError, ConnectionResetError) as exc:
            if attempt >= retries:
                if isinstance(exc, urllib.error.URLError):
                    raise RuntimeError(f"Could not reach Ollama at {host}: {exc.reason}") from exc
                raise RuntimeError(f"Ollama request failed: {exc}") from exc

        attempt += 1
        time.sleep(min(2.0 * attempt, 5.0))


def add_backend_args(parser):
    code_dir = Path(__file__).resolve().parent
    default_checkpoint = default_vanilla_checkpoint_path()
    checkpoint_help = "Path to the trained vanilla checkpoint (.pth)."
    if default_checkpoint is not None:
        checkpoint_help += f" Defaults to {default_checkpoint} on this machine for --backend vanilla."
    else:
        checkpoint_help += " Required for --backend vanilla."
    parser.add_argument(
        "--backend",
        choices=("vanilla", "ollama", "ollama_hybrid"),
        default="vanilla",
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--checkpoint",
        help=checkpoint_help,
    )
    parser.add_argument(
        "--labels-file",
        default=str(code_dir / "preprocess" / "wlasl_class_list.txt"),
        help="Path to the class label file for the vanilla backend.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Execution device for the vanilla backend.",
    )
    parser.add_argument(
        "--candidate-labels",
        help="Optional comma-separated shortlist of allowed labels for vanilla inference.",
    )
    parser.add_argument(
        "--candidate-labels-file",
        help="Optional file containing allowed labels, one per line or comma-separated.",
    )
    parser.add_argument(
        "--frame-normalization",
        choices=("none", "motion", "mediapipe"),
        default="none",
        help="Optional per-clip spatial normalization before vanilla inference.",
    )
    parser.add_argument(
        "--mediapipe-sample-count",
        type=int,
        default=12,
        help="How many frames to probe when estimating a MediaPipe-based crop.",
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:e2b",
        help="Local Ollama model name for the multimodal backend.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--ollama-sample-count",
        type=int,
        default=8,
        help="Number of evenly spaced frames to send to Ollama.",
    )
    parser.add_argument(
        "--ollama-frame-size",
        type=int,
        default=384,
        help="Maximum image side length for each frame sent to Ollama.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=120,
        help="Request timeout in seconds for Ollama inference.",
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the Ollama backend.",
    )
    parser.add_argument(
        "--ollama-top-p",
        type=float,
        default=0.95,
        help="top_p setting for the Ollama backend.",
    )
    parser.add_argument(
        "--ollama-top-k",
        type=int,
        default=64,
        help="top_k setting for the Ollama backend.",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default="10m",
        help="Ollama keep_alive value so repeated requests stay warm.",
    )
    parser.add_argument(
        "--ollama-retries",
        type=int,
        default=2,
        help="How many times to retry a failed Ollama request before giving up.",
    )
    parser.add_argument(
        "--ollama-proposal-checkpoint",
        help="Optional vanilla checkpoint used to propose candidate labels for --backend ollama_hybrid. Defaults to the local vanilla default checkpoint when available.",
    )
    parser.add_argument(
        "--ollama-proposal-top-k",
        type=int,
        default=5,
        help="How many proposal labels to pass from the vanilla checkpoint into ollama_hybrid.",
    )
    parser.add_argument(
        "--ollama-rerank-margin",
        type=float,
        default=0.05,
        help="Maximum proposal-score gap that still allows ollama_hybrid to override the top vanilla proposal.",
    )


def create_backend_from_args(args):
    default_checkpoint = default_vanilla_checkpoint_path()

    if args.backend == "vanilla":
        checkpoint_path = args.checkpoint or (str(default_checkpoint) if default_checkpoint is not None else None)
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint is required when --backend vanilla because no default checkpoint was found at "
                f"{DEFAULT_VANILLA_CHECKPOINT}."
            )
        args.checkpoint = checkpoint_path
        return VanillaASLBackend(
            checkpoint_path=checkpoint_path,
            labels_path=args.labels_file,
            device_name=args.device,
            frame_normalization=args.frame_normalization,
            mediapipe_sample_count=args.mediapipe_sample_count,
            candidate_labels=args.candidate_labels,
            candidate_labels_file=args.candidate_labels_file,
        )

    ollama_backend = OllamaASLBackend(
        model_name=args.ollama_model,
        host=args.ollama_host,
        sample_count=args.ollama_sample_count,
        frame_size=args.ollama_frame_size,
        timeout_seconds=args.ollama_timeout,
        temperature=args.ollama_temperature,
        top_p=args.ollama_top_p,
        top_k=args.ollama_top_k,
        keep_alive=args.ollama_keep_alive,
        request_retries=args.ollama_retries,
    )
    if args.backend == "ollama":
        return ollama_backend

    proposal_checkpoint = args.ollama_proposal_checkpoint or (str(default_checkpoint) if default_checkpoint is not None else None)
    if proposal_checkpoint is None:
        raise ValueError(
            "--ollama-proposal-checkpoint is required when --backend ollama_hybrid because no default checkpoint was found at "
            f"{DEFAULT_VANILLA_CHECKPOINT}."
        )
    args.ollama_proposal_checkpoint = proposal_checkpoint

    proposal_backend = VanillaASLBackend(
        checkpoint_path=proposal_checkpoint,
        labels_path=args.labels_file,
        device_name=args.device,
        frame_normalization=args.frame_normalization,
        mediapipe_sample_count=args.mediapipe_sample_count,
        candidate_labels=args.candidate_labels,
        candidate_labels_file=args.candidate_labels_file,
    )
    return OllamaHybridASLBackend(
        ollama_backend=ollama_backend,
        proposal_backend=proposal_backend,
        proposal_top_k=args.ollama_proposal_top_k,
        clip_length=64,
        stride=8,
        rerank_margin=args.ollama_rerank_margin,
    )
