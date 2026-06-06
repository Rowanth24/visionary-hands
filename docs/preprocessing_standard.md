# Preprocessing standard (Gate G-C)

There is **one** landmark preprocessing path, used identically at train and serve:

```
landmark_baseline/preprocess.py :: preprocess_landmark_sequence(
    sequence,                       # (num_frames, 543, 3) MediaPipe Holistic landmarks
    target_frames=64,
    mirror_right_to_left=True,
    layout="dominant_both_hands",   # the deployed dual-hand model's layout
)
```

What it does (dual-hand, handedness-canonical):
1. Infer the dominant hand from per-frame hand presence.
2. Select the dominant-hand (or both-hands) landmark subset.
3. Mirror right→left so a right-dominant signer is canonicalized to the left frame
   (removes handedness as a nuisance variable).
4. Drop empty frames, then pad-or-pool to `target_frames`.

Output: `PreprocessedSequence(frames, non_empty_frame_idxs, dominant_hand, layout,
source_num_frames, kept_num_frames)`. Models consume `frames[..., :2]`
(`xy_only=True`) reshaped to `(T, landmark_count*2)`.

## Single path — no train/serve skew

| Stage | Caller |
|---|---|
| Train (dataset prep) | `pipeline/prepare_landmark_dataset.py` |
| Serve — TorchScript | `landmark_baseline/inference.py` |
| Serve — ONNX | `landmark_baseline/inference_onnx.py` |
| Serve — benchmark harness | `eval/asl_backends.py :: LandmarkBackend` → `inference.py` |

All call `preprocess_landmark_sequence` with the same `layout`/`target_frames`.
`tests/test_preprocessing_standard.py` asserts the train-side and serve-side
feature tensors are **byte-identical** for the same raw landmarks — it fails if the
two paths ever diverge.

## Deprecated — do not use for the trained model

`eval/landmark_utils.py` (`normalize_landmark_points`, `HolisticSequenceEncoder`)
is a **different**, legacy normalization (shoulder-centered, 49 points, fixed
sequence length 16, nearest-neighbor `landmark_distance`). It does **not** match the
trained dual-hand transformer and would introduce train/serve skew if used for it.
Its only consumer, `eval/benchmark_landmark_closed_set.py`, is superseded by
`eval/benchmark_landmark_vs_vanilla.py` (Gate G-B). Both are kept for historical
reference only and are marked DEPRECATED in-file.
