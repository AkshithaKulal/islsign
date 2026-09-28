"""174-dim per-frame features in the exact layout of the datasettt clips.

Layout (decoded from the dataset and verified against every one of its 28,080
frames):

    0:63     hand A  21x3, wrist-relative, divided by max landmark distance
    63:126   hand B  21x3, same
    126:162  pose    12x3 (MediaPipe indices POSE_IDX), minus shoulder midpoint,
                     divided by 3D shoulder distance
    162:165  hand A wrist (hand landmarker) in the same pose frame
    165:168  hand B wrist, same
    168:172  wrist B - wrist A (xyz) and its length; zero unless both hands
    172      hand A present flag
    173      hand B present flag

Hand A is the hand MediaPipe labels "Right", hand B "Left". Coordinates are
MediaPipe image-normalised (x, y, z) from the frame as processed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from config import FEAT_DIM, SEQ_LEN

POSE_IDX = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 24)
L_SHOULDER, R_SHOULDER = 11, 12

HAND_A = slice(0, 63)
HAND_B = slice(63, 126)
POSE = slice(126, 162)
WRIST_A = slice(162, 165)
WRIST_B = slice(165, 168)
INTER = slice(168, 171)
INTER_DIST = 171
FLAG_A = 172
FLAG_B = 173

_EPS = 1e-6


def _hand_shape(hand: np.ndarray) -> np.ndarray:
    rel = hand - hand[0]
    scale = float(np.linalg.norm(rel, axis=1).max())
    if scale < _EPS:
        return np.zeros(63, dtype=np.float32)
    return (rel / scale).reshape(-1)


def frame_features(
    pose33: Optional[np.ndarray],
    hand_a: Optional[np.ndarray],
    hand_b: Optional[np.ndarray],
    pose_idx: Sequence[int] = POSE_IDX,
) -> Optional[np.ndarray]:
    """Return one 174-dim frame, or None when the body pose is missing."""
    if pose33 is None:
        return None
    center = (pose33[L_SHOULDER] + pose33[R_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(pose33[L_SHOULDER] - pose33[R_SHOULDER]))
    if scale < _EPS:
        return None

    out = np.zeros(FEAT_DIM, dtype=np.float32)
    out[POSE] = ((pose33[list(pose_idx)] - center) / scale).reshape(-1)

    wrist_a = wrist_b = None
    if hand_a is not None:
        out[HAND_A] = _hand_shape(hand_a)
        wrist_a = (hand_a[0] - center) / scale
        out[WRIST_A] = wrist_a
        out[FLAG_A] = 1.0
    if hand_b is not None:
        out[HAND_B] = _hand_shape(hand_b)
        wrist_b = (hand_b[0] - center) / scale
        out[WRIST_B] = wrist_b
        out[FLAG_B] = 1.0
    if wrist_a is not None and wrist_b is not None:
        d = wrist_b - wrist_a
        out[INTER] = d
        out[INTER_DIST] = float(np.linalg.norm(d))
    return out


def assign_hands(
    hands: Sequence[np.ndarray],
    labels: Sequence[str],
    center_x: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Map detected hands to (A, B). A = label "Right", B = label "Left".

    If both hands get the same (or no) label, the one further toward +x
    (the side hand A sits on in the dataset) becomes A.
    """
    a = b = None
    pending: list[np.ndarray] = []
    for hand, label in zip(hands, labels):
        name = (label or "").lower()
        if name.startswith("right") and a is None:
            a = hand
        elif name.startswith("left") and b is None:
            b = hand
        else:
            pending.append(hand)
    for hand in sorted(pending, key=lambda h: -float(h[0, 0])):
        if a is None and float(hand[0, 0]) >= center_x:
            a = hand
        elif b is None:
            b = hand
        elif a is None:
            a = hand
    return a, b


def resample_indices(n: int, length: int = SEQ_LEN) -> np.ndarray:
    return np.round(np.linspace(0, n - 1, length)).astype(int)


def to_sequence(frames: Sequence[np.ndarray], length: int = SEQ_LEN) -> np.ndarray:
    """Stretch/squeeze real frames to a fixed-length clip the way datasettt was made."""
    if not frames:
        raise ValueError("no frames")
    arr = np.stack(frames).astype(np.float32)
    return arr[resample_indices(len(arr), length)]


def check_invariants(clip: np.ndarray, tol: float = 1e-3) -> list[str]:
    """Structural checks every datasettt clip satisfies. Returns problems found."""
    problems: list[str] = []
    if clip.shape != (SEQ_LEN, FEAT_DIM):
        return [f"shape {clip.shape} != ({SEQ_LEN}, {FEAT_DIM})"]
    pose = clip[:, POSE].reshape(-1, 12, 3)
    if np.abs(pose[:, 5] + pose[:, 6]).max() > tol:
        problems.append("shoulder midpoint not at origin")
    if np.abs(np.linalg.norm(pose[:, 5] - pose[:, 6], axis=1) - 1).max() > tol:
        problems.append("shoulder distance != 1")
    for sl, flag, name in ((HAND_A, FLAG_A, "A"), (HAND_B, FLAG_B, "B")):
        present = clip[:, flag] > 0.5
        hand = clip[:, sl].reshape(-1, 21, 3)
        if present.any():
            mx = np.linalg.norm(hand[present], axis=2).max(1)
            if np.abs(mx - 1).max() > tol:
                problems.append(f"hand {name} max landmark distance != 1")
        if np.abs(hand[~present]).max(initial=0) > 0:
            problems.append(f"hand {name} values present while flag is 0")
    both = (clip[:, FLAG_A] > 0.5) & (clip[:, FLAG_B] > 0.5)
    if np.abs(clip[~both, 168:172]).max(initial=0) > 0:
        problems.append("inter-hand features non-zero without both hands")
    return problems
