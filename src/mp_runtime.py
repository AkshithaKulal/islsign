"""MediaPipe Tasks (VIDEO mode) wrapper shared by recording and live detection."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import (
    DEFAULT_POSE_MODEL,
    HAND_MODEL_URL,
    MEDIAPIPE_DIR,
    MP_MIN_CONF,
    POSE_MODELS,
)
from features import assign_hands, frame_features

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _xyz(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


@dataclass
class FrameLandmarks:
    pose33: Optional[np.ndarray]
    hand_a: Optional[np.ndarray]
    hand_b: Optional[np.ndarray]

    def features(self) -> Optional[np.ndarray]:
        return frame_features(self.pose33, self.hand_a, self.hand_b)


class LandmarkExtractor:
    """Pose + hand landmarkers in VIDEO mode. One instance per camera stream."""

    def __init__(self, pose_model: str = DEFAULT_POSE_MODEL):
        pose_path = _download(POSE_MODELS[pose_model], MEDIAPIPE_DIR / f"pose_landmarker_{pose_model}.task")
        hand_path = _download(HAND_MODEL_URL, MEDIAPIPE_DIR / "hand_landmarker.task")
        self.pose_model = pose_model
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=MP_MIN_CONF,
                min_pose_presence_confidence=MP_MIN_CONF,
                min_tracking_confidence=MP_MIN_CONF,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=MP_MIN_CONF,
                min_hand_presence_confidence=MP_MIN_CONF,
                min_tracking_confidence=MP_MIN_CONF,
            )
        )
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameLandmarks:
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_res = self._pose.detect_for_video(image, ts)
        hand_res = self._hands.detect_for_video(image, ts)

        pose33 = _xyz(pose_res.pose_landmarks[0]) if pose_res.pose_landmarks else None
        hands = [_xyz(h) for h in hand_res.hand_landmarks]
        labels = [(h[0].category_name if h else "") for h in hand_res.handedness]
        center_x = float((pose33[11, 0] + pose33[12, 0]) / 2) if pose33 is not None else 0.5
        hand_a, hand_b = assign_hands(hands, labels, center_x)
        return FrameLandmarks(pose33, hand_a, hand_b)

    def close(self) -> None:
        self._pose.close()
        self._hands.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def draw_landmarks(frame: np.ndarray, lm: FrameLandmarks) -> None:
    h, w = frame.shape[:2]

    def pt(p):
        return int(p[0] * w), int(p[1] * h)

    if lm.pose33 is not None:
        for i, j in POSE_CONNECTIONS:
            cv2.line(frame, pt(lm.pose33[i]), pt(lm.pose33[j]), (200, 200, 200), 2)
    for hand, color in ((lm.hand_a, (0, 200, 255)), (lm.hand_b, (255, 160, 0))):
        if hand is None:
            continue
        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, pt(hand[i]), pt(hand[j]), color, 2)
        for p in hand:
            cv2.circle(frame, pt(p), 3, color, -1)
