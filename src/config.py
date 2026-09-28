"""Project-wide constants. Recording, training and live detection all read these."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASETTT_DIR = DATA_DIR / "datasettt"
RECORDED_DIR = DATA_DIR / "recorded"
RECORDED_RAW_DIR = DATA_DIR / "recorded_raw"
MODELS_DIR = ROOT / "models"
MEDIAPIPE_DIR = MODELS_DIR / "mediapipe"
REPORTS_DIR = ROOT / "reports"

SIGNS = (
    "alright", "friday", "friend", "good_afternoon", "good_morning", "hello",
    "hour", "how_are_you", "i", "minute", "monday", "month", "saturday",
    "sunday", "thursday", "today", "tomorrow", "tuesday", "wednesday", "week",
    "year", "yesterday", "you",
)
NONE_LABEL = "none"

SEQ_LEN = 40
FEAT_DIM = 174

# datasettt clips hold ~31-33 real frames each (stretched to 40), captured on a
# laptop webcam. These are the recording defaults that reproduce that.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CLIP_SECONDS = 2.0
MIRROR = True

MP_MIN_CONF = 0.5
POSE_MODELS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
DEFAULT_POSE_MODEL = "full"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MIN_HAND_COVERAGE = 0.5
MIN_POSE_COVERAGE = 0.9
