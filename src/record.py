"""Record sign clips from the laptop webcam in the datasettt format.

    python src/record.py --label hello --count 5
    python src/record.py --all --count 5        # every sign, then "none"

Keys:  SPACE start / pause   R delete last clip   N next sign   Q quit

Each clip is saved twice:
    data/recorded/<label>/<label>_NNNN.npy       (40, 174) training features
    data/recorded_raw/<label>/<label>_NNNN.npz   raw landmarks, so features can be
                                                 recomputed without re-recording
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    CLIP_SECONDS,
    DEFAULT_POSE_MODEL,
    MIN_HAND_COVERAGE,
    MIN_POSE_COVERAGE,
    MIRROR,
    NONE_LABEL,
    POSE_MODELS,
    RECORDED_DIR,
    RECORDED_RAW_DIR,
    SIGNS,
)
from features import check_invariants, to_sequence
from mp_runtime import FrameLandmarks, LandmarkExtractor, draw_landmarks

REST_SECONDS = 1.5
WINDOW = "ISLSignTalk recorder"


def next_index(label: str) -> int:
    pat = re.compile(rf"^{re.escape(label)}_(\d+)\.npy$")
    nums = [int(m.group(1)) for p in (RECORDED_DIR / label).glob("*.npy") if (m := pat.match(p.name))]
    return max(nums, default=0) + 1


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _stack(items, shape) -> np.ndarray:
    return np.stack([x if x is not None else np.full(shape, np.nan, np.float32) for x in items])


def save_clip(label: str, buf: list[FrameLandmarks], times_ms: list[int], meta: dict) -> tuple[bool, str, list[Path]]:
    n = len(buf)
    feats = [lm.features() for lm in buf]
    pose_cov = sum(f is not None for f in feats) / max(n, 1)
    hand_cov = sum(lm.hand_a is not None or lm.hand_b is not None for lm in buf) / max(n, 1)
    if pose_cov < MIN_POSE_COVERAGE:
        return False, f"REJECTED: body visible in {pose_cov:.0%} of frames - step back", []
    if label != NONE_LABEL and hand_cov < MIN_HAND_COVERAGE:
        return False, f"REJECTED: hands visible in {hand_cov:.0%} of frames - keep hands in view", []

    seq = to_sequence([f for f in feats if f is not None])
    problems = check_invariants(seq)
    if problems:
        return False, "REJECTED: " + "; ".join(problems), []

    idx = next_index(label)
    stem = f"{label}_{idx:04d}"
    feat_path = RECORDED_DIR / label / f"{stem}.npy"
    raw_path = RECORDED_RAW_DIR / label / f"{stem}.npz"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, seq)
    np.savez_compressed(
        raw_path,
        pose33=_stack([lm.pose33 for lm in buf], (33, 3)),
        hand_a=_stack([lm.hand_a for lm in buf], (21, 3)),
        hand_b=_stack([lm.hand_b for lm in buf], (21, 3)),
        t_ms=np.asarray(times_ms, dtype=np.int64),
    )
    record = {
        "file": str(feat_path.relative_to(RECORDED_DIR.parent)),
        "raw": str(raw_path.relative_to(RECORDED_DIR.parent)),
        "label": label,
        "raw_frames": n,
        "fps": round(n / max((times_ms[-1] - times_ms[0]) / 1000.0, 1e-3), 2) if n > 1 else 0,
        "pose_coverage": round(pose_cov, 3),
        "hand_a_coverage": round(sum(lm.hand_a is not None for lm in buf) / n, 3),
        "hand_b_coverage": round(sum(lm.hand_b is not None for lm in buf) / n, 3),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        **meta,
    }
    with (RECORDED_DIR / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return True, f"saved {stem}  ({n} frames, hands {hand_cov:.0%})", [feat_path, raw_path]


def put(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="Record clips in the datasettt (40x174) format")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", help=f"one of: {', '.join(SIGNS + (NONE_LABEL,))}")
    group.add_argument("--all", action="store_true", help="record every sign, then 'none'")
    ap.add_argument("--count", type=int, default=5, help="clips per sign")
    ap.add_argument("--seconds", type=float, default=CLIP_SECONDS, help="clip length")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    ap.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    ap.add_argument("--pose-model", choices=sorted(POSE_MODELS), default=DEFAULT_POSE_MODEL)
    args = ap.parse_args()

    if args.label and args.label not in SIGNS + (NONE_LABEL,):
        ap.error(f"unknown label {args.label!r}")
    labels = list(SIGNS) + [NONE_LABEL] if args.all else [args.label]
    mirror = MIRROR and not args.no_mirror

    cap = open_camera(args.camera, args.width, args.height)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Camera {args.camera}: {actual[0]}x{actual[1]}  mirror={mirror}  pose={args.pose_model}")
    meta = {"seconds": args.seconds, "resolution": list(actual), "mirror": mirror, "pose_model": args.pose_model}

    extractor = LandmarkExtractor(args.pose_model)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    t0 = time.monotonic()
    label_i, done = 0, 0
    state, state_t = "wait", 0.0
    buf: list[FrameLandmarks] = []
    times: list[int] = []
    saved_stack: list[list[Path]] = []
    message, fps, last_t = "", 0.0, time.monotonic()

    try:
        while label_i < len(labels):
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed", file=sys.stderr)
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now
            ts = int((now - t0) * 1000)
            lm = extractor.process(frame, ts)
            label = labels[label_i]

            if state == "rest" and now - state_t >= REST_SECONDS:
                state, state_t, buf, times = "recording", now, [], []
            if state == "recording":
                buf.append(lm)
                times.append(ts)
                if now - state_t >= args.seconds:
                    ok_save, message, paths = save_clip(label, buf, times, meta)
                    print(f"[{label}] {message}")
                    if ok_save:
                        saved_stack.append(paths)
                        done += 1
                    if done >= args.count:
                        label_i, done, state = label_i + 1, 0, "wait"
                        saved_stack.clear()
                    else:
                        state, state_t = "rest", now

            view = frame.copy()
            draw_landmarks(view, lm)
            h, w = view.shape[:2]
            if state == "recording":
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                frac = min((now - state_t) / args.seconds, 1.0)
                cv2.rectangle(view, (0, h - 12), (int(w * frac), h), (0, 0, 255), -1)
                put(view, "SIGN NOW", (w - 170, 40), 0.9, (0, 0, 255))
            elif state == "rest":
                put(view, f"get ready {REST_SECONDS - (now - state_t):.1f}", (w - 230, 40), 0.8, (0, 255, 255))
            else:
                put(view, "SPACE = start", (w - 210, 40), 0.8, (0, 255, 0))

            shown = labels[label_i] if label_i < len(labels) else label
            hint = "rest / move naturally, do NOT sign" if shown == NONE_LABEL else "sign once, hands in view"
            put(view, f"{shown}  clip {done + 1}/{args.count}", (12, 34), 0.9)
            put(view, hint, (12, 64), 0.6, (220, 220, 220))
            hands = f"A:{'yes' if lm.hand_a is not None else '--'}  B:{'yes' if lm.hand_b is not None else '--'}"
            body = "body ok" if lm.pose33 is not None else "NO BODY"
            put(view, f"{hands}  {body}  {fps:4.1f} fps", (12, h - 24), 0.6)
            if message:
                put(view, message, (12, 94), 0.6, (0, 255, 0) if message.startswith("saved") else (0, 0, 255))
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if state == "wait":
                    state, state_t, message = "rest", now, ""
                else:
                    state, message = "wait", "paused"
            elif key == ord("r") and saved_stack and state != "recording":
                for p in saved_stack.pop():
                    p.unlink(missing_ok=True)
                done = max(done - 1, 0)
                message = "deleted last clip"
                print(f"[{label}] {message}")
            elif key == ord("n") and state != "recording":
                label_i, done, state = label_i + 1, 0, "wait"
                saved_stack.clear()
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
