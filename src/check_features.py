"""Check that our feature pipeline reproduces the datasettt format.

    python src/check_features.py

1. Verifies every datasettt clip satisfies the decoded layout.
2. If clips were recorded (src/record.py), compares them with datasettt:
   - value statistics for the same signs
   - a nearest-neighbour test: trained on datasettt, tested on your clips,
     next to datasettt's own cross-validation score as the ceiling
   - formula variants recomputed from the raw landmarks, to confirm the
     chosen one matches best
Writes reports/feature_check.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import DATASETTT_DIR, RECORDED_DIR, RECORDED_RAW_DIR, REPORTS_DIR, SIGNS
from features import FLAG_A, FLAG_B, POSE, POSE_IDX, WRIST_A, check_invariants, frame_features, to_sequence

K = 5
FOLDS = 5


def load_dir(root: Path, labels=SIGNS) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    xs, ys, paths = [], [], []
    for label in labels:
        for p in sorted((root / label).glob("*.npy")):
            xs.append(np.load(p).astype(np.float32))
            ys.append(label)
            paths.append(p)
    if not xs:
        return np.zeros((0, 40, 174), np.float32), np.array([]), []
    return np.stack(xs), np.array(ys), paths


def descriptor(X: np.ndarray) -> np.ndarray:
    segs = np.array_split(X, 4, axis=1)
    return np.concatenate([s.mean(1) for s in segs] + [X.std(1)], axis=1)


class KNN:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        D = descriptor(X)
        self.mu, self.sd = D.mean(0), D.std(0) + 1e-6
        Z = (D - self.mu) / self.sd
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        self.y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (descriptor(X) - self.mu) / self.sd
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        nn = np.argsort(-(Z @ self.Z.T), axis=1)[:, :K]
        out = []
        for row in nn:
            labels, counts = np.unique(self.y[row], return_counts=True)
            out.append(labels[np.argmax(counts)])
        return np.array(out)


def cv_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    rng = np.random.default_rng(0)
    fold = np.zeros(len(y), int)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        fold[rng.permutation(idx)] = np.arange(len(idx)) % FOLDS
    correct = 0
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        correct += int((KNN().fit(X[tr], y[tr]).predict(X[te]) == y[te]).sum())
    return correct / len(y)


def summary(X: np.ndarray) -> dict:
    F = X.reshape(-1, X.shape[-1])
    a, b = F[:, FLAG_A] > 0.5, F[:, FLAG_B] > 0.5
    uniq = [40 - int(np.all(np.isclose(c[1:], c[:-1]), axis=1).sum()) for c in X]
    wrist = X[:, :, 126 + 27:126 + 29]
    return {
        "clips": int(len(X)),
        "hand_A_coverage": round(float(a.mean()), 3),
        "hand_B_coverage": round(float(b.mean()), 3),
        "both_hands_coverage": round(float((a & b).mean()), 3),
        "hand_A_wrist_mean_xyz": np.round(F[a][:, WRIST_A].mean(0), 3).tolist() if a.any() else None,
        "pose_mean_xyz": np.round(F[:, POSE].reshape(-1, 12, 3).mean(0), 3).tolist(),
        "real_frames_per_clip_median": float(np.median(uniq)),
        "pose_wrist_motion_per_frame": round(float(np.linalg.norm(np.diff(wrist, axis=1), axis=2).mean()), 4),
    }


def stat_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Mean |difference| of per-column means over pose + wrist columns."""
    cols = list(range(126, 168))
    return float(np.abs(A.reshape(-1, 174)[:, cols].mean(0) - B.reshape(-1, 174)[:, cols].mean(0)).mean())


def recompute(raw: dict, variant: str, resolution) -> np.ndarray | None:
    pose, ha, hb = raw["pose33"].copy(), raw["hand_a"].copy(), raw["hand_b"].copy()
    pose_idx = POSE_IDX
    if variant == "hip23":
        pose_idx = POSE_IDX[:-1] + (23,)
    elif variant == "aspect_corrected" and resolution:
        w, h = resolution
        for arr in (pose, ha, hb):
            arr[..., 0] *= w / h
            arr[..., 2] *= w / h
    elif variant == "flipped":
        for arr in (pose, ha, hb):
            arr[..., 0] = 1.0 - arr[..., 0]
        pose = pose[:, [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21,
                        24, 23, 26, 25, 28, 27, 30, 29, 32, 31]]
        ha, hb = hb, ha
    frames = []
    for p, a, b in zip(pose, ha, hb):
        f = frame_features(
            None if np.isnan(p).any() else p,
            None if np.isnan(a).any() else a,
            None if np.isnan(b).any() else b,
            pose_idx=pose_idx,
        )
        if f is not None:
            frames.append(f)
    return to_sequence(frames) if frames else None


def load_manifest() -> dict[str, dict]:
    path = RECORDED_DIR / "manifest.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[Path(rec["file"]).stem] = rec
    return out


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    Xd, yd, _ = load_dir(DATASETTT_DIR)
    bad = sum(bool(check_invariants(c)) for c in Xd)
    print(f"datasettt: {len(Xd)} clips, {len(np.unique(yd))} signs, layout violations: {bad}")
    report["datasettt"] = {"violations": bad, **summary(Xd)}
    ceiling = cv_accuracy(Xd, yd)
    report["datasettt"]["knn_cv_accuracy"] = round(ceiling, 4)
    print(f"datasettt nearest-neighbour {FOLDS}-fold accuracy (ceiling): {ceiling:.1%}")

    Xr, yr, paths = load_dir(RECORDED_DIR)
    if len(Xr) == 0:
        print("\nNo recorded clips yet. Record a few first, e.g.:\n"
              "  python src/record.py --label hello --count 5")
        (REPORTS_DIR / "feature_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    bad_r = [(p.name, check_invariants(c)) for p, c in zip(paths, Xr) if check_invariants(c)]
    same = np.isin(yd, np.unique(yr))
    knn = KNN().fit(Xd, yd)
    pred = knn.predict(Xr)
    acc = float((pred == yr).mean())
    report["recorded"] = {
        "violations": [f"{n}: {'; '.join(v)}" for n, v in bad_r],
        "signs": sorted(np.unique(yr).tolist()),
        **summary(Xr),
        "knn_accuracy_vs_datasettt": round(acc, 4),
        "stat_distance_same_signs": round(stat_distance(Xd[same], Xr), 4),
        "per_clip": [{"file": p.name, "label": t, "pred": q} for p, t, q in zip(paths, yr, pred)],
    }
    report["datasettt_same_signs"] = summary(Xd[same])

    manifest = load_manifest()
    variants = {}
    for variant in ("chosen", "hip23", "aspect_corrected", "flipped"):
        Xv, yv = [], []
        for p, label in zip(paths, yr):
            raw_path = RECORDED_RAW_DIR / label / f"{p.stem}.npz"
            if not raw_path.exists():
                continue
            with np.load(raw_path) as z:
                raw = {k: z[k] for k in z.files}
            seq = recompute(raw, variant, manifest.get(p.stem, {}).get("resolution"))
            if seq is not None:
                Xv.append(seq)
                yv.append(label)
        if Xv:
            Xv, yv = np.stack(Xv), np.array(yv)
            variants[variant] = {
                "knn_accuracy": round(float((knn.predict(Xv) == yv).mean()), 4),
                "stat_distance": round(stat_distance(Xd[np.isin(yd, np.unique(yv))], Xv), 4),
            }
    report["variants"] = variants
    (REPORTS_DIR / "feature_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    d, r = report["datasettt_same_signs"], report["recorded"]
    print(f"\nrecorded: {len(Xr)} clips over {len(r['signs'])} signs, layout violations: {len(bad_r)}")
    print(f"{'':30s}{'datasettt':>14s}{'recorded':>14s}")
    for key in ("hand_A_coverage", "hand_B_coverage", "both_hands_coverage",
                "real_frames_per_clip_median", "pose_wrist_motion_per_frame"):
        print(f"{key:30s}{d[key]:>14}{r[key]:>14}")
    print(f"{'hand_A_wrist_mean_xyz':30s}{str(d['hand_A_wrist_mean_xyz']):>14}  {r['hand_A_wrist_mean_xyz']}")
    print(f"\nnearest-neighbour accuracy on your clips: {acc:.1%}  (datasettt ceiling {ceiling:.1%})")
    print(f"stat distance to datasettt (same signs): {r['stat_distance_same_signs']:.4f}  (lower is better)")
    if variants:
        print("\nformula variants (recomputed from raw landmarks):")
        for name, v in variants.items():
            print(f"  {name:18s} accuracy {v['knn_accuracy']:.1%}   stat distance {v['stat_distance']:.4f}")
    print(f"\nreport: {REPORTS_DIR / 'feature_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
