# ISLSignTalk

Live Indian Sign Language detection for 23 signs, built around the `datasettt`
landmark dataset (702 clips, laptop webcam).

Signs: alright, friday, friend, good_afternoon, good_morning, hello, hour,
how_are_you, i, minute, monday, month, saturday, sunday, thursday, today,
tomorrow, tuesday, wednesday, week, year, yesterday, you (+ `none` for "not signing").

## Setup

```powershell
cd C:\Sarastra\ISLSignTalk
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

## Layout

```
data/datasettt/       original dataset (read-only copy), <sign>/<sign>_NNNN.npy, each (40, 174)
data/recorded/        clips recorded with src/record.py, same format + manifest.jsonl
data/recorded_raw/    raw landmarks for every recorded clip (.npz)
src/config.py         constants shared by every script
src/features.py       MediaPipe landmarks -> 174 features (the single source of truth)
src/mp_runtime.py     MediaPipe pose + hand landmarkers (VIDEO mode)
src/record.py         webcam recorder
src/check_features.py checks our features against datasettt
models/  reports/
```

## Feature format (decoded from datasettt)

Every frame has 174 numbers. Every clip is 40 frames: about 32 real frames from a
~2 s recording, stretched with `round(linspace(0, n-1, 40))`.

| Columns | Content |
|---|---|
| 0-62 | hand A (MediaPipe label "Right"), 21 points x/y/z minus wrist, divided by the largest point distance |
| 63-125 | hand B (label "Left"), same |
| 126-161 | 12 body points (MediaPipe pose 0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 24), minus shoulder midpoint, divided by 3D shoulder distance |
| 162-164 | hand A wrist (from the hand model) in the same body frame |
| 165-167 | hand B wrist, same |
| 168-171 | wrist B - wrist A (x, y, z) and its length; zero unless both hands are visible |
| 172, 173 | hand A / hand B present (0 or 1) |

Column 170 is always 0 in the dataset. That's expected: the hand model reports
wrist z as 0, so the z difference between the two wrists is always 0.

`python src/check_features.py` confirms all 702 dataset clips follow this layout.

## Step 1: record a check set (about 10 minutes)

```powershell
python src/record.py --label hello --count 5
python src/record.py --label friday --count 5
python src/record.py --label how_are_you --count 5
python src/record.py --label i --count 5
python src/record.py --label you --count 5
python src/check_features.py
```

Keys: SPACE start/pause, R delete last clip, N next sign, Q quit.
Sit so your head, shoulders and both hands are in view. Sign once per red
recording window (2 s).

`check_features.py` reports:
- **nearest-neighbour accuracy on your clips** vs the dataset's own ceiling.
  Close to the ceiling means our features match the dataset.
- **stat distance** and **formula variants**: which formula choice (hip point,
  aspect ratio, mirroring) matches the dataset best.

## Step 2 (next): full recording + training + live detection

- `python src/record.py --all --count 20` records all 23 signs plus `none`
- `src/train.py` trains the model on datasettt + recorded clips and writes a report (coming next)
- `src/live.py` runs live detection in an OpenCV window, then a web page (coming next)
