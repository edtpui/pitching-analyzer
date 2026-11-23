# Pitch Mechanics Analyzer

Upload a phone video of one pitch and get back a skeleton overlay, the four key positions (peak knee
height, foot plant, max layback, release), 26 mechanics metrics with percentiles against 411 pitches from
100 Driveline pitchers, the kinematic sequence, model estimates of velocity and elbow torque, and a
coaching summary (rule-based, or written by Claude if an API key is set).

Everything runs locally: pose estimation is MediaPipe on the CPU, and the web app is FastAPI plus plain
HTML/JS. The only network call is the optional Claude report, which sends numbers, never the video.

## Run it

```powershell
.\run.ps1                      # http://127.0.0.1:8091
```

First-time setup, if `.venv` or `assets\` is missing:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart mediapipe==1.2.3 numpy scipy pandas scikit-learn joblib imageio-ffmpeg anthropic pytest httpx
.\.venv\Scripts\python.exe scripts\fetch_assets.py        # Driveline data (sha256-checked) + pose models, ~75 MB
.\.venv\Scripts\python.exe scripts\build_reference.py     # calibration, validation, reference, models (~5 min)
```

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to enable the Claude report
(model `claude-opus-5-5`, server-side refusal fallback enabled). Each report is generated only when you
click the button and is billed to that key.

## Docker (tailnet)

```powershell
docker compose up -d --build                      # container "pitching-analyzer" on 127.0.0.1:3202
tailscale serve --bg --https=8444 http://127.0.0.1:3202   # once; persists across reboots
```

Tailnet devices then use `https://alienware.tailbfde2b.ts.net:8444`. The container listens on
localhost only; it is not on the LAN and there is no Funnel. Analyses live in the `pitchlab_data`
volume, separate from the local `data\` folder used by `run.ps1`. Optional settings go in `.env`
(`PITCHLAB_PASSWORD` turns on a login prompt, `ANTHROPIC_API_KEY` enables Claude reports). Run
`docker compose up -d` again after changing it. `docker build --target test .` runs the test suite in the
Linux image. To stop sharing: `tailscale serve --https=8444 off`.

## Filming

- Slow motion: 240 fps if possible, 120 fps minimum. At 30/60 fps the arm at release is a blur.
- Tripod; no zooming or panning. Foot plant, stride and momentum are measured against the still image.
- Side-on from the arm side (third-base side for a right-hander), camera square to the mound-home line,
  about hip height. Straight behind also works but loses stride length and momentum.
- Whole body in frame from leg lift through follow-through; one pitch per clip.
- If a slow-motion clip was exported to play slowed down, enter the real capture rate (e.g. 240) on upload.

## How it works

1. **Pose:** MediaPipe Pose Landmarker (full / heavy / lite) gives 2D image landmarks and pelvis-centred 3D
   landmarks for every frame.
2. **Clean-up:** left/right swaps (common side-on) are repaired, short gaps interpolated, the throwing arm
   identified by wrist speed, and joints low-pass filtered (12 Hz body, 18 Hz arms).
3. **Lab frame:** the stride at foot plant sets the direction to home. A square camera keeps the image axis
   instead; left-handers are mirrored so every formula assumes a right-hander.
4. **Events:** release is when the throwing hand passes 0.40 leg lengths in front of the shoulder (side-on; peak wrist speed from other views), bridging frames where the arm is blurred. Peak knee
   height is the highest knee point before that. Foot plant is the first frame after peak knee height at
   which the lead ankle is slow and has arrived where it sits planted at release. Max layback is the peak
   external rotation between foot plant and release, or 31 ms before release (Driveline median) when the arm
   is blurred through layback - layback angles are then not reported.
5. **Metrics** are geometric definitions (`pitchlab/metrics.py`), identical for video and for the
   Driveline data.
6. **Reference** (`scripts/build_reference.py`): the same code runs on Driveline's 360 Hz joint centres
   after simulating phone video. That means resampling to 240/120/60 fps, per-frame plus slowly drifting
   joint noise, pelvis-centring, the app's filter, and a perspective camera at a random tripod position
   for the image tracks. Percentiles, confidence ratings, event calibration and the models all come from
   these simulated clips.

## Accuracy (measured on the Driveline data)

**Validity** is this app's definition at Driveline's own events, compared with Driveline's published
marker-based value. **Reliability** is the same metric from simulated video with the app's own event
detection, compared with the clean value. Metrics with validity ≥ 0.7 are shown on Driveline's scale
(linear mean/SD mapping). Low-confidence metrics are displayed but never drive coaching findings.

| Metric | Validity r | Reliability r (240 / 120 / 60 fps) | Confidence at 240 fps |
|---|---|---|---|
| Stride length | 0.87 | 0.96 / 0.95 / 0.94 | high |
| Stride direction | 0.98 | 0.90 / 0.91 / 0.88 | high |
| Lead knee flexion at foot plant | – | 0.83 / 0.79 / 0.75 | high |
| Lead knee flexion at release | – | 0.98 / 0.96 / 0.94 | high |
| Lead knee extension, contact to release | 0.97 | 0.89 / 0.83 / 0.78 | high |
| Peak pelvis speed toward home | 0.66 | 0.94 / 0.92 / 0.91 | medium |
| Pelvis rotation at foot plant | 0.81 | 0.49 / 0.47 / 0.38 | low |
| Torso rotation at foot plant | 0.75 | 0.57 / 0.51 / 0.37 | low |
| Hip-shoulder separation at foot plant | 0.83 | 0.82 / 0.77 / 0.66 | high |
| Peak hip-shoulder separation | 0.82 | 0.95 / 0.93 / 0.89 | high |
| Trunk forward tilt at foot plant | 0.83 | 0.66 / 0.68 / 0.54 | medium |
| Trunk forward tilt at release | 0.91 | 0.98 / 0.97 / 0.94 | high |
| Trunk side tilt at release | 0.81 | 0.93 / 0.90 / 0.80 | high |
| Shoulder abduction at foot plant | 0.86 | 0.87 / 0.79 / 0.71 | high |
| Shoulder horizontal abduction at foot plant | 0.91 | 0.85 / 0.80 / 0.73 | high |
| Shoulder external rotation at foot plant | 0.80 | 0.55 / 0.48 / 0.44 | low |
| Elbow flexion at foot plant | 0.96 | 0.81 / 0.78 / 0.70 | high |
| Max shoulder external rotation (layback) | 0.82 | 0.81 / 0.56 / 0.35 | high |
| Elbow flexion at max layback | 0.99 | 0.69 / 0.63 / 0.39 | medium |
| Arm slot | 0.92 | 0.98 / 0.96 / 0.94 | high |
| Peak pelvis rotation speed | 0.93 | 0.84 / 0.76 / 0.71 | high |
| Peak torso rotation speed | 0.52 | 0.91 / 0.86 / 0.78 | low |
| Peak elbow extension speed | 0.94 | 0.50 / 0.49 / 0.45 | low |
| Peak shoulder internal rotation speed | 0.58 | 0.46 / 0.29 / 0.03 | low |
| Pelvis-to-torso peak timing | 0.31 | 0.63 / 0.57 / 0.51 | low |
| Foot plant to release | – | 0.31 / 0.35 / 0.29 | low |

Event timing on simulated video, median / 90th-percentile absolute error, with the throwing arm blurred out of tracking around release in 10% / 35% / 70% of clips at 240 / 120 / 60 fps: 240 fps: release 2.8 / 8 ms, foot plant 10 / 34 ms, max layback 2.8 / 8 ms, peak knee height 7 / 19 ms; 60 fps: release 5.6 / 17 ms, foot plant 17 / 62 ms, max layback 5.6 / 19 ms, peak knee height 14 / 31 ms.

Models (5-fold cross-validation grouped by athlete, scored on simulated 240 fps clips):

| Target | Inputs | MAE | R² | Guessing the mean | Height + weight only |
|---|---|---|---|---|---|
| Pitch velocity | video metrics | 3.5 mph | 0.14 (hidden in app) | 3.8 mph | – |
| Pitch velocity | video metrics + height/weight | 3.2 mph | 0.27 | 3.8 mph | 3.7 mph |
| Elbow varus torque | video metrics | 15.4 N·m | 0.07 (hidden in app) | 15.6 N·m | – |
| Elbow varus torque | video metrics + height/weight | 10.8 N·m | 0.46 | 15.6 N·m | 11.8 N·m |

Predictions with cross-validated R² below 0.25 are not shown as numbers.

## Limitations

- All accuracy numbers above come from **simulated** video built from lab data. Real phone footage adds
  motion blur, occlusion and pose-model bias (MediaPipe's 3D depth in particular) that the simulation only
  approximates, so treat the ratings as best case.
- The reference group is 100 pitchers, mostly college (314 of 411 pitches), 70-94 mph. Youth pitchers sit
  outside it.
- Elbow torque is a model estimate driven mostly by body size, not a measurement. It is not a medical or
  injury-risk assessment.
- One pitcher in frame works best. With several people, the largest person in the first detection is
  tracked.

## Data licence

The Driveline OpenBiomechanics data is CC BY-NC-SA 4.0, with additional exclusions for professional sports
organisations and financial-analysis firms (https://openbiomechanics.org/#terms). This project uses it for
non-commercial analysis. Anything built on the reference bundle or the models inherits those terms.
MediaPipe models are Apache-2.0.

## Layout

```
pitchlab/            analysis package + FastAPI server (server.py) + static web UI (static/)
  pose.py            MediaPipe landmarker over a video
  preprocess.py      swap repair, gap fill, handedness, lab frame
  events.py          release / peak knee height / foot plant / max layback
  kinematics.py      angle and rotation time series
  metrics.py         the 26 metric definitions
  driveline.py       Driveline loader + phone-video simulation
  reference.py       percentiles, display mapping, confidence
  model.py           velocity / elbow-torque models
  coaching.py        rule findings + Claude report
  pipeline.py, jobs.py, render.py, video_io.py
scripts/             fetch_assets.py, build_reference.py
tests/               pytest suite (synthetic skeletons, virtual-camera Driveline clips, API, rendering)
assets/              downloaded data, models and the built reference (gitignored)
data/analyses/       uploads and results, one folder per analysis (gitignored)
```

## Tests

```powershell
$env:PYTHONUTF8 = '1'; .\.venv\Scripts\python.exe -m pytest
```

The camera tests render Driveline pitches through a virtual camera into MediaPipe-format landmarks and
check that the video path reproduces the lab metrics. That covers right- and left-handers, side and behind
views, angled cameras, a wrong handedness setting, and slow-motion capture rates. The Claude call is tested
against a fake client, so it makes no API requests.
