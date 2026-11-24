# Pitch Mechanics Analyzer

Upload a phone video of one pitch and get back a skeleton overlay, the four key positions (peak knee
height, foot plant, max layback, release), 26 mechanics metrics with percentiles against 411 pitches from
100 Driveline pitchers, the kinematic sequence, model estimates of velocity and elbow torque, and a
coaching summary (rule-based, or written by Claude if an API key is set).

A second section, **Pitch tracking**, measures the ball instead of the body: from a slow-motion clip filmed
from just behind the pitcher, aimed at the catcher (as PitchLab films), it gives release and plate velocity, induced vertical and horizontal break, release point,
plate location, and spin (rate, axis, efficiency and gyro from the seams; tilt and active spin from the
movement). See [Pitch tracking](#pitch-tracking-velocity-movement-and-spin).

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

Tailnet devices then use `https://<your-machine>.<your-tailnet>.ts.net:8444`. The container listens on
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
   identified by wrist speed, and joints low-pass filtered (12 Hz body, 18 Hz arms). Throwing-arm frames
   whose positions are guesses are dropped and bridged: low landmark confidence (blur; the bar drops from
   0.7 to 0.5 when the arm rarely reaches 0.7, as in dim light), frames where MediaPipe mirrored the whole
   body (the legs had to be swapped back), and single confident frames inside a blurred stretch.
3. **Lab frame:** the stride at foot plant sets the direction to home. A square camera keeps the image axis
   instead; left-handers are mirrored so every formula assumes a right-hander.
4. **Events:** release is when the throwing hand passes 0.40 leg lengths in front of the shoulder (side-on; peak wrist speed from other views), bridging frames where the arm is blurred. When the arm is
   lost from layback until well after release (dim light at 60 fps), release is extrapolated from the
   hand's last tracked position with Driveline's median arm path and marked *estimated*. Peak knee
   height is the highest knee point before that. Foot plant is the first frame after peak knee height at
   which the lead ankle is slow and has arrived where it sits planted at release. In side-on video the
   lead heel and toe are used too: the same test on them finds the touch-down (heel or toe, whichever
   lands first), and foot plant is Driveline's median 19 ms from first contact to full weight after it,
   if that comes first - MediaPipe's ankle point sits on the lower shin and keeps creeping forward while
   the shin rolls over the planted foot. Max layback is the peak
   external rotation between foot plant and release (skipping frames where the elbow is bent less than
   30 degrees: the rotation angle is taken from the forearm and is undefined with a straight arm), or 31 ms
   before release (Driveline median) when the arm is blurred through layback - layback angles are then not
   reported.
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
| Stride length | 0.87 | 0.96 / 0.95 / 0.95 | high |
| Stride direction | 0.98 | 0.91 / 0.88 / 0.88 | high |
| Lead knee flexion at foot plant | – | 0.82 / 0.80 / 0.77 | high |
| Lead knee flexion at release | – | 0.97 / 0.97 / 0.94 | high |
| Lead knee extension, contact to release | 0.97 | 0.87 / 0.84 / 0.82 | high |
| Peak pelvis speed toward home | 0.66 | 0.95 / 0.93 / 0.91 | medium |
| Pelvis rotation at foot plant | 0.81 | 0.49 / 0.38 / 0.34 | low |
| Torso rotation at foot plant | 0.75 | 0.54 / 0.50 / 0.36 | low |
| Hip-shoulder separation at foot plant | 0.83 | 0.83 / 0.81 / 0.65 | high |
| Peak hip-shoulder separation | 0.82 | 0.95 / 0.93 / 0.89 | high |
| Trunk forward tilt at foot plant | 0.83 | 0.67 / 0.65 / 0.58 | medium |
| Trunk forward tilt at release | 0.91 | 0.98 / 0.97 / 0.94 | high |
| Trunk side tilt at release | 0.81 | 0.92 / 0.90 / 0.80 | high |
| Shoulder abduction at foot plant | 0.86 | 0.83 / 0.78 / 0.71 | high |
| Shoulder horizontal abduction at foot plant | 0.91 | 0.84 / 0.81 / 0.69 | high |
| Shoulder external rotation at foot plant | 0.80 | 0.52 / 0.46 / 0.29 | low |
| Elbow flexion at foot plant | 0.96 | 0.82 / 0.79 / 0.68 | high |
| Max shoulder external rotation (layback) | 0.82 | 0.90 / 0.85 / 0.80 | high |
| Elbow flexion at max layback | 0.99 | 0.77 / 0.67 / 0.55 | medium |
| Arm slot | 0.92 | 0.98 / 0.96 / 0.93 | high |
| Peak pelvis rotation speed | 0.93 | 0.84 / 0.72 / 0.68 | high |
| Peak torso rotation speed | 0.52 | 0.91 / 0.83 / 0.81 | low |
| Peak elbow extension speed | 0.94 | 0.55 / 0.42 / 0.54 | low |
| Peak shoulder internal rotation speed | 0.58 | 0.40 / 0.17 / 0.31 | low |
| Pelvis-to-torso peak timing | 0.31 | 0.68 / 0.61 / 0.48 | low |
| Foot plant to release | – | 0.33 / 0.31 / 0.26 | low |

Event timing on simulated video, median / 90th-percentile absolute error, with the throwing arm blurred out of tracking around release in 10% / 35% / 70% of clips at 240 / 120 / 60 fps: 240 fps: release 2.8 / 8 ms, foot plant 11 / 40 ms, max layback 2.8 / 8 ms, peak knee height 7 / 22 ms; 60 fps: release 5.6 / 17 ms, foot plant 19 / 72 ms, max layback 5.6 / 17 ms, peak knee height 14 / 31 ms.

When the arm is lost from shortly before release until well after it (a separate simulation at 60 fps, arm
untracked until 6 frames after release), the estimated release is within 2 frames of the true one in 97% /
94% / 90% of pitches if tracking ended 1 / 2 / 3 frames before release, and in 40% if it ended 4 frames
before (then it lands about 2 frames early).

Models (5-fold cross-validation grouped by athlete, scored on simulated 240 fps clips):

| Target | Inputs | MAE | R² | Guessing the mean | Height + weight only |
|---|---|---|---|---|---|
| Pitch velocity | video metrics | 3.2 mph | 0.245 (hidden in app) | 3.8 mph | – |
| Pitch velocity | video metrics + height/weight | 3.0 mph | 0.34 | 3.8 mph | 3.7 mph |
| Elbow varus torque | video metrics | 15.4 N·m | -0.04 (hidden in app) | 15.6 N·m | – |
| Elbow varus torque | video metrics + height/weight | 10.7 N·m | 0.47 | 15.6 N·m | 11.8 N·m |

Predictions with cross-validated R² below 0.25 are not shown as numbers.

## Limitations

- All accuracy numbers above come from **simulated** video built from lab data. Real phone footage adds
  motion blur, occlusion and pose-model bias (MediaPipe's 3D depth in particular) that the simulation only
  approximates, so treat the ratings as best case.
- The reference group is 100 pitchers, mostly college (314 of 411 pitches), 70-94 mph. Youth pitchers sit
  outside it.
- Foot plant from the heel/toe touch-down is checked on real clips only (two so far, both within a frame of
  the foot going flat): the Driveline data has no foot markers, so the simulation cannot measure it. The
  19 ms from contact to full weight is Driveline's median; half of its pitchers take 14-28 ms.
- Night or indoor clips at 60 fps: MediaPipe can lose the throwing arm from layback until well after
  release and mirror the whole body for a few frames. The app drops those frames and estimates release
  from the arm's last tracked position (marked *estimated*), but filming in daylight or at 120-240 fps
  measures it instead.
- Elbow torque is a model estimate driven mostly by body size, not a measurement. It is not a medical or
  injury-risk assessment.
- One pitcher in frame works best. With several people, the largest person in the first detection is
  tracked.

## Pitch tracking (velocity, movement and spin)

### Filming

- **Just behind the pitcher (recommended, PitchLab's set-up):** tripod 3-6 ft behind the rubber and 1-2 ft
  to the throwing-arm side, about chest high, upright (portrait) or landscape, aimed at the catcher so
  **home plate is in the picture**. The ball is large and sharp as it leaves the hand, which is where the
  spin is read, and the release is always seen.
- **Keep filming until the ball is in the glove.** From behind, the ball flies straight away from the lens
  and is too small to size over its far half; the catch (just behind the plate) fixes the far end of the
  flight, the release the near end.
- **Beside the release** (tripod 5-8 ft to the arm side, level with the rubber, aimed down the line) also
  works, with wider velocity uncertainty.
- **Measure where the camera stands** with a tape: lens height above the ground, how far ahead of the front
  edge of the rubber along the line to home (negative behind it), and how far to the side of the
  mound-plate line. From up close the rubber is usually out of the picture and the plate is a thin sliver
  (about 40 x 3 px from behind the pitcher), too small to place the camera on its own, so these three
  numbers stand in for the rubber.
- **Behind home plate** (on the line to the mound, 10-15 ft back, 4-6 ft high) also works with no tape:
  mark the plate and the rubber instead. Spin is then read near the plate, where the ball is smaller.
- Home plate (and, behind the plate, the rubber) must be visible in at least one frame. No zoom, panning
  or tripod movement during the clip.
- **240 fps** and **1080p**. Seam spin needs 240 fps and the ball at least ~14 px across; at 120 fps you get
  velocity and movement, and the spin axis and active spin from the movement only. At 720p from the
  release the plate is only ~25 px wide, so the camera rests on the tape (see Limitations).
- Daylight and a fast shutter: the ball has to stand out from what is behind it.
- One pitch per clip; at most 10 s at 240 fps (trim on upload otherwise).

### Using it

1. **Pitch tracking -> New pitch**: upload the clip with the pitching distance (60 ft 6 in, 54, 50 or 46),
   throwing hand and, optionally, zone height, pitcher height and altitude.
2. **Mark the field**: click the five corners of home plate (any order) on the people-removed background
   or on any frame, zoomed in, and enter the taped camera position (height, ahead of the rubber, to the
   side). Filmed from behind the plate, click the two front corners of the rubber instead (the tape is then
   optional). *Check marks* solves the camera and draws the plate, rubber and strike zone over the picture;
   check that the reported camera position matches the tape. Marks from an earlier clip filmed from the
   same tripod position can be copied instead.
3. **Analyze pitch** (20-60 s): tracer video, numbers with one-SD uncertainties, strike-zone plot, and the
   seam fit (each ball crop with the fitted seam drawn on its stitches) for checking the spin by eye.
4. If the release is marked *not seen*, step the tracer to the frame where the ball leaves the hand and use
   **Set release here**: that turns the measurement into time of flight over a known distance.

### How it works (`pitchlab/flight/`)

1. **Camera** (`geometry.py`): focal length and pose from the marks (every assignment of clicks to corners
   is tried) and, when given, the taped camera position as a prior (height +-1.5 in, ground position +-3 in),
   then refined coarse-to-fine until the rendered outlines of the plate (and rubber) line up with the
   picture - hundreds of edge pixels instead of a few clicks, which matters because a 1 px error on a
   far object 40 px wide is a 2.5% error in every speed. Outlines under 32 px are not refined, and a fit
   whose marks miss by more than 3 px RMS is refused.
2. **Ball** (`detect.py`, `blob.py`, `track.py`, `measure.py`): blobs that differ from the static background
   inside the projected pitching corridor; chains of them that move smoothly; the chain that fits as a pitch
   (30-110 mph toward the plate, from the mound, through the zone area; anything motionless is rejected).
   Every frame is then re-measured where the fit predicts the ball: centre and size from half-level
   crossings of its outer edge on 48 radial lines and a circle fit (shading-, seam- and blur-tolerant).
3. **Flight** (`trajectory.py`): the nine-parameter constant-acceleration model (PITCHf/x) fitted through
   the calibrated camera. One camera cannot tell a pitch from a scaled copy of it about the lens, so the
   scale comes from the release distance (pitching distance minus extension) when the ball is seen leaving
   the hand, from the ball's apparent size (with a shared systematic factor, 5% prior), and from drag.
   Filmed from behind the release, the far end is anchored the same way at the catch: when the track stops
   inside the picture, before the video does, within 5 ft of the glove by the unanchored fit, its last point
   is taken as just short of the glove, 0.5 +- 1 ft behind the plate's back tip.
   Movement is TrackMan-style: release to the front of the plate, relative to a spinless pitch. Tilt and an
   active-spin estimate come from the Magnus acceleration (Nathan's 2008 lift model, about +-20%).
   Uncertainties combine the fit covariance with refits for 32 calibration samples.
4. **Spin** (`spin.py`): each large, sharp ball image becomes a seam map (darkness against a fitted shading
   surface, plus redness); 6000 orientations of a 3D seam curve are scored against it; frame-to-frame
   relative rotations (with the seam's 180-degree symmetries) vote for an angular velocity, refined jointly
   over up to 20 frames with the camera's view direction for each frame. Fits rated low are not shown.

### Accuracy (synthetic clips only)

There is no real footage with radar or tracking-unit truth in this repository, so every number below comes
from synthetic 240 fps clips (`tests/flight_synth.py`: drag-and-Magnus flight, a ray-traced seamed ball
whose seam shape and stitching differ from the estimator's, a moving pitcher, sensor noise, H.264 4:2:0),
with clicks perturbed by 1-1.5 px. They are a best case for the method, not a field validation.

| Measured on synthetic clips (1080p) | Behind the pitcher (PitchLab view, plate + tape) | Beside the release (plate + tape) | Behind home plate (plate + rubber) |
|---|---|---|---|
| Focal length after calibration | bias -1.9% | bias -1.9%; all 12 click draws within 4.5% | bias -0.55% |
| Release velocity (fastball / curveball) | -0.7 / -2.1 mph (reported +-1.5 / +-1.2) | +0.9 / -1.4 mph (reported +-3.7 / +-3.0) | +1.8 / +0.05 mph (reported +-1.6 / +-1.2) |
| IVB / HB | within 1.1 in / 0.6 in | within 0.3 in / 0.4 in | within 0.2 in / 0.8 in |
| Plate location | within 1.0 in | within 1.5 in | within 0.6 in |
| Extension | within 0.65 ft | within 0.4 ft | - |
| Seam spin rate / axis | within 0.3% / 2.4 degrees | within 0.2% / 2.5 degrees | curveball 0.2% / 1.2 degrees; fastball rated low (10.7 degrees, not shown) |

The synthetic catcher takes the ball 1 ft behind the plate's back tip, half a foot from the fit's 0.5 ft
assumption: that difference alone is about 1% of the speed. Velocity and spin figures from other apps
(PitchLab included) are estimates too, so a side-by-side on real pitches shows agreement, not error.
A camera raised well behind the pitcher (5 m back, 2.6 m up) measured +9.7 mph before the catch anchor
existed and has not been re-measured.

### Limitations

- Not validated on real footage. Real clips add rolling shutter (a few milliseconds of skew between the
  top and the bottom of the frame), lens distortion, autofocus and exposure changes, scuffed or printed
  balls, and backgrounds the ball blends into.
- Indoor facilities add LED flicker at 240 fps and other balls in flight in neighbouring tunnels; the
  tracker keeps the one chain that flies from the mound through the zone, but none of this is simulated.
- The ball's apparent size runs a few percent small when its lower half is in shadow; without a seen
  release the velocity rests on that size, hence the wider uncertainty.
- Seam spin needs 240 fps (unambiguous below a quarter turn per frame: 3600 rpm at 240 fps) and a close,
  sharp ball; logos and dirt on the ball are not modelled.
- From behind the pitcher, the speed leans on where the glove takes the ball (0.5 +- 1 ft behind the plate,
  about 0.8 mph per half foot). A pitch that gets past the catcher and is lost within 5 ft of the plate,
  or a catcher set well forward or back, shifts it; the result says when the catch was used.
- From the release, the camera's scale rests on the taped position and a small, foreshortened plate: the
  focal length runs about 2% short (so speeds about 1-2 mph off), and a tape error of a few inches moves
  it further. At 720p the plate is too small to refine and about 1 in 12 click draws is refused as
  inconsistent; film at 1080p.
- Air density comes from the altitude entered (sea level by default); temperature and humidity are ignored.

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
  flight/            pitch tracking: geometry (camera), detect/blob/track/measure (ball), trajectory, spin,
                     render (tracer, seam strip), pipeline, jobs, api (routes under /api/flights)
  static/flight.js   pitch-tracking pages (list, upload, field marking, results)
scripts/             fetch_assets.py, build_reference.py
tests/               pytest suite (synthetic skeletons, virtual-camera Driveline clips, API, rendering;
                     flight_synth.py renders synthetic ball-flight clips for test_flight.py)
assets/              downloaded data, models and the built reference (gitignored)
data/analyses/       uploads and results, one folder per analysis (gitignored)
data/flights/        pitch-tracking uploads and results, one folder per pitch (gitignored)
```

## Tests

```powershell
$env:PYTHONUTF8 = '1'; .\.venv\Scripts\python.exe -m pytest
```

The camera tests render Driveline pitches through a virtual camera into MediaPipe-format landmarks and
check that the video path reproduces the lab metrics. That covers right- and left-handers, side and behind
views, angled cameras, a wrong handedness setting, and slow-motion capture rates. The Claude call is tested
against a fake client, so it makes no API requests.
