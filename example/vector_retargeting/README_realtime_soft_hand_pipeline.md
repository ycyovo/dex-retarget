# Realtime Soft Hand Front-Half Pipeline

This script is the minimal Task 3 front-half pipeline:

```text
camera/video stream
-> MediaPipe hand detection
-> soft_hand DexPilot ref_value
-> MediaPipe wrist-frame -> soft_hand/base_link alignment
-> retargeting.retarget()
-> timing/FPS report
```

It does not start SAPIEN rendering yet. Use it first to check whether the
front half of the realtime chain can keep up with the video stream.

## Script

```bash
example/vector_retargeting/realtime_soft_hand_pipeline.py
```

The script defaults to:

- robot: `soft`
- retargeting type: `dexpilot`
- hand type: `right`
- camera source: OpenCV camera `0`
- visualization: off

## Run With Webcam

From the repository root:

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py
```

Run with explicit camera settings:

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py \
  --camera-source 0 \
  --width 640 \
  --height 480 \
  --fps 30
```

## Run With A Video File

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py \
  --camera-source example/vector_retargeting/data/hand_video_2.mp4 \
  --max-frames 300 \
  --report-every 30
```

## Show MediaPipe Overlay

Use `--display` to show the camera frame with the detected hand skeleton.
Press `q` in the OpenCV window to stop.

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py \
  --camera-source 0 \
  --display
```

For pure FPS testing, keep `--display` off because OpenCV UI rendering can
affect the measured throughput.

## Useful Options

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py --help
```

Common options:

- `--camera-source`: OpenCV camera index or video path. Defaults to camera `0`.
- `--max-frames`: stop after N captured frames. `0` means run until the stream ends.
- `--report-every`: print timing stats every N captured frames.
- `--display`: show MediaPipe skeleton overlay.
- `--width`, `--height`, `--fps`: requested capture settings.

## Output

The periodic report looks like:

```text
[soft_hand] elapsed=10.00s processed=300 detected=295 retargeted=295 failed=0 loop_fps=30.0 retarget_fps=29.5 avg_ms(capture=1.20, detect=12.40, ref=0.08, retarget=5.60)
```

Interpretation:

- `processed`: frames read from the stream.
- `detected`: frames where MediaPipe found the target hand.
- `retargeted`: frames that successfully produced a soft_hand qpos.
- `failed`: detection/ref construction/retargeting exceptions.
- `loop_fps`: total synchronous loop FPS.
- `retarget_fps`: successful qpos output FPS.
- `capture`: frame read time.
- `detect`: MediaPipe detection time.
- `ref`: ref_value construction plus soft_hand wrist alignment application time.
- `retarget`: DexPilot optimization time.

If `retarget_fps` is near or above the input stream FPS, the front half is
fast enough for the next SAPIEN integration step. If not, use the average
stage timings to identify whether the bottleneck is camera input, MediaPipe,
or DexPilot retargeting.

## Notes

The script reuses the Task 2 coordinate transform:

```text
MediaPipe wrist-frame -> soft_hand/base_link
```

This alignment is required for realtime results to match the already validated
offline `detect_from_video.py` soft_hand behavior.

## Live SAPIEN Display

Pass `--sim-display` to open a SAPIEN viewer alongside the optional OpenCV
camera window. Each successful DexPilot result is copied to the hand-only
soft_hand URDF by joint name and rendered immediately.

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py \
  --camera-source 0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --display \
  --sim-display \
  --sim-camera-view profile
```

This opens two windows:

- OpenCV: live camera image with the MediaPipe hand skeleton.
- SAPIEN: the retargeted soft_hand pose.

`--sim-camera-view` controls the initial SAPIEN view:

- `profile` (default): matches the side/profile view used by `render_robot_hand.py` (camera at `x=+0.5`).
- `palm`: looks from `-Z` toward the palm surface (`+Z` normal).
- `back`: looks from `+Z` toward the back of the hand.
- `top`: looks from `+Y` toward the fingertips (an end-on check).
- `bottom`: looks from `-Y` toward the wrist side (an end-on check).

For example, use `--sim-camera-view top` to start from the top view.

Click the OpenCV window and press `q` to stop normally. Closing the SAPIEN
viewer also ends the processing loop. The SAPIEN window stays responsive while
the hand is temporarily not detected; it holds the last valid qpos in that
case.
