# Realtime Allegro Genesis Pipeline

This example keeps the existing real-time front half and replaces the SAPIEN
pose preview with a DexMachina-style Genesis position-control scene:

```text
camera/video
-> MediaPipe right-hand landmarks
-> Allegro DexPilot retargeting
-> joint-name qpos mapping
-> Genesis position target and physics step
-> target/actual tracking-error report
```

It loads the hand-only URDF from:

```text
assets/robots/hands/allegro_hand/allegro_hand_right.urdf
```

The script uses the Genesis fork installed by the DexMachina environment. It
does not use DexMachina's `BaseRobot`, because that wrapper requires a robot
asset with six actuated wrist DOFs while this example intentionally uses the
hand-only Allegro asset from `dex-retarget`.

## Run With A Camera

Activate an environment containing the local `dex-retargeting` package,
MediaPipe, OpenCV, PyTorch, and the DexMachina Genesis fork. From the
`dex-retarget` repository root, run:

```bash
python3 example/vector_retargeting/realtime_allegro_genesis_pipeline.py \
  --camera-source 0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --display
```

The Genesis viewer is enabled by default. `--display` additionally opens the
OpenCV camera view with the MediaPipe skeleton.

## Run With A Video

```bash
python3 example/vector_retargeting/realtime_allegro_genesis_pipeline.py \
  --camera-source example/vector_retargeting/data/human_hand_video.mp4 \
  --max-frames 300 \
  --display
```

## Control Modes

The default mode sends qpos to the Genesis position controller using the
DexMachina Allegro finger gains (`kp=30`, `kv=2`):

```bash
python3 example/vector_retargeting/realtime_allegro_genesis_pipeline.py \
  --control-mode pd
```

Use direct state assignment only to isolate rendering or joint-order issues:

```bash
python3 example/vector_retargeting/realtime_allegro_genesis_pipeline.py \
  --control-mode set
```

Useful options:

- `--backend gpu|cpu`: Genesis backend; defaults to `gpu`.
- `--sim-hz`: physics time-step frequency; defaults to `120` Hz.
- `--physics-steps-per-frame`: Genesis steps after each captured frame.
- `--kp`, `--kv`, `--force-limit`: Allegro position-controller settings.
- `--self-collision`: enable self-collision at additional runtime cost.
- `--sim-camera-view perspective|palm|back|side`: initial viewer camera.
- `--no-show-viewer`: run Genesis without its viewer for timing checks.

For real-time use, start with one physics step per captured frame. Increasing
`--physics-steps-per-frame` gives the PD controller more simulation time to
settle, but lowers camera-loop throughput in this synchronous implementation.

The periodic report includes `tracking_abs_rad`, the mean absolute difference
between the latest target and actual Genesis joint positions.
