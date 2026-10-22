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

## Optional Bone Angle Objective

Configure the geometric angle loss in
`src/dex_retargeting/configs/teleop/soft_hand_right_dexpilot.yml`:

- `joint_angle_weight` controls the additional loss. Zero preserves the original
  DexPilot objective and does not compute or require bone angles. Start with
  `0.001` and compare against zero when tuning.
- `joint_angle_mapping` pairs three human landmark indices with three robot
  link names. Both sides measure the angle between `a -> b` and `b -> c`:
  straight is zero and fully folded back is pi. Robot point positions come
  from forward kinematics, so URDF axis signs and fixed rotations are handled
  by the model. There are no scale/sign corrections or qpos offsets.
- `joint_angle_ranges` optionally overrides each geometric angle's normalization
  span, in radians. Keys are angle names such as `index_pip`, not robot joint
  names. Missing or null spans default to pi, the unsigned angle domain.
  Set a positive measured span no greater than pi to balance relative errors
  over actual bone-angle motion ranges. This is independent of robot qpos
  limits and does not unlock joints.

Example pairing:

```yaml
joint_angle_mapping:
  index_pip:
    human_indices: [5, 6, 7]
    robot_links: [index_link_2, index_link_3, index_link_4]
joint_angle_ranges:
  index_pip: 1.5708  # example measured geometric span, radians
```

The SoftHand configuration uses three angles per finger, fifteen in total.
At each finger base, the incoming segment runs from the wrist (robot
`base_link` origin) to the flexion center (`*_link_2` origin), and the outgoing
segment ends at `*_link_3`. The wrist-to-base segment approximates a metacarpal.
The other two terms use the origins of `link_2/link_3/link_4/tip`.
Check these geometric point definitions against the physical hand when
calibrating; changing a joint axis does not require a sign change in the loss.

These fifteen geometric angles do not individually identify twenty robot
joint coordinates: base abduction and flexion can change the same angle.
Independent joints remain optimization variables; coupled followers are
derived from their source coordinates and participate through FK.
Unsigned angles cannot distinguish flexion from equal-magnitude extension;
DexPilot's directional vectors and robot limits supply additional information.
Exactly collinear bones use a finite zero subgradient at the unsigned-angle
cusp; the angle term alone cannot pick an initial bending direction there.
Coincident bone endpoints are rejected.

The combined objective is:

```text
L = L_DexPilot + joint_angle_weight / N
    * sum(((theta_robot(q) - theta_human) / range)^2)
```

Robot angles and their gradients share the existing link FK and Jacobians.
Human angles are computed once per frame. They are invariant to rigid
coordinate changes and uniform scale; no palm-frame decomposition is needed.

Both realtime SoftHand and offline video pipelines already supply landmarks:

```python
qpos = retargeting.retarget(ref_value, human_joint_pos=joint_pos)

# Or supply geometric reference angles (not robot qpos), in YAML mapping order:
qpos = retargeting.retarget(ref_value, reference_angles=human_bone_angles)
```

The previous `reference_qpos` argument and `type/indices/scale/offset` mapping
format have been removed to avoid mixing joint coordinates with geometry.
Supply exactly one reference source when enabled. Saved fingertip vectors
alone cannot recover intermediate bone angles; retain landmarks or reference
angles for replay. At weight zero, callers can supply only `ref_value`.

Run headless numerical and SoftHand integration tests:

```bash
python tests/test_joint_angle_prior.py
```

## Hard PIP/DIP Hardware Coupling

The SoftHand YAML enables the four-finger actuator relation
`q_dip = (2/3) * q_pip`. PIP is `*_joint_3`, DIP is `*_joint_4`; both use
negative flexion coordinates in this URDF. The thumb is independent.

```yaml
joint_couplings:
  index_joint_4: {source: index_joint_3, multiplier: 0.6666666666666666, offset: 0.0}
  middle_joint_4: {source: middle_joint_3, multiplier: 0.6666666666666666, offset: 0.0}
  ring_joint_4: {source: ring_joint_3, multiplier: 0.6666666666666666, offset: 0.0}
  little_joint_4: {source: little_joint_3, multiplier: 0.6666666666666666, offset: 0.0}
```

This is a hard kinematic constraint, independent of `joint_angle_weight`.
The optimizer has 16 independent variables and returns the same 20 named
robot joint coordinates. DIP is reconstructed at every FK evaluation;
the Jacobian includes its contribution to PIP. DIP limits restrict the
permitted PIP interval, and contradictory limits fail at startup. The affine
relation is preserved after low-pass filtering and optimization fallback.
Setting `joint_couplings: {}` restores independent PIP/DIP optimization.
Offsets are radians; only direct couplings from independent sources are
supported, with no chains or cycles. Explicit hardware couplings remain
active even if `ignore_mimic_joint` disables URDF mimic tags.

The relation applies to robot joint coordinates, not directly to the raw
geometric bone angles (URDF fixed offsets can make those differ). In the
PneumaticSystem output adapter, a coupled PIP actuator target is read from
the independent joint coordinate; it bypasses the legacy geometric estimate
whose final segment also includes DIP motion. This keeps predicted motion
consistent with the shared hardware controller. Human PIP/DIP reference
angles can still conflict with the hardware ratio: the optimizer finds a
compromise satisfying the coupling, rather than reproducing both exactly.

Offline coupling tests:

```bash
python tests/test_joint_couplings.py
```
