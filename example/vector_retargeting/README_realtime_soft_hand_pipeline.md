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

Use it first without visualization to check throughput; add `--sim-display`
to show the retargeted model in SAPIEN.

## 新手 hand 模型（先跑通 example）

### 先检查 URDF 零位

不运行检测和 retargeting，所有活动关节固定为 **0°**，根坐标为单位变换，
也不做物理步进。默认显示当前新手兼容模型，可用鼠标旋转、缩放：

```bash
python example/vector_retargeting/view_hand_zero_pose.py
```

查看含项目零位修正的 24 关节 CAD 模型，或原来的 soft_hand：

```bash
python example/vector_retargeting/view_hand_zero_pose.py \
  --urdf-path assets/robots/hands/hand/urdf/hand.urdf
python example/vector_retargeting/view_hand_zero_pose.py \
  --urdf-path assets/robots/hands/soft_hand/soft_hand_right_hand_only.urdf
```

`--view front|back|side|top` 设置初始视角（分别从 -Y、+Y、+X、+Z 观察）。
无窗口截图使用 `--headless --output /tmp/hand_zero.png`。
脚本的地面、灯光和环境贴图与 realtime viewer 一致。默认保留 URDF 材质；
可选 RGB 覆盖模型颜色（每个分量范围 0 到 1），例如：

```bash
python example/vector_retargeting/view_hand_zero_pose.py --hand-color 0.35 0.65 0.9
```

脚本会打印所有活动关节名称和 0° 值。这里的零位由 URDF 的 joint origin
决定，不等同于手指伸直；反折等装配问题也会直接显示出来。

### 运行 retargeting

在仓库根目录运行，使用已安装本项目及 MediaPipe/SAPIEN 的 Python 环境：

```bash
python example/vector_retargeting/realtime_soft_hand_pipeline.py \
  --config-path src/dex_retargeting/configs/teleop/hand_right_dexpilot.yml \
  --camera-source example/vector_retargeting/data/hand_video_2.mp4 \
  --max-frames 300 --report-every 30
```

查看新手模型时，在以上命令末尾添加 `--sim-display`；实时相机将
`--camera-source` 换成 `0`。`--display` 可选，用于显示检测骨架。
新配置的 URDF 同时用于求解和显示，不需要修改渲染代码或 PneumaticSystem。
不传 `--config-path` 时仍使用原 soft_hand 默认配置。

新手配置现已启用四指 PIP/DIP 硬联动：每指独立优化 PIP，DIP 由
`q4 = -2/3 * q3` 得到，正屈曲角满足 `DIP = 2/3 * PIP`。
新手 q3、q4 屈曲符号相反，因此 multiplier 为负。独立优化变量为 16 个；
源模型仍返回 24 维 qpos（包括四个保持零位的 mount），兼容模型返回 20 维。
PneumaticSystem 已使用新手配置，联动 PIP 输出直接读取关节坐标，跳过旧 FK 弯角估计。

当前只验证 example 链路，保留原 20 个关节名和 link 名。临时约定如下：

- `hand/urdf/hand.urdf` 中，中指、无名指和小指的 CAD joint3 坐标系 `rpy`
  已对齐食指 joint3，修复这三指在零位朝手掌反折的问题。原模型各指的
  横向分叉位置和网格保留；`hand_compat.urdf` 从该模型生成。
- 拇指四关节直接改为旧名称；其余四指暂将 CAD joint1 固定在 0，
  CAD joint2～joint5 对应旧 joint_1～joint_4，保留全部结构网格。
- 拇指零位修正在 `hand/urdf/hand.urdf` 的前两个 joint origin 中：相对修正前
  的 CAD 导出坐标，`q_export1 = q1 + pi`、`q_export2 = q2 + pi/2`；后两关节
  没有偏置。第一段改为朝食指侧，第二段消除反折，使拇指在 q=0 时向外、
  向指尖方向展开。安装点 xyz、旋转轴、网格、质量惯量和其余四指均未因此改动。
  这是经多视角检查的 **example 暂定零位，不是硬件标定**，也不保证旧气动角度语义。
  生成器原样复制修正后的 origin，不重复叠加零偏；零位查看器仍把 qpos 设为 0。
- `assets/robots/hands/hand/hand_origin.urdf` 保存拇指零位补偿前的 URDF 状态，
  可用于直接对照两项拇指 `rpy`。该快照保留了此前已做的中指、无名指、小指
  joint3 四指展开修正，因此不是最初未经修改的 SolidWorks 导出文件。
- 新手源 URDF 及兼容 URDF 的活动关节使用 `continuous`，没有 `<limit>`。
  `RobotWrapper` 为 Pinocchio 转换为内存中的标量角度表示，保留每关节一个 qpos，
  数值计算域为 ±10000 rad（不是机械限位）。实际软件角度范围由
  `hand_right_dexpilot.yml` 的 `joint_limits` 设置，单位 rad，目前为空、待标定。
  新手使用自己的 YAML 限位，联动采用上述适配符号后的 2/3 比例，几何角损失关闭；
  完整机械零位及实物角度映射尚未标定。
- 原始 `hand/urdf/hand.urdf` 与兼容模型均包含五个固定虚拟 `*_tip` frame；
  拇指挂在 `thumb_link4`，其余四指挂在各自 `*_link5`。tip 坐标暂用末节 STL
  包围盒的 +Y 端面中心估算，兼容模型生成器会复用源 URDF 的 frame，不重复添加；
  后续可用实测指尖位置替换。
- 保留语义关节命名不表示气动角度语义已校准；PneumaticSystem 已切换新手模型、
  修正屈曲符号并启用上述联动，拇指参考零位和实物映射仍待验证。

CAD 更新后可重新生成兼容模型；运行验证不需要相机和窗口：

```bash
python assets/scripts/build_hand_compat.py
python tests/test_hand_compat.py
```

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

With `--display`, the pipeline removes `SESSION_MANAGER` from its process
environment before initializing the preview. This prevents OpenCV's Qt backend
from using a stale desktop session connection that can terminate the process
with `ICE default IO error handler doing an exit()`. The parent terminal's
environment is unaffected.

```bash
python3 example/vector_retargeting/realtime_soft_hand_pipeline.py \
  --camera-source 0 \
  --display
```

For pure FPS testing, keep `--display` off because OpenCV UI rendering can
affect the measured throughput.

### 新手拇指根部输入适配

新手源模型与 `hand_compat.urdf` 使用完整腕部坐标对齐，并通过
`hand_input_adaptation.py` 补偿人手与模型拇指根部在掌面深度方向上的安装差异。
补偿作用于优化前的指尖目标，保留实际根部到指尖的深度变化；不会将
`thumb_joint_1` 锁零，也不要求启用 `joint_angle_weight`。近距离捏合时补偿逐渐
关闭，以保留 DexPilot 的接触目标。旧 soft_hand 继续使用原来的输入处理。

该处理已接入本实时脚本与 `detect_from_video.py`。直接调用库的其他程序
（包括 PneumaticSystem）需要在自己的输入路径中同步接入，不能只替换 YAML
就假设已有同样的补偿。模型识别依据当前 CAD link 拓扑；补偿使用新模型
`base_link` 的 Y 深度轴，后续更改基坐标系时需要一起核对。

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
