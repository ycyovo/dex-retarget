import os
import time
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
import sapien
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig


def _open_video_capture(
    camera_source: Optional[Union[int, str]],
    width: Optional[int],
    height: Optional[int],
    fps: Optional[float],
):
    source = 0 if camera_source is None else camera_source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _fixed_qpos_from_retargeting(retargeting):
    fixed_limits = retargeting.optimizer.robot.joint_limits[
        retargeting.optimizer.idx_pin2fixed
    ]
    return fixed_limits.mean(axis=1).astype("float32")


def _build_ref_value(retargeting, joint_pos: np.ndarray):
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting_type == "POSITION":
        return joint_pos[indices, :]

    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    return joint_pos[task_indices, :] - joint_pos[origin_indices, :]


def _set_viewer_camera(viewer: Viewer, camera_view: str):
    """Set a stable view centered on the hand root at the world origin."""
    views = {
        # Matches render_robot_hand.py: camera at +X looking toward the hand.
        "profile": ((-0.15, 0.1, 0.20), ( -np.pi/2.0 , -np.pi / 3.0 , 0.0)),
        # "profile": ((0.5, 0.0, 0.0), (0.0, 0.0, np.pi)),
        # The palm normal is +Z. Viewer forward is local +X, so the palm view
        # places the camera at -Z and pitches +X toward +Z.
        "palm": ((0.0, 0.0, -0.5), (0.0, np.pi / 2.0, 0.0)),
        # The back view places the camera at +Z and pitches +X toward -Z.
        "back": ((0.0, 0.0, 0.5), (0.0, -np.pi / 2.0 , 0.0)),
        # Fingers extend mainly along +Y. These views look from the finger
        # tips (+Y) or wrist side (-Y), and are useful as end-on checks.
        "top": ((0.0, 0.5, 0.0), (0.0, 0.0, np.pi / 2.0)),
        "bottom": ((0.0, -0.5, 0.0), (0.0, 0.0, -np.pi / 2.0)),
    }
    position, rpy = views[camera_view]
    viewer.set_camera_xyz(*position)
    viewer.set_camera_rpy(*rpy)


def _create_soft_hand_viewer(config, camera_view: str):
    """Create the Task 4 SAPIEN view for the hand-only soft_hand URDF."""
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.roughness = 0.9
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )

    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True
    urdf_path = Path(config.urdf_path)
    glb_path = urdf_path.with_name(f"{urdf_path.stem}_glb.urdf")
    robot = loader.load(str(glb_path if glb_path.exists() else urdf_path))
    if robot is None:
        raise RuntimeError(f"Failed to load soft_hand URDF: {urdf_path}")
    # Render root pose: world -> hand-only soft_hand/base_link.
    robot.set_pose(sapien.Pose([0, 0, -0.1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    _set_viewer_camera(viewer, camera_view)
    return viewer, robot


def _set_robot_qpos_by_name(robot, qpos: np.ndarray, retargeting_joint_names):
    """Copy qpos by joint name so URDF and retargeting joint orders may differ."""
    name_to_qpos = dict(zip(retargeting_joint_names, qpos))
    robot_qpos = robot.get_qpos()
    for sapien_index, joint in enumerate(robot.get_active_joints()):
        if joint.get_name() in name_to_qpos:
            robot_qpos[sapien_index] = name_to_qpos[joint.get_name()]
    robot.set_qpos(robot_qpos)


class RunningStats:
    def __init__(self):
        self.capture_s = 0.0
        self.detect_s = 0.0
        self.ref_s = 0.0
        self.retarget_s = 0.0
        self.frames = 0
        self.detected = 0
        self.retargeted = 0
        self.failed = 0

    def add_capture(self, dt: float):
        self.capture_s += dt
        self.frames += 1

    def add_detect(self, dt: float, detected: bool):
        self.detect_s += dt
        if detected:
            self.detected += 1

    def add_ref(self, dt: float):
        self.ref_s += dt

    def add_retarget(self, dt: float):
        self.retarget_s += dt
        self.retargeted += 1

    def add_failure(self):
        self.failed += 1


def _ms(total_s: float, count: int):
    return 1000.0 * total_s / max(count, 1)


def _print_stats(stats: RunningStats, elapsed_s: float, prefix: str):
    print(
        f"{prefix} elapsed={elapsed_s:.2f}s "
        f"processed={stats.frames} detected={stats.detected} "
        f"retargeted={stats.retargeted} failed={stats.failed} "
        f"loop_fps={stats.frames / max(elapsed_s, 1e-9):.1f} "
        f"retarget_fps={stats.retargeted / max(elapsed_s, 1e-9):.1f} "
        f"avg_ms(capture={_ms(stats.capture_s, stats.frames):.2f}, "
        f"detect={_ms(stats.detect_s, stats.frames):.2f}, "
        f"ref={_ms(stats.ref_s, stats.detected):.2f}, "
        f"retarget={_ms(stats.retarget_s, stats.retargeted):.2f})"
    )


def run_pipeline(
    camera_source: Optional[Union[int, str]] = None,
    max_frames: int = 0,
    report_every: int = 60,
    display: bool = False,
    sim_display: bool = False,
    sim_camera_view: Literal["profile", "palm", "back", "top", "bottom"] = "profile",
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    config_path: Optional[Path] = None,
):
    if display:
        # OpenCV's Qt backend otherwise connects to the inherited desktop
        # session manager. A stale/broken ICE connection can exit the process
        # from native code. This preview window needs no session restoration.
        os.environ.pop("SESSION_MANAGER", None)

    try:
        from detect_from_video import compute_soft_hand_wrist_alignment
        from hand_input_adaptation import adapt_new_hand_thumb_depth
        from single_hand_detector import SingleHandDetector
    except ModuleNotFoundError:
        from example.vector_retargeting.detect_from_video import (
            compute_soft_hand_wrist_alignment,
        )
        from example.vector_retargeting.hand_input_adaptation import (
            adapt_new_hand_thumb_depth,
        )
        from example.vector_retargeting.single_hand_detector import SingleHandDetector

    if config_path is None:
        config_path = get_default_config_path(
            RobotName.soft, RetargetingType.dexpilot, HandType.right,
        )
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    print(f"[soft_hand] Loading retargeting config: {config_path}", flush=True)
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    fixed_qpos = _fixed_qpos_from_retargeting(retargeting)
    print("[soft_hand] Initializing MediaPipe hand detector.", flush=True)
    detector = SingleHandDetector(hand_type="Right", selfie=False)
    # import ipdb; ipdb.set_trace()
    viewer = None
    robot = None
    if sim_display:
        try:
            config = RetargetingConfig.load_from_file(config_path)
            viewer, robot = _create_soft_hand_viewer(config, sim_camera_view)
            print(
                "[soft_hand] SAPIEN viewer ready "
                f"(camera view: {sim_camera_view}). Close it or press 'q' to stop."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SAPIEN viewer: {exc}") from exc

    print(f"[soft_hand] Opening camera/video source: {camera_source}", flush=True)
    cap = _open_video_capture(camera_source, width, height, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {camera_source}")

    print(f"[soft_hand] Config: {config_path}")
    print(
        "[soft_hand] Running realtime pipeline with SAPIEN display."
        if sim_display
        else "[soft_hand] Running front-half realtime pipeline without SAPIEN."
    )
    print("[soft_hand] Press 'q' in the OpenCV window to stop when --display is set.")

    stats = RunningStats()
    start_s = time.perf_counter()
    stream_start_s = time.monotonic()
    last_timestamp_ms = -1
    soft_hand_wrist_alignment = None

    try:
        while True:
            if viewer is not None and viewer.closed:
                print("[soft_hand] SAPIEN viewer closed.")
                break
            # Keep the SAPIEN event loop responsive even when no hand is detected.
            if viewer is not None:
                viewer.render()
            tic = time.perf_counter()
            ret, frame = cap.read()
            capture_dt = time.perf_counter() - tic
            if not ret:
                print("[soft_hand] End of stream or failed frame read.")
                break
            stats.add_capture(capture_dt)

            rgb = frame[..., ::-1]
            timestamp_ms = int((time.monotonic() - stream_start_s) * 1000.0)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            tic = time.perf_counter()
            try:
                num_box, joint_pos, keypoint_2d, _ = detector.detect(rgb, timestamp_ms)
            except Exception as exc:
                stats.add_detect(time.perf_counter() - tic, detected=False)
                stats.add_failure()
                print(f"[soft_hand][WARN] MediaPipe detection failed: {exc}")
                continue
            stats.add_detect(time.perf_counter() - tic, detected=num_box > 0)

            if display:
                shown = frame.copy()
                if num_box > 0 and keypoint_2d is not None:
                    shown = detector.draw_skeleton_on_image(shown, keypoint_2d)
                cv2.imshow("soft_hand_front_half_pipeline", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if num_box == 0:
                if report_every > 0 and stats.frames % report_every == 0:
                    _print_stats(stats, time.perf_counter() - start_s, "[soft_hand]")
                if max_frames > 0 and stats.frames >= max_frames:
                    break
                continue

            tic = time.perf_counter()
            try:
                ref_value = _build_ref_value(retargeting, joint_pos)
                if soft_hand_wrist_alignment is None:
                    soft_hand_wrist_alignment = compute_soft_hand_wrist_alignment(
                        retargeting,
                        ref_value,
                    )
                    print(
                        "[soft_hand] Calibrated MediaPipe wrist frame to "
                        "soft_hand/base_link rotation:\n"
                        f"{np.round(soft_hand_wrist_alignment, 6)}"
                    )
                # Coordinate transform: MediaPipe wrist-frame ref vectors ->
                # soft_hand/base_link ref vectors. This must match the Task 2
                # offline pipeline before the qpos is used in realtime.
                ref_value = ref_value @ soft_hand_wrist_alignment
                ref_value = adapt_new_hand_thumb_depth(
                    retargeting, ref_value,
                    joint_pos @ soft_hand_wrist_alignment,
                )
            except Exception as exc:
                stats.add_ref(time.perf_counter() - tic)
                stats.add_failure()
                print(f"[soft_hand][WARN] Failed to build retarget input: {exc}")
                continue
            stats.add_ref(time.perf_counter() - tic)

            tic = time.perf_counter()
            try:
                qpos = retargeting.retarget(
                    ref_value, fixed_qpos=fixed_qpos, human_joint_pos=joint_pos
                )
            except Exception as exc:
                stats.add_failure()
                print(f"[soft_hand][WARN] Retargeting failed: {exc}")
                continue
            stats.add_retarget(time.perf_counter() - tic)

            if robot is not None:
                _set_robot_qpos_by_name(robot, qpos, retargeting.joint_names)
            if viewer is not None:
                viewer.render()

            if report_every > 0 and stats.frames % report_every == 0:
                _print_stats(stats, time.perf_counter() - start_s, "[soft_hand]")
            if max_frames > 0 and stats.frames >= max_frames:
                break
    finally:
        cap.release()
        detector.hand_detector.close()
        if display:
            cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()

    elapsed_s = time.perf_counter() - start_s
    _print_stats(stats, elapsed_s, "[soft_hand][final]")
    retargeting.verbose()


def main(
    camera_source: Optional[Union[int, str]] = None,
    max_frames: int = 0,
    report_every: int = 60,
    display: bool = False,
    sim_display: bool = False,
    sim_camera_view: Literal["profile", "palm", "back", "top", "bottom"] = "profile",
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    config_path: Optional[Path] = None,
):
    """
    Run the realtime pipeline: camera/video -> MediaPipe -> DexPilot qpos.

    Args:
        camera_source: OpenCV camera index or video path. Defaults to camera 0.
        max_frames: Stop after this many captured frames. 0 means run until stream ends.
        report_every: Print timing stats every N captured frames. 0 disables periodic reports.
        display: Show the camera image with MediaPipe hand skeleton overlay.
        sim_display: Show the retargeted soft_hand qpos in a SAPIEN viewer.
        sim_camera_view: SAPIEN camera view: profile, palm, back, top, or bottom.
        width: Optional requested capture width.
        height: Optional requested capture height.
        fps: Optional requested capture FPS.
        config_path: Optional retargeting YAML. Defaults to the existing soft_hand config.
    """
    run_pipeline(
        camera_source=camera_source,
        max_frames=max_frames,
        report_every=report_every,
        display=display,
        sim_display=sim_display,
        sim_camera_view=sim_camera_view,
        width=width,
        height=height,
        fps=fps,
        config_path=config_path,
    )


if __name__ == "__main__":
    tyro.cli(main)
