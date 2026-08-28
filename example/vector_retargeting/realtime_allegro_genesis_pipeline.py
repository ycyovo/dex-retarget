import time
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import cv2
import numpy as np

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
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting.optimizer.retargeting_type == "POSITION":
        return joint_pos[indices, :]

    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    return joint_pos[task_indices, :] - joint_pos[origin_indices, :]


def build_joint_order_indices(
    source_joint_names: Sequence[str], target_joint_names: Sequence[str]
) -> np.ndarray:
    """Return indices that reorder source qpos into target joint order."""
    if len(set(source_joint_names)) != len(source_joint_names):
        raise ValueError("Source joint names contain duplicates.")
    if len(set(target_joint_names)) != len(target_joint_names):
        raise ValueError("Target joint names contain duplicates.")

    source_index = {name: index for index, name in enumerate(source_joint_names)}
    missing = [name for name in target_joint_names if name not in source_index]
    if missing:
        raise ValueError(
            "Genesis joints are missing from the retargeting output: "
            + ", ".join(missing)
        )
    return np.asarray([source_index[name] for name in target_joint_names], dtype=int)


def _camera_pose(view: str):
    views = {
        "perspective": ((0.38, -0.38, 0.28), (0.0, 0.0, 0.03)),
        "palm": ((0.0, 0.0, 0.48), (0.0, 0.0, 0.0)),
        "back": ((0.0, 0.0, -0.48), (0.0, 0.0, 0.0)),
        "side": ((0.48, 0.0, 0.08), (0.0, 0.0, 0.03)),
    }
    return views[view]


class GenesisAllegroSimulation:
    """Minimal DexMachina-style Genesis scene for a hand-only Allegro URDF."""

    def __init__(
        self,
        urdf_path: Path,
        retargeting_joint_names: Sequence[str],
        backend: Literal["gpu", "cpu"],
        control_mode: Literal["pd", "set"],
        sim_hz: float,
        substeps: int,
        kp: float,
        kv: float,
        force_limit: float,
        self_collision: bool,
        show_viewer: bool,
        camera_view: Literal["perspective", "palm", "back", "side"],
    ):
        if sim_hz <= 0:
            raise ValueError(f"sim_hz must be positive, got {sim_hz}")
        if substeps <= 0:
            raise ValueError(f"substeps must be positive, got {substeps}")
        if not urdf_path.exists():
            raise FileNotFoundError(f"Allegro URDF does not exist: {urdf_path}")

        try:
            import genesis as gs
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Genesis is unavailable. Activate the DexMachina environment and "
                "install its Genesis fork before running this script."
            ) from exc

        self.gs = gs
        self.torch = torch
        self.control_mode = control_mode
        self.show_viewer = show_viewer
        self.device = torch.device("cuda" if backend == "gpu" else "cpu")

        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)
        camera_pos, camera_lookat = _camera_pose(camera_view)
        dt = 1.0 / sim_hz
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                substeps=substeps,
                gravity=(0.0, 0.0, 0.0),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self_collision,
                batch_dofs_info=False,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                visualize_contact=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=35,
            ),
            show_viewer=show_viewer,
            use_visualizer=show_viewer,
            show_FPS=False,
        )
        self.hand = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf_path),
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=False,
                collision=True,
                visualization=True,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.scene.build(n_envs=1)

        self.actuated_joints = [
            joint
            for joint in self.hand.joints
            if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
        ]
        self.joint_names = [joint.name for joint in self.actuated_joints]
        self.dof_indices = [joint.dof_idx_local for joint in self.actuated_joints]
        self.retarget_to_sim_indices = build_joint_order_indices(
            retargeting_joint_names,
            self.joint_names,
        )

        num_dofs = len(self.dof_indices)
        gains_kp = torch.full((num_dofs,), kp, device=self.device)
        gains_kv = torch.full((num_dofs,), kv, device=self.device)
        force = torch.full((num_dofs,), force_limit, device=self.device)
        self.hand.set_dofs_kp(gains_kp, self.dof_indices)
        self.hand.set_dofs_kv(gains_kv, self.dof_indices)
        self.hand.set_dofs_force_range(-force, force, self.dof_indices)

        initial_qpos = np.concatenate(
            [joint.init_qpos for joint in self.actuated_joints]
        ).astype(np.float32)
        self.target = torch.as_tensor(initial_qpos, device=self.device)[None, :]
        self.lower_limits, self.upper_limits = self.hand.get_dofs_limit(
            self.dof_indices
        )
        self.hand.set_dofs_position(
            position=self.target,
            dofs_idx_local=self.dof_indices,
            zero_velocity=True,
        )

    def is_alive(self) -> bool:
        if not self.show_viewer:
            return True
        return self.scene.viewer.is_alive()

    def set_target(self, retargeting_qpos: np.ndarray):
        qpos = np.asarray(retargeting_qpos, dtype=np.float32)
        if qpos.ndim != 1:
            raise ValueError(f"Expected one-dimensional qpos, got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("Retargeting qpos contains non-finite values.")

        reordered = qpos[self.retarget_to_sim_indices]
        target = self.torch.as_tensor(reordered, device=self.device)[None, :]
        self.target = self.torch.maximum(
            self.torch.minimum(target, self.upper_limits), self.lower_limits
        )

        if self.control_mode == "set":
            self.hand.set_dofs_position(
                position=self.target,
                dofs_idx_local=self.dof_indices,
                zero_velocity=True,
            )
        else:
            self.hand.control_dofs_position(
                position=self.target,
                dofs_idx_local=self.dof_indices,
            )

    def step(self, steps: int) -> float:
        start = time.perf_counter()
        for _ in range(steps):
            self.scene.step()
        return time.perf_counter() - start

    def tracking_error(self) -> float:
        actual = self.hand.get_dofs_position(self.dof_indices)
        if actual.ndim == 1:
            actual = actual[None, :]
        return float(self.torch.mean(self.torch.abs(actual - self.target)).item())

    def close(self):
        if self.show_viewer and self.scene.viewer.is_alive():
            self.scene.viewer.stop()


class RunningStats:
    def __init__(self):
        self.capture_s = 0.0
        self.detect_s = 0.0
        self.ref_s = 0.0
        self.retarget_s = 0.0
        self.sim_s = 0.0
        self.tracking_error_sum = 0.0
        self.tracking_error_max = 0.0
        self.frames = 0
        self.detected = 0
        self.retargeted = 0
        self.failed = 0
        self.sim_updates = 0

    def add_tracking_error(self, error: float):
        self.tracking_error_sum += error
        self.tracking_error_max = max(self.tracking_error_max, error)
        self.sim_updates += 1


def _ms(total_s: float, count: int):
    return 1000.0 * total_s / max(count, 1)


def _print_stats(stats: RunningStats, elapsed_s: float):
    print(
        f"[allegro][genesis] elapsed={elapsed_s:.2f}s "
        f"processed={stats.frames} detected={stats.detected} "
        f"retargeted={stats.retargeted} failed={stats.failed} "
        f"loop_fps={stats.frames / max(elapsed_s, 1e-9):.1f} "
        f"retarget_fps={stats.retargeted / max(elapsed_s, 1e-9):.1f} "
        f"avg_ms(capture={_ms(stats.capture_s, stats.frames):.2f}, "
        f"detect={_ms(stats.detect_s, stats.frames):.2f}, "
        f"ref={_ms(stats.ref_s, stats.detected):.2f}, "
        f"retarget={_ms(stats.retarget_s, stats.retargeted):.2f}, "
        f"sim={_ms(stats.sim_s, stats.frames):.2f}) "
        f"tracking_abs_rad(avg={stats.tracking_error_sum / max(stats.sim_updates, 1):.4f}, "
        f"max={stats.tracking_error_max:.4f})"
    )


def run_pipeline(
    camera_source: Optional[Union[int, str]] = None,
    max_frames: int = 0,
    report_every: int = 60,
    display: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    backend: Literal["gpu", "cpu"] = "gpu",
    control_mode: Literal["pd", "set"] = "pd",
    sim_hz: float = 120.0,
    physics_steps_per_frame: int = 1,
    sim_substeps: int = 2,
    kp: float = 30.0,
    kv: float = 2.0,
    force_limit: float = 100.0,
    self_collision: bool = False,
    show_viewer: bool = True,
    sim_camera_view: Literal["perspective", "palm", "back", "side"] = "perspective",
):
    if physics_steps_per_frame <= 0:
        raise ValueError(
            "physics_steps_per_frame must be positive, got "
            f"{physics_steps_per_frame}"
        )

    try:
        from single_hand_detector import SingleHandDetector
    except ModuleNotFoundError:
        from example.vector_retargeting.single_hand_detector import SingleHandDetector

    config_path = get_default_config_path(
        RobotName.allegro,
        RetargetingType.dexpilot,
        HandType.right,
    )
    repository_root = Path(__file__).resolve().parents[2]
    robot_dir = repository_root / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    print(f"[allegro] Loading retargeting config: {config_path}", flush=True)
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()
    fixed_qpos = _fixed_qpos_from_retargeting(retargeting)

    print("[allegro] Initializing MediaPipe hand detector.", flush=True)
    detector = SingleHandDetector(hand_type="Right", selfie=False)
    print(f"[allegro] Initializing Genesis with URDF: {config.urdf_path}", flush=True)
    simulation = GenesisAllegroSimulation(
        urdf_path=Path(config.urdf_path),
        retargeting_joint_names=retargeting.joint_names,
        backend=backend,
        control_mode=control_mode,
        sim_hz=sim_hz,
        substeps=sim_substeps,
        kp=kp,
        kv=kv,
        force_limit=force_limit,
        self_collision=self_collision,
        show_viewer=show_viewer,
        camera_view=sim_camera_view,
    )

    cap = _open_video_capture(camera_source, width, height, fps)
    if not cap.isOpened():
        simulation.close()
        raise RuntimeError(f"Could not open camera/video source: {camera_source}")

    print(f"[allegro] Opening camera/video source: {camera_source}")
    print(
        f"[allegro] Genesis control={control_mode} sim_hz={sim_hz:g} "
        f"steps_per_frame={physics_steps_per_frame} backend={backend}"
    )
    print("[allegro] Press 'q' in the OpenCV window to stop when --display is set.")

    stats = RunningStats()
    start_s = time.perf_counter()
    stream_start_s = time.monotonic()
    last_timestamp_ms = -1

    try:
        while simulation.is_alive():
            tic = time.perf_counter()
            ret, frame = cap.read()
            stats.capture_s += time.perf_counter() - tic
            if not ret:
                print("[allegro] End of stream or failed frame read.")
                break
            stats.frames += 1

            rgb = frame[..., ::-1]
            timestamp_ms = int((time.monotonic() - stream_start_s) * 1000.0)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            tic = time.perf_counter()
            try:
                num_box, joint_pos, keypoint_2d, _ = detector.detect(rgb, timestamp_ms)
            except Exception as exc:
                stats.detect_s += time.perf_counter() - tic
                stats.failed += 1
                print(f"[allegro][WARN] MediaPipe detection failed: {exc}")
                num_box, joint_pos, keypoint_2d = 0, None, None
            else:
                stats.detect_s += time.perf_counter() - tic
                if num_box > 0:
                    stats.detected += 1

            if display:
                shown = frame.copy()
                if num_box > 0 and keypoint_2d is not None:
                    shown = detector.draw_skeleton_on_image(shown, keypoint_2d)
                cv2.imshow("allegro_genesis_realtime_pipeline", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if num_box > 0:
                tic = time.perf_counter()
                try:
                    # SingleHandDetector performs the coordinate transform:
                    # MediaPipe world-landmark frame -> wrist-centered MANO frame.
                    # The stock Allegro DexPilot config consumes this MANO convention.
                    ref_value = _build_ref_value(retargeting, joint_pos)
                except Exception as exc:
                    stats.failed += 1
                    print(f"[allegro][WARN] Failed to build retarget input: {exc}")
                else:
                    stats.ref_s += time.perf_counter() - tic
                    tic = time.perf_counter()
                    try:
                        qpos = retargeting.retarget(
                            ref_value,
                            fixed_qpos=fixed_qpos,
                        )
                        simulation.set_target(qpos)
                    except Exception as exc:
                        stats.failed += 1
                        print(f"[allegro][WARN] Retargeting/control failed: {exc}")
                    else:
                        stats.retarget_s += time.perf_counter() - tic
                        stats.retargeted += 1

            stats.sim_s += simulation.step(physics_steps_per_frame)
            stats.add_tracking_error(simulation.tracking_error())

            if report_every > 0 and stats.frames % report_every == 0:
                _print_stats(stats, time.perf_counter() - start_s)
            if max_frames > 0 and stats.frames >= max_frames:
                break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        simulation.close()

    elapsed_s = time.perf_counter() - start_s
    _print_stats(stats, elapsed_s)
    retargeting.verbose()


def main(
    camera_source: Optional[Union[int, str]] = None,
    max_frames: int = 0,
    report_every: int = 60,
    display: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    backend: Literal["gpu", "cpu"] = "gpu",
    control_mode: Literal["pd", "set"] = "pd",
    sim_hz: float = 120.0,
    physics_steps_per_frame: int = 1,
    sim_substeps: int = 2,
    kp: float = 30.0,
    kv: float = 2.0,
    force_limit: float = 100.0,
    self_collision: bool = False,
    show_viewer: bool = True,
    sim_camera_view: Literal["perspective", "palm", "back", "side"] = "perspective",
):
    """Run MediaPipe -> Allegro DexPilot -> DexMachina Genesis in real time."""
    run_pipeline(
        camera_source=camera_source,
        max_frames=max_frames,
        report_every=report_every,
        display=display,
        width=width,
        height=height,
        fps=fps,
        backend=backend,
        control_mode=control_mode,
        sim_hz=sim_hz,
        physics_steps_per_frame=physics_steps_per_frame,
        sim_substeps=sim_substeps,
        kp=kp,
        kv=kv,
        force_limit=force_limit,
        self_collision=self_collision,
        show_viewer=show_viewer,
        sim_camera_view=sim_camera_view,
    )


if __name__ == "__main__":
    try:
        import tyro
    except ModuleNotFoundError as exc:
        raise RuntimeError("The CLI requires tyro. Install the dex-retargeting example dependencies before running this script.") from exc
    tyro.cli(main)
