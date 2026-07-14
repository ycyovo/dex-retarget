import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import tyro

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
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
):
    try:
        from detect_from_video import compute_soft_hand_wrist_alignment
        from single_hand_detector import SingleHandDetector
    except ModuleNotFoundError:
        from example.vector_retargeting.detect_from_video import (
            compute_soft_hand_wrist_alignment,
        )
        from example.vector_retargeting.single_hand_detector import SingleHandDetector

    config_path = get_default_config_path(
        RobotName.soft,
        RetargetingType.dexpilot,
        HandType.right,
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

    print(f"[soft_hand] Opening camera/video source: {camera_source}", flush=True)
    cap = _open_video_capture(camera_source, width, height, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {camera_source}")

    print(f"[soft_hand] Config: {config_path}")
    print("[soft_hand] Running front-half realtime pipeline without SAPIEN.")
    print("[soft_hand] Press 'q' in the OpenCV window to stop when --display is set.")

    stats = RunningStats()
    start_s = time.perf_counter()
    stream_start_s = time.monotonic()
    last_timestamp_ms = -1
    soft_hand_wrist_alignment = None

    try:
        while True:
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
            except Exception as exc:
                stats.add_ref(time.perf_counter() - tic)
                stats.add_failure()
                print(f"[soft_hand][WARN] Failed to build retarget input: {exc}")
                continue
            stats.add_ref(time.perf_counter() - tic)

            tic = time.perf_counter()
            try:
                _ = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
            except Exception as exc:
                stats.add_failure()
                print(f"[soft_hand][WARN] Retargeting failed: {exc}")
                continue
            stats.add_retarget(time.perf_counter() - tic)

            if report_every > 0 and stats.frames % report_every == 0:
                _print_stats(stats, time.perf_counter() - start_s, "[soft_hand]")
            if max_frames > 0 and stats.frames >= max_frames:
                break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    elapsed_s = time.perf_counter() - start_s
    _print_stats(stats, elapsed_s, "[soft_hand][final]")
    retargeting.verbose()


def main(
    camera_source: Optional[Union[int, str]] = None,
    max_frames: int = 0,
    report_every: int = 60,
    display: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
):
    """
    Run the Task 3 front-half realtime pipeline:
    camera/video stream -> MediaPipe hand detection -> soft_hand DexPilot qpos.

    Args:
        camera_source: OpenCV camera index or video path. Defaults to camera 0.
        max_frames: Stop after this many captured frames. 0 means run until stream ends.
        report_every: Print timing stats every N captured frames. 0 disables periodic reports.
        display: Show the camera image with MediaPipe hand skeleton overlay.
        width: Optional requested capture width.
        height: Optional requested capture height.
        fps: Optional requested capture FPS.
    """
    run_pipeline(
        camera_source=camera_source,
        max_frames=max_frames,
        report_every=report_every,
        display=display,
        width=width,
        height=height,
        fps=fps,
    )


if __name__ == "__main__":
    tyro.cli(main)
