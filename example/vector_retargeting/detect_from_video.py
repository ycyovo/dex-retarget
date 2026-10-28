import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tqdm
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

try:
    from hand_input_adaptation import (
        adapt_new_hand_thumb_depth,
        new_hand_thumb_root_link,
    )
    from single_hand_detector import SingleHandDetector
    from hand_skeleton_visualization import save_hand_skeleton_debug_image
except ModuleNotFoundError:
    from example.vector_retargeting.hand_input_adaptation import (
        adapt_new_hand_thumb_depth,
        new_hand_thumb_root_link,
    )
    from example.vector_retargeting.single_hand_detector import SingleHandDetector
    from example.vector_retargeting.hand_skeleton_visualization import (
        save_hand_skeleton_debug_image,
    )


SOFT_HAND_WRIST_ALIGNMENT_ALPHA = 0.90


def normalize_vec(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def build_wrist_tip_frame_rows(wrist_to_tip_vecs: np.ndarray):
    """Build semantic wrist-frame axes as rows: lateral, forward, normal.

    The input order is thumb, index, middle, ring, little. The lateral axis is
    explicitly index-side minus little-side, so the transform preserves the
    four-finger order instead of allowing a point-set fit to mirror it.
    """
    thumb, index, middle, ring, little = wrist_to_tip_vecs
    del thumb

    forward = normalize_vec(np.mean([index, middle, ring, little], axis=0))
    lateral = index - little
    lateral = lateral - np.dot(lateral, forward) * forward
    lateral = normalize_vec(lateral)
    normal = normalize_vec(np.cross(forward, lateral))
    forward = normalize_vec(np.cross(lateral, normal))
    return np.stack([lateral, forward, normal], axis=0)


def estimate_palm_frame_row_rotation(
    source_vecs: np.ndarray, target_vecs: np.ndarray
):
    """Return R for row vectors such that source_vecs @ R ~= target_vecs."""
    source_frame = build_wrist_tip_frame_rows(source_vecs)
    target_frame = build_wrist_tip_frame_rows(target_vecs)
    return (source_frame.T @ target_frame).astype(np.float32)


def partial_row_rotation(rotation: np.ndarray, alpha: float):
    """Scale a row-vector rotation from identity to rotation by alpha."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Rotation alpha must be in [0, 1], got {alpha}")

    rotation = rotation.astype(np.float64)
    cos_theta = (np.trace(rotation) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return rotation.astype(np.float32)

    axis_skew = (rotation - rotation.T) / (2.0 * np.sin(theta))
    scaled_theta = theta * alpha
    partial = (
        np.eye(3)
        + np.sin(scaled_theta) * axis_skew
        + (1.0 - np.cos(scaled_theta)) * (axis_skew @ axis_skew)
    )
    return partial.astype(np.float32)


def compute_soft_hand_wrist_alignment(
    retargeting: SeqRetargeting, ref_value: np.ndarray
):
    optimizer = retargeting.optimizer
    len_proj = len(optimizer.projected)

    qpos = optimizer.robot.q0.copy()
    optimizer.robot.compute_forward_kinematics(qpos)
    body_pos = np.array(
        [
            optimizer.robot.get_link_pose(index)[:3, 3]
            for index in optimizer.computed_link_indices
        ]
    )
    origin_link_pos = body_pos[optimizer.origin_link_indices.numpy(), :]
    task_link_pos = body_pos[optimizer.task_link_indices.numpy(), :]
    robot_vec = task_link_pos - origin_link_pos

    # Coordinate transform: MediaPipe wrist-frame vectors -> soft_hand/base_link
    # vectors.  Use a semantic palm frame instead of Kabsch point-set fitting:
    # forward is the four-finger wrist-to-tip direction, lateral is index-side
    # minus little-side. This keeps the index/middle/ring/little order from
    # being mirrored while still aligning the URDF base frame convention.
    full_rotation = estimate_palm_frame_row_rotation(
        source_vecs=ref_value[len_proj:],
        target_vecs=robot_vec[len_proj:],
    )
    # return full_rotation
    # A partial rotation leaves different coordinate errors for the old XY
    # palm and new XZ palm. Use the full frame conversion for the new model.
    alpha = 1.0 if new_hand_thumb_root_link(retargeting) else SOFT_HAND_WRIST_ALIGNMENT_ALPHA
    return partial_row_rotation(full_rotation, alpha)


def retarget_video(
    retargeting: SeqRetargeting,
    video_path: str,
    output_path: str,
    config_path: str,
    robot_name: RobotName,
    debug_joint_vis_dir: Optional[str] = None,
    debug_joint_vis_every_n: int = 30,
    debug_joint_vis_max_frames: int = 8,
):
    cap = cv2.VideoCapture(video_path)

    data = []
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fixed arm qpos
        fixed_limits = retargeting.optimizer.robot.joint_limits[
            retargeting.optimizer.idx_pin2fixed
        ]
        fixed_qpos = fixed_limits.mean(axis=1).astype("float32")
        soft_hand_wrist_alignment = None
        debug_joint_vis_count = 0
        debug_joint_vis_path = (
            Path(debug_joint_vis_dir) if debug_joint_vis_dir is not None else None
        )

        last_timestamp_ms = -1
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(
                    rgb, timestamp_ms
                )
                # import ipdb; ipdb.set_trace()
                last_timestamp_ms = timestamp_ms
                if num_box == 0:
                    continue

                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                    if (
                        robot_name == RobotName.soft
                        and retargeting_type == "DEXPILOT"
                    ):
                        if soft_hand_wrist_alignment is None:
                            try:
                                soft_hand_wrist_alignment = (
                                    compute_soft_hand_wrist_alignment(
                                        retargeting, ref_value
                                    )
                                )
                                print(
                                    "[soft_hand] Calibrated MediaPipe wrist frame "
                                    "to soft_hand/base_link rotation:\n"
                                    f"{np.round(soft_hand_wrist_alignment, 6)}"
                                )
                            except Exception as exc:
                                print(
                                    "[soft_hand][WARN] Failed to calibrate wrist "
                                    f"alignment, using raw retarget input: {exc}"
                                )
                                soft_hand_wrist_alignment = np.eye(
                                    3, dtype=np.float32
                                )
                        ref_value = ref_value @ soft_hand_wrist_alignment
                        ref_value = adapt_new_hand_thumb_depth(
                            retargeting, ref_value,
                            joint_pos @ soft_hand_wrist_alignment,
                        )
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if (
                            debug_joint_vis_path is not None
                            and debug_joint_vis_count < debug_joint_vis_max_frames
                            and debug_joint_vis_every_n > 0
                            and current_frame % debug_joint_vis_every_n == 0
                        ):
                            # Coordinate transform for visualization only:
                            # MediaPipe wrist-frame landmarks -> soft_hand/base_link frame.
                            rotated_joint_pos = joint_pos @ soft_hand_wrist_alignment
                            # debug_path = save_hand_skeleton_debug_image(
                            #     rotated_joint_pos,
                            #     debug_joint_vis_path,
                            #     current_frame,
                            #     "soft_hand/base_link rotated MediaPipe landmarks",
                            # )
                            # debug_joint_vis_count += 1
                            # print(f"[soft_hand] Saved joint debug image: {debug_path}")

                # qpos = retargeting.retarget(ref_value)
                qpos = retargeting.retarget(
                    ref_value, fixed_qpos=fixed_qpos, human_joint_pos=joint_pos
                )
                
                data.append(qpos)
                pbar.update(1)


        meta_data = dict(
            config_path=config_path,
            dof=len(retargeting.optimizer.robot.dof_joint_names),
            joint_names=retargeting.optimizer.robot.dof_joint_names,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(dict(data=data, meta_data=meta_data), f)

        retargeting.verbose()
        cap.release()
        cv2.destroyAllWindows()


def main(
    robot_name: RobotName,
    video_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    debug_joint_vis_dir: Optional[str] = None,
    debug_joint_vis_every_n: int = 30,
    debug_joint_vis_max_frames: int = 8,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_video(
        retargeting,
        video_path,
        output_path,
        str(config_path),
        robot_name,
        debug_joint_vis_dir=debug_joint_vis_dir,
        debug_joint_vis_every_n=debug_joint_vis_every_n,
        debug_joint_vis_max_frames=debug_joint_vis_max_frames,
    )


if __name__ == "__main__":
    tyro.cli(main)
