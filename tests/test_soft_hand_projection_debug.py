from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "example" / "vector_retargeting"))

from dex_retargeting.constants import (  # noqa: E402
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig  # noqa: E402
from single_hand_detector import SingleHandDetector  # noqa: E402


SOFT_HAND_WRIST_ALIGNMENT_ALPHA = 0.95


def build_soft_retargeting(disable_pairwise: bool = False):
    config_path = get_default_config_path(
        RobotName.soft, RetargetingType.dexpilot, HandType.right
    )
    RetargetingConfig.set_default_urdf_dir(str(ROOT / "assets" / "robots" / "hands"))
    old_env = os.environ.get("DEXPILOT_DISABLE_ALL_PAIRWISE")
    try:
        if disable_pairwise:
            os.environ["DEXPILOT_DISABLE_ALL_PAIRWISE"] = "1"
        else:
            os.environ.pop("DEXPILOT_DISABLE_ALL_PAIRWISE", None)
        return RetargetingConfig.load_from_file(config_path).build()
    finally:
        if old_env is None:
            os.environ.pop("DEXPILOT_DISABLE_ALL_PAIRWISE", None)
        else:
            os.environ["DEXPILOT_DISABLE_ALL_PAIRWISE"] = old_env


def collect_first_ref_values(max_frames: int, max_video_frames_to_scan: int):
    cache_path = ROOT / "tests" / "soft_hand_debug_ref_values.npz"
    if cache_path.exists():
        obj = np.load(cache_path)
        ref_values = [ref_value for ref_value in obj["ref_values"][:max_frames]]
        return build_soft_retargeting().optimizer, ref_values

    video_path = ROOT / "example" / "vector_retargeting" / "data" / "human_hand_video.mp4"
    retargeting = build_soft_retargeting()
    indices = retargeting.optimizer.target_link_human_indices
    origin_indices = indices[0, :]
    task_indices = indices[1, :]

    cap = cv2.VideoCapture(str(video_path))
    ref_values: list[np.ndarray] = []
    last_timestamp_ms = -1

    vector_example_dir = ROOT / "example" / "vector_retargeting"
    old_cwd = Path.cwd()
    os.chdir(vector_example_dir)
    try:
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        # import ipdb; ipdb.set_trace()
        scanned_frames = 0
        while (
            cap.isOpened()
            and len(ref_values) < max_frames
            and scanned_frames < max_video_frames_to_scan
        ):
            ret, frame = cap.read()
            if not ret:
                break
            scanned_frames += 1

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            num_box, joint_pos, _keypoint_2d, _mediapipe_wrist_rot = detector.detect(
                frame[..., ::-1], timestamp_ms
            )
            last_timestamp_ms = timestamp_ms
            if num_box == 0:
                continue

            ref_values.append(joint_pos[task_indices, :] - joint_pos[origin_indices, :])
    finally:
        os.chdir(old_cwd)
        cap.release()

    # import ipdb; ipdb.set_trace()

    if not ref_values:
        raise RuntimeError("No hand frames detected in the requested scan window.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ref_values=np.asarray(ref_values))
    return retargeting.optimizer, ref_values


def run_retarget_variant(ref_values, disable_pairwise: bool):
    retargeting = build_soft_retargeting(disable_pairwise=disable_pairwise)
    fixed_limits = retargeting.optimizer.robot.joint_limits[
        retargeting.optimizer.idx_pin2fixed
    ]
    fixed_qpos = fixed_limits.mean(axis=1).astype("float32")
    outputs = []
    for ref_value in ref_values:
        outputs.append(retargeting.retarget(ref_value, fixed_qpos=fixed_qpos))
    return retargeting, np.asarray(outputs)


def dump_pairwise_comparison(ref_values):
    key_names = [
        "thumb_joint_2",
        "thumb_joint_3",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_2",
        "ring_joint_3",
        "little_joint_2",
        "little_joint_3",
    ]

    print("pairwise_comparison", flush=True)
    for label, disable_pairwise in [
        ("baseline", False),
        ("no_pairwise", True),
    ]:
        retargeting, arr = run_retarget_variant(
            ref_values, disable_pairwise=disable_pairwise
        )
        joint_names = retargeting.optimizer.robot.dof_joint_names
        key_idx = [joint_names.index(name) for name in key_names]
        j23_idx = [
            i
            for i, name in enumerate(joint_names)
            if name.endswith("_joint_2") or name.endswith("_joint_3")
        ]
        first = arr[0, key_idx]
        print(
            f"{label} abs_j23_mean={np.abs(arr[:, j23_idx]).mean():.4f} "
            f"abs_j23_max={np.abs(arr[:, j23_idx]).max():.4f}",
            flush=True,
        )
        print(
            f"{label} first["
            "thumb2 thumb3 index2 index3 middle2 middle3 ring2 ring3 little2 little3"
            f"]={np.round(first, 4).tolist()}",
            flush=True,
        )


def dump_wrist_to_tip_fk_comparison(opt, ref_values, frame_idx: int = 0):
    len_proj = len(opt.projected)
    target_vector = ref_values[frame_idx]
    ref_vec = target_vector * opt.scaling

    qpos = opt.robot.q0.copy()
    opt.robot.compute_forward_kinematics(qpos)
    body_pos = np.array(
        [
            opt.robot.get_link_pose(index)[:3, 3]
            for index in opt.computed_link_indices
        ]
    )
    origin_link_pos = body_pos[opt.origin_link_indices.numpy(), :]
    task_link_pos = body_pos[opt.task_link_indices.numpy(), :]
    robot_vec = task_link_pos - origin_link_pos

    print("wrist_to_tip_fk_comparison", flush=True)
    for i in range(len_proj, len(ref_vec)):
        robot_dist = np.linalg.norm(robot_vec[i])
        ref_dist = np.linalg.norm(ref_vec[i])
        cosine = float(
            np.dot(robot_vec[i], ref_vec[i])
            / ((robot_dist + 1e-9) * (ref_dist + 1e-9))
        )
        print(
            f"  {opt.origin_link_names[i]} -> {opt.task_link_names[i]} "
            f"robot_dist={robot_dist:.4f} ref_dist={ref_dist:.4f} "
            f"ratio_ref_over_robot={ref_dist / (robot_dist + 1e-9):.4f} "
            f"cosine={cosine:.4f} "
            f"robot_vec={np.round(robot_vec[i], 4).tolist()} "
            f"ref_vec={np.round(ref_vec[i], 4).tolist()}",
            flush=True,
        )


def get_zero_pose_robot_vectors(opt):
    qpos = opt.robot.q0.copy()
    opt.robot.compute_forward_kinematics(qpos)
    body_pos = np.array(
        [
            opt.robot.get_link_pose(index)[:3, 3]
            for index in opt.computed_link_indices
        ]
    )
    origin_link_pos = body_pos[opt.origin_link_indices.numpy(), :]
    task_link_pos = body_pos[opt.task_link_indices.numpy(), :]
    return task_link_pos - origin_link_pos


def estimate_row_vector_rotation(source_vecs: np.ndarray, target_vecs: np.ndarray):
    """Return R for row vectors such that source_vecs @ R ~= target_vecs."""
    source_norm = source_vecs / (np.linalg.norm(source_vecs, axis=1, keepdims=True) + 1e-9)
    target_norm = target_vecs / (np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-9)
    h = source_norm.T @ target_norm
    u, _s, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def normalize_vec(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def build_wrist_tip_frame_rows(wrist_to_tip_vecs: np.ndarray):
    """Build semantic wrist-frame axes as rows: lateral, forward, normal.

    The input order is thumb, index, middle, ring, little.  The lateral axis is
    explicitly index-side minus little-side, so this frame preserves the
    four-finger left/right order that plain Kabsch can flip.
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


def estimate_palm_frame_row_rotation(source_vecs: np.ndarray, target_vecs: np.ndarray):
    """Return R for MediaPipe wrist-frame row vectors -> soft_hand/base_link."""
    source_frame = build_wrist_tip_frame_rows(source_vecs)
    target_frame = build_wrist_tip_frame_rows(target_vecs)
    return source_frame.T @ target_frame


def partial_row_rotation(rotation: np.ndarray, alpha: float):
    """Scale a row-vector rotation from identity to rotation by alpha."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Rotation alpha must be in [0, 1], got {alpha}")

    rotation = rotation.astype(np.float64)
    cos_theta = (np.trace(rotation) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return rotation

    axis_skew = (rotation - rotation.T) / (2.0 * np.sin(theta))
    scaled_theta = theta * alpha
    return (
        np.eye(3)
        + np.sin(scaled_theta) * axis_skew
        + (1.0 - np.cos(scaled_theta)) * (axis_skew @ axis_skew)
    )


def rotate_ref_values(ref_values, rotation):
    return [ref_value @ rotation for ref_value in ref_values]


def dump_lateral_order(opt, robot_vec, values, label):
    len_proj = len(opt.projected)
    wrist_to_tip = values[len_proj:] * opt.scaling
    frame = build_wrist_tip_frame_rows(robot_vec[len_proj:])
    lateral = frame[0]
    names = [name.replace("_tip", "") for name in opt.task_link_names[len_proj:]]
    coords = wrist_to_tip @ lateral
    order = [name for _, name in sorted(zip(coords, names), reverse=True)]
    print(
        f"{label} lateral_coords={dict(zip(names, np.round(coords, 4).tolist()))} "
        f"order_high_to_low={order}",
        flush=True,
    )


def dump_rotated_wrist_to_tip_comparison(opt, ref_values):
    len_proj = len(opt.projected)
    robot_vec = get_zero_pose_robot_vectors(opt)
    source = ref_values[0][len_proj:]
    target = robot_vec[len_proj:] / opt.scaling
    palm_frame_rotation = estimate_palm_frame_row_rotation(source, target)
    rotations = [
        ("kabsch", estimate_row_vector_rotation(source, target)),
        ("palm_frame", palm_frame_rotation),
        (
            f"palm_frame_alpha_{SOFT_HAND_WRIST_ALIGNMENT_ALPHA:.2f}",
            partial_row_rotation(
                palm_frame_rotation, SOFT_HAND_WRIST_ALIGNMENT_ALPHA
            ),
        ),
    ]

    print("rotated_wrist_to_tip_alignment", flush=True)
    dump_lateral_order(opt, robot_vec, ref_values[0], "before")
    dump_lateral_order(opt, robot_vec, robot_vec / opt.scaling, "robot_zero")
    for rotation_label, rotation in rotations:
        rotated_ref_values = rotate_ref_values(ref_values, rotation)
        print(
            f"{rotation_label} rotation_det={np.linalg.det(rotation):.6f}",
            flush=True,
        )
        dump_lateral_order(opt, robot_vec, rotated_ref_values[0], rotation_label)
        for label, values in [("before", ref_values), (rotation_label, rotated_ref_values)]:
            print(label, flush=True)
            for i in range(len_proj, len(robot_vec)):
                ref_vec = values[0][i] * opt.scaling
                robot_dist = np.linalg.norm(robot_vec[i])
                ref_dist = np.linalg.norm(ref_vec)
                cosine = float(
                    np.dot(robot_vec[i], ref_vec)
                    / ((robot_dist + 1e-9) * (ref_dist + 1e-9))
                )
                print(
                    f"  {opt.origin_link_names[i]} -> {opt.task_link_names[i]} "
                    f"robot_dist={robot_dist:.4f} ref_dist={ref_dist:.4f} "
                    f"ratio_ref_over_robot={ref_dist / (robot_dist + 1e-9):.4f} "
                    f"cosine={cosine:.4f}",
                    flush=True,
                )

    print("rotated_retarget_comparison", flush=True)
    variants = [("baseline", ref_values)]
    variants.extend(
        (rotation_label, rotate_ref_values(ref_values, rotation))
        for rotation_label, rotation in rotations
    )
    for label, values in variants:
        retargeting, arr = run_retarget_variant(values, disable_pairwise=False)
        joint_names = retargeting.optimizer.robot.dof_joint_names
        key_names = [
            "thumb_joint_1",
            "thumb_joint_2",
            "thumb_joint_3",
            "index_joint_1",
            "index_joint_2",
            "index_joint_3",
            "middle_joint_1",
            "middle_joint_2",
            "middle_joint_3",
            "ring_joint_1",
            "ring_joint_2",
            "ring_joint_3",
            "little_joint_1",
            "little_joint_2",
            "little_joint_3",
        ]
        key_idx = [joint_names.index(name) for name in key_names]
        j1_idx = [i for i, name in enumerate(joint_names) if name.endswith("_joint_1")]
        j23_idx = [
            i
            for i, name in enumerate(joint_names)
            if name.endswith("_joint_2") or name.endswith("_joint_3")
        ]
        print(
            f"{label} abs_j1_mean={np.abs(arr[:, j1_idx]).mean():.4f} "
            f"abs_j1_max={np.abs(arr[:, j1_idx]).max():.4f} "
            f"abs_j23_mean={np.abs(arr[:, j23_idx]).mean():.4f} "
            f"abs_j23_max={np.abs(arr[:, j23_idx]).max():.4f}",
            flush=True,
        )
        for i in range(len_proj, len(robot_vec)):
            pass
        print(
            f"{label} first["
            "thumb1 thumb2 thumb3 index1 index2 index3 middle1 middle2 middle3 "
            "ring1 ring2 ring3 little1 little2 little3"
            f"]={np.round(arr[0, key_idx], 4).tolist()}",
            flush=True,
        )


def dump_projection_state(opt, ref_values):
    len_proj = len(opt.projected)
    len_s2 = len(opt.s2_project_index_task)
    len_s1 = len_proj - len_s2

    print(
        f"project_dist={opt.project_dist} escape_dist={opt.escape_dist} "
        f"len_proj={len_proj} len_s1={len_s1} len_s2={len_s2}",
        flush=True,
    )
    for frame_idx, target_vector in enumerate(ref_values):
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        projected = opt.projected.copy()
        projected[:len_s1][target_vec_dist[0:len_s1] < opt.project_dist] = True
        projected[:len_s1][target_vec_dist[0:len_s1] > opt.escape_dist] = False
        projected[len_s1:len_proj] = np.logical_and(
            projected[:len_s1][opt.s2_project_index_origin],
            projected[:len_s1][opt.s2_project_index_task],
        )
        projected[len_s1:len_proj] = np.logical_and(
            projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        print(f"frame={frame_idx}")
        print(f"  dist={np.round(target_vec_dist, 4).tolist()}")
        print(f"  projected={projected.astype(int).tolist()}")
        print(f"  projected_count={int(projected.sum())}/{len_proj}")


def dump_reference_vec_state(opt, ref_values, num_frames: int = 1):
    len_proj = len(opt.projected)
    origin = opt.origin_link_names
    task = opt.task_link_names
    print("reference_vec_debug", flush=True)
    for frame_idx, target_vector in enumerate(ref_values[:num_frames]):
        normal_vec = target_vector * opt.scaling
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        normal_vec_dist = np.linalg.norm(normal_vec[:len_proj], axis=1)
        print(f"frame={frame_idx}", flush=True)
        for i in range(len_proj):
            print(
                f"  {origin[i]} -> {task[i]} "
                f"target_dist={target_vec_dist[i]:.4f} "
                f"ref_dist={normal_vec_dist[i]:.4f} "
                f"target_vec={np.round(target_vector[i], 4).tolist()} "
                f"ref_vec={np.round(normal_vec[i], 4).tolist()}",
                flush=True,
            )


def main():
    # import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--max-video-frames-to-scan", type=int, default=30)
    parser.add_argument("--compare-pairwise", action="store_true")
    parser.add_argument("--compare-wrist-to-tip-fk", action="store_true")
    parser.add_argument("--compare-rotated-target", action="store_true")
    args = parser.parse_args()

    opt, ref_values = collect_first_ref_values(
        max_frames=args.max_frames,
        max_video_frames_to_scan=args.max_video_frames_to_scan,
    )
    dump_reference_vec_state(opt, ref_values, num_frames=1)
    dump_projection_state(opt, ref_values)
    if args.compare_wrist_to_tip_fk:
        dump_wrist_to_tip_fk_comparison(opt, ref_values)
    if args.compare_rotated_target:
        dump_rotated_wrist_to_tip_comparison(opt, ref_values)
    if args.compare_pairwise:
        dump_pairwise_comparison(ref_values)


if __name__ == "__main__":
    main()
