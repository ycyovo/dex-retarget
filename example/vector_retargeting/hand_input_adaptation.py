"""Input geometry corrections for the new CAD hand example models."""

import numpy as np


def new_hand_thumb_root_link(retargeting):
    """Recognize source/compatibility topology without relying on file paths."""
    links = set(retargeting.optimizer.robot.link_names)
    if {"thumb_link1", "index_link5"}.issubset(links):
        return "thumb_link1"
    if {"thumb_link_1", "index_mount_link"}.issubset(links):
        return "thumb_link_1"
    return None


def adapt_new_hand_thumb_depth(retargeting, ref_value, aligned_joint_pos):
    """Match thumb-tip depth relative to its root instead of the wrist.

    Inputs are already expressed in the robot base frame, before DexPilot's
    scaling. The new CAD palm lies in XZ, with depth along Y. Its thumb root
    sits behind the palm, unlike the detected human thumb root. Retaining that
    mounting-depth difference in a wrist-to-tip target produces false CMC
    flexion. Only translate the thumb target along depth; retain measured
    root-to-tip motion and both in-plane coordinates.
    """
    root_link = new_hand_thumb_root_link(retargeting)
    if root_link is None:
        return ref_value
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type != "DEXPILOT":
        return ref_value
    if not np.isfinite(optimizer.scaling) or optimizer.scaling <= 0:
        raise ValueError("New-hand thumb adaptation requires positive scaling")
    robot = optimizer.robot
    robot.compute_forward_kinematics(robot.q0)
    root = robot.get_link_pose(robot.get_link_index(root_link))[:3, 3].copy()
    wrist = robot.get_link_pose(robot.get_link_index("base_link"))[:3, 3]
    human_root_depth = aligned_joint_pos[1, 1] - aligned_joint_pos[0, 1]
    depth_offset = (root[1] - wrist[1]) / optimizer.scaling - human_root_depth

    # Update every vector involving the thumb tip, including pairwise terms,
    # so all targets still describe a single consistent set of tip positions.
    origins, tasks = optimizer.target_link_human_indices
    thumb_coefficients = (tasks == 4).astype(float) - (origins == 4).astype(float)
    # Preserve DexPilot's contact targets: near a pinch the measured tips must
    # still meet. Fade the mounting correction out across its projection band.
    thumb_pairs = (thumb_coefficients != 0) & (origins != 0) & (tasks != 0)
    distances = np.linalg.norm(ref_value[thumb_pairs], axis=1)
    if distances.size:
        band = max(optimizer.escape_dist - optimizer.project_dist, 1e-9)
        blend = np.clip((distances.min() - optimizer.project_dist) / band, 0, 1)
        depth_offset *= blend * blend * (3 - 2 * blend)
    adapted = np.array(ref_value, copy=True)
    adapted[:, 1] += thumb_coefficients * depth_offset
    return adapted
