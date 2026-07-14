from pathlib import Path
from typing import Optional, List, Union, Dict

import cv2
import numpy as np
import pinocchio as pin
import sapien
import tqdm
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.retargeting_config import RetargetingConfig

# Optional: same fixed rotation as `panda_to_soft_hand_joint` in `soft_hand_right.urdf` (panda_link8 -> base_link).
# Must match URDF `<origin rpy="..."/>` on that joint so hand-only preview aligns with the combined model.
# Applying it at the **render root** rotates palm + fingers together. For hand-only demos where you want
# the root like other hands / world axes, leave OFF (default).
_SOFT_HAND_PANDA_MOUNT_RPY = (1.5707963268/2, 0.0, 1.5707963268)


def _pose_from_rpy_xyz(rpy, xyz) -> sapien.Pose:
    R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    q = pin.Quaternion(R)
    x, y, z, w = q.coeffs()
    return sapien.Pose([xyz[0], xyz[1], xyz[2]], [x, y, z, w])



def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
    *,
    apply_soft_hand_panda_mount: bool = False,
):
    # Generate rendering config
    use_rt = headless
    if not use_rt:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
    else:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    # Config is loaded only to find the urdf path and robot name
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    # Viewer
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None
    record_video = output_video_path is not None

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)

    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5
    elif "soft_hand" in robot_name:
        loader.scale = 1.0

    if "glb" not in robot_name:
        glb_filepath = str(filepath).replace(".urdf", "_glb.urdf")
        filepath = glb_filepath if Path(glb_filepath).exists() else str(filepath)
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)

    # Debug: hide some links
    def _hide_robot_links(robot, link_names: set[str]):
        for link in robot.get_links():
            if link.name not in link_names:
                continue
            for c in link.entity.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    c.disable()  # 只关显示；关节/正运动学不变
        # 用法
    # _hide_robot_links(robot, {"base_link"})

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "soft_hand" in robot_name:
        # Hand-only: default root = identity + small Z (same pattern as other hands). Optional mount
        # matches Panda flange -> base_link for side-by-side with arm+hand URDF (rotates whole hand).
        if "hand_only" in robot_name and apply_soft_hand_panda_mount:
            robot.set_pose(
                _pose_from_rpy_xyz(_SOFT_HAND_PANDA_MOUNT_RPY, (0.0, 0.0, -0.1))
            )
        else:
            robot.set_pose(sapien.Pose([0, 0, -0.1]))

    # Video recorder
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (cam.get_width(), cam.get_height()),
        )

    # Different robot loader may have different orders for joints.
    # For combined robots (e.g. panda + hand), retargeting may only output hand joints,
    # so we map by intersection instead of requiring exact full-set match.
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    # retargeting_to_sapien = np.array(
    #     [retargeting_joint_names.index(name) for name in sapien_joint_names]
    # ).astype(int)
    retargeting_name_to_id = {
        name: idx for idx, name in enumerate(retargeting_joint_names)
    }
    matched_pairs = [
        (sapien_idx, retargeting_name_to_id[name])
        for sapien_idx, name in enumerate(sapien_joint_names)
        if name in retargeting_name_to_id
    ]
    if not matched_pairs:
        raise RuntimeError(
            "No overlapping joints between rendered robot and retargeting output."
        )

    for qpos in tqdm.tqdm(data):
        qpos_array = np.array(qpos)
        robot_qpos = robot.get_qpos()
        for sapien_idx, retargeting_idx in matched_pairs:
            robot_qpos[sapien_idx] = qpos_array[retargeting_idx]

        
        robot.set_qpos(robot_qpos)

        if not headless:
            for _ in range(2):
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str,
    output_video_path: Optional[str] = None,
    headless: bool = False,
    apply_soft_hand_panda_mount: bool = False,
):
    """
    Loads the preserved robot pose data and renders it either on screen or as an mp4 video.

    Args:
        pickle_path: Path to the .pickle file, created by `detect_from_video.py`.
        output_video_path: Path where the output video in .mp4 format would be saved.
            By default, it is set to None, implying no video will be saved.
        headless: Set to visualize the rendering on the screen by opening the viewer window.
        apply_soft_hand_panda_mount: If True, when rendering soft_hand hand-only URDF, apply the same
            fixed RPY as the Panda flange mount at the robot root (for comparison with the combined model).
            Default False keeps palm/finger axes closer to other hands (identity root + Z offset only).
    """
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]

    render_by_sapien(
        meta_data,
        data,
        output_video_path,
        headless,
        apply_soft_hand_panda_mount=apply_soft_hand_panda_mount,
    )


if __name__ == "__main__":
    tyro.cli(main)
