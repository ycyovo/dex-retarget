"""Inspect a URDF with every active joint held at exactly zero radians."""

import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import sapien
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from transforms3d.quaternions import mat2quat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "assets/robots/hands/hand/hand_compat.urdf"


def _camera_pose(robot, view):
    bounds = []
    for link in robot.get_links():
        body = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if body is not None and body.render_shapes:
            bounds.append(body.compute_global_aabb_tight())
    if not bounds:
        raise RuntimeError("The URDF contains no visible geometry.")
    lower = np.min([b[0] for b in bounds], axis=0)
    upper = np.max([b[1] for b in bounds], axis=0)
    center = (lower + upper) / 2
    # Fit the bounding sphere into a 60 degree vertical field of view.
    radius = max(float(np.linalg.norm(upper - lower)) / 2, 0.02)
    distance = radius / np.sin(np.pi / 6) * 1.2
    direction = np.array({
        "front": [0, -1, 0],
        "back": [0, 1, 0],
        "side": [1, 0, 0],
        "top": [0, 0, 1],
    }[view], dtype=float)
    forward = -direction
    up_hint = np.array([0, 1, 0] if view == "top" else [0, 0, 1])
    left = np.cross(up_hint, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation = np.column_stack([forward, left, up])
    return sapien.Pose(center + distance * direction, mat2quat(rotation))


def main(
    urdf_path: Path = DEFAULT_URDF,
    view: Literal["front", "back", "side", "top"] = "front",
    headless: bool = False,
    output: Path = Path("zero_pose.png"),
    hand_color: Optional[Tuple[float, float, float]] = None,
):
    """Display the exact URDF zero pose without retargeting or physics steps.

    Args:
        urdf_path: URDF to inspect; defaults to the new hand compatibility model.
        view: Initial view: front (-Y), back (+Y), side (+X), or top (+Z).
        headless: Save a single image instead of opening the interactive viewer.
        output: PNG output path when headless is enabled.
        hand_color: Optional RGB override in [0, 1], for example (0.35, 0.65, 0.9).
    """
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
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
    loader.fix_root_link = True
    robot = loader.load(str(urdf_path))
    if robot is None:
        raise RuntimeError(f"Failed to load URDF: {urdf_path}")
    if hand_color is not None:
        if len(hand_color) != 3 or any(c < 0 or c > 1 for c in hand_color):
            raise ValueError("hand_color must contain three values in [0, 1]")
        for link in robot.get_links():
            body = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if body is None:
                continue
            for shape in body.render_shapes:
                material = shape.get_material()
                material.base_color = [*hand_color, 1]
    robot.set_pose(sapien.Pose())
    zeros = np.zeros_like(robot.get_qpos())
    robot.set_qpos(zeros)
    robot.set_qvel(np.zeros_like(robot.get_qvel()))
    scene.update_render()
    pose = _camera_pose(robot, view)
    print(f"URDF: {urdf_path}")
    print(f"All {len(zeros)} active joints set to 0 degrees; base pose is identity.")
    for joint in robot.get_active_joints():
        print(f"  {joint.get_name()}: 0 deg")
    print("This is the URDF zero pose, which may differ from a straight hand.")

    if headless:
        import cv2

        camera = scene.add_camera("zero_pose", 1024, 1024, np.pi / 3, 0.001, 100)
        camera.set_entity_pose(pose)
        scene.update_render()
        camera.take_picture()
        color = camera.get_picture("Color")
        rgb = (np.clip(color[..., :3], 0, 1) * 255).astype(np.uint8)
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), rgb[..., ::-1]):
            raise RuntimeError(f"Failed to save image: {output}")
        np.testing.assert_array_equal(robot.get_qpos(), zeros)
        print(f"Saved: {output}")
        return

    viewer = Viewer()
    try:
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = True
        viewer.control_window.move_speed = 0.01
        viewer.set_camera_pose(pose)
        print("Rotate/zoom with the mouse; close the window to exit.")
        while not viewer.closed:
            # No scene.step(): gravity and joint limits must not change this pose.
            robot.set_qpos(zeros)
            scene.update_render()
            viewer.render()
            time.sleep(1 / 60)
    finally:
        viewer.close()


if __name__ == "__main__":
    tyro.cli(main)
