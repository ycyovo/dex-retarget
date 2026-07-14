from pathlib import Path

import cv2
import numpy as np


HAND_SKELETON_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


FINGER_COLORS_BGR = {
    "wrist": (80, 80, 80),
    "thumb": (40, 40, 240),
    "index": (40, 220, 220),
    "middle": (40, 220, 80),
    "ring": (240, 130, 40),
    "little": (220, 40, 220),
}


def get_joint_color_bgr(index: int):
    if index == 0:
        return FINGER_COLORS_BGR["wrist"]
    if index <= 4:
        return FINGER_COLORS_BGR["thumb"]
    if index <= 8:
        return FINGER_COLORS_BGR["index"]
    if index <= 12:
        return FINGER_COLORS_BGR["middle"]
    if index <= 16:
        return FINGER_COLORS_BGR["ring"]
    return FINGER_COLORS_BGR["little"]


def project_points_to_panel(
    joint_pos: np.ndarray, axes: tuple[int, int], panel_size: tuple[int, int]
):
    points_2d = joint_pos[:, axes].astype(np.float32)
    points_2d[:, 1] *= -1.0

    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    span = float(np.max(max_xy - min_xy))
    if span < 1e-6:
        span = 1.0

    width, height = panel_size
    margin = 34
    scale = min(width, height) - 2 * margin
    projected = (points_2d - center) / span * scale
    projected += np.array([width * 0.5, height * 0.5], dtype=np.float32)
    return projected.astype(np.int32)


def draw_hand_skeleton_views(joint_pos: np.ndarray, title: str):
    """Draw fixed orthographic views of 21 wrist-local hand landmarks."""
    if joint_pos.shape != (21, 3):
        raise ValueError(f"Expected joint_pos shape (21, 3), got {joint_pos.shape}")

    panel_width, panel_height = 360, 360
    views = [("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))]
    canvas = np.full(
        (panel_height, panel_width * len(views), 3), 255, dtype=np.uint8
    )

    for panel_idx, (view_name, axes) in enumerate(views):
        x0 = panel_idx * panel_width
        panel = canvas[:, x0 : x0 + panel_width]
        points = project_points_to_panel(joint_pos, axes, (panel_width, panel_height))

        cv2.putText(
            panel,
            view_name,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.line(panel, (0, 42), (panel_width, 42), (225, 225, 225), 1)

        for origin, task in HAND_SKELETON_CONNECTIONS:
            color = get_joint_color_bgr(task)
            cv2.line(
                panel,
                tuple(points[origin]),
                tuple(points[task]),
                color,
                3,
                cv2.LINE_AA,
            )

        for index, point in enumerate(points):
            color = get_joint_color_bgr(index)
            cv2.circle(panel, tuple(point), 5, color, -1, cv2.LINE_AA)
            cv2.circle(panel, tuple(point), 6, (30, 30, 30), 1, cv2.LINE_AA)
            label_pos = (int(point[0]) + 7, int(point[1]) - 7)
            cv2.putText(
                panel,
                str(index),
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        canvas,
        title,
        (12, canvas.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_hand_skeleton_debug_image(
    joint_pos: np.ndarray, output_dir: Path, frame_index: int, title: str
):
    output_dir.mkdir(parents=True, exist_ok=True)
    image = draw_hand_skeleton_views(joint_pos, title)
    output_path = output_dir / f"frame_{frame_index:06d}.png"
    cv2.imwrite(str(output_path), image)
    return output_path
