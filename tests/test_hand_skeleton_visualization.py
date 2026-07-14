from pathlib import Path

import cv2
import numpy as np

from example.vector_retargeting.hand_skeleton_visualization import (
    draw_hand_skeleton_views,
    save_hand_skeleton_debug_image,
)


def make_synthetic_hand_landmarks():
    joint_pos = np.zeros((21, 3), dtype=np.float32)
    fingers = {
        1: (-0.25, 0.12, [1, 2, 3, 4]),
        5: (-0.14, 0.23, [5, 6, 7, 8]),
        9: (0.00, 0.28, [9, 10, 11, 12]),
        13: (0.14, 0.23, [13, 14, 15, 16]),
        17: (0.25, 0.18, [17, 18, 19, 20]),
    }
    for _, (x, y_step, indices) in fingers.items():
        for step, index in enumerate(indices, start=1):
            joint_pos[index] = [x * step / len(indices), y_step * step, 0.02 * step]
    return joint_pos


def test_draw_hand_skeleton_views_returns_image():
    image = draw_hand_skeleton_views(
        make_synthetic_hand_landmarks(), "synthetic landmarks"
    )

    assert image.shape == (360, 1080, 3)
    assert image.dtype == np.uint8
    assert image.min() < 255


def test_save_hand_skeleton_debug_image(tmp_path: Path):
    output_path = save_hand_skeleton_debug_image(
        make_synthetic_hand_landmarks(), tmp_path, 3, "synthetic landmarks"
    )

    assert output_path.exists()
    image = cv2.imread(str(output_path))
    assert image is not None
    assert image.shape == (360, 1080, 3)
