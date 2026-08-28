from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "example" / "vector_retargeting"))

from realtime_allegro_genesis_pipeline import (  # noqa: E402
    build_joint_order_indices,
)


def test_build_joint_order_indices_reorders_by_name():
    indices = build_joint_order_indices(
        ["joint_b", "joint_c", "joint_a"],
        ["joint_a", "joint_b", "joint_c"],
    )

    np.testing.assert_array_equal(indices, [2, 0, 1])


def test_build_joint_order_indices_rejects_missing_joint():
    with pytest.raises(ValueError, match="joint_c"):
        build_joint_order_indices(
            ["joint_a", "joint_b"],
            ["joint_a", "joint_c"],
        )
