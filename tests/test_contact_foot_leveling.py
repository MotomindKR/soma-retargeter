import numpy as np
from scipy.spatial.transform import Rotation

from soma_retargeter.pipelines.newton_pipeline import (
    contact_foot_weights,
    level_contact_foot_targets,
    level_sole_quaternions,
)


def test_levels_only_low_stationary_sole_targets() -> None:
    targets = np.zeros((20, 1, 7), dtype=np.float64)
    targets[:, 0, 3:7] = Rotation.from_euler("x", 20.0, degrees=True).as_quat()
    targets[:10, 0, 2] = 0.0
    targets[10:, 0, 2] = 1.0

    leveled = level_contact_foot_targets(
        targets,
        100.0,
        [(0, np.asarray([0.0, 0.0, 0.0, 1.0]))],
        floor_percentile=5.0,
        maximum_height=0.05,
        maximum_speed=0.5,
        transition_seconds=0.0,
    )

    planted_up = Rotation.from_quat(leveled[2, 0, 3:7]).apply([0.0, 0.0, 1.0])
    swing_up = Rotation.from_quat(leveled[15, 0, 3:7]).apply([0.0, 0.0, 1.0])
    np.testing.assert_allclose(planted_up, [0.0, 0.0, 1.0], atol=1.0e-7)
    np.testing.assert_allclose(
        swing_up,
        Rotation.from_euler("x", 20.0, degrees=True).apply([0.0, 0.0, 1.0]),
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(leveled[:, :, :3], targets[:, :, :3])


def test_contact_weights_are_smoothed_without_changing_swing_center() -> None:
    positions = np.zeros((30, 3), dtype=np.float64)
    positions[10:20, 2] = 0.5

    weights = contact_foot_weights(
        positions,
        100.0,
        floor_percentile=5.0,
        maximum_height=0.05,
        maximum_speed=None,
        transition_seconds=0.03,
    )

    assert weights[4] == 1.0
    assert weights[14] == 0.0
    assert 0.0 < weights[9] < 1.0
    assert 0.0 < weights[20] < 1.0


def test_levels_sole_frame_instead_of_assuming_link_frame() -> None:
    link = Rotation.from_euler("xz", [20.0, 30.0], degrees=True)
    sole_local = Rotation.from_euler("y", 10.0, degrees=True)

    result = level_sole_quaternions(
        link.as_quat()[None], sole_local.as_quat(), np.asarray([1.0])
    )
    sole_up = (Rotation.from_quat(result[0]) * sole_local).apply([0.0, 0.0, 1.0])

    np.testing.assert_allclose(sole_up, [0.0, 0.0, 1.0], atol=1.0e-7)
