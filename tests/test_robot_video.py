import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from soma_retargeter.renderers.robot_video import actor_side_camera, render_times


class RobotVideoTests(unittest.TestCase):
    def test_actor_side_camera_tracks_root_heading(self):
        joint_q = np.zeros(7)
        joint_q[:3] = (1.0, 2.0, 0.5)
        joint_q[3:7] = Rotation.from_euler("z", 90.0, degrees=True).as_quat()

        position, yaw = actor_side_camera(joint_q, distance=3.0, height_offset=0.25)

        np.testing.assert_allclose(position, (1.0, -1.0, 0.75), atol=1.0e-12)
        self.assertAlmostEqual(yaw, 90.0)

    def test_actor_side_camera_can_use_opposite_side(self):
        joint_q = np.asarray((1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0))

        position, yaw = actor_side_camera(
            joint_q,
            distance=3.0,
            height_offset=0.25,
            opposite_side=True,
        )

        np.testing.assert_allclose(position, (4.0, 2.0, 0.75), atol=1.0e-12)
        self.assertAlmostEqual(abs(yaw), 180.0)

    def test_actor_side_camera_rejects_invalid_root(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            actor_side_camera(
                np.asarray((np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
                distance=3.0,
                height_offset=0.25,
            )
        with self.assertRaisesRegex(ValueError, "nonzero"):
            actor_side_camera(np.zeros(7), distance=3.0, height_offset=0.25)

    def test_render_times_include_source_endpoint(self):
        np.testing.assert_allclose(
            render_times(501, 120.0, 30.0), np.arange(126) / 30.0
        )

    def test_render_times_apply_bounds(self):
        np.testing.assert_allclose(
            render_times(
                601,
                120.0,
                25.0,
                start_seconds=1.0,
                maximum_seconds=2.0,
            ),
            1.0 + np.arange(51) / 25.0,
        )


if __name__ == "__main__":
    unittest.main()
