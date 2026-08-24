import os
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp

from soma_retargeter.robotics.robot_model import (
    BELLO_25_DOF_JOINT_NAMES,
    box_shape_support_points,
    build_robot_builder,
    minimum_support_height,
    resolve_robot_mjcf,
)


class BelloRobotModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("BELLO_MJCF_PATH")
        if path is None or not Path(path).is_file():
            raise unittest.SkipTest(
                "BELLO_MJCF_PATH does not point to an authorized Bello MJCF"
            )
        cls.path = Path(path)

    def test_external_model_contract_and_home_keyframe(self):
        builder = build_robot_builder("bello", self.path)
        self.assertEqual(builder.joint_coord_count, 32)
        self.assertEqual(builder.joint_dof_count, 31)
        self.assertEqual(len(BELLO_25_DOF_JOINT_NAMES), 25)
        self.assertAlmostEqual(builder.joint_q[2], 0.91)
        np.testing.assert_allclose(builder.joint_q[3:7], [0.0, 0.0, 0.0, 1.0])
        body_names = [label.split("/")[-1] for label in builder.body_label]
        self.assertIn("l_end_effector_sphere_link", body_names)
        self.assertIn("r_end_effector_sphere_link", body_names)

    def test_environment_path_resolution(self):
        self.assertEqual(resolve_robot_mjcf("bello"), self.path.resolve())

    def test_home_support_height_can_be_grounded_from_model_geometry(self):
        builder = build_robot_builder("bello", self.path)
        supports = box_shape_support_points(
            builder,
            (
                "left_ankle_roll_link_collision_box_1",
                "right_ankle_roll_link_collision_box_1",
            ),
        )
        model_builder = newton.ModelBuilder()
        model_builder.add_builder(builder, wp.transform_identity())
        model = model_builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        initial_height = minimum_support_height(state.body_q.numpy(), supports)
        self.assertGreater(initial_height, 0.04)
        self.assertLess(initial_height, 0.08)

        joint_q = model.joint_q.numpy()
        joint_q[2] -= initial_height
        wp.copy(model.joint_q, wp.array(joint_q, dtype=wp.float32))
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        self.assertAlmostEqual(
            minimum_support_height(state.body_q.numpy(), supports), 0.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
