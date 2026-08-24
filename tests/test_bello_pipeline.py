import os
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.assets.bvh import load_bvh
from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robotics.robot_model import minimum_support_height
from soma_retargeter.utils.space_conversion_utils import (
    FacingDirectionType,
    SpaceConverter,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class BelloPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("BELLO_MJCF_PATH")
        if path is None or not Path(path).is_file():
            raise unittest.SkipTest(
                "BELLO_MJCF_PATH does not point to an authorized Bello MJCF"
            )

    def test_two_frame_motion_is_finite_limited_and_uses_32_coordinates(self):
        skeleton, source = load_bvh(
            REPOSITORY_ROOT / "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh"
        )
        clip = AnimationBuffer(
            skeleton,
            2,
            source.sample_rate,
            np.copy(source.local_transforms[:2]),
        )
        config = pipeline_utils.get_retargeter_config(
            pipeline_utils.SourceType.SOMA,
            pipeline_utils.TargetType.BELLO,
        )
        config["initialization_pose"] = None
        config["offline_solver"]["initial_settle_passes"] = 1
        config["offline_solver"]["joint_smoothing_passes"] = 1
        pipeline = NewtonPipeline(
            skeleton,
            robot_type="bello",
            retarget_config=config,
        )
        source_to_mujoco = SpaceConverter(FacingDirectionType.MUJOCO).transform(
            wp.transform_identity()
        )
        pipeline.add_input_motions([clip], [source_to_mujoco], True)
        motion = np.asarray(pipeline.execute()[0].data)

        self.assertEqual(motion.shape, (2, 32))
        self.assertTrue(np.all(np.isfinite(motion)))
        maximum_delta = (
            config["offline_solver"]["max_joint_velocity"] / source.sample_rate
        )
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(motion[:, 7:], axis=0)))),
            maximum_delta + 1.0e-6,
        )
        lower = pipeline.ik_model.joint_limit_lower.numpy()[6:]
        upper = pipeline.ik_model.joint_limit_upper.numpy()[6:]
        self.assertTrue(np.all(motion[:, 7:] >= lower - 1.0e-5))
        self.assertTrue(np.all(motion[:, 7:] <= upper + 1.0e-5))

        model_builder = newton.ModelBuilder()
        model_builder.add_builder(pipeline.robot_builder, wp.transform_identity())
        model = model_builder.finalize()
        state = model.state()
        heights = []
        for frame in motion:
            wp.copy(model.joint_q, wp.array(frame, dtype=wp.float32))
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            heights.append(
                minimum_support_height(
                    state.body_q.numpy(), pipeline.ground_clearance_solves
                )
            )
        self.assertGreaterEqual(min(heights), -1.0e-6)
        self.assertLessEqual(min(heights), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
