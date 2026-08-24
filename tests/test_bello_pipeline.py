import os
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.assets.bvh import load_bvh
from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
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
        pipeline = NewtonPipeline(
            skeleton,
            robot_type="bello",
            retarget_config=config,
        )
        source_to_mujoco = SpaceConverter(FacingDirectionType.MUJOCO).transform(
            wp.transform_identity()
        )
        pipeline.add_input_motions([clip], [source_to_mujoco], True)
        motion = np.stack(pipeline.execute()[0].data)

        self.assertEqual(motion.shape, (2, 32))
        self.assertTrue(np.all(np.isfinite(motion)))
        lower = pipeline.ik_model.joint_limit_lower.numpy()[6:]
        upper = pipeline.ik_model.joint_limit_upper.numpy()[6:]
        self.assertTrue(np.all(motion[:, 7:] >= lower - 1.0e-5))
        self.assertTrue(np.all(motion[:, 7:] <= upper + 1.0e-5))

if __name__ == "__main__":
    unittest.main()
