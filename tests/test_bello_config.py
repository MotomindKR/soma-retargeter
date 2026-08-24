import json
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "soma_retargeter" / "configs" / "bello"


class BelloConfigTests(unittest.TestCase):
    def test_scaler_quaternions_are_normalized(self):
        config = json.loads(
            (CONFIG_ROOT / "soma_to_bello_scaler_config.json").read_text()
        )
        self.assertEqual(config["robot_type"], "bello")
        self.assertEqual(set(config["joint_scales"]), set(config["joint_offsets"]))
        for joint_name, (_, quaternion) in config["joint_offsets"].items():
            self.assertAlmostEqual(
                float(np.linalg.norm(quaternion)),
                1.0,
                places=9,
                msg=joint_name,
            )

    def test_two_stages_share_the_soma_mapping_contract(self):
        config = json.loads(
            (CONFIG_ROOT / "soma_to_bello_retargeter_config.json").read_text()
        )
        stages = config["ik_stages"]
        self.assertEqual(
            [stage["name"] for stage in stages], ["branch_selection", "final_tracking"]
        )
        self.assertEqual(tuple(stages[0]["ik_map"]), tuple(stages[1]["ik_map"]))
        for side in ("Left", "Right"):
            hand = stages[1]["ik_map"][f"{side}Hand"]
            prefix = "l" if side == "Left" else "r"
            self.assertEqual(hand["t_body"], f"{prefix}_end_effector_sphere_link")
            self.assertEqual(hand["t_weight"], 25.0)
            self.assertEqual(hand["r_weight"], [3.0, 3.0, 3.0])

    def test_feet_use_geometry_scale_and_motion_grounding(self):
        scaler = json.loads(
            (CONFIG_ROOT / "soma_to_bello_scaler_config.json").read_text()
        )
        self.assertEqual(scaler["joint_scales"]["LeftFoot"], 0.82)
        self.assertEqual(scaler["joint_scales"]["RightFoot"], 0.82)

        retargeter = json.loads(
            (CONFIG_ROOT / "soma_to_bello_retargeter_config.json").read_text()
        )
        ground = retargeter["ground_clearance"]
        self.assertTrue(ground["align_motion_to_ground"])
        self.assertEqual(ground["reference_percentile"], 50.0)

    def test_ablation_selected_solver_values_are_preserved(self):
        config = json.loads(
            (CONFIG_ROOT / "soma_to_bello_retargeter_config.json").read_text()
        )
        offline = config["offline_solver"]
        self.assertEqual(config["ik_iterations"], 12)
        self.assertEqual(offline["max_joint_velocity"], 7.5)
        self.assertEqual(
            offline["joint_smoothing_kernel"], [0.0625, 0.25, 0.375, 0.25, 0.0625]
        )
        self.assertEqual(offline["joint_smoothing_passes"], 3)


if __name__ == "__main__":
    unittest.main()
