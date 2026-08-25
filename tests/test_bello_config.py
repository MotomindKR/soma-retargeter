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

    def test_single_stage_uses_the_soma_mapping_contract(self):
        config = json.loads(
            (CONFIG_ROOT / "soma_to_bello_retargeter_config.json").read_text()
        )
        self.assertNotIn("ik_stages", config)
        self.assertNotIn("offline_solver", config)
        self.assertNotIn("ground_clearance", config)
        mapping = config["ik_map"]
        self.assertEqual(mapping["Hips"]["r_weight"], 7.5)
        for side in ("Left", "Right"):
            hand = mapping[f"{side}Hand"]
            prefix = "l" if side == "Left" else "r"
            self.assertEqual(hand["t_body"], f"{prefix}_end_effector_sphere_link")
            self.assertEqual(hand["t_weight"], 10.0)
            self.assertEqual(hand["r_weight"], 0.2)
            self.assertEqual(mapping[f"{side}ForeArm"]["r_weight"], 0.0)
            self.assertEqual(mapping[f"{side}Leg"]["r_weight"], 0.0)
            self.assertEqual(mapping[f"{side}Shin"]["r_weight"], 0.0)
            self.assertEqual(mapping[f"{side}Foot"]["r_weight"], 2.0)

    def test_feet_use_geometry_scale_and_vanilla_stabilizer(self):
        scaler = json.loads(
            (CONFIG_ROOT / "soma_to_bello_scaler_config.json").read_text()
        )
        self.assertEqual(scaler["joint_scales"]["Hips"], 0.87)
        self.assertEqual(scaler["joint_scales"]["LeftFoot"], 0.82)
        self.assertEqual(scaler["joint_scales"]["RightFoot"], 0.82)

        retargeter = json.loads(
            (CONFIG_ROOT / "soma_to_bello_retargeter_config.json").read_text()
        )
        self.assertEqual(
            retargeter["feet_stabilizer_config"],
            "bello/bello_feet_stabilizer_config.json",
        )
        stabilizer = json.loads(
            (CONFIG_ROOT / "bello_feet_stabilizer_config.json").read_text()
        )
        self.assertEqual(stabilizer["robot_type"], "bello")
        self.assertEqual(stabilizer["effectors"]["bello_root"], [30.0, 8.0])
        for side in ("left", "right"):
            self.assertEqual(
                stabilizer["effectors"][f"{side}_ankle_roll_link"],
                [10.0, 2.0],
            )
        self.assertTrue(retargeter["enable_post_processing"])

    def test_ablation_selected_solver_values_are_preserved(self):
        config = json.loads(
            (CONFIG_ROOT / "soma_to_bello_retargeter_config.json").read_text()
        )
        self.assertEqual(config["ik_iterations"], 24)
        self.assertEqual(config["joint_limit_weight"], 10.0)
        self.assertEqual(config["smooth_joint_filter_weight"], 10.0)
        self.assertEqual(
            config["smooth_joint_filter_objective_body_masks"]["l_upper_arm_link"],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
