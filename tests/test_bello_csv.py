import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from soma_retargeter.assets.csv import (
    Bello25DOF_CSVConfig,
    load_csv,
    save_csv,
    save_gmr_pickle,
)
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


class BelloCSVTests(unittest.TestCase):
    def setUp(self):
        self.config = Bello25DOF_CSVConfig()
        self.frames = np.zeros((2, 32), dtype=np.float32)
        self.frames[:, 2] = 0.91
        self.frames[:, 6] = 1.0
        self.frames[1, 7:] = np.linspace(-0.2, 0.2, 25)
        self.buffer = CSVAnimationBuffer.create_from_raw_data(self.frames, 120.0)

    def test_named_csv_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            save_csv(path, self.buffer, self.config)
            loaded = load_csv(path, fps=120.0, csv_config=self.config)
        np.testing.assert_allclose(np.asarray(loaded.data), self.frames, atol=1.0e-6)

    def test_gmr_pickle_uses_fixed_head_25_dof_schema(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            save_gmr_pickle(path, self.buffer)
            with path.open("rb") as input_file:
                payload = pickle.load(input_file)
        self.assertEqual(payload["fps"], 120.0)
        self.assertEqual(payload["root_pos"].shape, (2, 3))
        self.assertEqual(payload["root_rot"].shape, (2, 4))
        self.assertEqual(payload["dof_pos"].shape, (2, 25))
        np.testing.assert_array_equal(payload["dof_pos"], self.frames[:, 7:])

    def test_named_csv_rejects_a_different_robot_contract(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            save_csv(path, self.buffer, self.config)
            header, frames = path.read_text(encoding="utf-8").split("\n", 1)
            path.write_text(
                header.replace("left_hip_pitch_joint_dof", "wrong_joint")
                + "\n"
                + frames,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output contract"):
                load_csv(path, fps=120.0, csv_config=self.config)


if __name__ == "__main__":
    unittest.main()
