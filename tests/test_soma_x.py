# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_retargeter.assets.soma_x import (
    _load_resampled_motion,
    _stabilize_anatomical_joint_frames,
    load_amass_metadata,
)


def _write_motion(
    path: Path, *, model_type: str | bytes = "smplx", gender: str | bytes = "neutral"
) -> Path:
    frame_count = 7
    np.savez(
        path,
        surface_model_type=np.asarray(model_type),
        gender=np.asarray(gender),
        mocap_frame_rate=np.asarray(60.0),
        poses=np.zeros((frame_count, 165), dtype=np.float32),
        trans=np.column_stack(
            (np.linspace(0.0, 0.6, frame_count), np.zeros((frame_count, 2)))
        ),
        betas=np.zeros(16, dtype=np.float32),
    )
    return path


def test_load_and_resample_amass_metadata(tmp_path: Path) -> None:
    path = _write_motion(tmp_path / "motion.npz")
    metadata, poses, translations, betas, output_fps = _load_resampled_motion(
        path,
        target_fps=30.0,
        start_seconds=0.05,
        duration_seconds=0.05,
    )

    assert metadata.gender == "neutral"
    assert metadata.frame_count == 7
    assert poses.shape == (2, 55, 3)
    assert translations.shape == (2, 3)
    assert betas.shape == (16,)
    assert output_fps == 30.0
    assert translations[-1, 0] > translations[0, 0]


def test_preserves_native_frame_rate_by_default(tmp_path: Path) -> None:
    path = _write_motion(tmp_path / "motion.npz")
    _, poses, _, _, output_fps = _load_resampled_motion(
        path,
        target_fps=None,
        start_seconds=0.0,
        duration_seconds=None,
    )

    assert output_fps == 60.0
    assert len(poses) == 7


def test_rejects_non_smplx_motion(tmp_path: Path) -> None:
    path = _write_motion(tmp_path / "motion.npz", model_type="smplh")
    with pytest.raises(ValueError, match="surface model must be"):
        load_amass_metadata(path)


def test_accepts_bytes_encoded_metadata(tmp_path: Path) -> None:
    path = _write_motion(tmp_path / "motion.npz", model_type=b"smplx", gender=b"male")
    assert load_amass_metadata(path).gender == "male"


def test_anatomical_frames_follow_landmarks_without_changing_positions() -> None:
    names = ["Hips", "Chest", "LeftArm", "RightArm"]
    for side in ("Left", "Right"):
        names.extend(
            f"{side}{suffix}"
            for suffix in (
                "ForeArm",
                "Hand",
                "HandMiddle1",
                "HandIndex1",
                "HandPinky1",
            )
        )
    positions = {
        "Hips": [0.0, 0.0, 0.0],
        "Chest": [0.0, 0.0, 1.0],
        "LeftArm": [1.0, 0.0, 1.0],
        "RightArm": [-1.0, 0.0, 1.0],
    }
    for side, x in (("Left", 1.0), ("Right", -1.0)):
        positions.update(
            {
                f"{side}ForeArm": [x, 0.0, 0.0],
                f"{side}Hand": [x, 0.0, -1.0],
                f"{side}HandMiddle1": [x, 1.0, -1.0],
                f"{side}HandIndex1": [x + 0.2, 1.0, -1.0],
                f"{side}HandPinky1": [x - 0.2, 1.0, -1.0],
            }
        )
    canonical = np.repeat(np.eye(4)[None, ...], len(names), axis=0)
    canonical[:, :3, 3] = np.asarray([positions[name] for name in names])
    world_rotation = Rotation.from_euler("zy", [35.0, 20.0], degrees=True).as_matrix()
    poses = np.repeat(canonical[None, ...], 2, axis=0)
    poses[:, :, :3, 3] = canonical[:, :3, 3] @ world_rotation.T

    stabilized = _stabilize_anatomical_joint_frames(poses, names, names, canonical)

    for side in ("Left", "Right"):
        for suffix in ("Arm", "Hand"):
            index = names.index(f"{side}{suffix}")
            np.testing.assert_allclose(
                stabilized[:, index, :3, :3],
                np.broadcast_to(world_rotation, (2, 3, 3)),
                atol=1.0e-7,
            )
    np.testing.assert_array_equal(stabilized[:, :, :3, 3], poses[:, :, :3, 3])
