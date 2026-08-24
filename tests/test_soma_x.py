# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soma_retargeter.assets.soma_x import (
    _SOMA_X_TO_Z_UP,
    _align_soma_x_global_frames,
    _load_resampled_motion,
    load_amass_metadata,
)


def test_joint_frame_alignment_preserves_world_positions() -> None:
    global_matrices = np.tile(np.eye(4), (2, 2, 1, 1))
    global_matrices[:, :, :3, 3] = (
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        ((7.0, 8.0, 9.0), (10.0, 11.0, 12.0)),
    )
    pose_rotations = Rotation.from_euler(
        "zx", ((10.0, 5.0), (20.0, -5.0)), degrees=True
    ).as_matrix()
    global_matrices[0, :, :3, :3] = pose_rotations
    global_matrices[1, :, :3, :3] = pose_rotations[::-1]
    source_bind = np.stack(
        [
            Rotation.from_euler("x", angle, degrees=True).as_matrix()
            for angle in (30.0, -20.0)
        ]
    )
    canonical_bind = np.stack(
        [
            Rotation.from_euler("y", angle, degrees=True).as_matrix()
            for angle in (-10.0, 25.0)
        ]
    )

    aligned = _align_soma_x_global_frames(
        global_matrices, source_bind, canonical_bind
    )

    np.testing.assert_allclose(
        aligned[:, :, :3, 3], global_matrices[:, :, :3, 3], atol=0.0
    )
    expected = (
        global_matrices[:, :, :3, :3]
        @ np.swapaxes(_SOMA_X_TO_Z_UP[None] @ source_bind, -1, -2)[None]
        @ canonical_bind[None]
    )
    np.testing.assert_allclose(aligned[:, :, :3, :3], expected, atol=1.0e-12)


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
