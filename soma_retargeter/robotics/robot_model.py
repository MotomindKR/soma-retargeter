# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot model specifications and Newton model loading helpers."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
from scipy.spatial.transform import Rotation

BELLO_25_DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
)

GroundSupport = tuple[int, np.ndarray]


@dataclass(frozen=True)
class RobotSpec:
    """Static integration contract for a retargeting target."""

    name: str
    csv_config_name: str
    expected_joint_coord_count: int
    logical_joint_names: tuple[str, ...]
    mjcf_environment_variable: str | None = None
    default_keyframe: str | None = None


_ROBOT_SPECS = {
    "unitree_g1": RobotSpec(
        name="unitree_g1",
        csv_config_name="unitree_g1_29dof",
        expected_joint_coord_count=36,
        logical_joint_names=(),
    ),
    "bello": RobotSpec(
        name="bello",
        csv_config_name="bello_25dof",
        expected_joint_coord_count=32,
        logical_joint_names=BELLO_25_DOF_JOINT_NAMES,
        mjcf_environment_variable="BELLO_MJCF_PATH",
        default_keyframe="home",
    ),
}


def get_robot_spec(robot_type: str) -> RobotSpec:
    """Return the registered target specification."""

    try:
        return _ROBOT_SPECS[robot_type]
    except KeyError:
        allowed = ", ".join(_ROBOT_SPECS)
        raise ValueError(
            f"Unknown robot type [{robot_type}]. Allowed values: {allowed}"
        ) from None


def resolve_robot_mjcf(
    robot_type: str, robot_model_path: str | os.PathLike | None = None
) -> Path:
    """Resolve a robot MJCF without embedding proprietary assets in the package."""

    spec = get_robot_spec(robot_type)
    if robot_type == "unitree_g1" and robot_model_path is None:
        return (
            Path(newton.utils.download_asset("unitree_g1"))
            / "mjcf/g1_29dof_rev_1_0.xml"
        )

    configured_path = robot_model_path
    if configured_path is None and spec.mjcf_environment_variable is not None:
        configured_path = os.environ.get(spec.mjcf_environment_variable)
    if configured_path is None:
        variable_hint = (
            f" or set {spec.mjcf_environment_variable}"
            if spec.mjcf_environment_variable is not None
            else ""
        )
        raise ValueError(
            f"Robot [{robot_type}] requires an explicit robot_model_path{variable_hint}."
        )

    path = Path(os.path.expandvars(os.fspath(configured_path))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robot MJCF does not exist: {path}")
    return path


def build_robot_builder(
    robot_type: str,
    robot_model_path: str | os.PathLike | None = None,
    *,
    enable_self_collisions: bool = False,
) -> newton.ModelBuilder:
    """Load and validate one registered robot articulation."""

    spec = get_robot_spec(robot_type)
    mjcf_path = resolve_robot_mjcf(robot_type, robot_model_path)
    builder = newton.ModelBuilder()
    builder.add_mjcf(mjcf_path, enable_self_collisions=enable_self_collisions)
    if spec.default_keyframe is not None:
        apply_mjcf_keyframe(builder, mjcf_path, spec.default_keyframe)

    if builder.joint_coord_count != spec.expected_joint_coord_count:
        raise ValueError(
            f"Robot [{robot_type}] MJCF has {builder.joint_coord_count} joint coordinates; "
            f"expected {spec.expected_joint_coord_count}: {mjcf_path}"
        )
    if spec.logical_joint_names:
        movable_joint_names = tuple(
            label.split("/")[-1]
            for label, dof_dim in zip(
                builder.joint_label, builder.joint_dof_dim, strict=True
            )
            if int(dof_dim[0]) + int(dof_dim[1]) == 1
        )
        if movable_joint_names != spec.logical_joint_names:
            raise ValueError(
                f"Robot [{robot_type}] movable joint order does not match its output contract:\n"
                f"actual={movable_joint_names}\nexpected={spec.logical_joint_names}"
            )
    return builder


def box_shape_support_points(
    builder: newton.ModelBuilder,
    shape_names: Iterable[str],
) -> list[GroundSupport]:
    """Return body-local corners for named box shapes used as ground supports."""

    shape_names = tuple(shape_names)
    if not shape_names:
        raise ValueError("At least one ground-support shape is required.")
    labels = [label.split("/")[-1] for label in builder.shape_label]
    supports = []
    for shape_name in shape_names:
        try:
            shape_index = labels.index(shape_name)
        except ValueError:
            raise ValueError(
                f"Ground-support shape [{shape_name}] is missing from the robot model."
            ) from None
        if builder.shape_type[shape_index] != newton.GeoType.BOX:
            raise ValueError(f"Ground-support shape [{shape_name}] must be a box.")
        shape_transform = np.asarray(
            builder.shape_transform[shape_index], dtype=np.float64
        )
        half_extents = np.asarray(builder.shape_scale[shape_index], dtype=np.float64)
        corners = np.asarray(
            [
                [x * half_extents[0], y * half_extents[1], z * half_extents[2]]
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        local_points = (
            Rotation.from_quat(shape_transform[3:7]).apply(corners)
            + shape_transform[:3]
        )
        supports.append((int(builder.shape_body[shape_index]), local_points))
    return supports


def minimum_support_height(
    body_transforms: np.ndarray,
    supports: Sequence[GroundSupport],
    body_base: int = 0,
) -> float:
    """Return the lowest world-space point among configured support shapes."""

    if not supports:
        raise ValueError("At least one ground support is required.")
    height = np.inf
    for body_index, local_points in supports:
        transform = body_transforms[body_base + body_index]
        world_points = (
            Rotation.from_quat(transform[3:7]).apply(local_points) + transform[:3]
        )
        height = min(height, float(np.min(world_points[:, 2])))
    return height


def apply_mjcf_keyframe(
    builder: newton.ModelBuilder, mjcf_path: Path, keyframe_name: str
) -> None:
    """Apply an MJCF qpos keyframe because Newton 1.0 does not import keyframes."""

    root = ET.parse(mjcf_path).getroot()
    key = root.find(f"./keyframe/key[@name='{keyframe_name}']")
    if key is None or "qpos" not in key.attrib:
        raise ValueError(f"MJCF keyframe [{keyframe_name}] is missing from {mjcf_path}")
    qpos = [float(value) for value in key.attrib["qpos"].split()]
    if len(qpos) != builder.joint_coord_count:
        raise ValueError(
            f"MJCF keyframe [{keyframe_name}] contains {len(qpos)} coordinates; "
            f"expected {builder.joint_coord_count}: {mjcf_path}"
        )
    # MJCF stores free-joint quaternions scalar-first (wxyz), while Newton's
    # joint coordinate buffers use xyzw.
    for joint_index, joint_type in enumerate(builder.joint_type):
        if joint_type == newton.JointType.FREE:
            coord_start = builder.joint_q_start[joint_index]
            w, x, y, z = qpos[coord_start + 3 : coord_start + 7]
            qpos[coord_start + 3 : coord_start + 7] = [x, y, z, w]
    builder.joint_q[:] = qpos
