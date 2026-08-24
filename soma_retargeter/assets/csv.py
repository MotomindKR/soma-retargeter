# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import pickle
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.robotics.robot_model import get_robot_spec


class RobotCSVConfig(Protocol):
    name: str
    csv_header: list[str]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        ...
    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> list[float]:
        ...


@dataclass
class UnitreeG129DOF_CSVConfig:
    name: str = "unitree_g1_29dof"
    csv_header: ClassVar[list[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        """
        Convert one CSV row (including frame index) into one anim buffer frame.
        """
        # csv_row layout: [frame index, tx, ty, tz, rx, ry, rz, dof0, ...]
        num_joint_dofs = csv_row.shape[0] - 1 # Remove frame index
        anim_row = np.zeros(
            num_joint_dofs + 1, # euler rotate xyz values converted to quat
            dtype=np.float32)

        # translation (cm -> m)
        anim_row[0:3] = csv_row[1:4] * 0.01

        # rotation (euler deg -> quat)
        euler = np.deg2rad(csv_row[4:7])
        quat = wp.quat_rpy(euler[0], euler[1], euler[2])
        anim_row[3:7] = quat

        # remaining joints (deg -> rad)
        anim_row[7:] = np.deg2rad(csv_row[7:])

        return anim_row

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> list[float]:
        """
        Convert one anim buffer row into a CSV row with this config's layout.
        """
        # translation (m -> cm)
        t = wp.vec3(*anim_row[0:3]) * 100.0
        # root rotation (quat -> euler deg)
        q = wp.quat(*anim_row[3:7])
        euler = R.from_quat([q[0], q[1], q[2], q[3]]).as_euler("xyz", degrees=True)

        row = [frame_idx, t[0], t[1], t[2], euler[0], euler[1], euler[2]]

        # joints (rad -> deg)
        row.extend(np.rad2deg(anim_row[7:]))

        return row


@dataclass
class Bello25DOF_CSVConfig(UnitreeG129DOF_CSVConfig):
    """Named CSV contract for Bello's 25 logical hinge coordinates."""

    name: str = "bello_25dof"
    csv_header: ClassVar[list[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_pitch_joint_dof",
        "right_elbow_yaw_joint_dof", "right_wrist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_pitch_joint_dof",
        "left_elbow_yaw_joint_dof", "left_wrist_pitch_joint_dof",
    ]


def get_csv_config(robot_type: str) -> RobotCSVConfig:
    """Return the CSV codec matching a robot target."""

    configs = {
        "unitree_g1_29dof": UnitreeG129DOF_CSVConfig,
        "bello_25dof": Bello25DOF_CSVConfig,
    }
    config_name = get_robot_spec(robot_type).csv_config_name
    try:
        return configs[config_name]()
    except KeyError:
        raise ValueError(
            f"Robot [{robot_type}] references unknown CSV config [{config_name}]."
        ) from None


def load_csv(
    file_path: str,
    fps: float = 120.0,
    csv_config: RobotCSVConfig | None = None,
) -> CSVAnimationBuffer:
    """
    Load a robot motion CSV file into a ``CSVAnimationBuffer``.
    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.
        csv_config (RobotCSVConfig, optional): Configuration object that defines how to parse
            CSV rows into animation frames. Defaults to ``UnitreeG129DOF_CSVConfig``.
    Returns:
        CSVAnimationBuffer: An animation buffer containing the loaded and converted animation data.
    Raises:
        FileNotFoundError: If the CSV file at file_path does not exist.
    """
    if csv_config is None:
        csv_config = UnitreeG129DOF_CSVConfig()

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        print(f"[INFO]: Loading CSV [{file_path}] for robot [{csv_config.name}]")
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"CSV for [{csv_config.name}] is empty: {file_path}") from None
        header = next(csv.reader([header_line]))
        if header != csv_config.csv_header:
            raise ValueError(
                f"CSV header does not match the [{csv_config.name}] output contract: "
                f"{file_path}"
            )
        frame_data_position = f.tell()
        if not f.read().strip():
            raise ValueError(f"CSV for [{csv_config.name}] contains no frames: {file_path}")
        f.seek(frame_data_position)
        csv_data = np.loadtxt(f, delimiter=",", ndmin=2)
        num_frames = csv_data.shape[0]

        # Each anim row is derived by config, so infer size from first row
        first_row_anim = csv_config.to_anim_frame(csv_data[0])
        anim_data = np.zeros((num_frames, first_row_anim.shape[0]), dtype=np.float32)
        anim_data[0, :] = first_row_anim

        for i in range(1, num_frames):
            anim_data[i, :] = csv_config.to_anim_frame(csv_data[i])

        return CSVAnimationBuffer.create_from_raw_data(anim_data, fps)


def save_csv(
    file_path: str,
    buffer: CSVAnimationBuffer,
    csv_config: RobotCSVConfig | None = None,
) -> None:
    """
    Save a ``CSVAnimationBuffer`` to a robot motion CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.
        csv_config (RobotCSVConfig, optional): Configuration object that defines CSV format and headers.
            Defaults to ``UnitreeG129DOF_CSVConfig``.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
    """
    if csv_config is None:
        csv_config = UnitreeG129DOF_CSVConfig()
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")
    expected_anim_columns = len(csv_config.csv_header)
    first_frame_columns = len(buffer.get_data(0))
    if first_frame_columns != expected_anim_columns:
        raise ValueError(
            f"Animation for [{csv_config.name}] has {first_frame_columns} coordinates; "
            f"expected {expected_anim_columns}"
        )

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_config.csv_header)

        for i in range(buffer.num_frames):
            data = buffer.get_data(i)
            row = csv_config.to_csv_row(i, data)
            writer.writerow(row)


def save_gmr_pickle(file_path: str, buffer: CSVAnimationBuffer) -> None:
    """Write the legacy GMR motion dictionary consumed by Bello tooling."""

    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")
    data = np.asarray([buffer.get_data(i) for i in range(buffer.num_frames)], dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 32:
        raise ValueError(
            f"GMR Bello export requires [root xyz, root xyzw, 25 joints] (32 columns); "
            f"received {data.shape}"
        )
    payload = {
        "fps": float(buffer.sample_rate),
        "root_pos": data[:, :3],
        "root_rot": data[:, 3:7],
        "dof_pos": data[:, 7:],
    }
    with open(file_path, "wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)
