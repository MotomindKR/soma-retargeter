# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantitative metrics used to compare retargeting parameter profiles."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import newton
import numpy as np
import warp as wp
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def joint_jitter_metrics(
    motion: np.ndarray,
    sample_rate: float,
    *,
    evaluation_rate: float = 30.0,
) -> dict[str, float | int]:
    """Measure high-frequency hinge residuals at a common evaluation rate."""

    joints = np.asarray(motion, dtype=np.float64)[:, 7:]
    stride = max(1, round(float(sample_rate) / float(evaluation_rate)))
    joints = joints[::stride]
    window = min(7, len(joints) if len(joints) % 2 else len(joints) - 1)
    if window >= 3:
        trend = savgol_filter(
            joints,
            window_length=window,
            polyorder=min(2, window - 1),
            axis=0,
            mode="interp",
        )
        residual_degrees = np.rad2deg(joints - trend)
    else:
        residual_degrees = np.zeros_like(joints)
    steps = np.diff(np.asarray(motion, dtype=np.float64)[:, 7:], axis=0)
    velocities = np.rad2deg(steps) * float(sample_rate)
    return {
        "evaluation_rate_hz": float(sample_rate) / stride,
        "rms_degrees": float(np.sqrt(np.mean(residual_degrees**2))),
        "p99_degrees": float(np.percentile(np.abs(residual_degrees), 99)),
        "maximum_degrees": float(np.max(np.abs(residual_degrees))),
        "velocity_p99_degrees_per_second": (
            float(np.percentile(np.abs(velocities), 99)) if velocities.size else 0.0
        ),
        "maximum_velocity_degrees_per_second": (
            float(np.max(np.abs(velocities))) if velocities.size else 0.0
        ),
        "joint_samples": int(joints.size),
    }


def body_trajectories(
    robot_builder: newton.ModelBuilder,
    motions: list[np.ndarray],
) -> list[np.ndarray]:
    """Evaluate variable-length trajectories with one bounded world per motion."""

    if not motions:
        return []
    num_worlds = len(motions)
    builder = newton.ModelBuilder()
    for _ in range(num_worlds):
        builder.add_builder(robot_builder, xform=wp.transform_identity())
    model = builder.finalize()
    state = model.state()
    bodies_per_robot = robot_builder.body_count
    default_q = np.asarray(robot_builder.joint_q, dtype=np.float32)
    motions = [np.asarray(motion, dtype=np.float32) for motion in motions]
    results = [
        np.empty((len(motion), bodies_per_robot, 7), dtype=np.float32)
        for motion in motions
    ]
    joint_q = np.repeat(default_q[None, :], num_worlds, axis=0)
    for frame in range(max(len(motion) for motion in motions)):
        active = []
        for world, motion in enumerate(motions):
            if frame < len(motion):
                joint_q[world] = motion[frame]
                active.append(world)
        wp.copy(model.joint_q, wp.array(joint_q.reshape(-1), dtype=wp.float32))
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy().reshape(num_worlds, bodies_per_robot, 7)
        for world in active:
            results[world][frame] = body_q[world]
    return results


def task_tracking_metrics(
    motions: list[np.ndarray],
    targets: list[np.ndarray],
    pipeline,
) -> dict:
    """Measure unweighted task errors for each mapped source landmark."""

    if len(motions) != len(targets):
        raise ValueError("Motion and target batch sizes do not match.")
    trajectories = body_trajectories(pipeline.robot_builder, motions)
    per_motion = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    for motion_index, (motion, target, body_q) in enumerate(
        zip(motions, targets, trajectories, strict=True)
    ):
        if len(motion) != len(target):
            raise ValueError(
                f"Motion {motion_index} has {len(motion)} frames but {len(target)} targets."
            )
        motion_report = {}
        for landmark_index, landmark_name in enumerate(pipeline.mapped_joints):
            position_body, _ = pipeline.mapped_body_link_pos_data[landmark_index]
            position_errors = np.linalg.norm(
                body_q[:, position_body, :3] - target[:, landmark_index, :3], axis=1
            )
            rotation_body, _ = pipeline.mapped_body_link_rot_data[landmark_index]
            actual = Rotation.from_quat(body_q[:, rotation_body, 3:7])
            desired = Rotation.from_quat(target[:, landmark_index, 3:7])
            orientation_errors = np.rad2deg((actual * desired.inv()).magnitude())
            motion_report[landmark_name] = {
                "position_p50_meters": float(np.percentile(position_errors, 50)),
                "position_p95_meters": float(np.percentile(position_errors, 95)),
                "orientation_p50_degrees": float(np.percentile(orientation_errors, 50)),
                "orientation_p95_degrees": float(np.percentile(orientation_errors, 95)),
            }
            landmark_pool = pooled.setdefault(landmark_name, {})
            landmark_pool.setdefault("position", []).append(position_errors)
            landmark_pool.setdefault("orientation", []).append(orientation_errors)
        per_motion.append(motion_report)

    aggregate = {}
    for landmark_name, components in pooled.items():
        metrics = {}
        if "position" in components:
            values = np.concatenate(components["position"])
            metrics["position_p50_meters"] = float(np.percentile(values, 50))
            metrics["position_p95_meters"] = float(np.percentile(values, 95))
        if "orientation" in components:
            values = np.concatenate(components["orientation"])
            metrics["orientation_p50_degrees"] = float(np.percentile(values, 50))
            metrics["orientation_p95_degrees"] = float(np.percentile(values, 95))
        aggregate[landmark_name] = metrics
    return {"aggregate": aggregate, "per_motion": per_motion}


def mujoco_model_metrics(motions: list[np.ndarray], mjcf_path: str | Path) -> dict:
    """Measure self-collision and joint-limit proximity in the reference MJCF."""

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo metrics require the evaluation extra: uv sync --extra evaluation"
        ) from error

    model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).resolve()))
    if model.nq != 32:
        raise ValueError(f"Bello evaluation MJCF has nq={model.nq}; expected 32.")
    data = mujoco.MjData(model)
    collision_pairs = Counter()
    collision_frames = 0
    deepest_distance = 0.0
    minimum_margin = np.inf
    near_limit_samples = 0
    limited_joint_samples = 0
    margin = np.deg2rad(1.0)

    per_motion = []
    for motion in motions:
        motion_collision_pairs = Counter()
        motion_collision_frames = 0
        motion_deepest_distance = 0.0
        motion_minimum_margin = np.inf
        motion_near_limit_samples = 0
        motion_limited_joint_samples = 0
        qpos = np.asarray(motion, dtype=np.float64).copy()
        qpos[:, 3:7] = qpos[:, [6, 3, 4, 5]]
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            frame_collides = False
            for contact in data.contact[: data.ncon]:
                if contact.dist >= -1.0e-7:
                    continue
                frame_collides = True
                deepest_distance = min(deepest_distance, float(contact.dist))
                motion_deepest_distance = min(
                    motion_deepest_distance, float(contact.dist)
                )
                pair = tuple(
                    sorted(
                        (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
                    )
                )
                collision_pairs[pair] += 1
                motion_collision_pairs[pair] += 1
            collision_frames += int(frame_collides)
            motion_collision_frames += int(frame_collides)

        for joint_id in range(model.njnt):
            if (
                model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
                or not model.jnt_limited[joint_id]
            ):
                continue
            address = int(model.jnt_qposadr[joint_id])
            lower, upper = model.jnt_range[joint_id]
            values = qpos[:, address]
            distances = np.minimum(values - lower, upper - values)
            minimum_margin = min(minimum_margin, float(np.min(distances)))
            motion_minimum_margin = min(motion_minimum_margin, float(np.min(distances)))
            near_limit_samples += int(np.sum(distances <= margin))
            limited_joint_samples += len(values)
            motion_near_limit_samples += int(np.sum(distances <= margin))
            motion_limited_joint_samples += len(values)

        per_motion.append(
            {
                "maximum_self_collision_penetration_meters": float(
                    -motion_deepest_distance
                ),
                "self_collision_frame_fraction": float(
                    motion_collision_frames / len(motion)
                ),
                "minimum_joint_limit_margin_degrees": float(
                    np.rad2deg(max(0.0, motion_minimum_margin))
                ),
                "near_joint_limit_sample_fraction": float(
                    motion_near_limit_samples / motion_limited_joint_samples
                ),
                "most_common_self_collision_pairs": [
                    {"geoms": list(pair), "samples": samples}
                    for pair, samples in motion_collision_pairs.most_common(5)
                ],
            }
        )

    total_frames = sum(len(motion) for motion in motions)
    return {
        "maximum_self_collision_penetration_meters": float(-deepest_distance),
        "self_collision_frame_fraction": float(collision_frames / total_frames),
        "minimum_joint_limit_margin_degrees": float(
            np.rad2deg(max(0.0, minimum_margin))
        ),
        "near_joint_limit_sample_fraction": float(
            near_limit_samples / limited_joint_samples
        ),
        "most_common_self_collision_pairs": [
            {"geoms": list(pair), "samples": samples}
            for pair, samples in collision_pairs.most_common(10)
        ],
        "per_motion": per_motion,
    }
