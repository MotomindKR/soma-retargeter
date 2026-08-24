#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ablate Bello Newton solver settings over the bundled SOMA motion suite."""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import warp as wp

from soma_retargeter.assets import bvh as bvh_utils
from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.pipelines.retarget_quality import (
    joint_jitter_metrics,
    mujoco_model_metrics,
    task_tracking_metrics,
)
from soma_retargeter.robotics.robot_model import resolve_robot_mjcf
from soma_retargeter.utils.space_conversion_utils import (
    FacingDirectionType,
    SpaceConverter,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRE_TUNING_MAX_JOINT_VELOCITY = 3.0 * np.pi


def wrist_task_weights(translation: float, rotation: float) -> dict:
    return {
        "final_tracking": {
            "LeftHand": {
                "translation": translation,
                "rotation": [rotation, rotation, rotation],
            },
            "RightHand": {
                "translation": translation,
                "rotation": [rotation, rotation, rotation],
            },
        }
    }


def pre_tuning_profile(name: str, **overrides) -> dict:
    """Pin the old defaults so historical presets remain reproducible."""

    offline_solver = {"max_joint_velocity": PRE_TUNING_MAX_JOINT_VELOCITY}
    offline_solver.update(overrides.pop("offline_solver", {}))
    profile = {
        "name": name,
        "ik_iterations": 10,
        "offline_solver": offline_solver,
        "task_weights": wrist_task_weights(50.0, 30.0),
    }
    profile.update(overrides)
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-directory",
        type=Path,
        default=REPOSITORY_ROOT / "assets/motions/bvh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/bello_ablation.json",
    )
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=("baseline", "broad", "focused", "interaction", "confirmation"),
        default="broad",
        help="Built-in candidate set to run when --profiles is not supplied.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def default_profiles() -> list[dict]:
    profiles = [pre_tuning_profile("baseline")]
    profiles.extend(
        pre_tuning_profile(f"iterations_{value}", ik_iterations=value)
        for value in (4, 6, 8, 12, 16)
    )
    profiles.extend(
        pre_tuning_profile(f"joint_limit_{value:g}", joint_limit_weight=value)
        for value in (2.5, 5.0, 20.0, 40.0)
    )
    profiles.extend(
        pre_tuning_profile(
            f"smoothing_{passes}",
            offline_solver={"joint_smoothing_passes": passes},
        )
        for passes in (0, 1, 2, 4)
    )
    profiles.extend(
        pre_tuning_profile(
            f"iterations_{iterations}_passes_2",
            ik_iterations=iterations,
            offline_solver={"passes_per_frame": 2},
        )
        for iterations in (6, 10)
    )
    profiles.extend(
        pre_tuning_profile(
            f"final_hands_{scale:g}",
            task_scales={
                "final_tracking": {
                    "LeftHand": {"translation": scale, "rotation": scale},
                    "RightHand": {"translation": scale, "rotation": scale},
                }
            },
        )
        for scale in (0.5, 0.75, 1.25, 1.5)
    )
    profiles.extend(
        pre_tuning_profile(
            f"final_elbows_{scale:g}",
            task_scales={
                "final_tracking": {
                    "LeftForeArm": {"translation": scale},
                    "RightForeArm": {"translation": scale},
                }
            },
        )
        for scale in (0.5, 1.5)
    )
    profiles.extend(
        pre_tuning_profile(
            f"branch_arms_{scale:g}",
            task_scales={
                "branch_selection": {
                    "LeftArm": {"rotation": scale},
                    "RightArm": {"rotation": scale},
                }
            },
        )
        for scale in (0.5, 1.5)
    )
    profiles.extend(
        pre_tuning_profile(
            f"final_feet_{scale:g}",
            task_scales={
                "final_tracking": {
                    "LeftFoot": {"translation": scale, "rotation": scale},
                    "RightFoot": {"translation": scale, "rotation": scale},
                }
            },
        )
        for scale in (0.5, 0.75, 1.25)
    )
    return profiles


def focused_profiles() -> list[dict]:
    """Probe wrist translation/orientation independently and solver warm-up."""

    profiles = [pre_tuning_profile("baseline")]
    profiles.extend(
        pre_tuning_profile(
            f"wrist_rotation_{scale:g}",
            task_scales={
                "final_tracking": {
                    "LeftHand": {"rotation": scale},
                    "RightHand": {"rotation": scale},
                }
            },
        )
        for scale in (0.0, 0.1, 0.25, 0.5, 0.75)
    )
    profiles.extend(
        pre_tuning_profile(
            f"wrist_translation_{translation:g}_rotation_{rotation:g}",
            task_scales={
                "final_tracking": {
                    "LeftHand": {
                        "translation": translation,
                        "rotation": rotation,
                    },
                    "RightHand": {
                        "translation": translation,
                        "rotation": rotation,
                    },
                }
            },
        )
        for translation, rotation in (
            (0.5, 0.1),
            (0.75, 0.1),
            (1.25, 0.1),
            (0.5, 0.25),
            (0.75, 0.25),
            (1.25, 0.25),
        )
    )
    profiles.extend(
        (
            pre_tuning_profile(
                "initialization_20_stabilization_10",
                num_initialization_frames=20,
                num_stabilization_frames=10,
            ),
            pre_tuning_profile(
                "initialization_30_stabilization_15",
                num_initialization_frames=30,
                num_stabilization_frames=15,
            ),
            pre_tuning_profile(
                "initial_settle_40",
                offline_solver={"initial_settle_passes": 40},
            ),
        )
    )
    return profiles


def interaction_profiles() -> list[dict]:
    """Refine the strongest focused candidate against solver regularization."""

    def wrist_profile(name: str, **overrides) -> dict:
        translation = overrides.pop("wrist_translation", 25.0)
        rotation = overrides.pop("wrist_rotation", 3.0)
        profile = pre_tuning_profile(name, **overrides)
        profile["task_weights"] = wrist_task_weights(translation, rotation)
        return profile

    profiles = [
        pre_tuning_profile("baseline"),
        wrist_profile("wrist_t0.5_r0.1"),
    ]
    for translation, rotation in (
        (0.4, 0.1),
        (0.6, 0.1),
        (0.5, 0.05),
        (0.5, 0.15),
    ):
        profile = wrist_profile(
            f"wrist_t{translation:g}_r{rotation:g}",
            wrist_translation=50.0 * translation,
            wrist_rotation=30.0 * rotation,
        )
        profiles.append(profile)
    profiles.extend(
        wrist_profile(f"candidate_iterations_{value}", ik_iterations=value)
        for value in (6, 8, 12)
    )
    profiles.extend(
        wrist_profile(f"candidate_joint_limit_{value:g}", joint_limit_weight=value)
        for value in (5.0, 15.0, 20.0)
    )
    profiles.extend(
        wrist_profile(
            f"candidate_smoothing_{passes}",
            offline_solver={"joint_smoothing_passes": passes},
        )
        for passes in (2, 4, 5)
    )
    profiles.extend(
        wrist_profile(
            f"candidate_velocity_{value:g}",
            offline_solver={"max_joint_velocity": value},
        )
        for value in (6.0, 7.5)
    )
    profiles.extend(
        (
            wrist_profile(
                "candidate_joint_limit_15_smoothing_4",
                joint_limit_weight=15.0,
                offline_solver={"joint_smoothing_passes": 4},
            ),
            wrist_profile(
                "candidate_iterations_8_joint_limit_15_smoothing_4",
                ik_iterations=8,
                joint_limit_weight=15.0,
                offline_solver={"joint_smoothing_passes": 4},
            ),
        )
    )
    return profiles


def confirmation_profiles() -> list[dict]:
    """Confirm the velocity-cap result and promising cross-parameter effects."""

    def candidate(name: str, **overrides) -> dict:
        profile = pre_tuning_profile(name, **overrides)
        profile["task_weights"] = wrist_task_weights(25.0, 3.0)
        return profile

    return [
        pre_tuning_profile("baseline"),
        candidate(
            "candidate_velocity_7",
            offline_solver={"max_joint_velocity": 7.0},
        ),
        candidate(
            "candidate_velocity_7.5",
            offline_solver={"max_joint_velocity": 7.5},
        ),
        candidate(
            "candidate_velocity_8",
            offline_solver={"max_joint_velocity": 8.0},
        ),
        candidate(
            "candidate_velocity_8.5",
            offline_solver={"max_joint_velocity": 8.5},
        ),
        candidate(
            "candidate_iterations_12_velocity_7",
            ik_iterations=12,
            offline_solver={"max_joint_velocity": 7.0},
        ),
        candidate(
            "candidate_iterations_12_velocity_7.5",
            ik_iterations=12,
            offline_solver={"max_joint_velocity": 7.5},
        ),
        candidate(
            "candidate_iterations_12_velocity_8",
            ik_iterations=12,
            offline_solver={"max_joint_velocity": 8.0},
        ),
        candidate(
            "candidate_joint_limit_15_smoothing_4_velocity_7.5",
            joint_limit_weight=15.0,
            offline_solver={
                "joint_smoothing_passes": 4,
                "max_joint_velocity": 7.5,
            },
        ),
    ]


def scale_task_weights(config: dict, rules: dict) -> None:
    """Apply named task multipliers without obscuring the resulting configuration."""

    stages = {stage["name"]: stage for stage in config["ik_stages"]}
    for stage_name, landmarks in rules.items():
        if stage_name not in stages:
            raise ValueError(f"Unknown IK stage in task scales: {stage_name}")
        mappings = stages[stage_name]["ik_map"]
        for landmark_name, components in landmarks.items():
            if landmark_name not in mappings:
                raise ValueError(
                    f"Unknown landmark [{landmark_name}] in IK stage [{stage_name}]."
                )
            mapping = mappings[landmark_name]
            if "translation" in components:
                mapping["t_weight"] *= float(components["translation"])
            if "rotation" in components:
                scale = float(components["rotation"])
                if isinstance(mapping["r_weight"], list):
                    mapping["r_weight"] = [
                        scale * value for value in mapping["r_weight"]
                    ]
                else:
                    mapping["r_weight"] *= scale


def set_task_weights(config: dict, rules: dict) -> None:
    """Set absolute task weights for reproducible historical profiles."""

    stages = {stage["name"]: stage for stage in config["ik_stages"]}
    for stage_name, landmarks in rules.items():
        if stage_name not in stages:
            raise ValueError(f"Unknown IK stage in task weights: {stage_name}")
        mappings = stages[stage_name]["ik_map"]
        for landmark_name, components in landmarks.items():
            if landmark_name not in mappings:
                raise ValueError(
                    f"Unknown landmark [{landmark_name}] in IK stage [{stage_name}]."
                )
            mapping = mappings[landmark_name]
            if "translation" in components:
                mapping["t_weight"] = float(components["translation"])
            if "rotation" in components:
                value = components["rotation"]
                mapping["r_weight"] = (
                    [float(axis) for axis in value]
                    if isinstance(value, list)
                    else float(value)
                )


def merge_profile(base: dict, profile: dict) -> dict:
    config = deepcopy(base)
    for key, value in profile.items():
        if key == "name":
            continue
        if key == "task_scales":
            scale_task_weights(config, value)
            continue
        if key == "task_weights":
            set_task_weights(config, value)
            continue
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def load_suite(directory: Path, max_frames: int | None):
    paths = sorted(directory.glob("*.bvh"))
    if not paths:
        raise ValueError(f"No BVH files found in {directory}.")
    skeleton = None
    motions = []
    for path in paths:
        loaded_skeleton, motion = bvh_utils.load_bvh(path, skeleton)
        if skeleton is None:
            skeleton = loaded_skeleton
        if max_frames is not None and motion.num_frames > max_frames:
            from soma_retargeter.animation.animation_buffer import AnimationBuffer

            motion = AnimationBuffer(
                skeleton,
                max_frames,
                motion.sample_rate,
                np.copy(motion.local_transforms[:max_frames]),
            )
        motions.append(motion)
    return paths, skeleton, motions


def maximum_metric(tracking: list[dict], names: tuple[str, ...], metric: str) -> float:
    values = [
        motion[name][metric]
        for motion in tracking
        for name in names
        if name in motion and metric in motion[name]
    ]
    return max(values) if values else 0.0


def selection_metrics(metrics: dict) -> dict[str, float]:
    tracking = metrics["tracking"]["per_motion"]
    model_per_motion = metrics["model"]["per_motion"]
    return {
        "jitter_rms_degrees": max(item["rms_degrees"] for item in metrics["jitter"]),
        "velocity_p99_degrees_per_second": max(
            item["velocity_p99_degrees_per_second"] for item in metrics["jitter"]
        ),
        "collision_penetration_meters": metrics["model"][
            "maximum_self_collision_penetration_meters"
        ],
        "collision_frame_fraction": metrics["model"]["self_collision_frame_fraction"],
        "worst_motion_collision_frame_fraction": max(
            item["self_collision_frame_fraction"] for item in model_per_motion
        ),
        "near_joint_limit_sample_fraction": metrics["model"][
            "near_joint_limit_sample_fraction"
        ],
        "worst_motion_near_joint_limit_sample_fraction": max(
            item["near_joint_limit_sample_fraction"] for item in model_per_motion
        ),
        "wrist_position_p95_meters": maximum_metric(
            tracking, ("LeftHand", "RightHand"), "position_p95_meters"
        ),
        "wrist_orientation_p95_degrees": maximum_metric(
            tracking, ("LeftHand", "RightHand"), "orientation_p95_degrees"
        ),
        "foot_orientation_p95_degrees": maximum_metric(
            tracking, ("LeftFoot", "RightFoot"), "orientation_p95_degrees"
        ),
        "foot_position_p95_meters": maximum_metric(
            tracking, ("LeftFoot", "RightFoot"), "position_p95_meters"
        ),
        "knee_position_p95_meters": maximum_metric(
            tracking, ("LeftShin", "RightShin"), "position_p95_meters"
        ),
    }


def rank_profiles(results: list[dict]) -> None:
    """Rank candidates against the observed population and mark its Pareto front."""

    groups = {
        "physical": (
            "collision_penetration_meters",
            "collision_frame_fraction",
            "worst_motion_collision_frame_fraction",
            "near_joint_limit_sample_fraction",
            "worst_motion_near_joint_limit_sample_fraction",
        ),
        "tracking": (
            "wrist_position_p95_meters",
            "wrist_orientation_p95_degrees",
            "foot_position_p95_meters",
            "foot_orientation_p95_degrees",
            "knee_position_p95_meters",
        ),
        "motion": (
            "jitter_rms_degrees",
            "velocity_p99_degrees_per_second",
        ),
    }
    metric_names = tuple(name for names in groups.values() for name in names)
    values = np.asarray(
        [
            [result["selection_metrics"][name] for name in metric_names]
            for result in results
        ],
        dtype=np.float64,
    )
    minima = np.min(values, axis=0)
    spans = np.max(values, axis=0) - minima
    normalized = np.divide(
        values - minima,
        spans,
        out=np.zeros_like(values),
        where=spans > 1.0e-12,
    )
    group_indices = {
        group: [metric_names.index(name) for name in names]
        for group, names in groups.items()
    }
    group_scores = np.stack(
        [np.mean(normalized[:, indices], axis=1) for indices in group_indices.values()],
        axis=1,
    )
    distances = np.sqrt(np.mean(group_scores**2, axis=1))
    for index, result in enumerate(results):
        dominated = any(
            other != index
            and np.all(group_scores[other] <= group_scores[index] + 1.0e-12)
            and np.any(group_scores[other] < group_scores[index] - 1.0e-12)
            for other in range(len(results))
        )
        result["selection"] = {
            "data_normalized_metrics": {
                name: float(normalized[index, metric_index])
                for metric_index, name in enumerate(metric_names)
            },
            "group_scores": {
                group: float(group_scores[index, group_index])
                for group_index, group in enumerate(groups)
            },
            "utopia_distance": float(distances[index]),
            "pareto_optimal": not dominated,
        }


def run_profile(paths, skeleton, source_motions, base_config, profile, mjcf_path):
    config = merge_profile(base_config, profile)
    pipeline = NewtonPipeline(
        skeleton,
        robot_type="bello",
        retarget_config=config,
        robot_model_path=str(mjcf_path),
    )
    source_to_mujoco = SpaceConverter(FacingDirectionType.MUJOCO).transform(
        wp.transform_identity()
    )
    start = time.perf_counter()
    pipeline.add_input_motions(
        source_motions,
        [source_to_mujoco] * len(source_motions),
        True,
    )
    buffers = pipeline.execute()
    elapsed = time.perf_counter() - start
    motions = [np.asarray(buffer.data, dtype=np.float32) for buffer in buffers]
    removed_frames = (
        pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    )
    targets = [
        np.asarray(
            pipeline.input_targets[index][removed_frames : removed_frames + len(motion)]
        )
        for index, motion in enumerate(motions)
    ]
    tracking = task_tracking_metrics(motions, targets, pipeline)
    jitter = [
        joint_jitter_metrics(motion, buffer.sample_rate)
        for motion, buffer in zip(motions, buffers, strict=True)
    ]
    model = mujoco_model_metrics(motions, mjcf_path)
    metrics = {"tracking": tracking, "jitter": jitter, "model": model}
    return {
        "name": profile["name"],
        "overrides": {key: value for key, value in profile.items() if key != "name"},
        "selection_metrics": selection_metrics(metrics),
        "elapsed_seconds": elapsed,
        "motion_names": [path.name for path in paths],
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    mjcf_path = resolve_robot_mjcf("bello", os.environ.get("BELLO_MJCF_PATH"))
    paths, skeleton, motions = load_suite(args.motion_directory, args.max_frames)
    base_config = pipeline_utils.get_retargeter_config(
        pipeline_utils.SourceType.SOMA,
        pipeline_utils.TargetType.BELLO,
    )
    built_in_profiles = {
        "baseline": lambda: [{"name": "baseline"}],
        "broad": default_profiles,
        "focused": focused_profiles,
        "interaction": interaction_profiles,
        "confirmation": confirmation_profiles,
    }
    profiles = built_in_profiles[args.preset]()
    if args.profiles is not None:
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))

    results = []
    for profile in profiles:
        print(f"[ABLATION] Running {profile['name']} ({len(motions)} motions)")
        result = run_profile(paths, skeleton, motions, base_config, profile, mjcf_path)
        results.append(result)
        print(f"[ABLATION] {profile['name']}: elapsed={result['elapsed_seconds']:.2f}s")
    rank_profiles(results)
    results.sort(key=lambda item: item["selection"]["utopia_distance"])
    report = {
        "mjcf_path": str(mjcf_path),
        "motion_directory": str(args.motion_directory.resolve()),
        "ranking": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ABLATION] Best observed balance: {results[0]['name']} "
        f"(distance={results[0]['selection']['utopia_distance']:.4f})"
    )
    print(f"[ABLATION] Wrote {args.output}")


if __name__ == "__main__":
    main()
