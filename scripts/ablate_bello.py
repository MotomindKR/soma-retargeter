#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ablate Bello's vanilla Newton IK configuration on bundled SOMA motions."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import warp as wp

from soma_retargeter.animation.animation_buffer import AnimationBuffer
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
    parser.add_argument("--profiles", type=Path)
    parser.add_argument(
        "--preset",
        choices=("baseline", "broad", "confirmation"),
        default="broad",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=240,
        help="Frames retained from each clip; use 0 for complete motions.",
    )
    return parser.parse_args()


def symmetric_weights(
    landmark: str,
    *,
    translation: float | None = None,
    rotation: float | None = None,
) -> dict:
    values = {}
    if translation is not None:
        values["translation"] = translation
    if rotation is not None:
        values["rotation"] = rotation
    return {f"{side}{landmark}": values for side in ("Left", "Right")}


def broad_profiles() -> list[dict]:
    """Small, interpretable probes around the checked-in single-stage map."""

    return [
        {"name": "baseline"},
        {"name": "iterations_16", "ik_iterations": 16},
        {"name": "iterations_32", "ik_iterations": 32},
        {"name": "smoothing_3", "smooth_joint_filter_weight": 3.0},
        {"name": "smoothing_8", "smooth_joint_filter_weight": 8.0},
        {
            "name": "position_only_wrists",
            "task_weights": symmetric_weights("Hand", rotation=0.0),
        },
        {
            "name": "light_wrist_orientation",
            "task_weights": symmetric_weights("Hand", rotation=0.1),
        },
        {
            "name": "stronger_elbow_position",
            "task_weights": symmetric_weights("ForeArm", translation=8.0),
        },
        {
            "name": "forearm_rotation_0.1",
            "task_weights": symmetric_weights("ForeArm", rotation=0.1),
        },
        {
            "name": "initial_forearm_rotation_0.25",
            "task_weights": symmetric_weights("ForeArm", rotation=0.25),
        },
        {
            "name": "lighter_arm_positions",
            "task_weights": {
                **symmetric_weights("ForeArm", translation=2.0),
                **symmetric_weights("Hand", translation=6.0),
            },
        },
        {
            "name": "stronger_arm_positions",
            "task_weights": {
                **symmetric_weights("ForeArm", translation=8.0),
                **symmetric_weights("Hand", translation=16.0),
            },
        },
        {
            "name": "stronger_feet",
            "task_weights": symmetric_weights(
                "Foot", translation=45.0, rotation=3.0
            ),
        },
    ]


def confirmation_profiles() -> list[dict]:
    """Cross-check conservative arm refinements at full solver quality."""

    return [
        {"name": "baseline"},
        {
            "name": "initial_forearm_rotation_0.25",
            "task_weights": symmetric_weights("ForeArm", rotation=0.25),
        },
        {
            "name": "wrist_r0",
            "task_weights": symmetric_weights("Hand", rotation=0.0),
        },
        {
            "name": "wrist_r0.1",
            "task_weights": symmetric_weights("Hand", rotation=0.1),
        },
        {
            "name": "wrist_r0.3",
            "task_weights": symmetric_weights("Hand", rotation=0.3),
        },
        {
            "name": "elbow_t8",
            "task_weights": symmetric_weights("ForeArm", translation=8.0),
        },
        {
            "name": "hand_t16",
            "task_weights": symmetric_weights("Hand", translation=16.0),
        },
        {
            "name": "upper_arm_r0.3",
            "task_weights": symmetric_weights("Arm", rotation=0.3),
        },
    ]


def merge_profile(base: dict, profile: dict) -> dict:
    config = deepcopy(base)
    for key, value in profile.items():
        if key == "name":
            continue
        if key == "task_weights":
            for landmark_name, components in value.items():
                try:
                    mapping = config["ik_map"][landmark_name]
                except KeyError:
                    raise ValueError(
                        f"Unknown landmark in task weights: {landmark_name}"
                    ) from None
                if "translation" in components:
                    mapping["t_weight"] = float(components["translation"])
                if "rotation" in components:
                    mapping["r_weight"] = float(components["rotation"])
            continue
        config[key] = deepcopy(value)
    return config


def load_suite(directory: Path, max_frames: int):
    paths = sorted(directory.glob("*.bvh"))
    if not paths:
        raise ValueError(f"No BVH files found in {directory}.")
    skeleton = None
    motions = []
    for path in paths:
        loaded_skeleton, motion = bvh_utils.load_bvh(path, skeleton)
        if skeleton is None:
            skeleton = loaded_skeleton
        retained_frames = min(max_frames, motion.num_frames) if max_frames else motion.num_frames
        motions.append(
            AnimationBuffer(
                skeleton,
                retained_frames,
                motion.sample_rate,
                np.copy(motion.local_transforms[:retained_frames]),
            )
        )
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
    groups = {
        "physical": (
            "collision_penetration_meters",
            "collision_frame_fraction",
            "worst_motion_collision_frame_fraction",
            "near_joint_limit_sample_fraction",
        ),
        "tracking": (
            "wrist_position_p95_meters",
            "wrist_orientation_p95_degrees",
            "foot_position_p95_meters",
            "foot_orientation_p95_degrees",
            "knee_position_p95_meters",
        ),
        "motion": ("jitter_rms_degrees", "velocity_p99_degrees_per_second"),
    }
    metric_names = tuple(name for names in groups.values() for name in names)
    values = np.asarray(
        [[result["selection_metrics"][name] for name in metric_names] for result in results]
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
    for index, result in enumerate(results):
        result["selection"] = {
            "group_scores": {
                name: float(group_scores[index, group_index])
                for group_index, name in enumerate(groups)
            },
            "utopia_distance": float(np.sqrt(np.mean(group_scores[index] ** 2))),
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
    motions = [np.stack(buffer.data).astype(np.float32) for buffer in buffers]
    removed_frames = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    targets = [
        np.asarray(pipeline.input_targets[index][removed_frames : removed_frames + len(motion)])
        for index, motion in enumerate(motions)
    ]
    metrics = {
        "tracking": task_tracking_metrics(motions, targets, pipeline),
        "jitter": [
            joint_jitter_metrics(motion, buffer.sample_rate)
            for motion, buffer in zip(motions, buffers, strict=True)
        ],
        "model": mujoco_model_metrics(motions, mjcf_path),
    }
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
    presets = {
        "baseline": lambda: [{"name": "baseline"}],
        "broad": broad_profiles,
        "confirmation": confirmation_profiles,
    }
    profiles = presets[args.preset]()
    if args.profiles:
        profiles = json.loads(args.profiles.read_text(encoding="utf-8"))

    results = []
    for profile in profiles:
        print(f"[ABLATION] Running {profile['name']} ({len(motions)} motions)")
        results.append(
            run_profile(paths, skeleton, motions, base_config, profile, mjcf_path)
        )
        print(
            f"[ABLATION] {profile['name']}: "
            f"elapsed={results[-1]['elapsed_seconds']:.2f}s"
        )
        gc.collect()
    rank_profiles(results)
    results.sort(key=lambda item: item["selection"]["utopia_distance"])
    report = {
        "mjcf_path": str(mjcf_path),
        "motion_directory": str(args.motion_directory.resolve()),
        "maximum_frames_per_motion": args.max_frames or None,
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
