# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retarget AMASS SMPL-X motions through SOMA-X and the Newton pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import warp as wp

from soma_retargeter.assets import csv as csv_utils
from soma_retargeter.assets.soma_x import (
    SOMAXAMASSConverter,
    load_amass_metadata,
    resolve_smplx_model,
)
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", nargs="+", type=Path, help="AMASS SMPL-X .npz files")
    parser.add_argument("--body-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--soma-assets",
        type=Path,
        default=None,
        help="SOMA-X assets directory; downloaded and cached when omitted",
    )
    parser.add_argument("--robot-model-path", type=Path, default=None)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--soma-batch-size", type=int, default=32)
    parser.add_argument(
        "--export-gmr-pickle",
        action="store_true",
        help="also write a renderer-compatible Bello pickle",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot_model_path = args.robot_model_path
    if robot_model_path is None:
        configured_path = os.environ.get("BELLO_MJCF_PATH")
        robot_model_path = Path(configured_path) if configured_path else None

    csv_config = csv_utils.get_csv_config("bello")
    converters: dict[str, SOMAXAMASSConverter] = {}
    with wp.ScopedDevice(args.device):
        for motion_path in args.motions:
            metadata = load_amass_metadata(motion_path)
            if metadata.gender not in converters:
                model_path = resolve_smplx_model(args.body_model_dir, metadata.gender)
                converters[metadata.gender] = SOMAXAMASSConverter(
                    model_path,
                    gender=metadata.gender,
                    assets_dir=args.soma_assets,
                    device=args.device,
                    batch_size=args.soma_batch_size,
                )

            print(f"[INFO]: Converting [{motion_path}] with py-soma-x==0.2.1")
            skeleton, animation, metrics = converters[metadata.gender].convert(
                motion_path,
                target_fps=args.target_fps,
                start_seconds=args.start_seconds,
                duration_seconds=args.max_seconds,
            )
            print(
                "[INFO]: SOMA-X mean/max vertex error: "
                f"{metrics.mean_vertex_error_meters:.4f}/"
                f"{metrics.maximum_vertex_error_meters:.4f} m"
            )
            pipeline = NewtonPipeline(
                skeleton,
                source_type="soma",
                robot_type="bello",
                robot_model_path=str(robot_model_path) if robot_model_path else None,
            )
            pipeline.add_input_motions([animation], [wp.transform_identity()], True)
            outputs = pipeline.execute()
            if len(outputs) != 1:
                raise RuntimeError(
                    f"Expected one retargeted motion; received {len(outputs)}"
                )

            output_path = args.output_dir / f"{motion_path.stem}.csv"
            csv_utils.save_csv(output_path, outputs[0], csv_config=csv_config)
            if args.export_gmr_pickle:
                csv_utils.save_gmr_pickle(output_path.with_suffix(".pkl"), outputs[0])
            print(f"[INFO]: Wrote [{output_path}]")


if __name__ == "__main__":
    main()
