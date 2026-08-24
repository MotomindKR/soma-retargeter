# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render canonical Bello CSV motion through Newton's actual OpenGL viewer."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import newton
import numpy as np
import warp as wp

from soma_retargeter.assets import csv as csv_utils
from soma_retargeter.robotics.robot_model import build_robot_builder


def actor_side_camera(
    joint_q: np.ndarray,
    *,
    distance: float,
    height_offset: float,
    opposite_side: bool = False,
) -> tuple[np.ndarray, float]:
    """Return camera position and yaw in Bello's root-relative side frame."""
    joint_q = np.asarray(joint_q, dtype=np.float64)
    if joint_q.ndim != 1 or len(joint_q) < 7:
        raise ValueError("joint_q must contain root position and xyzw quaternion")
    if not np.all(np.isfinite(joint_q[:7])):
        raise ValueError("root position and quaternion must be finite")
    if distance <= 0.0:
        raise ValueError("camera distance must be positive")

    quaternion = joint_q[3:7]
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-8:
        raise ValueError("root quaternion must be nonzero")
    x, y, z, w = quaternion / norm
    root_x_axis = np.asarray(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )
    planar_axis = root_x_axis[:2]
    planar_norm = np.linalg.norm(planar_axis)
    if planar_norm < 1.0e-8:
        raise ValueError("root x-axis cannot be vertical for a side camera")
    planar_axis /= planar_norm
    if opposite_side:
        planar_axis *= -1.0

    position = np.asarray(joint_q[:3], dtype=np.float64).copy()
    position[:2] -= planar_axis * distance
    position[2] += height_offset
    yaw_degrees = float(np.rad2deg(np.arctan2(planar_axis[1], planar_axis[0])))
    return position, yaw_degrees


def render_times(
    source_frame_count: int,
    source_fps: float,
    render_fps: float,
    *,
    start_seconds: float = 0.0,
    maximum_seconds: float | None = None,
) -> np.ndarray:
    """Return inclusive output timestamps bounded by the source trajectory."""
    if source_frame_count < 1:
        raise ValueError("source motion must contain at least one frame")
    if source_fps <= 0.0 or render_fps <= 0.0:
        raise ValueError("source and render frame rates must be positive")
    duration = (source_frame_count - 1) / source_fps
    if start_seconds < 0.0 or start_seconds > duration:
        raise ValueError("start time must lie within the source motion")
    end_seconds = duration
    if maximum_seconds is not None:
        if maximum_seconds <= 0.0:
            raise ValueError("maximum duration must be positive")
        end_seconds = min(end_seconds, start_seconds + maximum_seconds)
    count = int(np.floor((end_seconds - start_seconds) * render_fps + 1.0e-9)) + 1
    return start_seconds + np.arange(count, dtype=np.float64) / render_fps


def _open_video_writer(
    ffmpeg: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def render_bello_video(
    motion_path: Path,
    output_path: Path,
    *,
    robot_model_path: Path | None = None,
    source_fps: float = 120.0,
    render_fps: float = 30.0,
    width: int = 768,
    height: int = 720,
    camera_distance: float = 3.6,
    camera_height_offset: float = 0.3,
    camera_pitch: float = -3.0,
    opposite_side: bool = False,
    start_seconds: float = 0.0,
    maximum_seconds: float | None = None,
) -> None:
    """Render one canonical Bello CSV without translating through another engine."""
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("video dimensions must be positive even integers")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("video output must use the .mp4 extension")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode Newton viewer frames")

    motion = csv_utils.load_csv(
        str(motion_path), fps=source_fps, csv_config=csv_utils.get_csv_config("bello")
    )
    times = render_times(
        motion.num_frames,
        motion.sample_rate,
        render_fps,
        start_seconds=start_seconds,
        maximum_seconds=maximum_seconds,
    )
    robot_builder = build_robot_builder("bello", robot_model_path)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_builder(robot_builder, wp.transform_identity())
    model = builder.finalize()
    state = model.state()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.rendering{output_path.suffix}"
    )
    temporary_output.unlink(missing_ok=True)
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    writer = None
    try:
        viewer.set_model(model)
        writer = _open_video_writer(
            ffmpeg, temporary_output, width=width, height=height, fps=render_fps
        )
        assert writer.stdin is not None
        for timestamp in times:
            joint_q = np.asarray(motion.sample(float(timestamp)), dtype=np.float32)
            if joint_q.shape != (model.joint_coord_count,):
                raise ValueError(
                    f"Bello motion has {len(joint_q)} coordinates; "
                    f"model expects {model.joint_coord_count}"
                )
            wp.copy(model.joint_q, wp.array(joint_q, dtype=wp.float32))
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            camera_position, camera_yaw = actor_side_camera(
                joint_q,
                distance=camera_distance,
                height_offset=camera_height_offset,
                opposite_side=opposite_side,
            )
            viewer.set_camera(
                wp.vec3(*camera_position), pitch=camera_pitch, yaw=camera_yaw
            )
            viewer.begin_frame(float(timestamp))
            viewer.log_state(state)
            viewer.end_frame()
            writer.stdin.write(viewer.get_frame().numpy().tobytes())
        writer.stdin.close()
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        temporary_output.replace(output_path)
    finally:
        viewer.close()
        if writer is not None and writer.poll() is None:
            if writer.stdin is not None and not writer.stdin.closed:
                try:
                    writer.stdin.close()
                except BrokenPipeError:
                    pass
            writer.terminate()
            writer.wait()
        temporary_output.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="canonical Bello CSV")
    parser.add_argument("video_output", type=Path)
    parser.add_argument("--robot-model-path", type=Path, default=None)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.6)
    parser.add_argument("--camera-height-offset", type=float, default=0.3)
    parser.add_argument("--camera-pitch", type=float, default=-3.0)
    parser.add_argument("--opposite-side", action="store_true")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with wp.ScopedDevice(args.device):
        render_bello_video(
            args.motion,
            args.video_output,
            robot_model_path=args.robot_model_path,
            source_fps=args.source_fps,
            render_fps=args.render_fps,
            width=args.width,
            height=args.height,
            camera_distance=args.camera_distance,
            camera_height_offset=args.camera_height_offset,
            camera_pitch=args.camera_pitch,
            opposite_side=args.opposite_side,
            start_seconds=args.start_seconds,
            maximum_seconds=args.max_seconds,
        )
    print(args.video_output.resolve())


if __name__ == "__main__":
    main()
