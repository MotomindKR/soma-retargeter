# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA-X adapters for AMASS SMPL-X motion files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import Skeleton

_EXPECTED_SMPLX_POSE_WIDTH = 165
# HeadEnd-to-toe stature of the uniform SOMA rig shipped with this retargeter.
# Identity-backed SOMA-X rigs are normalized to this scale before retargeting.
REFERENCE_SOMA_STATURE_METERS = 1.7668
_CANONICAL_SOMA_POSE = (
    Path(__file__).resolve().parents[1] / "configs" / "soma" / "soma_zero_frame0.bvh"
)
_SOMA_X_TO_Z_UP = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()


@dataclass(frozen=True)
class AMASSMetadata:
    gender: str
    frame_count: int
    frame_rate: float


@dataclass(frozen=True)
class SOMAXConversionMetrics:
    mean_vertex_error_meters: float
    maximum_vertex_error_meters: float
    identity_stature_meters: float
    applied_identity_scale: float
    source_frame_rate: float
    output_frame_rate: float


def _scalar_string(value: np.ndarray, name: str) -> str:
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f"AMASS field [{name}] must be scalar; received {value.shape}")
    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return str(scalar).strip().lower()


def _frame_rate(motion: Any) -> float:
    for name in ("mocap_frame_rate", "mocap_framerate"):
        if name in motion:
            frame_rate = float(np.asarray(motion[name]).item())
            if np.isfinite(frame_rate) and frame_rate > 0.0:
                return frame_rate
            raise ValueError(
                f"AMASS frame rate must be positive; received {frame_rate}"
            )
    raise ValueError("AMASS motion is missing mocap_frame_rate")


def load_amass_metadata(motion_path: str | Path) -> AMASSMetadata:
    motion_path = Path(motion_path).expanduser()
    if not motion_path.is_file():
        raise FileNotFoundError(f"AMASS motion not found: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as motion:
        if "surface_model_type" in motion:
            model_type = _scalar_string(
                motion["surface_model_type"], "surface_model_type"
            )
            if model_type != "smplx":
                raise ValueError(
                    f"AMASS surface model must be [smplx]; received [{model_type}]"
                )
        gender = _scalar_string(motion["gender"], "gender")
        if gender not in {"female", "male", "neutral"}:
            raise ValueError(f"Unsupported AMASS gender [{gender}]")
        poses = np.asarray(motion["poses"])
        if poses.ndim != 2 or poses.shape[1] < _EXPECTED_SMPLX_POSE_WIDTH:
            raise ValueError(
                "AMASS SMPL-X poses must have shape "
                f"(frames, {_EXPECTED_SMPLX_POSE_WIDTH}+); received {poses.shape}"
            )
        return AMASSMetadata(gender, len(poses), _frame_rate(motion))


def resolve_smplx_model(body_model_dir: str | Path, gender: str) -> Path:
    root = Path(body_model_dir).expanduser()
    filename = f"SMPLX_{gender.upper()}.npz"
    candidates = (root / filename, root / "smplx" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"SMPL-X body model [{filename}] not found in: {searched}")


def _sample_times(
    frame_count: int,
    source_fps: float,
    target_fps: float,
    start_seconds: float,
    duration_seconds: float | None,
) -> np.ndarray:
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"Target frame rate must be positive; received {target_fps}")
    if not np.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError(f"Start time must be non-negative; received {start_seconds}")
    if duration_seconds is not None and (
        not np.isfinite(duration_seconds) or duration_seconds <= 0.0
    ):
        raise ValueError(f"Duration must be positive; received {duration_seconds}")

    source_end = (frame_count - 1) / source_fps
    if start_seconds > source_end:
        raise ValueError(
            f"Start time {start_seconds:.3f}s exceeds motion duration {source_end:.3f}s"
        )
    target_end = source_end
    if duration_seconds is not None:
        target_end = min(target_end, start_seconds + duration_seconds)
    count = int(np.floor((target_end - start_seconds) * target_fps + 1.0e-8)) + 1
    return start_seconds + np.arange(count, dtype=np.float64) / target_fps


def _resample_rotvecs(
    values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape((len(source_times), -1, 3))
    if len(source_times) == 1:
        return np.broadcast_to(flat[0], (len(target_times), *flat.shape[1:])).copy()
    output = np.empty((len(target_times), flat.shape[1], 3), dtype=np.float32)
    for joint_index in range(flat.shape[1]):
        output[:, joint_index] = Slerp(
            source_times,
            Rotation.from_rotvec(flat[:, joint_index]),
        )(target_times).as_rotvec()
    return output


def _load_resampled_motion(
    motion_path: Path,
    *,
    target_fps: float | None,
    start_seconds: float,
    duration_seconds: float | None,
) -> tuple[AMASSMetadata, np.ndarray, np.ndarray, np.ndarray, float]:
    metadata = load_amass_metadata(motion_path)
    output_fps = metadata.frame_rate if target_fps is None else target_fps
    with np.load(motion_path, allow_pickle=False) as motion:
        poses = np.asarray(
            motion["poses"][:, :_EXPECTED_SMPLX_POSE_WIDTH], dtype=np.float64
        )
        translations = np.asarray(motion["trans"], dtype=np.float64)
        betas = np.asarray(motion["betas"], dtype=np.float32).reshape(-1)
    if translations.shape != (metadata.frame_count, 3):
        raise ValueError(f"Invalid AMASS trans shape: {translations.shape}")
    if not all(np.all(np.isfinite(value)) for value in (poses, translations, betas)):
        raise ValueError("AMASS motion contains non-finite values")

    source_times = (
        np.arange(metadata.frame_count, dtype=np.float64) / metadata.frame_rate
    )
    target_times = _sample_times(
        metadata.frame_count,
        metadata.frame_rate,
        output_fps,
        start_seconds,
        duration_seconds,
    )
    poses = _resample_rotvecs(poses, source_times, target_times)
    translations = np.stack(
        [
            np.interp(target_times, source_times, translations[:, axis])
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.float32)
    return metadata, poses, translations, betas, float(output_fps)


def _identity_stature_meters(rig: Any) -> float:
    joint_indices = {name: index for index, name in enumerate(rig.joint_names)}
    required = ("HeadEnd", "LeftToeEnd", "RightToeEnd")
    missing = [name for name in required if name not in joint_indices]
    if missing:
        raise ValueError(f"SOMA-X public rig is missing stature joints: {missing}")
    positions = rig.bind_transforms_world[0, :, :3, 3].detach().cpu().numpy()
    head = positions[joint_indices["HeadEnd"]]
    toes = 0.5 * (
        positions[joint_indices["LeftToeEnd"]] + positions[joint_indices["RightToeEnd"]]
    )
    delta = np.abs(head - toes)
    stature = float(np.max(delta))
    if not np.isfinite(stature) or stature <= 0.0:
        raise ValueError(f"Invalid SOMA-X identity stature [{stature}]")
    return stature


def _global_matrices_to_local_transforms(
    global_matrices: np.ndarray, parent_indices: np.ndarray
) -> np.ndarray:
    local_matrices = np.empty_like(global_matrices)
    for joint_index, parent_index in enumerate(parent_indices):
        if parent_index == -1:
            local_matrices[:, joint_index] = global_matrices[:, joint_index]
        else:
            local_matrices[:, joint_index] = (
                np.linalg.inv(global_matrices[:, parent_index])
                @ global_matrices[:, joint_index]
            )
    local_transforms = np.empty((*local_matrices.shape[:2], 7), dtype=np.float32)
    local_transforms[:, :, :3] = local_matrices[:, :, :3, 3]
    local_transforms[:, :, 3:7] = (
        Rotation.from_matrix(local_matrices[:, :, :3, :3].reshape(-1, 3, 3))
        .as_quat()
        .reshape(*local_matrices.shape[:2], 4)
    )
    return local_transforms


def _align_soma_x_global_frames(
    global_matrices: np.ndarray,
    source_bind_rotations: np.ndarray,
    canonical_bind_rotations: np.ndarray,
) -> np.ndarray:
    """Align Y-up SOMA-X bind axes with canonical Z-up SOMA joint axes.

    AMASS pose transforms already carry the motion's Z-up world orientation, so
    only the SOMA-X bind reference is changed into the canonical basis. Joint
    positions are deliberately left untouched.
    """

    global_matrices = np.asarray(global_matrices)
    source_bind_rotations = np.asarray(source_bind_rotations)
    canonical_bind_rotations = np.asarray(canonical_bind_rotations)
    if global_matrices.ndim != 4 or global_matrices.shape[2:] != (4, 4):
        raise ValueError(
            "global_matrices must have shape (frames, joints, 4, 4); "
            f"received {global_matrices.shape}"
        )
    joint_count = global_matrices.shape[1]
    expected_rotation_shape = (joint_count, 3, 3)
    if source_bind_rotations.shape != expected_rotation_shape:
        raise ValueError(
            "source bind rotations must have shape "
            f"{expected_rotation_shape}; received {source_bind_rotations.shape}"
        )
    if canonical_bind_rotations.shape != expected_rotation_shape:
        raise ValueError(
            "canonical bind rotations must have shape "
            f"{expected_rotation_shape}; received {canonical_bind_rotations.shape}"
        )

    source_bind_rotations_z_up = _SOMA_X_TO_Z_UP[None] @ source_bind_rotations
    corrections = (
        np.swapaxes(source_bind_rotations_z_up, -1, -2)
        @ canonical_bind_rotations
    )
    aligned = np.array(global_matrices, copy=True)
    aligned[:, :, :3, :3] = global_matrices[:, :, :3, :3] @ corrections[None]
    return aligned


def _canonical_soma_global_rotations(joint_names: list[str]) -> np.ndarray:
    """Return the repository's canonical Z-up global frame for each SOMA joint."""

    from soma_retargeter.assets.bvh import load_bvh

    skeleton, animation = load_bvh(_CANONICAL_SOMA_POSE)
    missing = [name for name in joint_names if skeleton.joint_index(name) < 0]
    if missing:
        raise ValueError(f"Canonical SOMA pose is missing joints: {missing}")

    local = Rotation.from_quat(animation.local_transforms[0, :, 3:7])
    global_rotations: list[Rotation] = []
    source_to_z_up = Rotation.from_matrix(_SOMA_X_TO_Z_UP)
    for joint_index, parent_index in enumerate(skeleton.parent_indices):
        if parent_index == -1:
            global_rotations.append(source_to_z_up * local[joint_index])
        else:
            global_rotations.append(
                global_rotations[parent_index] * local[joint_index]
            )
    return np.stack(
        [global_rotations[skeleton.joint_index(name)].as_matrix() for name in joint_names]
    )


class SOMAXAMASSConverter:
    """Convert AMASS SMPL-X motion into a native SOMA animation buffer."""

    def __init__(
        self,
        body_model_path: str | Path,
        *,
        gender: str,
        assets_dir: str | Path | None = None,
        device: str = "cuda:0",
        batch_size: int = 32,
    ) -> None:
        try:
            import torch
            from soma import SMPLXLayer, SOMALayer, get_assets_dir
            from soma.pose_inversion import PoseInversion
        except ImportError as error:
            raise ImportError(
                "AMASS conversion requires py-soma-x==0.2.1; "
                "install it with `uv sync --extra amass`"
            ) from error

        try:
            from soma import __version__ as soma_x_version
        except ImportError as error:
            raise ImportError(
                "Unable to determine the installed py-soma-x version"
            ) from error
        if soma_x_version != "0.2.1":
            raise RuntimeError(
                f"AMASS conversion requires py-soma-x==0.2.1; found {soma_x_version}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive; received {batch_size}")

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device [{device}] requested but CUDA is unavailable"
            )
        self.batch_size = batch_size
        self.body_model_path = Path(body_model_path).expanduser()
        if not self.body_model_path.is_file():
            raise FileNotFoundError(
                f"SMPL-X body model not found: {self.body_model_path}"
            )
        self.gender = gender.strip().lower()
        if self.gender not in {"female", "male", "neutral"}:
            raise ValueError(f"Unsupported SMPL-X gender [{gender}]")
        self.assets_dir = (
            Path(assets_dir).expanduser() if assets_dir else get_assets_dir()
        )

        common_args = {
            "device": self.device,
            "mode": "warp",
        }
        self.source_layer = SMPLXLayer(
            self.assets_dir,
            gender=self.gender,
            model_path=self.body_model_path,
            num_betas=16,
            **common_args,
        )
        self.soma_layer = SOMALayer(
            self.assets_dir,
            identity_model_type="smplx",
            identity_model_kwargs={
                "model_path": self.body_model_path,
                "gender": self.gender,
                "num_betas": 16,
            },
            low_lod=True,
            enable_procedural_transforms=False,
            correctives_model_path=None,
            **common_args,
        )
        self.inverter = PoseInversion(self.soma_layer, low_lod=True)

    def convert(
        self,
        motion_path: str | Path,
        *,
        target_fps: float | None = None,
        start_seconds: float = 0.0,
        duration_seconds: float | None = None,
        normalize_stature: bool = True,
    ) -> tuple[Skeleton, AnimationBuffer, SOMAXConversionMetrics]:
        metadata, poses, translations, betas, output_fps = _load_resampled_motion(
            Path(motion_path).expanduser(),
            target_fps=target_fps,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        if metadata.gender != self.gender:
            raise ValueError(
                f"Converter gender is [{self.gender}], motion gender is [{metadata.gender}]"
            )

        torch = self.torch
        beta_count = min(len(betas), 16)
        identity = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        identity[:, :beta_count] = torch.from_numpy(betas[:beta_count]).to(self.device)
        self.source_layer.prepare_identity(identity)
        self.inverter.prepare_identity(identity)
        rig = self.soma_layer.public_rig_view()
        identity_stature = _identity_stature_meters(rig)
        identity_scale = (
            REFERENCE_SOMA_STATURE_METERS / identity_stature
            if normalize_stature
            else 1.0
        )

        transform_chunks = []
        error_chunks = []
        for start in range(0, len(poses), self.batch_size):
            stop = min(start + self.batch_size, len(poses))
            pose_chunk = torch.from_numpy(poses[start:stop]).to(self.device)
            translation_chunk = torch.from_numpy(translations[start:stop]).to(
                self.device
            )
            with torch.inference_mode():
                source_vertices = self.source_layer.pose(
                    pose_chunk,
                    global_translation=translation_chunk,
                )["vertices"]
            result = self.inverter.fit(
                source_vertices,
                body_iters=2,
                finger_iters=0,
                full_iters=1,
                lie_iters=3,
                lie_lambda=1.0e-1,
            )
            with torch.inference_mode():
                posed = self.soma_layer.pose(
                    result.rotations[:, 1:],
                    pose2rot=False,
                    transl=result.root_translation,
                    absolute_pose=True,
                    apply_correctives=False,
                    fk_only=True,
                )
            transform_chunks.append(posed.transforms.detach().cpu().numpy())
            error_chunks.append(result.per_vertex_error.detach().cpu().numpy())
            del pose_chunk, translation_chunk, source_vertices, result, posed

        global_matrices = np.concatenate(transform_chunks, axis=0)
        joint_names = list(rig.joint_names)
        source_bind_matrices = rig.bind_transforms_world[0].detach().cpu().numpy()
        global_matrices = _align_soma_x_global_frames(
            global_matrices,
            source_bind_matrices[:, :3, :3],
            _canonical_soma_global_rotations(joint_names),
        )
        global_matrices[:, :, :3, 3] *= identity_scale
        errors = np.concatenate(error_chunks, axis=0)
        parent_indices = rig.joint_parent_ids.detach().cpu().numpy().astype(np.int32)
        parent_indices[0] = -1
        local_transforms = _global_matrices_to_local_transforms(
            global_matrices, parent_indices
        )
        skeleton = Skeleton(
            len(joint_names),
            joint_names,
            parent_indices,
            local_transforms[0],
        )
        animation = AnimationBuffer(
            skeleton,
            len(local_transforms),
            output_fps,
            local_transforms,
        )
        metrics = SOMAXConversionMetrics(
            mean_vertex_error_meters=float(np.mean(errors)),
            maximum_vertex_error_meters=float(np.max(errors)),
            identity_stature_meters=identity_stature,
            applied_identity_scale=identity_scale,
            source_frame_rate=metadata.frame_rate,
            output_frame_rate=output_fps,
        )
        return skeleton, animation, metrics
