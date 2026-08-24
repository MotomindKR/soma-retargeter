# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import newton
import numpy as np
import warp as wp
from newton import ik
from scipy.spatial.transform import Rotation
from tqdm import trange

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer
from soma_retargeter.pipelines.ik_objectives import (
    IKJointReferenceObjective,
    IKObjectiveRotationAxisWeighted,
    IKSmoothJointFilter,
)
from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.robot_model import (
    box_shape_support_points,
    build_robot_builder,
    minimum_support_height,
)
from soma_retargeter.utils import io_utils, newton_utils

_DEFAULT_IK_SOLVER_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT = 5.5
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5


def contact_foot_weights(
    positions: np.ndarray,
    sample_rate: float,
    *,
    floor_percentile: float,
    maximum_height: float,
    maximum_speed: float | None,
    transition_seconds: float,
) -> np.ndarray:
    """Return smoothly blended contact weights for one foot trajectory."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Foot positions must have shape (frames, 3)")
    if len(positions) == 0:
        raise ValueError("Foot positions must contain at least one frame")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Foot positions must be finite")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("Contact leveling sample rate must be positive")
    if not 0.0 <= floor_percentile <= 100.0:
        raise ValueError("Contact floor percentile must be in [0, 100]")
    if maximum_height < 0.0 or transition_seconds < 0.0:
        raise ValueError("Contact leveling thresholds must be non-negative")
    if maximum_speed is not None and maximum_speed < 0.0:
        raise ValueError("Contact leveling thresholds must be non-negative")

    if len(positions) > 1:
        speeds = np.linalg.norm(
            np.gradient(positions, 1.0 / sample_rate, axis=0), axis=1
        )
    else:
        speeds = np.zeros(1, dtype=np.float64)
    floor_height = float(np.percentile(positions[:, 2], floor_percentile))
    planted = positions[:, 2] <= floor_height + maximum_height
    if maximum_speed is not None:
        planted &= speeds <= maximum_speed
    weights = planted.astype(np.float64)

    transition_radius = round(transition_seconds * sample_rate)
    if transition_radius > 0:
        blend_kernel = np.hanning(2 * transition_radius + 3)[1:-1]
        blend_kernel /= np.sum(blend_kernel)
        padded = np.pad(weights, (transition_radius, transition_radius), mode="edge")
        weights = np.convolve(padded, blend_kernel, mode="valid")
    return np.clip(weights, 0.0, 1.0)


def level_sole_quaternions(
    link_quaternions: np.ndarray,
    sole_local_quaternion: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Flatten sole roll/pitch while preserving link heading and blend weight."""
    link_quaternions = np.asarray(link_quaternions)
    weights = np.asarray(weights, dtype=np.float64)
    if link_quaternions.ndim != 2 or link_quaternions.shape[1] != 4:
        raise ValueError("Link quaternions must have shape (frames, 4)")
    if weights.shape != (len(link_quaternions),):
        raise ValueError("Sole leveling weights must have shape (frames,)")
    if not np.all(np.isfinite(weights)) or np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("Sole leveling weights must be finite and in [0, 1]")

    link_rotation = Rotation.from_quat(link_quaternions)
    sole_local = Rotation.from_quat(np.asarray(sole_local_quaternion))
    sole_rotation = link_rotation * sole_local
    sole_forward = sole_rotation.apply(np.asarray([0.0, 1.0, 0.0]))
    horizontal_forward = sole_forward.copy()
    horizontal_forward[:, 2] = 0.0
    horizontal_norm = np.linalg.norm(horizontal_forward, axis=1)
    valid = horizontal_norm > 1.0e-8
    effective_weights = np.array(weights, copy=True)
    effective_weights[~valid] = 0.0
    horizontal_forward[valid] /= horizontal_norm[valid, None]
    horizontal_forward[~valid] = np.asarray([0.0, 1.0, 0.0])
    up = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), horizontal_forward.shape)
    right = np.cross(horizontal_forward, up)
    flat_sole = Rotation.from_matrix(np.stack((right, horizontal_forward, up), axis=-1))
    flat_link = flat_sole * sole_local.inv()
    correction = flat_link * link_rotation.inv()
    return (
        Rotation.from_rotvec(correction.as_rotvec() * effective_weights[:, None])
        * link_rotation
    ).as_quat()


def level_contact_foot_targets(
    targets: np.ndarray,
    sample_rate: float,
    foot_frames: list[tuple[int, np.ndarray]],
    *,
    floor_percentile: float,
    maximum_height: float,
    maximum_speed: float | None,
    transition_seconds: float,
) -> np.ndarray:
    """Level robot sole targets only while their source landmarks are planted."""
    targets = np.asarray(targets)
    if targets.ndim != 3 or targets.shape[2] != 7:
        raise ValueError("IK targets must have shape (frames, effectors, 7)")

    output = np.array(targets, copy=True)
    for target_index, sole_local_quaternion in foot_frames:
        if target_index < 0 or target_index >= targets.shape[1]:
            raise ValueError(f"Contact foot target index is invalid: {target_index}")
        weights = contact_foot_weights(
            targets[:, target_index, :3],
            sample_rate,
            floor_percentile=floor_percentile,
            maximum_height=maximum_height,
            maximum_speed=maximum_speed,
            transition_seconds=transition_seconds,
        )
        if np.any(weights > 1.0e-8):
            output[:, target_index, 3:7] = level_sole_quaternions(
                targets[:, target_index, 3:7], sole_local_quaternion, weights
            )
    return output


class NewtonPipeline:
    """
    Newton-based motion retargeting pipeline.

    This pipeline retargets human motion captured on a common skeleton
    to a target robot using inverse kinematics (IK),
    custom objectives, and optional post-processing filters such as
    joint limit clamping and feet stabilization.
    """
    def __init__(
        self,
        skeleton: Skeleton,
        source_type='soma',
        robot_type='unitree_g1',
        retarget_config: dict | None = None,
        robot_model_path: str | None = None,
    ):
        """
        Initialize the Newton retargeting pipeline.

        Args:
            skeleton: Common skeleton definition used by the input clips to be retargeted.
            source_type: Source skeleton type name. Currently only "soma" is supported.
            robot_type: Registered target robot type name.
            retarget_config: Optional configuration dictionary. If None, a
                configuration is loaded from disk based on the source/target
                types.

        Raises:
            ValueError: If the target robot type is not supported.
        """
        self.source_type = pipeline_utils.get_source_type_from_str(source_type)
        self.target_type = pipeline_utils.get_target_type_from_str(robot_type)
        self.input_targets = []
        self.input_sample_rates = []
        self.input_contact_weights = []
        self.max_frames = -1

        if retarget_config is None:
            retargeter_config = pipeline_utils.get_retargeter_config(self.source_type, self.target_type)
        else:
            retargeter_config = retarget_config

        self.ik_iterations = retargeter_config.get('ik_iterations', _DEFAULT_IK_SOLVER_ITERATIONS)
        self.joint_limit_weight = retargeter_config.get('joint_limit_weight', _DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT)
        self.smooth_joint_filter_weight = retargeter_config.get('smooth_joint_filter_weight', _DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT)
        self.post_processing_enabled = retargeter_config.get('enable_post_processing', True)
        self.enable_self_penetration = False
        self.smooth_joint_filter_coord_masks = None
        self.joint_limit_clamper = None
        self.robot_type = robot_type
        self.robot_model_path = robot_model_path or retargeter_config.get('robot_model_path')
        self.robot_builder = build_robot_builder(
            robot_type,
            self.robot_model_path,
            enable_self_collisions=self.enable_self_penetration,
        )

        self.human_robot_scaler = HumanToRobotScaler(
            skeleton,
            retargeter_config['model_height'],
            io_utils.get_config_file(retargeter_config['human_robot_scaler_config']),
        )
        self.num_body_count = self.robot_builder.body_count
        self.num_dofs = self.robot_builder.joint_dof_count
        self.ik_model = self._build_model(1)

        ik_stage_configs = retargeter_config.get('ik_stages')
        if ik_stage_configs is None:
            ik_stage_configs = [{'name': 'default', 'ik_map': retargeter_config['ik_map']}]
        if not ik_stage_configs:
            raise ValueError("Retargeter configuration must define at least one IK stage.")
        self.ik_stage_configs = ik_stage_configs
        self.ik_stage_mappings = [
            self._build_target_mapping(self.human_robot_scaler.skeleton, stage['ik_map'])
            for stage in self.ik_stage_configs
        ]
        self.mapped_joints = self.ik_stage_mappings[0]['mapped_joints']
        for stage_mapping in self.ik_stage_mappings[1:]:
            if stage_mapping['mapped_joints'] != self.mapped_joints:
                raise ValueError("Every IK stage must map the same SOMA joints in the same order.")

        smooth_body_masks = retargeter_config.get('smooth_joint_filter_objective_body_masks')
        if smooth_body_masks is not None:
            self.smooth_joint_filter_coord_masks = newton_utils.create_joint_coord_masks(
                self.ik_model, smooth_body_masks, 0.0)

        effector_names = self.human_robot_scaler.effector_names()
        self.target_effector_indices = [effector_names.index(name) for name in self.mapped_joints]
        self.joint_limit_clamper = JointLimitClamper(self.ik_model)
        self.contact_foot_leveling_config = retargeter_config.get('contact_foot_leveling')

        self.feet_stabilizer = None
        self.contact_feet_stabilizer = None
        self.feet_effector_indices = []
        feet_config = retargeter_config.get('feet_stabilizer_config')
        if self.post_processing_enabled and feet_config:
            self.feet_effector_indices = [
                self.mapped_joints.index("LeftFoot"),
                self.mapped_joints.index("RightFoot"),
            ]
            self.feet_stabilizer = FeetStabilizer(
                io_utils.get_config_file(feet_config),
                self.robot_model_path,
            )

        offline_config = retargeter_config.get('offline_solver', {})
        self.passes_per_frame = max(1, int(offline_config.get('passes_per_frame', 1)))
        self.initial_settle_passes = max(1, int(offline_config.get('initial_settle_passes', 1)))
        self.max_joint_velocity = offline_config.get('max_joint_velocity')
        if self.max_joint_velocity is not None and float(self.max_joint_velocity) <= 0.0:
            raise ValueError("offline_solver.max_joint_velocity must be positive")
        self.joint_smoothing_kernel = offline_config.get('joint_smoothing_kernel')
        self.root_orientation_smoothing_kernel = offline_config.get('root_orientation_smoothing_kernel')
        self.joint_smoothing_passes = max(0, int(offline_config.get('joint_smoothing_passes', 0)))
        self.planar_relative_yaw_config = retargeter_config.get('planar_relative_yaw_task')
        self.ground_clearance_config = retargeter_config.get('ground_clearance')
        self.ground_clearance_solves = self._build_ground_clearance_data()
        self.contact_foot_frames = self._build_contact_foot_frames()
        if self.feet_stabilizer is not None and self.contact_foot_frames:
            weight = float(self.contact_foot_leveling_config.get(
                'stabilizer_rotation_weight', 100.0))
            self.contact_feet_stabilizer = FeetStabilizer(
                io_utils.get_config_file(feet_config),
                self.robot_model_path,
                effector_rotation_weight_overrides={
                    'left_ankle_roll_link': weight,
                    'right_ankle_roll_link': weight,
                },
            )

        self.initialization_pose = None
        self.num_initialization_frames = 0
        self.num_stabilization_frames = 0
        initialization_pose = retargeter_config.get('initialization_pose')
        if initialization_pose:
            init_skel, init_anim = bvh_utils.load_bvh(io_utils.get_config_file(initialization_pose))
            self.initialization_pose = SkeletonInstance(init_skel, [0, 0, 0], wp.transform_identity())
            self.initialization_pose.set_local_transforms(init_anim.get_local_transforms(0))
            self.num_initialization_frames = retargeter_config.get(
                'num_initialization_frames', _DEFAULT_NUM_INITIALIZATION_FRAMES)
            self.num_stabilization_frames = retargeter_config.get(
                'num_stabilization_frames', _DEFAULT_NUM_STABILIZATION_FRAMES)

    def clear(self):
        """
        Clear all accumulated input motions and reset internal state.

        This removes all previously added motions set for retargeting.
        It does not modify static configuration such as the robot model or IK settings.
        """
        self.input_targets = []
        self.input_sample_rates = []
        self.input_contact_weights = []
        self.max_frames = -1

    def add_input_motions(self, buffers: list[AnimationBuffer], offsets: list[wp.transform], scale_animation: bool):
        """
        Add input motions to be retargeted.
        Each buffer is converted into IK targets using the human-to-robot scaler.

        Args:
            buffers: List of input animation buffers defined on the common skeleton.
            offsets: List of root transforms applied to each buffer. If the
                length does not match `buffers`, identity transforms are used
                for all.
            scale_animation: Whether to rescale the source motion using the
                configured HumanToRobotScaler.
        """
        offsets = offsets if len(offsets) == len(buffers) else [wp.transform_identity()] * len(buffers)
        for i in trange(len(buffers), desc="[INFO] Converting Motions for Newton"):
            buffer = buffers[i]
            source_frame_offset = 0
            if self.initialization_pose and self.num_initialization_frames > 0:
                buffer = newton_utils.create_buffer_with_initialization_frames(
                    self.initialization_pose, buffers[i], self.num_initialization_frames, self.num_stabilization_frames)
                source_frame_offset = buffer.num_frames - buffers[i].num_frames

            self.max_frames = max(self.max_frames, buffer.num_frames)
            buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(buffer, scale_animation, offsets[i])
            buffer_effectors = buffer_effectors[:, self.target_effector_indices, :]
            if self.contact_foot_frames:
                contact = self.contact_foot_leveling_config
                contact_kwargs = {
                    'floor_percentile': float(contact.get('floor_percentile', 5.0)),
                    'maximum_height': float(contact.get('maximum_height', 0.06)),
                    'maximum_speed': (
                        None if contact.get('maximum_speed') is None
                        else float(contact['maximum_speed'])
                    ),
                    'transition_seconds': float(contact.get('transition_seconds', 0.08)),
                }
                weights = np.stack([
                    contact_foot_weights(
                        buffer_effectors[source_frame_offset:, target_index, :3],
                        buffer.sample_rate,
                        **contact_kwargs,
                    )
                    for target_index, _, _ in self.contact_foot_frames
                ], axis=1)
                if source_frame_offset:
                    weights = np.concatenate((
                        np.broadcast_to(weights[0], (source_frame_offset, weights.shape[1])),
                        weights,
                    ))
                buffer_effectors = np.array(buffer_effectors, copy=True)
                for foot_index, (target_index, _, sole_quaternion) in enumerate(
                    self.contact_foot_frames
                ):
                    buffer_effectors[:, target_index, 3:7] = level_sole_quaternions(
                        buffer_effectors[:, target_index, 3:7],
                        sole_quaternion,
                        weights[:, foot_index],
                    )
                self.input_contact_weights.append(weights)

            self.input_targets.append(buffer_effectors)
            self.input_sample_rates.append(buffers[i].sample_rate)

    def execute(self):
        """
        Run the retargeting pipeline on all added input motions.

        This method builds a multi-environment Newton model, sets up IK
        objectives, and performs frame-by-frame IK solving.

        Returns:
            list[CSVAnimationBuffer]: A list of retargeted robot motions, one per input motion.
        """
        num_envs = len(self.input_targets)
        if num_envs == 0:
            return []

        # Clamp objective weights to valid values
        self.ik_iterations = max(1, self.ik_iterations)
        self.joint_limit_weight = max(0.0, self.joint_limit_weight)
        self.smooth_joint_filter_weight = max(0.0, self.smooth_joint_filter_weight)

        print("[INFO] Newton Retargeter Settings: ")
        print(f"[INFO]\t  Source Skeleton Type: {pipeline_utils.get_source_str_from_type(self.source_type)}")
        print(f"[INFO]\t  Target Robot Type: {pipeline_utils.get_target_str_from_type(self.target_type)}")
        print(f"[INFO]\t  Post-Processing Enabled: {self.post_processing_enabled}")
        print(f"[INFO]\t  Initialization Pose: {self.initialization_pose is not None}")
        print(f"[INFO]\t  Initialization Frame Count: {self.num_initialization_frames}")
        print(f"[INFO]\t  Constraint Stabilization Frame Count: {self.num_stabilization_frames}")
        print(f"[INFO]\t  IK Solver Iterations: {self.ik_iterations}")
        print(f"[INFO]\t  Joint Limit Objective Weight: {self.joint_limit_weight}")
        print(f"[INFO]\t  Smooth Joint Filter Objective Weight: {self.smooth_joint_filter_weight}")

        model = self._build_model(num_envs)
        state = model.state()

        if self.feet_stabilizer is not None:
            self.feet_stabilizer.setup_num_envs(num_envs)
            env_feet_tx = np.empty((num_envs, len(self.feet_effector_indices), 7), dtype=np.float32)
        if self.contact_feet_stabilizer is not None:
            self.contact_feet_stabilizer.setup_num_envs(num_envs)

        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ik_stages = [
            self._create_ik_stage(
                num_envs,
                model,
                state,
                mapping,
                is_final_stage=stage_index == len(self.ik_stage_mappings) - 1,
            )
            for stage_index, mapping in enumerate(self.ik_stage_mappings)
        ]

        joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
        wp.copy(joint_q, model.joint_q)

        for stage in ik_stages:
            stage['solver'].reset()

        graph_capture = None

        def single_step():
            for stage in ik_stages:
                stage['solver'].step(joint_q, joint_q, iterations=self.ik_iterations)

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                single_step()
            graph_capture = cap.graph
        else:
            single_step()
        wp.copy(joint_q, model.joint_q)

        num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames
        final_smooth_objective = ik_stages[-1]['smooth_objective']
        joint_q_data = [np.full((len(self.input_targets[i]),), None) for i in range(num_envs)]
        previous_outputs = [None] * num_envs
        for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions"):
            if final_smooth_objective is not None and num_frames_to_remove > 0 and frame <= num_frames_to_remove:
                final_smooth_objective.set_weight(
                    self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove)))

            active_envs = []
            for env in range(num_envs):
                if frame > (len(self.input_targets[env])-1):
                    continue
                active_envs.append(env)
                frame_targets = self.input_targets[env][frame]
                for stage in ik_stages:
                    for objective_index, target in enumerate(frame_targets):
                        stage['position_objectives'][objective_index].set_target_position(
                            env, wp.vec3(*target[0:3]))
                        stage['rotation_objectives'][objective_index].set_target_rotation(
                            env, wp.quat(*target[3:7]))
                yaw_objective = ik_stages[-1]['yaw_objective']
                if yaw_objective is not None:
                    yaw_objective.set_target(env, self._planar_relative_yaw_target(frame_targets))

            solve_passes = (
                self.initial_settle_passes
                if frame in (0, num_frames_to_remove)
                else self.passes_per_frame
            )
            for _ in range(solve_passes):
                if graph_capture is not None:
                    wp.capture_launch(graph_capture)
                else:
                    single_step()

            if self.feet_stabilizer is not None:
                self.feet_stabilizer.reset_state(joint_q)

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env])-1):
                        env_feet_tx[env] = np.asarray(self.input_targets[env][-1][self.feet_effector_indices])
                    else:
                        env_feet_tx[env] = np.asarray(self.input_targets[env][frame][self.feet_effector_indices])

                self.feet_stabilizer.solve(env_feet_tx)
                data = self.joint_limit_clamper.apply(self.feet_stabilizer.current_state()).numpy()
            else:
                data = self.joint_limit_clamper.apply(joint_q).numpy()

            self._apply_velocity_limits(data, previous_outputs, active_envs)
            wp.copy(joint_q, wp.array(data, dtype=wp.float32))
            self._enforce_ground_clearance(data, joint_q, model, state, active_envs)

            for env in active_envs:
                previous_outputs[env] = np.array(data[env], copy=True)
                joint_q_data[env][frame] = np.array(data[env], copy=True)

        smoothed_motions = []
        for env in range(num_envs):
            motion = np.stack(joint_q_data[env])
            motion = self._smooth_trajectory(motion, self.input_sample_rates[env])
            smoothed_motions.append(motion)
        self._post_stabilize_contact_feet(smoothed_motions, joint_q)
        self._enforce_ground_clearance_trajectories(
            smoothed_motions, joint_q, model, state)
        return [
            CSVAnimationBuffer.create_from_raw_data(
                motion[num_frames_to_remove:], self.input_sample_rates[env])
            for env, motion in enumerate(smoothed_motions)
        ]

    def _build_model(self, num_envs: int):
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self.robot_builder, xform=wp.transform_identity())

        builder.add_ground_plane()
        model = builder.finalize(requires_grad=True)

        return model

    def _build_target_mapping(self, skeleton, ik_map):
        mapped_joints = []
        mapped_body_link_pos_data = []
        mapped_body_link_rot_data = []
        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        for joint, mapping_data in ik_map.items():
            mapped_joints.append(joint)
            skeleton.joint_index(joint)
            mapped_body_link_pos_data.append((body_names.index(mapping_data['t_body']), mapping_data['t_weight']))
            mapped_body_link_rot_data.append((body_names.index(mapping_data['r_body']), mapping_data['r_weight']))

        return {
            'mapped_joints': mapped_joints,
            'position_data': mapped_body_link_pos_data,
            'rotation_data': mapped_body_link_rot_data,
        }

    def _create_ik_stage(self, num_envs, model, state, mapping, *, is_final_stage):

        # Gather default body position and rotation based on model state to initialize
        # position and rotation objectives
        position_data = mapping['position_data']
        rotation_data = mapping['rotation_data']
        num_body_link_pos = len(position_data)
        num_body_link_rot = len(rotation_data)
        pos_targets = np.zeros((num_envs, num_body_link_pos), dtype=wp.vec3)
        rot_targets = np.zeros((num_envs, num_body_link_rot), dtype=wp.quat)

        body_q = state.body_q.numpy()
        for env in range(num_envs):
            base = env * self.num_body_count
            for ee_idx, (link_idx, _) in enumerate(position_data):
                pos_targets[env, ee_idx] = body_q[base + link_idx][0:3]

            for ee_idx, (link_idx, _) in enumerate(rotation_data):
                rot_wp = wp.quat(body_q[base + link_idx][3:7])
                rot_targets[env, ee_idx] = wp.normalize(rot_wp)

        pos_num_ees = len(position_data)
        rot_num_ees = len(rotation_data)
        pos_target_arrays, rot_target_arrays = [], []
        for ee_idx in range(pos_num_ees):
            pos_wp = wp.array(pos_targets[:, ee_idx], dtype=wp.vec3)
            pos_target_arrays.append(pos_wp)

        for ee_idx in range(rot_num_ees):
            rot_wp = wp.array(rot_targets[:, ee_idx], dtype=wp.vec4)
            rot_target_arrays.append(rot_wp)

        position_objectives = []
        for i, (link_idx, w) in enumerate(position_data):
            objective = ik.IKObjectivePosition(
                link_index=link_idx,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=pos_target_arrays[i],
                weight=w)
            position_objectives.append(objective)

        rotation_objectives = []
        for i, (link_idx, weight) in enumerate(rotation_data):
            objective_args = {
                'link_index': link_idx,
                'link_offset_rotation': wp.quat_identity(),
                'target_rotations': rot_target_arrays[i],
            }
            if isinstance(weight, list):
                objective = IKObjectiveRotationAxisWeighted(
                    axis_weights=weight,
                    **objective_args,
                )
            else:
                objective = ik.IKObjectiveRotation(weight=weight, **objective_args)
            rotation_objectives.append(objective)

        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        active_objectives = [*position_objectives, *rotation_objectives]
        if self.joint_limit_weight > 0.0:
            active_objectives.append(joint_limit_objective)

        smooth_objective = None
        if is_final_stage and self.smooth_joint_filter_weight > 0.0:
            initial_smooth_weight = (
                0.0
                if self.num_initialization_frames + self.num_stabilization_frames > 0
                else self.smooth_joint_filter_weight
            )
            smooth_objective = IKSmoothJointFilter(
                joint_limit_lower=self.ik_model.joint_limit_lower,
                joint_limit_upper=self.ik_model.joint_limit_upper,
                weight=initial_smooth_weight,
                coord_masks=self.smooth_joint_filter_coord_masks,
            )
            active_objectives.append(smooth_objective)

        yaw_objective = None
        if is_final_stage and self.planar_relative_yaw_config is not None:
            yaw_objective = self._create_planar_relative_yaw_objective(num_envs)
            active_objectives.append(yaw_objective)

        solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=num_envs,
            objectives=active_objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        return {
            'position_objectives': position_objectives,
            'rotation_objectives': rotation_objectives,
            'smooth_objective': smooth_objective,
            'yaw_objective': yaw_objective,
            'solver': solver,
        }

    def _create_planar_relative_yaw_objective(self, num_envs):
        config = self.planar_relative_yaw_config
        joint_name = config['robot_joint_name']
        joint_names = [newton_utils.get_name_from_label(label) for label in self.ik_model.joint_label]
        try:
            joint_index = joint_names.index(joint_name)
        except ValueError:
            raise ValueError(f"Planar-yaw joint [{joint_name}] is missing from the robot model.") from None

        coord_index = int(self.ik_model.joint_q_start.numpy()[joint_index])
        dof_index = int(self.ik_model.joint_qd_start.numpy()[joint_index])
        self.planar_relative_yaw_limits = (
            float(self.ik_model.joint_limit_lower.numpy()[dof_index]),
            float(self.ik_model.joint_limit_upper.numpy()[dof_index]),
        )
        return IKJointReferenceObjective(
            coord_index=coord_index,
            dof_index=dof_index,
            targets=wp.zeros(num_envs, dtype=wp.float32),
            weight=float(config['orientation_cost']),
        )

    def _planar_relative_yaw_target(self, frame_targets):
        config = self.planar_relative_yaw_config
        frame_left, frame_right = (
            self.mapped_joints.index(name) for name in config['human_frame_landmarks'])
        root_left, root_right = (
            self.mapped_joints.index(name) for name in config['human_root_landmarks'])
        frame_axis = frame_targets[frame_right, :2] - frame_targets[frame_left, :2]
        root_axis = frame_targets[root_right, :2] - frame_targets[root_left, :2]
        frame_norm = np.linalg.norm(frame_axis)
        root_norm = np.linalg.norm(root_axis)
        if frame_norm < 1.0e-8 or root_norm < 1.0e-8:
            return 0.0
        frame_axis /= frame_norm
        root_axis /= root_norm
        cross_z = root_axis[0] * frame_axis[1] - root_axis[1] * frame_axis[0]
        angle = np.arctan2(cross_z, np.dot(root_axis, frame_axis))
        angle *= float(config.get('scale', 1.0))
        return float(np.clip(angle, *self.planar_relative_yaw_limits))

    def _apply_velocity_limits(self, data, previous_outputs, active_envs):
        if self.max_joint_velocity is None:
            return
        for env in active_envs:
            previous = previous_outputs[env]
            if previous is None:
                continue
            max_delta = float(self.max_joint_velocity) / float(self.input_sample_rates[env])
            data[env, 7:] = previous[7:] + np.clip(
                data[env, 7:] - previous[7:], -max_delta, max_delta)

    def _smooth_trajectory(self, motion, sample_rate):
        if self.joint_smoothing_passes == 0 or len(motion) < 2:
            return motion
        output = np.asarray(motion, dtype=np.float32).copy()
        joint_kernel = self._validated_smoothing_kernel(self.joint_smoothing_kernel)
        root_kernel = self._validated_smoothing_kernel(self.root_orientation_smoothing_kernel)

        for _ in range(self.joint_smoothing_passes):
            if joint_kernel is not None:
                output[:, 7:] = self._convolve_edge_padded(output[:, 7:], joint_kernel)
            if root_kernel is not None:
                output[:, 3:7] = self._smooth_quaternions(output[:, 3:7], root_kernel)

        if self.max_joint_velocity is not None:
            max_delta = float(self.max_joint_velocity) / float(sample_rate)
            for frame in range(1, len(output)):
                output[frame, 7:] = output[frame - 1, 7:] + np.clip(
                    output[frame, 7:] - output[frame - 1, 7:], -max_delta, max_delta)
        return output

    @staticmethod
    def _validated_smoothing_kernel(values):
        if values is None:
            return None
        kernel = np.asarray(values, dtype=np.float64)
        if kernel.ndim != 1 or len(kernel) % 2 != 1 or len(kernel) == 0:
            raise ValueError("Smoothing kernels must be non-empty, odd-length vectors.")
        if not np.all(np.isfinite(kernel)) or np.sum(kernel) <= 0.0:
            raise ValueError("Smoothing kernels must contain finite values with a positive sum.")
        return kernel / np.sum(kernel)

    @staticmethod
    def _convolve_edge_padded(values, kernel):
        radius = len(kernel) // 2
        padded = np.pad(values, ((radius, radius), (0, 0)), mode='edge')
        return np.stack([
            np.sum(padded[frame:frame + len(kernel)] * kernel[:, None], axis=0)
            for frame in range(len(values))
        ])

    @staticmethod
    def _smooth_quaternions(quaternions, kernel):
        radius = len(kernel) // 2
        padded = np.pad(quaternions, ((radius, radius), (0, 0)), mode='edge')
        result = np.empty_like(quaternions)
        for frame in range(len(quaternions)):
            window = padded[frame:frame + len(kernel)].copy()
            reference = quaternions[frame]
            window[np.sum(window * reference, axis=1) < 0.0] *= -1.0
            average = np.sum(window * kernel[:, None], axis=0)
            norm = np.linalg.norm(average)
            result[frame] = average / norm if norm > 1.0e-8 else reference
        return result

    def _build_ground_clearance_data(self):
        if self.ground_clearance_config is None:
            return []
        percentile = float(
            self.ground_clearance_config.get('reference_percentile', 50.0)
        )
        if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
            raise ValueError("ground_clearance.reference_percentile must be in [0, 100]")
        return box_shape_support_points(
            self.robot_builder,
            self.ground_clearance_config['sole_shapes'],
        )

    def _build_contact_foot_frames(self):
        if self.contact_foot_leveling_config is None:
            return []
        configured_shapes = self.contact_foot_leveling_config.get('sole_shapes', {})
        if set(configured_shapes) != {'LeftFoot', 'RightFoot'}:
            raise ValueError(
                "contact_foot_leveling.sole_shapes must map LeftFoot and RightFoot"
            )
        shape_names = [label.split('/')[-1] for label in self.robot_builder.shape_label]
        result = []
        for landmark in ('LeftFoot', 'RightFoot'):
            shape_name = configured_shapes[landmark]
            try:
                shape_index = shape_names.index(shape_name)
            except ValueError:
                raise ValueError(
                    f"Contact sole shape [{shape_name}] is missing from the robot model"
                ) from None
            if self.robot_builder.shape_type[shape_index] != newton.GeoType.BOX:
                raise ValueError(f"Contact sole shape [{shape_name}] must be a box")
            target_index = self.mapped_joints.index(landmark)
            shape_transform = np.asarray(
                self.robot_builder.shape_transform[shape_index], dtype=np.float64
            )
            result.append((
                target_index,
                int(self.robot_builder.shape_body[shape_index]),
                shape_transform[3:7],
            ))
        return result

    def _post_stabilize_contact_feet(self, motions, joint_q):
        if self.contact_feet_stabilizer is None:
            return
        passes = int(self.contact_foot_leveling_config.get('post_smoothing_passes', 0))
        if passes < 0:
            raise ValueError(
                "contact_foot_leveling.post_smoothing_passes must be non-negative"
            )
        if passes == 0:
            return

        num_envs = len(motions)
        work = np.stack([motion[0] for motion in motions])
        foot_targets = np.empty(
            (num_envs, len(self.feet_effector_indices), 7), dtype=np.float32
        )
        for frame in range(max(len(motion) for motion in motions)):
            active_envs = []
            for env, motion in enumerate(motions):
                if frame < len(motion):
                    work[env] = motion[frame]
                    active_envs.append(env)
            wp.copy(joint_q, wp.array(work, dtype=wp.float32))
            self.contact_feet_stabilizer.reset_state(joint_q)
            body_q = self.contact_feet_stabilizer.state.body_q.numpy().reshape(
                num_envs, self.contact_feet_stabilizer.num_body_count, 7
            )
            for env in range(num_envs):
                weight_frame = min(frame, len(self.input_contact_weights[env]) - 1)
                for foot_index, (_, body_index, sole_quaternion) in enumerate(
                    self.contact_foot_frames
                ):
                    current = body_q[env, body_index]
                    foot_targets[env, foot_index, :3] = current[:3]
                    foot_targets[env, foot_index, 3:7] = level_sole_quaternions(
                        current[None, 3:7],
                        sole_quaternion,
                        self.input_contact_weights[env][
                            weight_frame, foot_index:foot_index + 1
                        ],
                    )[0]
            for _ in range(passes):
                self.contact_feet_stabilizer.reset_state(joint_q)
                self.contact_feet_stabilizer.solve(foot_targets)
                work = self.joint_limit_clamper.apply(
                    self.contact_feet_stabilizer.current_state()
                ).numpy()
                wp.copy(joint_q, wp.array(work, dtype=wp.float32))
            for env in active_envs:
                motions[env][frame] = work[env]

    def _enforce_ground_clearance(self, data, joint_q, model, state, active_envs):
        if not self.ground_clearance_solves or not active_envs:
            return
        flat_joint_q = joint_q.reshape((joint_q.shape[0] * joint_q.shape[1],))
        newton.eval_fk(model, flat_joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        ground_height = float(self.ground_clearance_config.get('ground_height', 0.0))
        for env in active_envs:
            body_base = env * self.num_body_count
            minimum_height = minimum_support_height(
                body_q, self.ground_clearance_solves, body_base
            )
            if minimum_height < ground_height:
                data[env, 2] += ground_height - minimum_height
        wp.copy(joint_q, wp.array(data, dtype=wp.float32))

    def _enforce_ground_clearance_trajectories(self, motions, joint_q, model, state):
        if not self.ground_clearance_solves:
            return
        work = np.stack([motion[0] for motion in motions])
        minimum_heights = [np.empty(len(motion), dtype=np.float64) for motion in motions]
        for frame in range(max(len(motion) for motion in motions)):
            active_envs = []
            for env, motion in enumerate(motions):
                if frame < len(motion):
                    work[env] = motion[frame]
                    active_envs.append(env)
            wp.copy(joint_q, wp.array(work, dtype=wp.float32))
            flat_joint_q = joint_q.reshape((joint_q.shape[0] * joint_q.shape[1],))
            newton.eval_fk(model, flat_joint_q, model.joint_qd, state)
            body_q = state.body_q.numpy()
            for env in active_envs:
                body_base = env * self.num_body_count
                minimum_heights[env][frame] = minimum_support_height(
                    body_q, self.ground_clearance_solves, body_base
                )

        ground_height = float(self.ground_clearance_config.get('ground_height', 0.0))
        align_motion = bool(
            self.ground_clearance_config.get('align_motion_to_ground', False)
        )
        percentile = float(
            self.ground_clearance_config.get('reference_percentile', 50.0)
        )
        for env, motion in enumerate(motions):
            heights = minimum_heights[env]
            alignment = 0.0
            if align_motion:
                alignment = ground_height - float(np.percentile(heights, percentile))
            penetration_correction = ground_height - heights
            corrections = np.maximum(alignment, penetration_correction)
            motion[:, 2] += corrections.astype(motion.dtype, copy=False)
