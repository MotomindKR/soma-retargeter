# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from newton import ik
from newton._src.sim.ik.ik_common import IKJacobianType


@wp.kernel
def _axis_weighted_rotation_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_rot: wp.array1d(dtype=wp.vec4),
    link_index: int,
    link_offset_rotation: wp.quat,
    canonicalize_quat_err: wp.bool,
    start_idx: int,
    axis_weights: wp.vec3,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    base = problem_idx_map[row]
    body_tf = body_q[row, link_index]
    actual_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6]) * link_offset_rotation
    target_vec = target_rot[base]
    target_quat = wp.quat(target_vec[0], target_vec[1], target_vec[2], target_vec[3])
    q_err = actual_rot * wp.quat_inverse(target_quat)
    if canonicalize_quat_err and wp.dot(actual_rot, target_quat) < 0.0:
        q_err = -q_err

    vector_norm = wp.sqrt(q_err[0] * q_err[0] + q_err[1] * q_err[1] + q_err[2] * q_err[2])
    axis_angle = wp.vec3(2.0 * q_err[0], 2.0 * q_err[1], 2.0 * q_err[2])
    if vector_norm > 1.0e-8:
        axis_angle = wp.vec3(q_err[0], q_err[1], q_err[2]) * (
            2.0 * wp.atan2(vector_norm, q_err[3]) / vector_norm
        )

    residuals[row, start_idx] = axis_weights[0] * axis_angle[0]
    residuals[row, start_idx + 1] = axis_weights[1] * axis_angle[1]
    residuals[row, start_idx + 2] = axis_weights[2] * axis_angle[2]


@wp.kernel
def _axis_weighted_rotation_jacobian(
    affects_dof: wp.array1d(dtype=wp.uint8),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    start_idx: int,
    axis_weights: wp.vec3,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, dof_idx = wp.tid()
    if affects_dof[dof_idx] == 0:
        return
    motion = joint_S_s[problem_idx, dof_idx]
    jacobian[problem_idx, start_idx, dof_idx] = axis_weights[0] * motion[3]
    jacobian[problem_idx, start_idx + 1, dof_idx] = axis_weights[1] * motion[4]
    jacobian[problem_idx, start_idx + 2, dof_idx] = axis_weights[2] * motion[5]


class IKObjectiveRotationAxisWeighted(ik.IKObjectiveRotation):
    """Newton rotation objective with one residual weight per world axis."""

    def __init__(self, *, axis_weights, **kwargs):
        super().__init__(weight=1.0, **kwargs)
        self.axis_weights = wp.vec3(*axis_weights)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _axis_weighted_rotation_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self.target_rotations,
                self.link_index,
                self.link_offset_rotation,
                self.canonicalize_quat_err,
                start_idx,
                self.axis_weights,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _axis_weighted_rotation_jacobian,
            dim=[joint_q.shape[0], model.joint_dof_count],
            inputs=[self.affects_dof, joint_S_s, start_idx, self.axis_weights],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _joint_reference_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    coord_index: int,
    target: wp.array1d(dtype=wp.float32),
    weight: float,
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    residuals[row, start_idx] = weight * (joint_q[row, coord_index] - target[problem_idx_map[row]])


@wp.kernel
def _joint_reference_jacobian(
    dof_index: int,
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx = wp.tid()
    jacobian[problem_idx, start_idx, dof_index] = weight


@wp.kernel
def _set_joint_reference_target(
    problem_idx: int,
    value: float,
    target: wp.array1d(dtype=wp.float32),
):
    target[problem_idx] = value


class IKJointReferenceObjective(ik.IKObjective):
    """Track one scalar joint coordinate with an analytic Jacobian."""

    def __init__(self, coord_index: int, dof_index: int, targets, weight: float):
        super().__init__()
        self.coord_index = coord_index
        self.dof_index = dof_index
        self.targets = targets
        self.weight = weight

    def residual_dim(self):
        return 1

    def supports_analytic(self):
        return True

    def set_target(self, problem_idx: int, value: float):
        self._require_batch_layout()
        wp.launch(
            _set_joint_reference_target,
            dim=1,
            inputs=[problem_idx, value],
            outputs=[self.targets],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _joint_reference_residuals,
            dim=joint_q.shape[0],
            inputs=[
                joint_q,
                self.coord_index,
                self.targets,
                self.weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _joint_reference_jacobian,
            dim=joint_q.shape[0],
            inputs=[self.dof_index, self.weight, start_idx],
            outputs=[jacobian],
            device=self.device,
        )


@wp.func
def _wp_smooth_joint_filter_func(
    x            : wp.float32,
    lower_limit  : wp.float32,
    upper_limit  : wp.float32,
    padding_limit: wp.float32,
    m            : wp.float32,
    p            : wp.float32
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += (padding_limit - c)
    upper_limit -= (padding_limit + c)
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit-x, x-upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    dof_to_coord: wp.array1d(dtype=wp.int32),        # (n_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),     # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    if mask > 0.0:
        lower = joint_limit_lower[dof_idx]
        upper = joint_limit_upper[dof_idx]
        c = (lower + upper) * 0.5

        q = joint_q[problem, coord_idx]
        error = (q - c)

        smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
        residuals[problem, start_idx + dof_idx] = error * smoother * weight[0] * mask
    else:
        residuals[problem, start_idx + dof_idx] = 0.0


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),  # (1)
):
    out_weight[0] = in_value


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),    # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),   # (n_coords)
    n_dofs: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32), # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),      # (n_batch, n_residuals, n_dofs)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    # Jacobian is diagonal: dr[dof]/dq[dof] = weight
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask


class IKSmoothJointFilter(ik.IKObjective):
    """
    An IK objective that applies a smooth penalty to joint coordinates that approach or exceed specified limits
    using an inverse gaussian filter.

    Args:
        joint_limit_lower (wp.array1d): An array of shape (n_dofs,) containing the lower limits for each joint degree of freedom.
        joint_limit_upper (wp.array1d): An array of shape (n_dofs,) containing the upper limits for each joint degree of freedom.
        weight (float, optional): A scalar weight that controls the strength of the joint limit penalty. Defaults to 0.01.
        coord_masks (wp.array1d, optional): An array of shape (n_coords,) containing mask values for each joint coordinate.
            Mask values should be in the range [0, 1], where 0 means the coordinate is ignored by this objective and 1 means it is fully considered.
            All coords are used by default if no masks are specified.
    """
    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.01, coord_masks=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
                self.coord_masks = None
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks
                self.coord_masks_np = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)

        # All coords are considered if no coord masks have been declared
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        # Build DOF to coordinate mapping
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k

        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        # For autodiff mode
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return

        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        # Use the analytic Jacobian fill since it's simple
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
