from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .export import DebugSample


TensorLike = torch.Tensor | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class SMPLXForwardResult:
    vertices: torch.Tensor
    joints: torch.Tensor
    faces: torch.Tensor


def batch_rodrigues(theta: torch.Tensor) -> torch.Tensor:
    batch_size = theta.shape[0]
    num_rotations = 1 if theta.ndim == 2 else theta.shape[1]
    theta = theta.reshape(-1, 3)

    angle = torch.norm(theta, dim=1, keepdim=True) + 1.0e-8
    r = theta / angle
    angle_cos = torch.cos(angle)
    angle_sin = torch.sin(angle)

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    k = torch.zeros((theta.shape[0], 3, 3), dtype=theta.dtype, device=theta.device)
    k[:, 0, 1] = -z
    k[:, 0, 2] = y
    k[:, 1, 0] = z
    k[:, 1, 2] = -x
    k[:, 2, 0] = -y
    k[:, 2, 1] = x

    identity = torch.eye(3, dtype=theta.dtype, device=theta.device).unsqueeze(0)
    rot = identity + angle_sin.unsqueeze(-1) * k + (1.0 - angle_cos.unsqueeze(-1)) * torch.matmul(k, k)
    if num_rotations > 1:
        return rot.reshape(batch_size, num_rotations, 3, 3)
    return rot.reshape(batch_size, 3, 3)


def transform_mat(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    batch_size = rot.shape[0]
    mat = torch.eye(4, dtype=rot.dtype, device=rot.device).unsqueeze(0).repeat(batch_size, 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


class SMPLXDebugModel:
    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self._load_model()

    def _load_model(self) -> None:
        try:
            data = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:
            data = torch.load(self.model_path, map_location=self.device)

        if not isinstance(data, dict):
            raise TypeError(f"Expected a tensor dictionary in {self.model_path}, got {type(data)!r}.")

        self.v_template = data["v_template"].to(self.device, dtype=torch.float32).contiguous()
        self.shapedirs = data["shapedirs"].to(self.device, dtype=torch.float32).contiguous()
        self.exprdirs = data["exprdirs"].to(self.device, dtype=torch.float32).contiguous()
        self.blenddirs = torch.cat((self.shapedirs, self.exprdirs), dim=2).contiguous()
        self.posedirs = data["posedirs"].to(self.device, dtype=torch.float32).contiguous()
        self.j_regressor = data["J_regressor"].to(self.device, dtype=torch.float32).contiguous()
        self.weights = data["weights"].to(self.device, dtype=torch.float32).contiguous()
        self.faces = data["faces"].to(self.device, dtype=torch.long).contiguous()
        self.faces_cpu = self.faces.cpu().numpy()

        parents = data["parents"].to(self.device, dtype=torch.long).contiguous()
        parents = parents.clone()
        parents[0] = 0
        self.parents = parents
        self.parents_host = parents.cpu().tolist()
        self.expr_count = int(self.exprdirs.shape[2])
        self.joint_count = int(self.j_regressor.shape[0])

    def _to_tensor(self, value: TensorLike | None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=dtype)
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def _expand_param(self, value: TensorLike | None, width: int, batch_size: int) -> torch.Tensor:
        tensor = self._to_tensor(value)
        if tensor is None or tensor.numel() == 0:
            return torch.zeros((batch_size, width), dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, tensor.shape[1])
        if tensor.shape[1] < width:
            pad = torch.zeros((batch_size, width - tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat((tensor, pad), dim=1)
        elif tensor.shape[1] > width:
            tensor = tensor[:, :width]
        return tensor.contiguous()

    def forward(
        self,
        betas: TensorLike,
        pose_axis_angle: TensorLike,
        *,
        transl: TensorLike | None = None,
        expression: TensorLike | None = None,
        jaw_pose: TensorLike | None = None,
        eye_pose: TensorLike | None = None,
        left_hand_pose: TensorLike | None = None,
        right_hand_pose: TensorLike | None = None,
    ) -> SMPLXForwardResult:
        with torch.no_grad():
            betas_tensor = self._to_tensor(betas)
            if betas_tensor is None:
                raise ValueError("betas are required")
            if betas_tensor.ndim == 1:
                betas_tensor = betas_tensor.unsqueeze(0)
            betas_tensor = betas_tensor.reshape(betas_tensor.shape[0], -1).contiguous()
            batch_size = int(betas_tensor.shape[0])

            pose_tensor = self._to_tensor(pose_axis_angle)
            if pose_tensor is None:
                raise ValueError("pose_axis_angle is required")
            if pose_tensor.ndim == 1:
                pose_tensor = pose_tensor.unsqueeze(0)
            if pose_tensor.ndim == 2:
                if pose_tensor.shape[0] == 1 and batch_size > 1:
                    pose_tensor = pose_tensor.expand(batch_size, pose_tensor.shape[1])
                pose_tensor = pose_tensor.reshape(pose_tensor.shape[0], -1, 3)
            elif pose_tensor.ndim == 3 and pose_tensor.shape[0] == 1 and batch_size > 1:
                pose_tensor = pose_tensor.expand(batch_size, pose_tensor.shape[1], pose_tensor.shape[2])
            if pose_tensor.ndim != 3:
                raise ValueError(f"Expected pose_axis_angle to be rank-3 after reshaping, got {pose_tensor.ndim}.")
            pose_tensor = pose_tensor.contiguous()

            trans_tensor = self._to_tensor(transl)
            if trans_tensor is None or trans_tensor.numel() == 0:
                trans_tensor = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
            elif trans_tensor.ndim == 1:
                trans_tensor = trans_tensor.unsqueeze(0)
            if trans_tensor.ndim == 2 and trans_tensor.shape[0] == 1 and batch_size > 1:
                trans_tensor = trans_tensor.expand(batch_size, trans_tensor.shape[1])
            trans_tensor = trans_tensor.contiguous()

            expression_tensor = self._expand_param(expression, self.expr_count, batch_size)
            jaw_pose_tensor = self._expand_param(jaw_pose, 3, batch_size)
            eye_pose_tensor = self._expand_param(eye_pose, 6, batch_size)
            left_hand_pose_tensor = self._expand_param(left_hand_pose, 45, batch_size)
            right_hand_pose_tensor = self._expand_param(right_hand_pose, 45, batch_size)

            global_pose = pose_tensor[:, 0].contiguous()
            body_pose = pose_tensor[:, 1:22].contiguous()
            pose_all = torch.cat(
                (
                    global_pose.reshape(batch_size, 3),
                    body_pose.reshape(batch_size, 63),
                    jaw_pose_tensor,
                    eye_pose_tensor,
                    left_hand_pose_tensor,
                    right_hand_pose_tensor,
                ),
                dim=1,
            ).reshape(batch_size, self.joint_count, 3)

            betas_full = torch.cat((betas_tensor, expression_tensor), dim=1)
            v_shaped = self.v_template.unsqueeze(0) + torch.einsum("bl,mkl->bmk", betas_full, self.blenddirs)
            joints = torch.einsum("jv,bvi->bji", self.j_regressor, v_shaped)

            rot_mats = batch_rodrigues(pose_all)
            pose_feature = (rot_mats[:, 1:] - torch.eye(3, device=self.device)).reshape(batch_size, -1)
            v_posed = v_shaped + torch.einsum("bl,mkl->bmk", pose_feature, self.posedirs)

            transforms: list[torch.Tensor] = [torch.empty(0, device=self.device)] * self.joint_count
            parent_joints = joints.index_select(1, self.parents)
            joint_offsets = joints - parent_joints

            for joint_idx in range(self.joint_count):
                rot = rot_mats[:, joint_idx]
                offset = joints[:, 0] if joint_idx == 0 else joint_offsets[:, joint_idx]
                local_transform = transform_mat(rot, offset)
                if joint_idx == 0:
                    transforms[joint_idx] = local_transform
                else:
                    parent_idx = self.parents_host[joint_idx]
                    if parent_idx < 0 or parent_idx >= joint_idx:
                        transforms[joint_idx] = local_transform
                    else:
                        transforms[joint_idx] = torch.matmul(transforms[parent_idx], local_transform)

            results_g = torch.stack(transforms, dim=1)

            neg_joints = -joints
            zeros = torch.zeros((batch_size, self.joint_count, 1), dtype=neg_joints.dtype, device=self.device)
            ones = torch.ones((batch_size, self.joint_count, 1), dtype=neg_joints.dtype, device=self.device)

            row0 = torch.cat((ones, zeros, zeros, neg_joints[:, :, 0:1]), dim=2).unsqueeze(2)
            row1 = torch.cat((zeros, ones, zeros, neg_joints[:, :, 1:2]), dim=2).unsqueeze(2)
            row2 = torch.cat((zeros, zeros, ones, neg_joints[:, :, 2:3]), dim=2).unsqueeze(2)
            row3 = torch.cat((zeros, zeros, zeros, ones), dim=2).unsqueeze(2)

            rest_inv = torch.cat((row0, row1, row2, row3), dim=2)
            skinning_transforms = torch.matmul(results_g, rest_inv)
            vertex_transforms = torch.einsum("bkij,vk->bvij", skinning_transforms, self.weights)

            v_posed_homo = torch.cat(
                (v_posed, torch.ones((batch_size, v_posed.shape[1], 1), dtype=v_posed.dtype, device=self.device)),
                dim=2,
            )
            vertices_homo = torch.matmul(vertex_transforms, v_posed_homo.unsqueeze(-1)).squeeze(-1)
            vertices = vertices_homo[:, :, :3] + trans_tensor.unsqueeze(1)

            joints_posed = results_g[:, :, :3, 3] + trans_tensor.unsqueeze(1)
            return SMPLXForwardResult(vertices=vertices, joints=joints_posed, faces=self.faces)

    def vertices_for_sample(self, sample: DebugSample, *, apply_y_sign: bool = False) -> tuple[np.ndarray, np.ndarray]:
        result = self.forward(
            sample.params.betas,
            sample.params.pose_axis_angle,
            transl=sample.params.transl,
            expression=sample.params.expression,
            jaw_pose=sample.params.jaw_pose,
            eye_pose=sample.params.eye_pose,
            left_hand_pose=sample.params.left_hand_pose,
            right_hand_pose=sample.params.right_hand_pose,
        )
        if apply_y_sign:
            result.vertices[:, :, 1] *= sample.y_sign
            result.joints[:, :, 1] *= sample.y_sign
        vertices = result.vertices[0].detach().cpu().numpy()
        return vertices, self.faces_cpu

    def result_for_sample(self, sample: DebugSample, *, apply_y_sign: bool = False) -> SMPLXForwardResult:
        result = self.forward(
            sample.params.betas,
            sample.params.pose_axis_angle,
            transl=sample.params.transl,
            expression=sample.params.expression,
            jaw_pose=sample.params.jaw_pose,
            eye_pose=sample.params.eye_pose,
            left_hand_pose=sample.params.left_hand_pose,
            right_hand_pose=sample.params.right_hand_pose,
        )
        if apply_y_sign:
            result.vertices[:, :, 1] *= sample.y_sign
            result.joints[:, :, 1] *= sample.y_sign
        return result
