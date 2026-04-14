from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyrender
import trimesh

from .export import DebugSample
from .model import SMPLXDebugModel


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32) if up is None else up.astype(np.float32)
    forward = eye - target
    forward /= np.linalg.norm(forward) + 1.0e-8
    right = np.cross(up_vec, forward)
    right /= np.linalg.norm(right) + 1.0e-8
    true_up = np.cross(forward, right)
    true_up /= np.linalg.norm(true_up) + 1.0e-8

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


@dataclass
class MeshBundle:
    mesh: trimesh.Trimesh
    joints: np.ndarray
    center: np.ndarray
    extent: float
    world_up: np.ndarray
    support_y: float
    down_sign: float


class InteractiveSmplxViewer:
    def __init__(
        self,
        model: SMPLXDebugModel,
        samples: Sequence[DebugSample],
        *,
        apply_y_sign: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("No samples were found in the export JSON.")
        self.model = model
        self.samples = list(samples)
        self.apply_y_sign = apply_y_sign
        self.current_index = 0
        self.scene = pyrender.Scene(
            bg_color=np.array([248, 248, 248, 255], dtype=np.uint8),
            ambient_light=np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float32),
        )
        self.mesh_node: pyrender.Node | None = None
        self.ground_node: pyrender.Node | None = None
        self.camera_node: pyrender.Node | None = None
        self.light_nodes: list[pyrender.Node] = []

    def _bundle_for_index(self, index: int) -> MeshBundle:
        sample = self.samples[index]
        result = self.model.result_for_sample(sample, apply_y_sign=self.apply_y_sign)
        vertices = result.vertices[0].detach().cpu().numpy()
        joints = result.joints[0].detach().cpu().numpy()
        faces = self.model.faces_cpu
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = np.tile(np.array([[196, 202, 220, 255]], dtype=np.uint8), (vertices.shape[0], 1))
        bounds = mesh.bounds
        center = bounds.mean(axis=0).astype(np.float32)
        extent = float(max(np.max(bounds[1] - bounds[0]), 1.0))

        head_joint_idx = 15
        foot_joint_indices = (7, 8, 10, 11)
        if joints.shape[0] > max((head_joint_idx, *foot_joint_indices)):
            head_y = float(joints[head_joint_idx, 1])
            foot_ys = np.array([joints[joint_idx, 1] for joint_idx in foot_joint_indices], dtype=np.float32)
            down_sign = 1.0 if float(np.mean(foot_ys)) >= head_y else -1.0
            support_y = float(np.max(foot_ys) if down_sign > 0.0 else np.min(foot_ys))
        else:
            min_y = float(bounds[0, 1])
            max_y = float(bounds[1, 1])
            down_sign = 1.0 if abs(max_y - center[1]) >= abs(min_y - center[1]) else -1.0
            support_y = max_y if down_sign > 0.0 else min_y

        world_up = np.array([0.0, -down_sign, 0.0], dtype=np.float32)
        return MeshBundle(
            mesh=mesh,
            joints=joints,
            center=center,
            extent=extent,
            world_up=world_up,
            support_y=support_y,
            down_sign=down_sign,
        )

    def _replace_camera(self, bundle: MeshBundle) -> None:
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
            self.camera_node = None
        for light_node in self.light_nodes:
            self.scene.remove_node(light_node)
        self.light_nodes.clear()

        eye = bundle.center + np.array([0.0, bundle.extent * 0.2, bundle.extent * 2.7], dtype=np.float32)
        camera_pose = _look_at(eye, bundle.center, up=bundle.world_up)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.camera_node = self.scene.add(camera, pose=camera_pose)

        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        self.light_nodes.append(self.scene.add(key_light, pose=camera_pose))

        side_eye = bundle.center + np.array([bundle.extent * 1.2, bundle.extent * 0.4, bundle.extent * 1.8], dtype=np.float32)
        side_pose = _look_at(side_eye, bundle.center, up=bundle.world_up)
        fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        self.light_nodes.append(self.scene.add(fill_light, pose=side_pose))

    def _replace_ground(self, bundle: MeshBundle) -> None:
        if self.ground_node is not None:
            self.scene.remove_node(self.ground_node)
            self.ground_node = None

        bounds = bundle.mesh.bounds
        plane_size = max(bundle.extent * 2.5, 1.5)
        plane_thickness = max(bundle.extent * 0.015, 0.01)
        if bundle.down_sign > 0.0:
            support_y = max(float(bounds[1, 1]), bundle.support_y)
        else:
            support_y = min(float(bounds[0, 1]), bundle.support_y)
        ground_y = support_y + (bundle.down_sign * plane_thickness * 0.5)
        ground_center = np.array(
            [bundle.center[0], ground_y, bundle.center[2]],
            dtype=np.float32,
        )

        ground = trimesh.creation.box(extents=(plane_size, plane_thickness, plane_size))
        ground.apply_translation(ground_center)
        ground.visual.face_colors = np.tile(
            np.array([[225, 228, 232, 255]], dtype=np.uint8),
            (len(ground.faces), 1),
        )
        ground_mesh = pyrender.Mesh.from_trimesh(ground, smooth=False)
        self.ground_node = self.scene.add(ground_mesh)

    def _apply_index(self, index: int) -> None:
        bundle = self._bundle_for_index(index)
        mesh = pyrender.Mesh.from_trimesh(bundle.mesh, smooth=False)
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)
        self._replace_ground(bundle)
        self._replace_camera(bundle)
        self.current_index = index
        self._print_current_sample()

    def _step(self, viewer: pyrender.Viewer, delta: int) -> None:
        next_index = (self.current_index + delta) % len(self.samples)
        with viewer.render_lock:
            self._apply_index(next_index)

    def _print_current_sample(self) -> None:
        sample = self.samples[self.current_index]
        print(
            f"[sample {self.current_index + 1}/{len(self.samples)}] "
            f"sample_index={sample.sample_index} frame={sample.frame} "
            f"image={sample.image_file} crop={sample.crop_path}"
        )

    def show(self, start_index: int = 0) -> None:
        start_index = max(0, min(start_index, len(self.samples) - 1))
        self._apply_index(start_index)

        keymap = {
            "n": lambda viewer: self._step(viewer, 1),
            "p": lambda viewer: self._step(viewer, -1),
            "h": lambda viewer: self._print_current_sample(),
        }

        print("Controls: mouse orbit/pan/zoom, 'n' next sample, 'p' previous sample, 'h' current sample info.")
        pyrender.Viewer(
            self.scene,
            use_raymond_lighting=False,
            run_in_thread=False,
            viewport_size=(1280, 960),
            title="SMPL-X Debug Renderer",
            registered_keys=keymap,
        )
