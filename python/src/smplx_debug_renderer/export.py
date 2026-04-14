from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in values]


@dataclass(frozen=True)
class EffectiveSmplxParams:
    betas: list[float]
    transl: list[float]
    global_orient: list[float]
    body_pose: list[float]
    pose_axis_angle: list[float]
    expression: list[float]
    jaw_pose: list[float]
    left_eye_pose: list[float]
    right_eye_pose: list[float]
    eye_pose: list[float]
    left_hand_pose: list[float]
    right_hand_pose: list[float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EffectiveSmplxParams":
        return cls(
            betas=_float_list(payload.get("betas")),
            transl=_float_list(payload.get("transl")),
            global_orient=_float_list(payload.get("global_orient")),
            body_pose=_float_list(payload.get("body_pose")),
            pose_axis_angle=_float_list(payload.get("pose_axis_angle")),
            expression=_float_list(payload.get("expression")),
            jaw_pose=_float_list(payload.get("jaw_pose")),
            left_eye_pose=_float_list(payload.get("left_eye_pose")),
            right_eye_pose=_float_list(payload.get("right_eye_pose")),
            eye_pose=_float_list(payload.get("eye_pose")),
            left_hand_pose=_float_list(payload.get("left_hand_pose")),
            right_hand_pose=_float_list(payload.get("right_hand_pose")),
        )


@dataclass(frozen=True)
class DebugSample:
    sample_index: int
    frame: int
    image_file: str
    crop_path: str
    body_model: str
    y_sign: float
    params: EffectiveSmplxParams

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DebugSample":
        return cls(
            sample_index=int(payload.get("sample_index", -1)),
            frame=int(payload.get("frame", -1)),
            image_file=str(payload.get("image_file", "")),
            crop_path=str(payload.get("crop_path", "")),
            body_model=str(payload.get("body_model", "smplx")),
            y_sign=float(payload.get("y_sign", 1.0)),
            params=EffectiveSmplxParams.from_dict(payload.get("effective_smplx_params", {})),
        )


def load_debug_export(json_path: str | Path) -> tuple[dict[str, Any], list[DebugSample]]:
    path = Path(json_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = [DebugSample.from_dict(sample) for sample in payload.get("samples", [])]
    metadata = {key: value for key, value in payload.items() if key != "samples"}
    return metadata, samples
