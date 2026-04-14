from __future__ import annotations

import argparse
from pathlib import Path

import torch
import trimesh

from .export import load_debug_export
from .model import SMPLXDebugModel


def _find_model_path(explicit_path: str | None, json_path: Path) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"SMPL-X model not found: {path}")
        return path

    for directory in (json_path.parent, *json_path.parents):
        candidate = directory / "smplx_data.pt"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find smplx_data.pt. Pass it explicitly with --model <path>."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render src_uv SMPL-X debug exports.")
    parser.add_argument("--json", required=True, help="Path to effective_smplx_params.json")
    parser.add_argument("--model", help="Path to smplx_data.pt")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to start from")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cpu or cuda",
    )
    parser.add_argument(
        "--apply-y-sign",
        action="store_true",
        help="Apply the saved y_sign flip to the rendered mesh",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Load the mesh and print a summary instead of opening a viewer",
    )
    parser.add_argument(
        "--obj-out",
        help="Optional OBJ export path for the selected sample in headless mode",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    json_path = Path(args.json).expanduser().resolve()
    metadata, samples = load_debug_export(json_path)
    if not samples:
        raise RuntimeError(f"No samples found in {json_path}")

    start_index = max(0, min(args.sample, len(samples) - 1))
    model_path = _find_model_path(args.model, json_path)
    model = SMPLXDebugModel(model_path, device=args.device)

    if args.headless:
        sample = samples[start_index]
        vertices, faces = model.vertices_for_sample(sample, apply_y_sign=args.apply_y_sign)
        print(
            f"epoch={metadata.get('epoch')} sample_index={sample.sample_index} frame={sample.frame} "
            f"vertices={vertices.shape[0]} faces={faces.shape[0]} device={args.device}"
        )
        if args.obj_out:
            out_path = Path(args.obj_out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out_path)
            print(f"Wrote OBJ to {out_path}")
        return

    from .viewer import InteractiveSmplxViewer

    viewer = InteractiveSmplxViewer(model, samples, apply_y_sign=args.apply_y_sign)
    viewer.show(start_index=start_index)
