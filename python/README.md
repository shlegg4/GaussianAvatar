# SMPL-X Debug Renderer

This package loads the epoch-level `effective_smplx_params.json` files written by `src_uv` and renders the saved SMPL-X parameters interactively.

## Setup

From the repo root:

```powershell
python -m venv --system-site-packages python/.venv
python/.venv/Scripts/python -m pip install --upgrade pip
python/.venv/Scripts/python -m pip install -e ./python
```

`--system-site-packages` lets the venv reuse the machine's existing PyTorch install, which avoids reinstalling a large wheel if Torch is already present.

## Usage

Headless smoke test:

```powershell
python/.venv/Scripts/python -m smplx_debug_renderer --json outputs_sobel/pairs/epoch_9/effective_smplx_params.json --headless
```

Interactive viewer:

```powershell
python/.venv/Scripts/python -m smplx_debug_renderer --json outputs_sobel/pairs/epoch_9/effective_smplx_params.json
```

Optional arguments:

- `--model <path>` overrides the default `smplx_data.pt` lookup.
- `--sample <index>` starts the viewer on a specific sample.
- `--device cpu|cuda` selects the Torch device. Default is `cuda` when available, otherwise `cpu`.
- `--apply-y-sign` applies the saved `y_sign` flip before rendering the mesh.
- `--obj-out <path>` exports the selected sample as an `.obj` in headless mode.

## Viewer Controls

- Mouse drag: orbit
- Shift + mouse drag: pan
- Mouse wheel: zoom
- `n`: next sample
- `p`: previous sample
- `h`: print current sample info
