import torch
import numpy as np

# 1. Provide the exact path to your .npz file
npz_path = r"C:\Users\Sam\Downloads\smplx_lockedhead_20230207\models_lockedhead\smplx\SMPLX_NEUTRAL.npz"

print(f"Loading {npz_path}...")
data = np.load(npz_path, allow_pickle=True)

print(f"Available keys in the NPZ file: {data.files}")

# 2. Extract J_regressor safely
J_reg = data['J_regressor']
if J_reg.ndim == 0:  # If it's wrapped in a 0-d object array
    J_reg = J_reg.item().todense()
elif hasattr(J_reg, 'todense'):
    J_reg = J_reg.todense()

print("Converting arrays to PyTorch tensors...")

# 3. Handle the shape/expression slicing logic
full_shapedirs = data['shapedirs']

# Check if expressions are baked into shapedirs (Locked Head logic)
if full_shapedirs.shape[-1] > 10:
    print(f"Detected combined shapedirs with {full_shapedirs.shape[-1]} dimensions.")
    # The first 10 are body shape (identity), the rest are facial expressions
    shape_data = full_shapedirs[:, :, :10]
    expr_data = full_shapedirs[:, :, 10:]
    print(f"Sliced into -> Shape: {shape_data.shape[-1]} | Expressions: {expr_data.shape[-1]}")
else:
    # Standard fallback if the file has them separated
    print("Detected standard shapedirs (10 dimensions). Looking for separate expression key...")
    shape_data = full_shapedirs
    if 'exprdirs' in data:
        expr_data = data['exprdirs']
    elif 'expr_dirs' in data:
        expr_data = data['expr_dirs']
    else:
        raise KeyError("Could not find expression blendshapes in shapedirs or as a separate key.")

# 4. Package exactly the tensors your C++ code expects
export_dict = {
    "v_template": torch.tensor(data['v_template'], dtype=torch.float32),
    "shapedirs": torch.tensor(shape_data, dtype=torch.float32),
    "exprdirs": torch.tensor(expr_data, dtype=torch.float32),
    "posedirs": torch.tensor(data['posedirs'], dtype=torch.float32),
    "J_regressor": torch.tensor(J_reg, dtype=torch.float32),
    "weights": torch.tensor(data['weights'], dtype=torch.float32),
    
    # kintree_table is a [2, num_joints] array. The first row contains the parent indices.
    "parents": torch.tensor(data['kintree_table'][0], dtype=torch.long),
    "jaw_index": torch.tensor([22], dtype=torch.long)
}

# 5. Save using PyTorch's pickle format
out_path = "smplx_neutral.pt"
torch.save(export_dict, out_path)

print(f"Success! Generated {out_path} for your C++ SMPLLayer.")