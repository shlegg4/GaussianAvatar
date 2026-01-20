import torch
import numpy as np
import pickle
import sys
import scipy.sparse  # Import scipy to handle the sparse matrix check

def main(model_path):
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        try:
            params = pickle.load(f, encoding='latin1')
        except TypeError:
            params = pickle.load(f)

    # Helper: Convert to float32 tensor
    def to_tensor(array):
        # 1. Handle SciPy Sparse Matrices explicitly
        if scipy.sparse.issparse(array):
            array = array.toarray()
            
        # 2. Handle Chumpy objects (if present)
        elif hasattr(array, 'r'):  # Chumpy objects often have an 'r' attribute
             array = np.array(array)
             
        return torch.tensor(np.array(array), dtype=torch.float32)

    # --- Processing ---

    # 1. v_template
    v_template = to_tensor(params['v_template'])
    
    # 2. shapedirs
    shapedirs = to_tensor(params['shapedirs'])
    
    # 3. posedirs
    posedirs = to_tensor(params['posedirs'])
    
    # 4. J_regressor (The logic is now moved inside to_tensor)
    J_regressor = to_tensor(params['J_regressor'])
        
    # 5. Parents (Ensure Int64/Long)
    parents = torch.tensor(params['kintree_table'][0].astype(np.int64), dtype=torch.long)
    
    # 6. Weights
    weights = to_tensor(params['weights'])

    print("Saving to smpl_data.pt...")
    
    output_dict = {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": J_regressor,
        "parents": parents,
        "weights": weights
    }
    
    torch.save(output_dict, "smpl_data.pt")
    print("Done! 'smpl_data.pt' created.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_smpl.py SMPL_NEUTRAL.pkl")
    else:
        main(sys.argv[1])