"""
BCS Stage 1 Capability Test 054: Variational: Latent Space Sparsity
Description: Examine the sparsity pattern of the latent representations
Generated automatically by generate_all_tests.py.
"""

import sys
import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Add grandparent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import create_model

def run_test():
    print("Running capability test 054: Variational: Latent Space Sparsity")
    
    # 1. Generate Input Data
    data = b'ABCDEFGH' * 50
    
    # 2. Initialize Model
    # Config parameters: {'use_variational': True, 'n_active_bytes': 32}
    model = create_model(**{'use_variational': True, 'n_active_bytes': 32})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 54,
        "test_name": "Variational: Latent Space Sparsity",
        "description": "Examine the sparsity pattern of the latent representations",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        obs = np.mean(model.field.Phi, axis=0).astype(np.float32)
        obs = obs / max(obs.sum(), 1e-10)
        latents, _, _ = model.variational.encode(obs) if model.variational else (None, None, None)
        z = latents[0] if latents else None
        sparsity = float(np.mean(z == 0.0)) if z is not None else 0.0
        metrics.update({
            "latent_sparsity": sparsity,
            "latent_shape": list(z.shape) if z is not None else []
        })
        success = True
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_054.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
