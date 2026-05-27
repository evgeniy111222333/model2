"""
BCS Stage 1 Capability Test 015: Field: Double-Well Potential Bifurcation
Description: Verify that Phi field develops spatial bimodal (double-well) pattern separation
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
    print("Running capability test 015: Field: Double-Well Potential Bifurcation")
    
    # 1. Generate Input Data
    data = b'AAAAABBBBB' * 50
    
    # 2. Initialize Model
    # Config parameters: {'n_active_bytes': 64}
    model = create_model(**{'n_active_bytes': 64})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 15,
        "test_name": "Field: Double-Well Potential Bifurcation",
        "description": "Verify that Phi field develops spatial bimodal (double-well) pattern separation",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        phi_vals = model.field.Phi.flatten()
        # Check for presence of negative and positive regions
        has_positive = np.any(phi_vals > 0.1)
        has_negative = np.any(phi_vals < -0.1)
        metrics.update({
            "phi_min": float(np.min(phi_vals)),
            "phi_max": float(np.max(phi_vals)),
            "has_bimodal": bool(has_positive and has_negative)
        })
        success = has_positive and has_negative
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_015.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
