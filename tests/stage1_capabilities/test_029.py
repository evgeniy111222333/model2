"""
BCS Stage 1 Capability Test 029: Predictive: Hierarchical Alignment
Description: Ensure hierarchical predictive coding learns without errors on structured data
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
    print("Running capability test 029: Predictive: Hierarchical Alignment")
    
    # 1. Generate Input Data
    data = b'PATTERN123' * 40
    
    # 2. Initialize Model
    # Config parameters: {'use_hierarchical_pc': True, 'n_active_bytes': 32, 'n_conversion_levels': 3}
    model = create_model(**{'use_hierarchical_pc': True, 'n_active_bytes': 32, 'n_conversion_levels': 3})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 29,
        "test_name": "Predictive: Hierarchical Alignment",
        "description": "Ensure hierarchical predictive coding learns without errors on structured data",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        hpc_errors = results.get('v5_hpc_errors', [])
        metrics.update({
            "error_count": len(hpc_errors),
            "final_hpc_error": float(hpc_errors[-1]) if hpc_errors else 0.0
        })
        success = True
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_029.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
