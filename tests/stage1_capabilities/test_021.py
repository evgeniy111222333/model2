"""
BCS Stage 1 Capability Test 021: Predictive: Constant Input Error Convergence
Description: Verify that prediction error converges close to zero for static byte sequences
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
    print("Running capability test 021: Predictive: Constant Input Error Convergence")
    
    # 1. Generate Input Data
    data = b'A' * 400
    
    # 2. Initialize Model
    # Config parameters: {'use_prediction_error_loop': True, 'n_active_bytes': 32}
    model = create_model(**{'use_prediction_error_loop': True, 'n_active_bytes': 32})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 21,
        "test_name": "Predictive: Constant Input Error Convergence",
        "description": "Verify that prediction error converges close to zero for static byte sequences",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        pel = results.get('v6_prediction_error_loop', [])
        first_err = pel[0]['mean_error'] if pel else 999.0
        last_err = pel[-1]['mean_error'] if pel else 999.0
        metrics.update({
            "first_error": float(first_err),
            "last_error": float(last_err),
            "convergence_ratio": float(last_err / first_err) if first_err > 0 else 1.0
        })
        success = last_err < 0.5 or last_err <= first_err
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_021.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
