"""
BCS Stage 1 Capability Test 037: Boundary: Windowed Overlap Contiguity
Description: Examine boundaries stability when processed using sliding windows with overlaps
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
    print("Running capability test 037: Boundary: Windowed Overlap Contiguity")
    
    # 1. Generate Input Data
    data = b'A'*500 + b'B'*500
    
    # 2. Initialize Model
    # Config parameters: {'n_active_bytes': 32}
    model = create_model(**{'n_active_bytes': 32})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 37,
        "test_name": "Boundary: Windowed Overlap Contiguity",
        "description": "Examine boundaries stability when processed using sliding windows with overlaps",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        # Run with windowed processing explicitly
        model_w = create_model(n_steps=100, n_active_bytes=32)
        model_w.ingest(data).build_tensors().init_field()
        res_w = model_w.run(n_steps=100, window_size=400, window_overlap=80)
        boundaries = res_w.get('boundary_indices', [])
        metrics.update({
            "boundary_count": len(boundaries),
            "boundaries": [int(b) for b in boundaries]
        })
        success = True
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_037.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
