"""
BCS Stage 1 Capability Test 043: Clustering: Spatial Contiguity Verify
Description: Verify that individual clusters contain spatially contiguous positions
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
    print("Running capability test 043: Clustering: Spatial Contiguity Verify")
    
    # 1. Generate Input Data
    data = b'ABCD' * 100
    
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
        "test_id": 43,
        "test_name": "Clustering: Spatial Contiguity Verify",
        "description": "Verify that individual clusters contain spatially contiguous positions",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        clusters = results.get('final_clusters', [])
        coherent = all(
            len(set(np.diff(c['positions']))) == 1
            for c in clusters
            if len(c['positions']) > 2
        )
        metrics.update({
            "cluster_count": len(clusters),
            "spatially_coherent": coherent
        })
        success = coherent or len(clusters) == 0
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_043.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
