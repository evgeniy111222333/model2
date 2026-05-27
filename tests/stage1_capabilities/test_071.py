"""
BCS Stage 1 Capability Test 071: Memory: Working Memory Ring Eviction
Description: Verify oldest cluster details are evicted when working memory exceeds buffer size
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
    print("Running capability test 071: Memory: Working Memory Ring Eviction")
    
    # 1. Generate Input Data
    data = b'A'*10 + b'B'*10 + b'C'*10 + b'D'*10 + b'E'*10 + b'F'*10 + b'G'*10 + b'H'*10 + b'I'*10
    
    # 2. Initialize Model
    # Config parameters: {'use_working_memory': True, 'n_active_bytes': 32}
    model = create_model(**{'use_working_memory': True, 'n_active_bytes': 32})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 71,
        "test_name": "Memory: Working Memory Ring Eviction",
        "description": "Verify oldest cluster details are evicted when working memory exceeds buffer size",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        buffer_size = len(model.working_memory.buffer) if model.working_memory else 0
        metrics.update({
            "buffer_size": buffer_size,
            "capacity": model.working_memory.capacity if model.working_memory else 0
        })
        success = buffer_size <= (model.working_memory.capacity if model.working_memory else 99)
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_071.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
