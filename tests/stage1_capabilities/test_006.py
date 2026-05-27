"""
BCS Stage 1 Capability Test 006: Modality: Dense Binary Data
Description: Check high entropy uniform random bytes detection
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

# Add paths for both bcs package and tests module
import os
import sys

# Get paths
test_dir = os.path.dirname(os.path.abspath(__file__))  # E:\arc\bcs\tests\stage1_capabilities
tests_dir = os.path.dirname(test_dir)  # E:\arc\bcs\tests
bcs_dir = os.path.dirname(tests_dir)  # E:\arc\bcs
arc_dir = os.path.dirname(bcs_dir)  # E:\arc

# Insert in correct order - E:\arc first so 'bcs' is found, then E:\arc\bcs for 'tests'
if arc_dir not in sys.path:
    sys.path.insert(0, arc_dir)
if bcs_dir not in sys.path:
    sys.path.insert(0, bcs_dir)

from bcs.model import BCSModelV6
from tests import create_model

def run_test():
    print("Running capability test 006: Modality: Dense Binary Data")
    
    # 1. Generate Input Data
    np.random.seed(42); data = bytes(np.random.randint(0, 256, 1000))
    
    # 2. Initialize Model
    # Config parameters: {'use_bayesian_modality': True, 'n_active_bytes': 64}
    model = create_model(**{'use_bayesian_modality': True, 'n_active_bytes': 64})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {"error_run": str(e)}
    
    # 4. Extract metrics & check success status
    metrics = {
        "test_id": 6,
        "test_name": "Modality: Dense Binary Data",
        "description": "Check high entropy uniform random bytes detection",
        "success": False
    }
    
    success = False
    
    # Custom evaluation code
    try:
        modality = model.detected_modality
        entropy = float(model.substrate._shannon_entropy(model.substrate.byte_distribution))
        metrics.update({
            "detected_modality": modality,
            "entropy": entropy
        })
        success = modality == "binary"
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {e}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_006.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {out_path}")
    print(f"Status: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    run_test()
