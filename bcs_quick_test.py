"""
BCS Stage 1 Real Data Experiments - Quick Version
"""

import sys
import os
import numpy as np

# Add E:\arc to path so 'bcs' package is found
parent_dir = os.path.dirname(os.path.abspath(__file__))  # E:\arc\bcs
grandparent = os.path.dirname(parent_dir)  # E:\arc
if grandparent not in sys.path:
    sys.path.insert(0, grandparent)

from bcs.model import BCSModelV6


def quick_experiment():
    print("="*60)
    print("  BCS STAGE 1: REAL DATA EXPERIMENTS")
    print("="*60)
    
    # Load real text data
    with open('E:\\arc\\text.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    data = text_data.encode('utf-8')
    print(f"\nLoaded {len(data)} bytes of real text")
    
    # Quick test
    print("\n[1/5] Testing basic processing...")
    model = BCSModelV6(
        n_active_bytes=32,
        use_crystallized_memory=True,
        use_prediction_error_loop=True
    )
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=100, record_every=50)
    
    print(f"    Clusters: {len(results['final_clusters'])}")
    print(f"    Crystals: {len(model.crystal_memory.crystals)}")
    
    # Check prediction convergence
    pel = results.get('v6_prediction_error_loop', [])
    if pel:
        first_err = pel[0]['mean_error']
        last_err = pel[-1]['mean_error']
        print(f"    Prediction error: {first_err:.4f} -> {last_err:.4f}")
    
    # Test modality detection
    print("\n[2/5] Modality detection...")
    model2 = BCSModelV6(n_active_bytes=32, use_bayesian_modality=True)
    model2.ingest(data)
    print(f"    Detected: {model2.detected_modality}")
    
    # Test clustering quality
    print("\n[3/5] Cluster analysis...")
    clusters = results['final_clusters']
    if clusters:
        qualities = [c['quality_score'] for c in clusters]
        print(f"    Quality range: [{min(qualities):.3f}, {max(qualities):.3f}]")
        print(f"    Avg cluster size: {np.mean([c['size'] for c in clusters]):.1f}")
    
    # Test memory
    print("\n[4/5] Memory consolidation...")
    print(f"    Working memory: {len(model.working_memory.buffer)} items")
    print(f"    Crystallized: {len(model.crystal_memory.crystals)} patterns")
    
    # Test free energy
    print("\n[5/5] Free energy...")
    fe = results.get('free_energy_over_time', [])
    if fe:
        print(f"    Initial: {fe[0]:.4f}, Final: {fe[-1]:.4f}")
        if fe[-1] < fe[0]:
            print("    ✓ FE decreasing (system learning)")
        else:
            print("    ! FE stable (equilibrium)")
    
    print("\n" + "="*60)
    print("  EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    quick_experiment()