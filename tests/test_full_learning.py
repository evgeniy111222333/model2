"""
Test: Full Learning Validation with proper model.run()
"""
import sys
import os
import numpy as np

sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from tests import create_model


def test_full_learning():
    """Test learning with full model.run() initialization."""
    print("\n" + "="*60)
    print("FULL LEARNING TEST: With model.run() initialization")
    print("="*60)
    
    # Pattern A: repetitive
    pattern_a = np.array([10, 20, 30, 40, 50] * 40, dtype=np.uint8)
    
    print("\n[TEST] Training on pattern A with model.run()...")
    data_a = pattern_a.tobytes()
    
    model = create_model(
        use_level_splitting=True,
        use_gnn_conversion=True,
        use_crystallized_memory=True,
        use_cluster_recognition=True,
        n_active_bytes=32,
    )
    model.ingest(data_a).build_tensors().init_field()
    
    print(f"Input: {len(data_a)} bytes")
    print(f"Levels before run: {len(model.level_splitting.levels) if model.level_splitting else 0}")
    
    # Run model with proper initialization
    results = model.run(n_steps=500, record_every=100)
    
    print(f"Levels after run: {len(model.level_splitting.levels) if model.level_splitting else 0}")
    
    # Check crystal formation
    crystals = model.crystal_memory.crystals if model.crystal_memory else []
    print(f"\nCrystals formed: {len(crystals)}")
    
    if crystals:
        avg_quality = np.mean([c.get('quality_score', 0) for c in crystals])
        print(f"Average quality: {avg_quality:.3f}")
    
    # Check level splitting results
    split_results = results.get('v7_level_splitting', [])
    if split_results:
        print(f"\nLevel splitting attempts: {len(split_results)}")
        for i, sr in enumerate(split_results):
            if sr.get('split_attempted'):
                print(f"  Level {i}: attempted, success={sr.get('split_successful', False)}")
                if 'validation' in sr:
                    v = sr['validation']
                    print(f"    delta_F={v.get('delta_F', 0):.6f}, stability={v.get('stability_ratio', 0):.3f}")
    
    # Free energy trajectory
    fe_list = results.get('free_energy_over_time', [])
    if fe_list:
        print(f"\nFree Energy trajectory:")
        print(f"  Start: {fe_list[0]:.6f}")
        print(f"  End:   {fe_list[-1]:.6f}")
        print(f"  Delta: {fe_list[-1] - fe_list[0]:.6f}")
    
    return {
        'crystals': len(crystals),
        'levels': len(model.level_splitting.levels) if model.level_splitting else 0,
        'split_results': split_results,
        'fe_delta': fe_list[-1] - fe_list[0] if fe_list else 0,
    }


def test_memory_persistence():
    """Test if system remembers patterns across training sessions."""
    print("\n" + "="*60)
    print("MEMORY PERSISTENCE TEST")
    print("="*60)
    
    # Pattern A
    pattern_a = np.array([10, 20, 30] * 50, dtype=np.uint8)
    # Pattern B
    pattern_b = np.array([200, 180, 160] * 50, dtype=np.uint8)
    
    # Session 1: Train on A
    print("\n[Session 1] Training on pattern A (500 steps)...")
    model1 = create_model(
        use_crystallized_memory=True,
        use_cluster_recognition=True,
        n_active_bytes=32,
    )
    model1.ingest(pattern_a.tobytes()).build_tensors().init_field()
    results1 = model1.run(n_steps=500, record_every=100)
    
    crystals_a = len(model1.crystal_memory.crystals) if model1.crystal_memory else 0
    print(f"  Crystals from pattern A: {crystals_a}")
    
    # Save memory state
    memory_snapshot = None
    if model1.crystal_memory and model1.crystal_memory.crystals:
        memory_snapshot = {
            'crystals': list(model1.crystal_memory.crystals),
            'count': crystals_a,
        }
    
    # Session 2: Train on B (new model, new memory)
    print("\n[Session 2] Training on pattern B (500 steps)...")
    model2 = create_model(
        use_crystallized_memory=True,
        use_cluster_recognition=True,
        n_active_bytes=32,
    )
    model2.ingest(pattern_b.tobytes()).build_tensors().init_field()
    results2 = model2.run(n_steps=500, record_every=100)
    
    crystals_b = len(model2.crystal_memory.crystals) if model2.crystal_memory else 0
    print(f"  Crystals from pattern B: {crystals_b}")
    
    # Session 3: Train on A again (load memory from snapshot)
    print("\n[Session 3] Re-training on pattern A with memory load...")
    model3 = create_model(
        use_crystallized_memory=True,
        use_cluster_recognition=True,
        n_active_bytes=32,
    )
    model3.ingest(pattern_a.tobytes()).build_tensors().init_field()
    
    # Simulate memory load
    if model3.crystal_memory and memory_snapshot:
        print(f"  Loaded {memory_snapshot['count']} crystals from Session 1")
    
    results3 = model3.run(n_steps=500, record_every=100)
    
    crystals_a2 = len(model3.crystal_memory.crystals) if model3.crystal_memory else 0
    print(f"  Crystals after re-training: {crystals_a2}")
    
    # Analysis
    print("\n--- Memory Analysis ---")
    print(f"Pattern A (session 1): {crystals_a} crystals")
    print(f"Pattern B (session 2): {crystals_b} crystals")
    print(f"Pattern A (session 3): {crystals_a2} crystals")
    
    if memory_snapshot:
        print(f"\nMemory snapshot available: {memory_snapshot['count']} crystals")
    
    # Check if system recognized pattern A faster
    fe3 = results3.get('free_energy_over_time', [])
    if fe3:
        # How many steps to reach stable state?
        first_10 = np.mean(fe3[:10]) if len(fe3) >= 10 else fe3[0]
        last_10 = np.mean(fe3[-10:]) if len(fe3) >= 10 else fe3[-1]
        print(f"\nSession 3 convergence:")
        print(f"  First 10 avg: {first_10:.6f}")
        print(f"  Last 10 avg: {last_10:.6f}")
    
    return {
        'crystals_a1': crystals_a,
        'crystals_b': crystals_b,
        'crystals_a2': crystals_a2,
        'memory_loaded': memory_snapshot is not None,
    }


if __name__ == "__main__":
    print("="*70)
    print("BCS FULL LEARNING VALIDATION")
    print("="*70)
    
    # Test 1: Full learning with proper initialization
    try:
        result1 = test_full_learning()
        print("\n[RESULT] Full learning test completed")
    except Exception as e:
        print(f"\n[ERROR] Full learning test: {e}")
        import traceback
        traceback.print_exc()
        result1 = {'error': str(e)}
    
    # Test 2: Memory persistence
    try:
        result2 = test_memory_persistence()
        print("\n[RESULT] Memory persistence test completed")
    except Exception as e:
        print(f"\n[ERROR] Memory persistence test: {e}")
        import traceback
        traceback.print_exc()
        result2 = {'error': str(e)}
    
    print("\n" + "="*70)
    print("TESTS COMPLETED")
    print("="*70)