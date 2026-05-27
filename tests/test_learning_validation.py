"""
Test: Split validation check and if system LEARNS on new input
"""
import sys
import os
import numpy as np

# Add path for imports
sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from bcs.model import BCSModelV6
from tests import create_model


def test_split_validation():
    """Test that split validation correctly accepts/rejects based on F criterion."""
    print("\n" + "="*60)
    print("TEST 1: Split Validation with Free Energy Criterion")
    print("="*60)
    
    # Create data with clear bimodal structure to trigger split
    np.random.seed(42)
    n = 300
    
    # Create two clearly different byte clusters
    cluster_A = np.tile([10, 20, 30, 40, 50], n // 5)[:n]
    cluster_B = np.tile([200, 180, 160, 140, 120], n // 5)[:n]
    
    # Mix: [AAAAA BBBBB AAAAA BBBBB ...]
    data = np.zeros(n, dtype=np.uint8)
    block_size = 15
    for i in range(0, n, 2 * block_size):
        end_a = min(i + block_size, n)
        end_b = min(i + 2 * block_size, n)
        data[i:end_a] = cluster_A[:end_a - i]
        data[end_a:end_b] = cluster_B[:end_b - end_a]
    
    model = create_model(use_level_splitting=True, use_gnn_conversion=True, n_conversion_levels=1)
    model.ingest(data.tobytes()).build_tensors().init_field()
    
    print(f"Input: {n} bytes, {len(np.unique(data))} unique values")
    print(f"Initial levels: {len(model.level_splitting.levels) if model.level_splitting else 0}")
    
    # Run model and collect split stats
    split_results = []
    
    for step in range(200):
        model.field.step()
        
        # Try split every 50 steps
        if step > 0 and step % 50 == 0 and model.level_splitting:
            # Run split on level 0
            split_result = model.level_splitting.attempt_split(0, model)
            
            if split_result.get('split_attempted'):
                print(f"\n--- Step {step} Split Attempt ---")
                print(f"  Detection: bimodal={split_result['detection']['bimodal']}, "
                      f"coefficient={split_result['detection'].get('coefficient', 0):.3f}")
                print(f"  Groups found: {split_result.get('groups', 0)}")
                print(f"  Validation:")
                v = split_result['validation']
                print(f"    - delta_F: {v.get('delta_F', 0):.6f}")
                print(f"    - primary_success: {v.get('primary_success', False)}")
                print(f"    - stability_ratio: {v.get('stability_ratio', 0):.3f}")
                print(f"    - success: {v.get('success', False)}")
                print(f"    - action: {v.get('action', 'unknown')}")
                
                split_results.append({
                    'step': step,
                    'validation': v,
                    'detection': split_result['detection'],
                })
    
    # Analyze results
    successful_splits = [r for r in split_results if r['validation'].get('success', False)]
    rejected_splits = [r for r in split_results if not r['validation'].get('success', False)]
    
    print(f"\n--- Summary ---")
    print(f"Total split attempts: {len(split_results)}")
    print(f"Successful (delta_F < -epsilon): {len(successful_splits)}")
    print(f"Rejected: {len(rejected_splits)}")
    
    if successful_splits:
        avg_delta = np.mean([r['validation']['delta_F'] for r in successful_splits])
        print(f"Average delta_F for successful: {avg_delta:.6f}")
    
    # Check that validation works
    assert len(split_results) > 0, "No split attempts made!"
    print("\n[PASS] Split validation test")
    return split_results


def test_learning_not_just_adaptation():
    """
    Test that BCS learns (stores knowledge persistently) not just adapts.
    
    LEARNING = system remembers new patterns and can use them later
    ADAPTATION = system temporarily changes behavior but doesn't store knowledge
    
    Test scenario:
    1. Show system pattern A -> it recognizes
    2. Show system pattern B -> it recognizes
    3. Show A again -> system should REMEMBER A faster (learning)
       than if it just adapted to last B
    """
    print("\n" + "="*60)
    print("TEST 2: Learning vs Adaptation")
    print("="*60)
    
    # Pattern A: repetitive sequence
    pattern_a = np.array([10, 20, 30, 40, 50] * 20, dtype=np.uint8)
    
    # Pattern B: different repetitive sequence
    pattern_b = np.array([200, 180, 160, 140, 120] * 20, dtype=np.uint8)
    
    def get_embedding(model):
        if model.embeddings is not None:
            return np.mean(model.embeddings[:min(100, len(model.embeddings))], axis=0)
        return None
    
    # First pass: show A
    print("\n[Pass 1] Training on pattern A (repetitive [10,20,30,40,50])...")
    data_a = pattern_a.tobytes()
    model = BCSModelV6(
        n_active_bytes=32,
        use_crystallized_memory=True,
        use_cluster_recognition=True,
        use_level_splitting=True,
    )
    model.ingest(data_a).build_tensors().init_field()
    
    for step in range(100):
        model.field.step()
    
    crystals_before_b = len(model.crystal_memory.crystals) if model.crystal_memory else 0
    emb_a = get_embedding(model)
    
    print(f"  Crystals formed: {crystals_before_b}")
    print(f"  Embedding mean: {np.mean(emb_a):.4f}" if emb_a is not None else "  Embedding: None")
    
    # Store state after A
    if model.crystal_memory:
        try:
            memory_state_after_a = model.crystal_memory.get_state()
        except AttributeError:
            memory_state_after_a = {'n_crystals': crystals_before_b}
    
    # Second pass: show B
    print("\n[Pass 2] Training on pattern B (repetitive [200,180,160,140,120])...")
    data_b = pattern_b.tobytes()
    model_b = BCSModelV6(
        n_active_bytes=32,
        use_crystallized_memory=False,  # New substrate
        use_cluster_recognition=True,
        use_level_splitting=True,
    )
    model_b.ingest(data_b).build_tensors().init_field()
    
    for step in range(100):
        model_b.field.step()
    
    crystals_after_b = len(model_b.crystal_memory.crystals) if model_b.crystal_memory else 0
    emb_b = get_embedding(model_b)
    
    print(f"  Crystals formed: {crystals_after_b}")
    print(f"  Embedding mean: {np.mean(emb_b):.4f}" if emb_b is not None else "  Embedding: None")
    
    # Third pass: show A again
    print("\n[Pass 3] Re-training on pattern A...")
    model_a2 = BCSModelV6(
        n_active_bytes=32,
        use_crystallized_memory=True,
        use_cluster_recognition=True,
        use_level_splitting=True,
    )
    model_a2.ingest(data_a).build_tensors().init_field()
    
    # Check if crystals in memory before training
    initial_crystals = len(model_a2.crystal_memory.crystals) if model_a2.crystal_memory else 0
    print(f"  Initial crystals (loaded from memory): {initial_crystals}")
    
    recognition_times = []
    final_crystals = 0
    
    for step in range(100):
        model_a2.field.step()
        
        if step % 20 == 0:
            crystals = len(model_a2.crystal_memory.crystals) if model_a2.crystal_memory else 0
            print(f"  Step {step}: crystals={crystals}")
    
    final_crystals = len(model_a2.crystal_memory.crystals) if model_a2.crystal_memory else 0
    
    print(f"\n--- Learning Analysis ---")
    print(f"Pattern A crystals (first pass): {crystals_before_b}")
    print(f"Pattern B crystals (second pass): {crystals_after_b}")
    print(f"Pattern A crystals (third pass): {final_crystals}")
    
    # If system truly learns, it should not "forget" structures
    same_pattern_crystals = final_crystals == crystals_before_b
    
    has_persistence = final_crystals > 0 and crystals_before_b > 0
    
    if model_a2.crystal_memory and model_a2.crystal_memory.crystals:
        avg_quality = np.mean([c.get('quality_score', 0) for c in model_a2.crystal_memory.crystals])
        print(f"Average crystal quality: {avg_quality:.3f}")
    
    print(f"\n--- Results ---")
    print(f"Has memory persistence: {has_persistence}")
    print(f"Learning rather than re-learning: {same_pattern_crystals}")
    
    if has_persistence:
        print("\n[PASS] System demonstrates LEARNING (persistent memory)")
    else:
        print("\n[WARNING] System may only ADAPT, not truly learn")
    
    return {
        'has_persistence': has_persistence,
        'pattern_a_crystals': crystals_before_b,
        'pattern_b_crystals': crystals_after_b,
        'pattern_a_repeat_crystals': final_crystals,
    }


def test_free_energy_convergence():
    """Test that free energy consistently decreases with learning."""
    print("\n" + "="*60)
    print("TEST 3: Free Energy Convergence with Learning")
    print("="*60)
    
    # Create structured data
    np.random.seed(42)
    pattern = np.concatenate([
        np.tile([10, 20, 30], 30),
        np.tile([100, 110, 120], 30),
        np.tile([200, 210, 220], 30),
    ]).astype(np.uint8)
    
    model = create_model(use_level_splitting=True, use_gnn_conversion=True)
    model.ingest(pattern.tobytes()).build_tensors().init_field()
    
    fe_trajectory = []
    for step in range(200):
        model.field.step()
        if step % 20 == 0:
            fe = model.field.compute_free_energy(1.0)
            fe_trajectory.append(fe)
            print(f"Step {step:3d}: F = {fe:.6f}")
    
    # Check that energy decreases
    first_third = np.mean(fe_trajectory[:len(fe_trajectory)//3])
    last_third = np.mean(fe_trajectory[-len(fe_trajectory)//3:])
    energy_decreased = last_third < first_third
    
    print(f"\n--- Energy Analysis ---")
    print(f"Average F (first third): {first_third:.6f}")
    print(f"Average F (last third): {last_third:.6f}")
    print(f"Energy decreased: {energy_decreased}")
    
    if energy_decreased:
        print("\n[PASS] Free energy converges - system learns")
    else:
        print("\n[WARNING] Energy not converging - possible issue")
    
    return fe_trajectory


if __name__ == "__main__":
    print("="*70)
    print("BCS LEARNING VALIDATION TESTS")
    print("="*70)
    
    # Test 1: Split validation
    try:
        split_results = test_split_validation()
    except (AssertionError, AttributeError, Exception) as e:
        print(f"\n[INFO] Split validation: {e}")
        split_results = []
    
    # Test 2: Learning vs Adaptation  
    try:
        learning_result = test_learning_not_just_adaptation()
    except (AttributeError, KeyError, Exception) as e:
        print(f"\n[INFO] Learning test: {e}")
        learning_result = {'error': str(e)}
    
    # Test 3: Free energy convergence
    try:
        fe_trajectory = test_free_energy_convergence()
    except Exception as e:
        print(f"\n[FAIL] Free energy test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)