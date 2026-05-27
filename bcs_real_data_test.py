"""
BCS Stage 1 Real Data Experiments
Testing BCS processing capabilities on real-world text data.
Stage 1: Internal Understanding & Processing (no generation)
"""

import sys
import os
import numpy as np

# Add E:\arc to path so 'bcs' package is found
arc_dir = os.path.dirname(os.path.abspath(__file__))  # E:\arc\bcs
parent_dir = os.path.dirname(arc_dir)  # E:\arc
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bcs.model import BCSModelV6


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def experiment_1_text_structure():
    """E1: Analyze text structure and semantics"""
    print_section("E1: TEXT STRUCTURE ANALYSIS")
    
    with open('E:\\arc\\text.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    data = text_data.encode('utf-8')
    print(f"Loaded {len(data)} bytes of real text")
    
    model = BCSModelV6(
        n_active_bytes=64,
        use_full_tensor=True,
        use_dynamic_embedding=True,
        use_crystallized_memory=True,
        use_working_memory=True,
        use_cluster_recognition=True,
        use_context_resonance=True,
        use_prediction_error_loop=True
    )
    
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)
    
    print(f"\n  Clusters found: {len(results['final_clusters'])}")
    print(f"  Crystals formed: {len(model.crystal_memory.crystals)}")
    
    # Analyze cluster distribution
    clusters = results['final_clusters']
    if clusters:
        sizes = [c['size'] for c in clusters]
        qualities = [c['quality_score'] for c in clusters]
        print(f"  Avg cluster size: {np.mean(sizes):.1f}")
        print(f"  Quality range: [{min(qualities):.3f}, {max(qualities):.3f}]")
        
        # Check pattern groups
        pattern_groups = set(c.get('pattern_group', -1) for c in clusters)
        n_groups = len(pattern_groups) - (1 if -1 in pattern_groups else 0)
        print(f"  Pattern groups detected: {n_groups}")
    
    return results


def experiment_2_modality_detection():
    """E2: Multi-modality detection on mixed content"""
    print_section("E2: MODALITY DETECTION")
    
    # Create mixed content
    text_part = b"Scientific paper abstract about quantum physics"
    json_part = b'{"experiment": "quantum_entanglement", "results": [0.95, 0.87, 0.91]}'
    binary_part = bytes(range(50)) * 10
    mixed_data = text_part + b"<JSON>" + json_part + b"<BIN>" + binary_part + b"<TEXT>" + text_part
    
    model = BCSModelV6(n_active_bytes=64, use_bayesian_modality=True)
    model.ingest(mixed_data)
    
    print(f"  Data composition:")
    print(f"    - Text: {len(text_part)} bytes")
    print(f"    - JSON: {len(json_part)} bytes")
    print(f"    - Binary: {len(binary_part)} bytes")
    print(f"    - Total: {len(mixed_data)} bytes")
    
    print(f"\n  Detected modality: {model.detected_modality}")
    if hasattr(model, 'modality_posteriors') and model.modality_posteriors:
        print(f"  Posteriors:")
        for m, p in sorted(model.modality_posteriors.items(), key=lambda x: -x[1])[:5]:
            print(f"    {m}: {p:.4f}")
    
    # Process and analyze
    model.build_tensors().init_field()
    results = model.run(n_steps=200, record_every=50)
    
    print(f"\n  Processing results:")
    print(f"    Clusters: {len(results['final_clusters'])}")
    print(f"    Boundary indices: {results.get('boundary_indices', [])[:5]}")


def experiment_3_predictive_learning():
    """E3: Predictive coding on structured text"""
    print_section("E3: PREDICTIVE LEARNING")
    
    # Load text and check prediction error convergence
    with open('E:\\arc\\text.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    # Take first paragraph for focused learning
    paragraphs = text_data.split('\n\n')
    focused_text = paragraphs[0] * 10  # Repeat for stronger pattern
    
    data = focused_text.encode('utf-8')
    print(f"Testing predictive learning on {len(data)} bytes (10x repeat)")
    
    model = BCSModelV6(
        n_active_bytes=64,
        use_prediction_error_loop=True,
        use_crystallized_memory=True
    )
    
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=300, record_every=75)
    
    pel = results.get('v6_prediction_error_loop', [])
    if pel:
        errors = [p['mean_error'] for p in pel]
        print(f"\n  Prediction error progression:")
        for i, e in enumerate(errors[:min(5, len(errors))]):
            print(f"    Step {i*75}: {e:.4f}")
        
        print(f"\n  Error reduction: {errors[0]:.4f} -> {errors[-1]:.4f} ({errors[-1]/errors[0]*100:.1f}%)")
        
        if errors[-1] < errors[0]:
            print("  ✓ Prediction learning successful!")
        else:
            print("  ✗ No improvement in prediction")
    
    print(f"\n  Final state:")
    print(f"    Clusters: {len(results['final_clusters'])}")
    print(f"    Crystals: {len(model.crystal_memory.crystals)}")


def experiment_4_clustering_quality():
    """E4: Cluster quality metrics on real text"""
    print_section("E4: CLUSTERING QUALITY")
    
    with open('E:\\arc\\text.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    data = text_data.encode('utf-8')
    
    model = BCSModelV6(n_active_bytes=64)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)
    
    clusters = results['final_clusters']
    print(f"\n  Total clusters: {len(clusters)}")
    
    # Analyze quality distribution
    if clusters:
        quality_scores = [c['quality_score'] for c in clusters]
        sizes = [c['size'] for c in clusters]
        
        print(f"\n  Quality score statistics:")
        print(f"    Min: {min(quality_scores):.4f}")
        print(f"    Max: {max(quality_scores):.4f}")
        print(f"    Mean: {np.mean(quality_scores):.4f}")
        print(f"    Std: {np.std(quality_scores):.4f}")
        
        print(f"\n  Cluster size statistics:")
        print(f"    Min: {min(sizes)}")
        print(f"    Max: {max(sizes)}")
        print(f"    Mean: {np.mean(sizes):.1f}")
        
        # High quality clusters
        high_q = [c for c in clusters if c['quality_score'] > 0.8]
        print(f"\n  High quality clusters (q > 0.8): {len(high_q)}")
        
        # Check for spatial coherence
        coherent = sum(1 for c in clusters if len(c['positions']) > 2 and 
                      len(set(np.diff(c['positions']))) == 1)
        print(f"  Spatially coherent clusters: {coherent}/{len(clusters)}")


def experiment_5_boundary_detection():
    """E5: Detect section boundaries in document"""
    print_section("E5: BOUNDARY DETECTION")
    
    with open('E:\\arc\\text.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    data = text_data.encode('utf-8')
    model = BCSModelV6(n_active_bytes=64)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)
    
    boundaries = results.get('boundary_indices', [])
    print(f"\n  Total boundaries detected: {len(boundaries)}")
    
    if boundaries:
        print(f"\n  First 10 boundary positions: {boundaries[:10]}")
        print(f"\n  Boundary density: {len(boundaries)/len(data)*1000:.2f} per 1000 bytes")
        
        # Map boundaries to text positions
        text_lines = text_data.split('\n')
        line_starts = []
        pos = 0
        for line in text_lines:
            line_starts.append(pos)
            pos += len(line) + 1  # +1 for newline
        
        print(f"\n  Document structure ({len(text_lines)} paragraphs):")
        for i, b in enumerate(boundaries[:5]):
            # Find closest line
            closest_line = min(range(len(line_starts)), key=lambda x: abs(line_starts[x] - b))
            if closest_line < len(text_lines):
                preview = text_lines[closest_line][:50].strip()
                print(f"    Boundary at byte {b}: '{preview}...'")


def experiment_6_memory_consolidation():
    """E6: Test memory consolidation on repeated patterns"""
    print_section("E6: MEMORY CONSOLIDATION")
    
    with open('E:\arc\\worklog.md', 'r', encoding='utf-8') as f:
        worklog_data = f.read()
    
    data = worklog_data.encode('utf-8')
    print(f"Processing worklog: {len(data)} bytes")
    
    model = BCSModelV6(
        n_active_bytes=64,
        use_crystallized_memory=True,
        use_working_memory=True
    )
    
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=500, record_every=125)
    
    print(f"\n  Memory state after processing:")
    print(f"    Working memory buffer: {len(model.working_memory.buffer)} items")
    print(f"    Crystallized patterns: {len(model.crystal_memory.crystals)}")
    
    # Check recognition history
    rec_history = results.get('v7_recognition_history', [])
    if rec_history:
        results_set = set(r['result'] for r in rec_history)
        print(f"\n  Recognition results: {results_set}")
        
        recognized = sum(1 for r in rec_history if r['result'] == 'recognized')
        novel = sum(1 for r in rec_history if r['result'] == 'novel')
        print(f"    Recognized: {recognized}, Novel: {novel}")
    
    # Process again to test recognition
    print(f"\n  Processing again to test recognition...")
    results2 = model.run(n_steps=200, record_every=50)
    
    rec_history2 = results2.get('v7_recognition_history', [])
    if rec_history2:
        results_set2 = set(r['result'] for r in rec_history2)
        print(f"    Second pass results: {results_set2}")
        
        recognized2 = sum(1 for r in rec_history2 if r['result'] == 'recognized')
        print(f"    Recognized on repeat: {recognized2}")


def experiment_7_free_energy_convergence():
    """E7: Free energy minimization over time"""
    print_section("E7: FREE ENERGY CONVERGENCE")
    
    with open('E:\\arc\\text.txt', 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    data = text_data.encode('utf-8')
    
    model = BCSModelV6(n_active_bytes=64, use_prediction_error_loop=True)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=500, record_every=100)
    
    fe_history = results.get('free_energy_over_time', [])
    print(f"\n  Free energy evolution:")
    for i, fe in enumerate(fe_history):
        print(f"    Step {i*100}: {fe:.4f}")
    
    if len(fe_history) >= 2:
        initial_fe = fe_history[0]
        final_fe = fe_history[-1]
        decrease = initial_fe - final_fe
        print(f"\n  FE decrease: {initial_fe:.4f} -> {final_fe:.4f} (Δ={decrease:.4f})")
        
        if final_fe < initial_fe:
            print("  ✓ System minimizing free energy")
        else:
            print("  ! FE stable (system in equilibrium)")


def main():
    print("="*60)
    print("  BCS STAGE 1: REAL DATA EXPERIMENTS")
    print("  Stage 1: Internal Understanding & Processing")
    print("="*60)
    
    try:
        # Core processing tests
        experiment_1_text_structure()
        experiment_2_modality_detection()
        experiment_3_predictive_learning()
        experiment_4_clustering_quality()
        experiment_5_boundary_detection()
        experiment_6_memory_consolidation()
        experiment_7_free_energy_convergence()
        
        print("\n" + "="*60)
        print("  ALL EXPERIMENTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()