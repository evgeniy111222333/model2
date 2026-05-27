"""
BCS Practical Applications Test
Test real-world use cases without deep algorithm inspection.
"""
import sys
import os
import numpy as np

sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from tests import create_model


def test_file_type_detection():
    """
    PRACTICAL: Detect file type WITHOUT magic bytes.
    BCS should understand structure, not just byte patterns.
    """
    print("\n" + "="*60)
    print("TEST: File Type Detection (No Magic Bytes)")
    print("="*60)
    
    # Real files
    files = [
        (r"E:\arc\text.txt", "Text file"),
        (r"E:\arc\worklog.md", "Markdown file"),
        (r"E:\arc\bcs\model.py", "Python code"),
    ]
    
    results = []
    for filepath, description in files:
        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filepath} not found")
            continue
            
        with open(filepath, 'rb') as f:
            data = f.read()
        
        print(f"\n[{description}] {os.path.basename(filepath)}")
        print(f"  Size: {len(data)} bytes")
        
        model = create_model(
            use_bayesian_modality=True,
            use_cluster_recognition=True,
            n_active_bytes=64,
        )
        model.ingest(data).build_tensors().init_field()
        results_model = model.run(n_steps=200, record_every=100)
        
        # Key metrics
        modality = results_model.get('detected_modality', 'unknown')
        entropy = results_model.get('substrate_info', {}).get('entropy', 0)
        clusters = len(results_model.get('final_clusters', []))
        
        print(f"  Detected modality: {modality}")
        print(f"  Entropy: {entropy:.2f} bits")
        print(f"  Clusters found: {clusters}")
        
        results.append({
            'file': os.path.basename(filepath),
            'modality': modality,
            'entropy': entropy,
            'clusters': clusters,
        })
    
    return results


def test_anomaly_detection():
    """
    PRACTICAL: Find anomalies in real data.
    BCS should detect unusual patterns through free energy spikes.
    """
    print("\n" + "="*60)
    print("TEST: Anomaly Detection")
    print("="*60)
    
    # Create data with injected anomalies
    base_data = np.tile([10, 20, 30, 40, 50] * 20, dtype=np.uint8)
    
    # Inject anomalies
    anomaly_positions = [50, 100, 150]
    data = base_data.copy()
    for pos in anomaly_positions:
        data[pos] = 255  # Unusual byte
    
    print(f"\nData: {len(data)} bytes, {len(anomaly_positions)} anomalies injected")
    print(f"Anomaly positions: {anomaly_positions}")
    
    model = create_model(
        use_prediction_error_loop=True,
        use_cluster_recognition=True,
        n_active_bytes=32,
    )
    model.ingest(data.tobytes()).build_tensors().init_field()
    
    # Run and track prediction errors
    pel_history = []
    for step in range(300):
        model.field.step()
        
        # Check PEL at anomaly positions
        if step % 20 == 0 and hasattr(model, 'prediction_error_loop'):
            pel = model.prediction_error_loop
            if pel:
                pel_error = pel.get('prediction_error', 0)
                pel_history.append((step, pel_error))
                print(f"  Step {step:3d}: PEL error = {pel_error:.4f}")
    
    # Find where anomalies detected
    print(f"\nPEL history length: {len(pel_history)}")
    
    return {'anomaly_positions': anomaly_positions, 'pel_history': pel_history}


def test_structure_analysis():
    """
    PRACTICAL: Analyze structure of complex data.
    BCS should find boundaries, segments, patterns.
    """
    print("\n" + "="*60)
    print("TEST: Structure Analysis")
    print("="*60)
    
    # Mixed content: header + body + footer pattern
    header = b'<html><body>'
    body = b'This is some content with various bytes.'
    footer = b'</body></html>'
    
    data = header + body + footer
    print(f"\nMixed content: {len(data)} bytes")
    print(f"  Header: {len(header)} bytes")
    print(f"  Body: {len(body)} bytes")
    print(f"  Footer: {len(footer)} bytes")
    
    model = create_model(
        use_boundary_detection=True,
        use_level_splitting=True,
        use_gnn_conversion=True,
        n_active_bytes=64,
    )
    model.ingest(data).build_tensors().init_field()
    
    results = model.run(n_steps=300, record_every=100)
    
    # Get boundaries
    boundaries = results.get('v5_boundaries', [])
    print(f"\nBoundaries detected: {len(boundaries)}")
    if boundaries:
        for b in boundaries[:10]:
            print(f"  Position {b.get('position', '?')}")
    
    # Get clusters
    clusters = results.get('final_clusters', [])
    print(f"Clusters: {len(clusters)}")
    
    # Get levels
    if model.level_splitting:
        print(f"Hierarchy levels: {len(model.level_splitting.levels)}")
    
    return {
        'boundaries': len(boundaries),
        'clusters': len(clusters),
        'levels': len(model.level_splitting.levels) if model.level_splitting else 0,
    }


def test_compression_potential():
    """
    PRACTICAL: Estimate compression potential.
    BCS should identify redundancies that could be compressed.
    """
    print("\n" + "="*60)
    print("TEST: Compression Potential Estimation")
    print("="*60)
    
    # High redundancy
    redundant_data = np.tile([1, 2, 3, 4, 5], 100).tobytes()
    
    # Low redundancy (random-ish)
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 500, dtype=np.uint8).tobytes()
    
    results = []
    
    for name, data in [("Redundant [1,2,3,4,5] x100", redundant_data), 
                        ("Random 500 bytes", random_data)]:
        print(f"\n[{name}]")
        print(f"  Size: {len(data)} bytes")
        
        model = create_model(n_active_bytes=32)
        model.ingest(data).build_tensors().init_field()
        
        # Quick run
        res = model.run(n_steps=100, record_every=50)
        
        entropy = res.get('substrate_info', {}).get('entropy', 0)
        clusters = len(res.get('final_clusters', []))
        
        # Estimate compression ratio
        unique_bytes = len(set(data))
        estimated_ratio = 1.0 * unique_bytes / 256
        
        print(f"  Entropy: {entropy:.2f} bits")
        print(f"  Unique bytes: {unique_bytes}")
        print(f"  Clusters: {clusters}")
        print(f"  Estimated compressible: {100 * (1 - estimated_ratio):.0f}%")
        
        results.append({
            'name': name,
            'entropy': entropy,
            'clusters': clusters,
            'compressible': 100 * (1 - estimated_ratio),
        })
    
    return results


def test_modality_transfer():
    """
    PRACTICAL: How fast does BCS switch modalities?
    """
    print("\n" + "="*60)
    print("TEST: Modality Switching Speed")
    print("="*60)
    
    # Text-like data
    text_data = b'This is a sample text with many repeated words and patterns.'
    
    # Image-like data (grayscale gradient)
    image_data = bytes(range(256)) * 10
    
    # Audio-like (sine wave)
    import math
    audio_data = bytes([int(128 + 127 * math.sin(i * 0.1)) for i in range(1000)])
    
    results = []
    for name, data in [("Text", text_data), ("Image (gradient)", image_data), ("Audio (sine)", audio_data)]:
        print(f"\n[{name}] {len(data)} bytes")
        
        model = create_model(use_bayesian_modality=True, n_active_bytes=64)
        model.ingest(data).build_tensors().init_field()
        
        res = model.run(n_steps=100, record_every=50)
        
        modality = res.get('detected_modality', 'unknown')
        confidence = res.get('substrate_info', {}).get('modality_posteriors', {}).get(modality, 0)
        
        print(f"  Detected: {modality} ({confidence:.2f})")
        
        results.append({'name': name, 'detected': modality, 'confidence': confidence})
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("BCS PRACTICAL APPLICATIONS")
    print("="*70)
    
    # Test 1: File type detection
    try:
        file_results = test_file_type_detection()
        print("\n[OK] File type detection completed")
    except Exception as e:
        print(f"\n[ERROR] File type detection: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Anomaly detection
    try:
        anomaly_results = test_anomaly_detection()
        print("\n[OK] Anomaly detection completed")
    except Exception as e:
        print(f"\n[ERROR] Anomaly detection: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Structure analysis
    try:
        structure_results = test_structure_analysis()
        print("\n[OK] Structure analysis completed")
    except Exception as e:
        print(f"\n[ERROR] Structure analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Compression potential
    try:
        compression_results = test_compression_potential()
        print("\n[OK] Compression potential estimation completed")
    except Exception as e:
        print(f"\n[ERROR] Compression potential: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Modality transfer
    try:
        modality_results = test_modality_transfer()
        print("\n[OK] Modality switching test completed")
    except Exception as e:
        print(f"\n[ERROR] Modality switching: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL PRACTICAL TESTS COMPLETED")
    print("="*70)