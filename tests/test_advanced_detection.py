"""
Test: Advanced Modality & Boundary Detection with Information Geometry v2.0
"""
import sys
import os
import numpy as np

sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from bcs.information.geometry import (
    AdvancedModalityDetector,
    AdvancedBoundaryDetector,
    fisher_rao_distance,
    alpha_divergence,
    compute_spectral_signature,
    compute_entropy_profile,
    compute_compression_fingerprint,
)


def test_modality_detection():
    """Test the advanced Information Geometry modality detector."""
    print("\n" + "="*60)
    print("TEST: Info Geometry Modality Detection v2.0")
    print("="*60)
    
    detector = AdvancedModalityDetector(
        use_spectral=True,
        use_compression=True,
        adaptive=True,
    )
    
    # Comprehensive test cases with REALISTIC modality-specific data
    test_cases = [
        ("text_ascii", b"This is a sample text with many words and characters for testing." * 20),
        
        ("text_utf8", b"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd1\x96\xd1\x82! Hello \xd0\xa1\xd0\xb2\xd1\x96\xd1\x82" * 20),
        
        ("image", bytes([i % 256 for i in range(3000)])),  # Gradient = near-uniform (all values)
        
        # Audio: Real audio sine wave - centered at 0x80, varies smoothly
        ("audio", bytes([128 + int(50 * np.sin(i * 0.1)) for i in range(5000)])),  
        
        ("binary", bytes([0] * 2500 + [255] * 500)),  # Sparse: mostly null bytes
        
        # Structured JSON with clear markers (no high bytes)
        ("structured", b'{"key": "value", "num": 42, "flag": true}' * 50),
    ]
    
    results = []
    
    for expected_mod, data in test_cases:
        # Compute byte distribution
        dist = np.zeros(256)
        for b in data:
            dist[b] += 1
        dist = dist / dist.sum()
        
        # Extract features for debug
        spec_feat = compute_spectral_signature(dist)
        ent_feat = compute_entropy_profile(dist)
        
        print(f"\n[{expected_mod}]")
        print(f"  Entropy: {ent_feat['shannon_entropy']:.2f} bits")
        print(f"  Spectral flatness: {spec_feat.get('spectral_flatness', 0):.3f}")
        print(f"  Top5 bytes: {[f'0x{b:02x}' for b in np.argsort(dist)[-5:][::-1]]}")
        
        # Detect modality with raw data
        modality, posteriors = detector.detect(dist, raw_data=data)
        confidence = posteriors.get(modality, 0)
        
        # Sort posteriors for display
        sorted_posts = sorted(posteriors.items(), key=lambda x: -x[1])
        print(f"  Posteriors: " + ", ".join([f"{m}={v:.3f}" for m, v in sorted_posts[:3]]))
        
        # Check if correct
        correct = modality == expected_mod
        print(f"  -> {modality} ({confidence:.3f}) {'OK' if correct else 'FAIL'}")
        
        results.append({
            'expected': expected_mod,
            'detected': modality,
            'confidence': confidence,
            'correct': correct,
        })
    
    # Summary
    n_correct = sum(1 for r in results if r['correct'])
    print(f"\n--- Summary ---")
    print(f"Correct: {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    
    return results


def test_boundary_detection():
    """Test the advanced Geometric Boundary Detector."""
    print("\n" + "="*60)
    print("TEST: Advanced Boundary Detection v2.0")
    print("="*60)
    
    # Create test data with clear boundaries
    data = bytes([0x00] * 50 + [0xFF] * 50 + [0x80] * 50)
    
    class MockSubstrate:
        def __init__(self, data):
            self.length = len(data)
            self.byte_values = np.array(list(data))
            self.one_hot = np.eye(256)[list(data)]
    
    substrate = MockSubstrate(data)
    detector = AdvancedBoundaryDetector(
        scales=[16],
        use_fisher_rao=True,
        use_geodesic=True,
        decay_rate=0.9,
    )
    
    # Detect boundaries
    boundaries = detector.detect_boundary_positions(
        substrate,
        percentile=98.0,  # Very high threshold for clean peaks
        min_gap=5,
    )
    
    print(f"\nData: {len(data)} bytes, 3 blocks of 50 bytes each")
    print(f"Expected boundaries around: 50, 100")
    print(f"Detected boundaries: {boundaries}")
    
    # Check detection accuracy
    expected_boundaries = [50, 100]
    detected_expected = sum(1 for eb in expected_boundaries 
                           if any(abs(b - eb) < 10 for b in boundaries))
    
    print(f"Boundary accuracy: {detected_expected}/{len(expected_boundaries)}")
    
    return {
        'boundaries': boundaries.tolist(),
        'accuracy': detected_expected / len(expected_boundaries),
    }


def test_fisher_rao_metric():
    """Test Fisher-Rao distance computation."""
    print("\n" + "="*60)
    print("TEST: Fisher-Rao Distance Metric")
    print("="*60)
    
    # Uniform distribution
    u = np.ones(256) / 256.0
    
    # Text distribution
    text = np.zeros(256)
    for b in b"This is a test":
        text[b] += 1
    text = text / text.sum()
    
    # Audio distribution (Gaussian)
    x = np.arange(256)
    audio = np.exp(-0.5 * ((x - 128) / 25) ** 2)
    audio = audio / audio.sum()
    
    # Compute distances
    dist_text_audio = fisher_rao_distance(text, audio)
    dist_text_uniform = fisher_rao_distance(text, u)
    dist_audio_uniform = fisher_rao_distance(audio, u)
    
    print(f"Fisher-Rao distances:")
    print(f"  Text -> Audio: {dist_text_audio:.4f}")
    print(f"  Text -> Uniform: {dist_text_uniform:.4f}")
    print(f"  Audio -> Uniform: {dist_audio_uniform:.4f}")
    
    # Verify triangle inequality
    print(f"  Triangle inequality check:")
    print(f"    d(Text,Audio) + d(Audio,Uniform) >= d(Text,Uniform): ", end="")
    if dist_text_audio + dist_audio_uniform >= dist_text_uniform - 0.01:
        print("OK")
    else:
        print("FAIL")
    
    return {'fr_distance': dist_text_audio}


def test_alpha_divergence():
    """Test alpha-divergence family."""
    print("\n" + "="*60)
    print("TEST: Alpha-Divergence Family")
    print("="*60)
    
    p = np.random.rand(256)
    p = p / p.sum()
    q = np.random.rand(256)
    q = q / q.sum()
    
    for alpha in [-0.5, 0.0, 0.5, 0.9, 1.0]:
        div = alpha_divergence(p, q, alpha)
        print(f"  α = {alpha:+.1f}: D = {div:.4f}")
    
    return True


def test_spectral_features():
    """Test spectral signature extraction."""
    print("\n" + "="*60)
    print("TEST: Spectral Signature Extraction")
    print("="*60)
    
    test_cases = [
        ("text_ascii", b"This is a sample text with many words and characters."),
        ("uniform", bytes(range(256))),
        ("sparse", bytes([0] * 100 + [255] * 100)),
    ]
    
    for name, data in test_cases:
        dist = np.zeros(256)
        for b in data:
            dist[b] += 1
        dist = dist / dist.sum()
        
        spec = compute_spectral_signature(dist)
        print(f"\n[{name}]")
        print(f"  DC: {spec['dc_component']:.3f}")
        print(f"  Spectral spread: {spec['spectral_spread']:.3f}")
        print(f"  Low freq ratio: {spec['low_freq_ratio']:.3f}")
        print(f"  Flatness: {spec['spectral_flatness']:.3f}")
        print(f"  N peaks: {spec['n_peaks']}")
    
    return True


def test_compression_fingerprint():
    """Test compression-based fingerprinting."""
    print("\n" + "="*60)
    print("TEST: Compression Fingerprint")
    print("="*60)
    
    test_cases = [
        ("sparse_binary", bytes([0] * 500 + [255] * 500)),
        ("text", b"This is a sample text with many repeated words and patterns." * 50),
        ("random", bytes(np.random.randint(0, 256, 1000))),
    ]
    
    for name, data in test_cases:
        fp = compute_compression_fingerprint(data)
        print(f"\n[{name}]")
        print(f"  Compression ratio: {fp['compression_ratio_mean']:.3f}")
        print(f"  Entropy: {fp['entropy_mean']:.2f} bits")
    
    return True


def test_cross_modality():
    """Test cross-modality distance analysis."""
    print("\n" + "="*60)
    print("TEST: Cross-Modality Distance Analysis")
    print("="*60)
    
    detector = AdvancedModalityDetector()
    
    # Compute centroids
    centroids = detector.centroids
    
    # Compute pairwise Fisher-Rao distances
    modalities = list(centroids.keys())
    n = len(modalities)
    
    print("\nFisher-Rao Distance Matrix:")
    print("         ", end="")
    for m in modalities:
        print(f"{m[:6]:>8s}", end="")
    print()
    
    dist_matrix = np.zeros((n, n))
    for i, m1 in enumerate(modalities):
        print(f"{m1[:8]:>8s}", end="")
        for j, m2 in enumerate(modalities):
            d = fisher_rao_distance(centroids[m1], centroids[m2])
            dist_matrix[i, j] = d
            print(f"{d:>8.4f}", end="")
        print()
    
    print("\nNearest neighbors (by Fisher-Rao):")
    for i, m1 in enumerate(modalities):
        others = [(modalities[j], dist_matrix[i, j]) for j in range(n) if i != j]
        others.sort(key=lambda x: x[1])
        print(f"  {m1}: {others[0][0]} ({others[0][1]:.4f}), {others[1][0]} ({others[1][1]:.4f})")
    
    return {'dist_matrix': dist_matrix}


if __name__ == "__main__":
    print("="*60)
    print("BCS ADVANCED MODALITY & BOUNDARY DETECTION TESTS v2.0")
    print("="*60)
    
    # Run all tests
    test_fisher_rao_metric()
    test_alpha_divergence()
    test_cross_modality()
    test_spectral_features()
    test_compression_fingerprint()
    test_modality_detection()
    test_boundary_detection()
    
    print("\n" + "="*60)
    print("ALL ADVANCED TESTS COMPLETED")
    print("="*60)
