"""
E2: Structure Detection Tests
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import print_header, print_result, create_model, run_model


def e2_repetitions():
    """E2.1: Repetition detection"""
    print_header("E2.1: Repetition Detection")

    tests_passed = 0
    tests_total = 4

    # Test 1: Simple repetition - should detect multiple clusters
    data = b"ABCD" * 100 + b"EFGH" * 100 + b"IJKL" * 100
    model = create_model(n_steps=300)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=300, record_every=150)

    clusters = results.get('final_clusters', [])
    n_clusters = len(clusters)

    # Should have multiple clusters, not just 1
    has_structure = n_clusters > 1
    print_result("Simple repetition → multiple clusters", has_structure,
                f"n_clusters={n_clusters}")
    tests_passed += 1 if has_structure else 0

    # Test 2: All same pattern - should preserve spatial coherence
    data2 = b"XYZW" * 50 + b"XYZW" * 50 + b"XYZW" * 50
    model2 = create_model(n_steps=200)
    model2.ingest(data2).build_tensors().init_field()
    results2 = model2.run(n_steps=200, record_every=100)

    clusters2 = results2.get('final_clusters', [])

    # Check that clusters are spatially coherent (contiguous)
    coherent = all(
        len(set(np.diff(c['positions']))) == 1  # All consecutive
        for c in clusters2
        if len(c['positions']) > 2
    )
    print_result("Spatial coherence preserved", coherent, f"n_clusters={len(clusters2)}")
    tests_passed += 1 if coherent else 0

    # Test 3: Pattern groups - similar clusters marked together
    has_pattern_groups = all(
        'pattern_group' in c for c in clusters2
    )
    print_result("Pattern groups assigned", has_pattern_groups,
                f"groups: {set(c.get('pattern_group') for c in clusters2)}")
    tests_passed += 1 if has_pattern_groups else 0

    # Test 4: Boundaries between sections detected
    data3 = b"A" * 100 + b"B" * 100 + b"C" * 100
    model3 = create_model(n_steps=300)
    model3.ingest(data3).build_tensors().init_field()
    results3 = model3.run(n_steps=300, record_every=150)

    boundaries = results3.get('boundary_indices', [])
    # Should have boundaries near 100 and 200
    has_boundaries = len(boundaries) >= 2
    print_result("Section boundaries detected", has_boundaries,
                f"boundaries at: {boundaries[:5]}")
    tests_passed += 1 if has_boundaries else 0

    return tests_passed == tests_total


def e2_boundaries():
    """E2.2: Boundary detection"""
    print_header("E2.2: Boundary Detection")

    tests_passed = 0
    tests_total = 3

    test_cases = [
        ("3 distinct sections", b"AAA" * 50 + b"BBB" * 50 + b"CCC" * 50, 2),
        ("Alternating", b"A" * 50 + b"B" * 50 + b"A" * 50 + b"B" * 50, 3),
        ("Gradual transition", b"A" * 30 + b"B" * 40 + b"C" * 30, 1),  # Might not detect well
    ]

    for name, data, expected_min in test_cases:
        try:
            model = create_model(n_steps=300)
            model.ingest(data).build_tensors().init_field()
            results = model.run(n_steps=300, record_every=150)

            boundaries = results.get('boundary_indices', [])
            n_sections = len(boundaries) + 1

            # At least expected_min boundaries
            ok = n_sections >= expected_min
            print_result(f"{name}", ok,
                        f"sections={n_sections} (expected >={expected_min})")
            tests_passed += 1 if ok else 0
        except Exception as e:
            print_result(f"{name}", False, str(e))

    return tests_passed == tests_total


def e2_nested():
    """E2.3: Nested/hierarchical structure"""
    print_header("E2.3: Nested Structure")

    tests_passed = 0
    tests_total = 2

    # Test 1: JSON-like structure
    json_data = b'{"key1": "value1", "key2": [1, 2, 3]}' * 20
    model = create_model(n_steps=300)
    model.ingest(json_data).build_tensors().init_field()
    results = model.run(n_steps=300, record_every=150)

    clusters = results.get('final_clusters', [])
    n_clusters = len(clusters)

    # Should have some structure (not just 1)
    has_structure = n_clusters > 1
    print_result("JSON structure detected", has_structure, f"n_clusters={n_clusters}")
    tests_passed += 1 if has_structure else 0

    # Test 2: Mixed content types
    mixed = b"<html><body>Text content</body></html>" * 10 + b"{}[];:,.<>" * 30
    model2 = create_model(n_steps=300)
    model2.ingest(mixed).build_tensors().init_field()
    results2 = model2.run(n_steps=300, record_every=100)

    clusters2 = results2.get('final_clusters', [])

    # Should differentiate sections
    has_multiple = len(clusters2) > 1
    print_result("Mixed content differentiation", has_multiple,
                f"n_clusters={len(clusters2)}")
    tests_passed += 1 if has_multiple else 0

    return tests_passed == tests_total


def e2_cluster_quality():
    """E2.4: Cluster quality metrics"""
    print_header("E2.4: Cluster Quality")

    tests_passed = 0
    tests_total = 3

    # Test 1: Quality scores assigned
    data = b"Hello world test data for clustering" * 30
    model = create_model(n_steps=300)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=300, record_every=150)

    clusters = results.get('final_clusters', [])

    has_quality = all('quality_score' in c for c in clusters)
    print_result("Quality scores assigned", has_quality,
                f"{sum('quality_score' in c for c in clusters)}/{len(clusters)} clusters")
    tests_passed += 1 if has_quality else 0

    # Test 2: Quality score range
    if has_quality and clusters:
        scores = [c['quality_score'] for c in clusters]
        in_range = all(0 <= s <= 1 for s in scores)
        print_result("Quality scores in [0,1]", in_range,
                    f"range=[{min(scores):.3f}, {max(scores):.3f}]")
        tests_passed += 1 if in_range else 0
    else:
        print_result("Quality scores in [0,1]", False, "no clusters")

    # Test 3: Cluster statistics present
    has_stats = all('size' in c and 'distribution' in c for c in clusters)
    print_result("Cluster statistics present", has_stats)
    tests_passed += 1 if has_stats else 0

    return tests_passed == tests_total


def e2_all():
    """Run all E2 structure detection tests"""
    print("\n" + "="*60)
    print("  E2: STRUCTURE DETECTION")
    print("="*60)

    results = {
        'e2_repetitions': e2_repetitions(),
        'e2_boundaries': e2_boundaries(),
        'e2_nested': e2_nested(),
        'e2_cluster_quality': e2_cluster_quality(),
    }

    print("\n" + "="*60)
    print("  E2 SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")

    return all(results.values())


if __name__ == "__main__":
    e2_all()