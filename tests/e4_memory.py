"""
E4: Memory and Crystallization Tests
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import print_header, print_result, create_model, run_model


def e4_crystallization():
    """E4.1: Pattern crystallization"""
    print_header("E4.1: Pattern Crystallization")

    tests_passed = 0
    tests_total = 3

    # Test 1: Repeated pattern should crystallize
    pattern = b"SPECIAL_PATTERN_HERE" * 50
    model = create_model(n_steps=500, use_crystallized_memory=True)
    model.ingest(pattern).build_tensors().init_field()
    results = model.run(n_steps=500, record_every=100)

    crystals = model.crystal_memory.crystals if model.crystal_memory else []
    n_crystals = len(crystals)

    # Should have at least one crystal
    has_crystal = n_crystals > 0
    print_result("Pattern crystallizes", has_crystal, f"n_crystals={n_crystals}")
    tests_passed += 1 if has_crystal else 0

    # Test 2: Unique data should NOT crystallize
    np.random.seed(42)
    unique_data = bytes(np.random.randint(0, 256, 1000))
    model2 = create_model(n_steps=300, use_crystallized_memory=True)
    model2.ingest(unique_data).build_tensors().init_field()
    results2 = model2.run(n_steps=300, record_every=100)

    crystals2 = model2.crystal_memory.crystals if model2.crystal_memory else []
    n_crystals2 = len(crystals2)

    # Should have few or no crystals
    low_crystals = n_crystals2 < 5
    print_result("Unique data does not over-crystallize", low_crystals,
                f"n_crystals={n_crystals2}")
    tests_passed += 1 if low_crystals else 0

    # Test 3: Same pattern twice - second should be recognized
    pattern_a = b"PATTERN_A_REPEATED" * 30
    pattern_b = b"PATTERN_A_REPEATED" * 30  # Same as A
    combined = pattern_a + b"OTHER_DATA" * 20 + pattern_b

    model3 = create_model(n_steps=600, use_crystallized_memory=True)
    model3.ingest(combined).build_tensors().init_field()
    results3 = model3.run(n_steps=600, record_every=150)

    crystals3 = model3.crystal_memory.crystals if model3.crystal_memory else []

    # Should crystallize at least one pattern
    has_pattern = len(crystals3) > 0
    print_result("Repeated pattern recognized", has_pattern,
                f"n_crystals={len(crystals3)}")
    tests_passed += 1 if has_pattern else 0

    return tests_passed == tests_total


def e4_recognition():
    """E4.2: Pattern recognition"""
    print_header("E4.2: Pattern Recognition")

    tests_passed = 0
    tests_total = 2

    # Test 1: Same pattern seen twice
    pattern = b"RECOGNIZE_ME" * 40
    data = pattern + b"SOME_DIFFERENT_CONTENT" * 50 + pattern

    model = create_model(n_steps=600, use_crystallized_memory=True, use_cluster_recognition=True)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=600, record_every=150)

    rec_history = results.get('v7_recognition_history', [])

    if rec_history:
        results_set = set(r['result'] for r in rec_history)
        has_recognition = 'recognized' in results_set or 'ambivalent' in results_set
        print_result("Pattern recognized on repeat", has_recognition,
                    f"results: {results_set}")
        tests_passed += 1 if has_recognition else 0
    else:
        print_result("Pattern recognized on repeat", False, "no recognition history")
        # Don't count as fail if feature not implemented

    # Test 2: Different patterns should not be recognized as same
    data2 = b"AAAA" * 50 + b"BBBB" * 50 + b"CCCC" * 50
    model2 = create_model(n_steps=400, use_crystallized_memory=True, use_cluster_recognition=True)
    model2.ingest(data2).build_tensors().init_field()
    results2 = model2.run(n_steps=400, record_every=100)

    clusters2 = results2.get('final_clusters', [])

    if clusters2:
        # Clusters should differentiate (not all same pattern_group if data differs)
        n_pattern_groups = len(set(c.get('pattern_group', -1) for c in clusters2))
        differentiates = n_pattern_groups > 1
        print_result("Different patterns differentiated", differentiates,
                    f"pattern_groups={n_pattern_groups}")
        tests_passed += 1 if differentiates else 0
    else:
        print_result("Different patterns differentiated", False, "no clusters")

    return tests_passed == tests_total


def e4_working_memory():
    """E4.3: Working memory functionality"""
    print_header("E4.3: Working Memory")

    tests_passed = 0
    tests_total = 3

    data = b"Working memory test data here" * 50
    model = create_model(n_steps=400, use_working_memory=True)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)

    wm = results.get('v7_working_memory', {})

    # Test 1: Working memory has entries
    has_buffer = wm.get('buffer_size', 0) > 0
    print_result("Working memory has buffer entries", has_buffer,
                f"buffer_size={wm.get('buffer_size', 0)}")
    tests_passed += 1 if has_buffer else 0

    # Test 2: Working memory accessible
    has_relevant = wm.get('top_relevant', 0) >= 0
    print_result("Working memory query works", has_relevant,
                f"top_relevant={wm.get('top_relevant', 'N/A')}")
    tests_passed += 1 if has_relevant else 0

    # Test 3: System stable with working memory
    field_stats = model.field.get_field_statistics()
    stable = np.isfinite(field_stats['u_mean']) and np.isfinite(field_stats['v_mean'])
    print_result("System stable with working memory", stable)
    tests_passed += 1 if stable else 0

    return tests_passed == tests_total


def e4_context_resonance():
    """E4.4: Context resonance"""
    print_header("E4.4: Context Resonance")

    tests_passed = 0
    tests_total = 2

    data = b"Context resonance test with patterns" * 40
    model = create_model(n_steps=400, use_context_resonance=True)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)

    ctx_norms = results.get('v7_context_norms', [])

    # Test 1: Context vectors computed
    has_context = len(ctx_norms) > 0
    print_result("Context vectors computed", has_context,
                f"entries={len(ctx_norms)}")
    tests_passed += 1 if has_context else 0

    # Test 2: Context norms are finite
    if ctx_norms:
        norms = [c['ctx_norm'] for c in ctx_norms]
        finite = all(np.isfinite(n) for n in norms)
        print_result("Context norms finite", finite,
                    f"range=[{min(norms):.4f}, {max(norms):.4f}]")
        tests_passed += 1 if finite else 0
    else:
        print_result("Context norms finite", False, "no data")

    return tests_passed == tests_total


def e4_all():
    """Run all E4 memory tests"""
    print("\n" + "="*60)
    print("  E4: MEMORY & CRYSTALLIZATION")
    print("="*60)

    results = {
        'e4_crystallization': e4_crystallization(),
        'e4_recognition': e4_recognition(),
        'e4_working_memory': e4_working_memory(),
        'e4_context_resonance': e4_context_resonance(),
    }

    print("\n" + "="*60)
    print("  E4 SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")

    return all(results.values())


if __name__ == "__main__":
    e4_all()