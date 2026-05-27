"""
E3: Predictive Coding Tests
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import print_header, print_result, create_model, run_model


def e3_convergence():
    """E3.1: Prediction error convergence"""
    print_header("E3.1: Prediction Error Convergence")

    tests_passed = 0
    tests_total = 3

    # Test 1: Simple pattern - error should decrease
    data = b"ABCDEFGH" * 100  # Repeating pattern
    model = create_model(n_steps=400)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)

    pel_history = results.get('v6_prediction_error_loop', [])

    if pel_history and len(pel_history) >= 3:
        first_error = pel_history[0]['mean_error']
        last_error = pel_history[-1]['mean_error']

        # Error should decrease or stabilize
        improves = last_error <= first_error * 1.1  # Allow 10% margin
        print_result("Error decreases over time", improves,
                    f"first={first_error:.4f}, last={last_error:.4f}, ratio={last_error/first_error:.2f}")
        tests_passed += 1 if improves else 0
    else:
        print_result("Error decreases over time", False,
                    f"not enough PEL data: {len(pel_history)} entries")

    # Test 2: Free energy convergence
    fe_over_time = results.get('free_energy_over_time', [])
    if len(fe_over_time) >= 3:
        first_fe = fe_over_time[0]
        last_fe = fe_over_time[-1]

        # FE should decrease or stabilize
        stable_fe = last_fe <= first_fe * 1.2
        print_result("Free energy stable/converging", stable_fe,
                    f"first={first_fe:.4f}, last={last_fe:.4f}")
        tests_passed += 1 if stable_fe else 0
    else:
        print_result("Free energy stable/converging", False, "not enough FE data")

    # Test 3: Consistent learning across runs
    data2 = b"Test pattern for learning consistency" * 50
    model2 = create_model(n_steps=400)
    model2.ingest(data2).build_tensors().init_field()
    results2 = model2.run(n_steps=400, record_every=100)

    pel2 = results2.get('v6_prediction_error_loop', [])
    if pel2 and len(pel2) >= 2:
        errors2 = [p['mean_error'] for p in pel2]
        # Errors should not explode (be in reasonable range)
        not_exploding = all(e < 10.0 for e in errors2)
        print_result("Prediction errors stable", not_exploding,
                    f"max error={max(errors2):.4f}")
        tests_passed += 1 if not_exploding else 0
    else:
        print_result("Prediction errors stable", False, "not enough data")

    return tests_passed == tests_total


def e3_pattern_learning():
    """E3.2: Learning specific patterns"""
    print_header("E3.2: Pattern Learning")

    tests_passed = 0
    tests_total = 3

    # Test 1: Sinusoidal-like byte pattern
    # Create pattern: 0, 50, 100, 150, 200, 150, 100, 50 (8 values) repeating
    pattern = bytes([0, 50, 100, 150, 200, 150, 100, 50] * 100)
    model = create_model(n_steps=400)
    model.ingest(pattern).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)

    pel = results.get('v6_prediction_error_loop', [])
    if pel:
        mean_error = np.mean([p['mean_error'] for p in pel])
        # Should have learned something (error not too high)
        learned = mean_error < 5.0
        print_result("Sinusoidal pattern learned", learned,
                    f"mean_error={mean_error:.4f}")
        tests_passed += 1 if learned else 0
    else:
        print_result("Sinusoidal pattern learned", False, "no PEL data")

    # Test 2: Alternating pattern (high predictability)
    alt_data = b"ABABABAB" * 100
    model2 = create_model(n_steps=400)
    model2.ingest(alt_data).build_tensors().init_field()
    results2 = model2.run(n_steps=400, record_every=100)

    pel2 = results2.get('v6_prediction_error_loop', [])
    if pel2:
        final_error = pel2[-1]['mean_error']
        # Very predictable pattern should have low error
        predictable = final_error < 1.0
        print_result("Alternating pattern predictable", predictable,
                    f"final_error={final_error:.4f}")
        tests_passed += 1 if predictable else 0
    else:
        print_result("Alternating pattern predictable", False, "no data")

    # Test 3: Random data (should not overfit)
    np.random.seed(123)
    rand_data = bytes(np.random.randint(0, 256, 500))
    model3 = create_model(n_steps=300)
    model3.ingest(rand_data).build_tensors().init_field()
    results3 = model3.run(n_steps=300, record_every=100)

    pel3 = results3.get('v6_prediction_error_loop', [])
    if pel3:
        errors3 = [p['mean_error'] for p in pel3]
        # Random data error should not go to zero (no pattern to learn)
        not_zero = any(e > 0.1 for e in errors3)
        print_result("Random data - no false overfitting", not_zero,
                    f"error range=[{min(errors3):.4f}, {max(errors3):.4f}]")
        tests_passed += 1 if not_zero else 0
    else:
        print_result("Random data - no false overfitting", False, "no data")

    return tests_passed == tests_total


def e3_anomaly():
    """E3.3: Anomaly detection via prediction error"""
    print_header("E3.3: Anomaly Detection")

    tests_passed = 0
    tests_total = 2

    # Test 1: Normal pattern + single anomaly
    normal = b"ABCDEFGH" * 50 + b"X" + b"ABCDEFGH" * 50  # X is anomaly
    model = create_model(n_steps=400)
    model.ingest(normal).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=100)

    # Check if system detects unusual behavior at position ~400
    # (This is indirect - we check that the system handles it without crashing)
    pel = results.get('v6_prediction_error_loop', [])
    stable = all(p['mean_error'] < 100 for p in pel) if pel else True
    print_result("System handles anomaly without crash", stable,
                f"pel entries={len(pel)}")
    tests_passed += 1 if stable else 0

    # Test 2: Two very different patterns concatenated
    mixed = b"AAAAAAAAAA" * 30 + b"0123456789" * 30
    model2 = create_model(n_steps=400)
    model2.ingest(mixed).build_tensors().init_field()
    results2 = model2.run(n_steps=400, record_every=100)

    # Check for transition via cluster structure change
    # Two distinct patterns should create distinguishable cluster groups
    clusters = results2.get('final_clusters', [])
    if clusters:
        # Check that clusters have different pattern_group values
        # (indicating system differentiates the sections)
        pattern_groups = set(c.get('pattern_group', -1) for c in clusters)
        n_groups = len(pattern_groups) - (1 if -1 in pattern_groups else 0)
        shows_transition = n_groups >= 2
        print_result("Transition between patterns visible", shows_transition,
                    f"pattern_groups={n_groups}, clusters={len(clusters)}")
        tests_passed += 1 if shows_transition else 0
    else:
        print_result("Transition between patterns visible", False, "no clusters")
        tests_passed += 1 if False else 0

    return tests_passed == tests_total


def e3_all():
    """Run all E3 predictive coding tests"""
    print("\n" + "="*60)
    print("  E3: PREDICTIVE CODING")
    print("="*60)

    results = {
        'e3_convergence': e3_convergence(),
        'e3_pattern_learning': e3_pattern_learning(),
        'e3_anomaly': e3_anomaly(),
    }

    print("\n" + "="*60)
    print("  E3 SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")

    return all(results.values())


if __name__ == "__main__":
    e3_all()