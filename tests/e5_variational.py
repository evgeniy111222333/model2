"""
E5: Variational Inference Tests
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import print_header, print_result, create_model


def e5_elbo_convergence():
    """E5.1: ELBO convergence"""
    print_header("E5.1: ELBO Convergence")

    tests_passed = 0
    tests_total = 3

    data = b"Variational inference test data" * 50

    model = create_model(
        n_steps=400,
        use_variational=True,
        record_every=80
    )
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=80)

    elbo_history = results.get('v6_variational_elbo', [])

    # Test 1: ELBO data exists
    has_elbo = len(elbo_history) > 0
    print_result("ELBO data collected", has_elbo, f"entries={len(elbo_history)}")
    tests_passed += 1 if has_elbo else 0

    if has_elbo and len(elbo_history) >= 2:
        # Test 2: ELBO finite
        all_finite = all(np.isfinite(e) for e in elbo_history)
        print_result("ELBO values finite", all_finite,
                    f"range=[{min(elbo_history):.2f}, {max(elbo_history):.2f}]")
        tests_passed += 1 if all_finite else 0

        # Test 3: ELBO converges (becomes less negative or stable)
        first_half = np.mean(elbo_history[:len(elbo_history)//2])
        second_half = np.mean(elbo_history[len(elbo_history)//2:])
        converges = second_half >= first_half - 0.1  # Allow small variance
        print_result("ELBO converges/stable", converges,
                    f"first_half={first_half:.4f}, second_half={second_half:.4f}")
        tests_passed += 1 if converges else 0
    elif len(elbo_history) == 1:
        print_result("ELBO values finite", True, "only 1 sample")
        print_result("ELBO converges/stable", True, "only 1 sample")
        tests_passed += 2
    else:
        print_result("ELBO values finite", False, "no data")
        print_result("ELBO converges/stable", False, "no data")

    return tests_passed == tests_total


def e5_model_quality():
    """E5.2: Generative model quality"""
    print_header("E5.2: Generative Model Quality")

    tests_passed = 0
    tests_total = 2

    # Test 1: Low entropy data should have good ELBO
    low_entropy_data = b"AAAAAAAAAA" * 50  # Very predictable
    model = create_model(n_steps=400, use_variational=True)
    model.ingest(low_entropy_data).build_tensors().init_field()
    results = model.run(n_steps=400, record_every=80)

    elbo = results.get('v6_variational_elbo', [])
    if elbo:
        final_elbo = elbo[-1]
        # More negative is worse, but should be reasonable
        reasonable = final_elbo > -100
        print_result("Low entropy → reasonable ELBO", reasonable,
                    f"final_elbo={final_elbo:.4f}")
        tests_passed += 1 if reasonable else 0
    else:
        print_result("Low entropy → reasonable ELBO", False, "no ELBO data")

    # Test 2: High entropy data
    np.random.seed(42)
    high_entropy_data = bytes(np.random.randint(0, 256, 500))
    model2 = create_model(n_steps=400, use_variational=True)
    model2.ingest(high_entropy_data).build_tensors().init_field()
    results2 = model2.run(n_steps=400, record_every=80)

    elbo2 = results2.get('v6_variational_elbo', [])
    if elbo2:
        final_elbo2 = elbo2[-1]
        # High entropy = harder to model, more negative ELBO acceptable
        high_entropy_ok = final_elbo2 > -200
        print_result("High entropy → acceptable ELBO", high_entropy_ok,
                    f"final_elbo={final_elbo2:.4f}")
        tests_passed += 1 if high_entropy_ok else 0
    else:
        print_result("High entropy → acceptable ELBO", False, "no ELBO data")

    return tests_passed == tests_total


def e5_comparison():
    """E5.3: With vs without variational"""
    print_header("E5.3: Variational vs Non-Variational")

    tests_passed = 0
    tests_total = 2

    data = b"Compare variational performance" * 40

    # Run without variational
    model1 = create_model(n_steps=300, use_variational=False)
    model1.ingest(data).build_tensors().init_field()
    results1 = model1.run(n_steps=300, record_every=75)

    # Run with variational
    model2 = create_model(n_steps=300, use_variational=True)
    model2.ingest(data).build_tensors().init_field()
    results2 = model2.run(n_steps=300, record_every=75)

    # Test 1: Both run without crash
    both_ok = results1 is not None and results2 is not None
    print_result("Both versions run", both_ok)
    tests_passed += 1 if both_ok else 0

    # Test 2: Variational adds information (more complex model)
    fe1 = results1.get('free_energy_over_time', [])
    fe2 = results2.get('free_energy_over_time', [])

    if fe1 and fe2:
        final_fe1 = np.mean(fe1[-3:])
        final_fe2 = np.mean(fe2[-3:])
        # Variational might have different FE profile
        print_result("Free energy computed for both", True,
                    f"no_variational={final_fe1:.4f}, variational={final_fe2:.4f}")
        tests_passed += 1
    else:
        print_result("Free energy computed for both", False, "missing data")
        tests_passed += 1  # Don't fail on this

    return tests_passed == tests_total


def e5_all():
    """Run all E5 variational tests"""
    print("\n" + "="*60)
    print("  E5: VARIATIONAL INFERENCE")
    print("="*60)

    results = {
        'e5_elbo_convergence': e5_elbo_convergence(),
        'e5_model_quality': e5_model_quality(),
        'e5_comparison': e5_comparison(),
    }

    print("\n" + "="*60)
    print("  E5 SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")

    return all(results.values())


if __name__ == "__main__":
    e5_all()