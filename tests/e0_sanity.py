"""
E0: Sanity Checks - Basic system functionality
"""

import numpy as np
import sys
import os

# Add grandparent (E:\arc) to path so 'bcs' package works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import print_header, print_result, create_model, run_model


def e0_empty():
    """E0.1: Empty/single byte input"""
    print_header("E0.1: Empty input handling")

    tests_passed = 0
    tests_total = 3

    # Test 1: Empty data
    try:
        model = create_model(n_steps=50)
        model.ingest(b'')
        model.build_tensors()
        model.init_field()
        stats = model.field.get_field_statistics()
        print_result("Empty bytes accepted", True, f"stats: {stats['u_mean']:.4f}")
        tests_passed += 1
    except Exception as e:
        print_result("Empty bytes accepted", False, str(e))

    # Test 2: Single byte
    try:
        model = create_model(n_steps=50)
        model.ingest(b'X')
        model.build_tensors()
        model.init_field()
        model.field.step()
        print_result("Single byte accepted", True, f"length={model.substrate.length}")
        tests_passed += 1
    except Exception as e:
        print_result("Single byte accepted", False, str(e))

    # Test 3: Very short data
    try:
        data = bytes(range(10))
        model = create_model(n_steps=50)
        model.ingest(data)
        model.build_tensors()
        model.init_field()
        for _ in range(20):
            model.field.step()
        print_result("Short data (10 bytes)", True, f"length={model.substrate.length}")
        tests_passed += 1
    except Exception as e:
        print_result("Short data", False, str(e))

    return tests_passed == tests_total


def e0_noise():
    """E0.2: High entropy random data"""
    print_header("E0.2: Random noise input")

    tests_passed = 0
    tests_total = 3

    # Test 1: Pure random (uniform distribution)
    try:
        np.random.seed(42)
        data = bytes(np.random.randint(0, 256, size=500))
        model = create_model(n_steps=100)
        model.ingest(data)
        model.build_tensors()
        model.init_field()

        for _ in range(100):
            model.field.step()

        # Check no NaN/Inf
        has_nan = np.any(np.isnan(model.field.Phi))
        has_inf = np.any(np.isinf(model.field.Phi))
        stable = not has_nan and not has_inf

        print_result("Uniform random stable", stable,
                    f"nan={has_nan}, inf={has_inf}")
        tests_passed += 1 if stable else 0
    except Exception as e:
        print_result("Uniform random stable", False, str(e))

    # Test 2: All same byte
    try:
        data = bytes([0x41] * 500)
        model = create_model(n_steps=100)
        model.ingest(data)
        model.build_tensors()
        model.init_field()

        for _ in range(100):
            model.field.step()

        print_result("All same byte stable", True,
                    f"unique_bytes={len(model.substrate.byte_distribution[model.substrate.byte_distribution > 0])}")
        tests_passed += 1
    except Exception as e:
        print_result("All same byte stable", False, str(e))

    # Test 3: Full byte range
    try:
        data = bytes(range(256)) * 10  # Each byte 10 times
        model = create_model(n_steps=100)
        model.ingest(data)
        model.build_tensors()
        model.init_field()

        for _ in range(100):
            model.field.step()

        unique = int(np.count_nonzero(model.substrate.byte_distribution))
        print_result("Full byte range", True, f"unique={unique}/256")
        tests_passed += 1
    except Exception as e:
        print_result("Full byte range", False, str(e))

    return tests_passed == tests_total


def e0_field_dynamics():
    """E0.3: Field evolution basic checks"""
    print_header("E0.3: Field dynamics")

    tests_passed = 0
    tests_total = 4

    data = b"Test string for field dynamics" * 20
    model = create_model(n_steps=200)
    model.ingest(data).build_tensors().init_field()

    u_means = []
    phi_means = []
    fe_values = []

    for step in range(200):
        model.field.step()
        if step % 50 == 0:
            u_means.append(float(np.mean(model.field.u)))
            phi_means.append(float(np.mean(model.field.Phi)))
            fe_values.append(model.field.compute_free_energy(1.0))

    # Test 1: u_mean changes over time (not stuck)
    u_changes = len(set(round(u, 4) for u in u_means))
    u_ok = u_changes > 1
    print_result("u_mean changes over time", u_ok, f"{u_changes} unique values")
    tests_passed += 1 if u_ok else 0

    # Test 2: Phi evolves (not stuck at init)
    phi_changes = len(set(round(p, 3) for p in phi_means))
    phi_ok = phi_changes > 1
    print_result("Phi evolves", phi_ok, f"{phi_changes} unique values")
    tests_passed += 1 if phi_ok else 0

    # Test 3: Free energy finite
    fe_finite = all(np.isfinite(f) for f in fe_values)
    print_result("Free energy finite", fe_finite)
    tests_passed += 1 if fe_finite else 0

    # Test 4: u values in reasonable range
    u_reasonable = all(abs(u) < 100 for u in u_means)  # Not exploding
    print_result("u values reasonable", u_reasonable, f"range=[{min(u_means):.2f}, {max(u_means):.2f}]")
    tests_passed += 1 if u_reasonable else 0

    return tests_passed == tests_total


def e0_all():
    """Run all E0 sanity tests"""
    print("\n" + "="*60)
    print("  E0: SANITY CHECKS")
    print("="*60)

    results = {
        'e0_empty': e0_empty(),
        'e0_noise': e0_noise(),
        'e0_field_dynamics': e0_field_dynamics(),
    }

    print("\n" + "="*60)
    print("  E0 SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")

    return all(results.values())


if __name__ == "__main__":
    e0_all()