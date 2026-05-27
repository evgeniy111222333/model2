"""
E1: Modality Detection Tests
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import print_header, print_result, create_model, run_model


def e1_ascii_text():
    """E1.1: ASCII text detection"""
    print_header("E1.1: ASCII Text Modality")

    tests_passed = 0
    tests_total = 3

    test_cases = [
        ("Plain text", b"Hello world, this is a test message with common words!"),
        ("Repeated text", b"The quick brown fox jumps over the lazy dog. " * 30),
        ("Low entropy text", b"AAAAAA BBBBBB CCCCCC DDDDDD " * 25),
    ]

    for name, data in test_cases:
        try:
            model = create_model(n_steps=50)
            model.ingest(data)

            modality = model.detected_modality
            entropy = float(model.substrate._shannon_entropy(model.substrate.byte_distribution))

            is_ascii = modality == "text_ascii"
            print_result(f"{name}", is_ascii, f"modality={modality}, entropy={entropy:.2f}")
            tests_passed += 1 if is_ascii else 0
        except Exception as e:
            print_result(f"{name}", False, str(e))

    return tests_passed == tests_total


def e1_binary():
    """E1.2: Binary/sparse data detection"""
    print_header("E1.2: Binary/Sparse Modality")

    tests_passed = 0
    tests_total = 3

    test_cases = [
        ("High null ratio", bytes([0x00] * 100 + [0xFF] * 50 + list(range(50)))),
        ("Mixed binary", bytes([0x00, 0x01, 0x02] * 200 + [0xFF] * 100)),
        ("Structured numbers", bytes(range(10)) * 100 + bytes([0x00] * 50)),
    ]

    for name, data in test_cases:
        try:
            model = create_model(n_steps=50)
            model.ingest(data)

            modality = model.detected_modality
            null_ratio = float(model.substrate.byte_distribution[0])

            # Not text_ascii
            is_binary = modality != "text_ascii"
            print_result(f"{name}", is_binary,
                        f"modality={modality}, null_ratio={null_ratio:.2f}")
            tests_passed += 1 if is_binary else 0
        except Exception as e:
            print_result(f"{name}", False, str(e))

    return tests_passed == tests_total


def e1_mixed():
    """E1.3: Ambiguous/mixed data"""
    print_header("E1.3: Mixed/Ambiguous Data")

    tests_passed = 0
    tests_total = 3

    test_cases = [
        ("High entropy random", bytes(np.random.randint(0, 256, 500))),
        ("All bytes equal", bytes(range(256)) * 2),
        ("Medium entropy", bytes([i % 50 for i in range(500)])),
    ]

    for name, data in test_cases:
        try:
            model = create_model(n_steps=50)
            model.ingest(data)

            modality = model.detected_modality
            entropy = float(model.substrate._shannon_entropy(model.substrate.byte_distribution))
            unique = int(np.count_nonzero(model.substrate.byte_distribution))

            # Should not crash on any modality
            print_result(f"{name}", True,
                        f"modality={modality}, entropy={entropy:.2f}, unique={unique}")
            tests_passed += 1
        except Exception as e:
            print_result(f"{name}", False, str(e))

    return tests_passed == tests_total


def e1_bayesian():
    """E1.4: Bayesian modality detector"""
    print_header("E1.4: Bayesian Modality Detection")

    tests_passed = 0
    tests_total = 3

    if not hasattr(BCSModelV6, '__init__'):
        print_result("Bayesian available", False, "Model init issue")
        return False

    test_cases = [
        ("ASCII text", b"Simple test message here", "text_ascii"),
        ("Binary with nulls", bytes([0x00] * 50 + [0x01] * 50), "sparse_binary"),
        ("Structured", bytes(range(20)) * 10, "structured_data"),
    ]

    for name, data, expected in test_cases:
        try:
            model = create_model(
                use_bayesian_modality=True,
                n_steps=50
            )
            model.ingest(data)

            # Check posteriors exist
            has_posteriors = hasattr(model, 'modality_posteriors') and model.modality_posteriors

            if has_posteriors:
                top_modality = max(model.modality_posteriors.items(), key=lambda x: x[1])[0]
                matches = top_modality == expected
                print_result(f"{name}", matches,
                            f"expected={expected}, got={top_modality}, posteriors={model.modality_posteriors}")
            else:
                matches = model.detected_modality == expected
                print_result(f"{name}", matches,
                            f"fallback modality={model.detected_modality}")

            tests_passed += 1 if matches else 0
        except Exception as e:
            print_result(f"{name}", False, str(e))

    return tests_passed == tests_total


def e1_all():
    """Run all E1 modality tests"""
    print("\n" + "="*60)
    print("  E1: MODALITY DETECTION")
    print("="*60)

    results = {
        'e1_ascii_text': e1_ascii_text(),
        'e1_binary': e1_binary(),
        'e1_mixed': e1_mixed(),
        'e1_bayesian': e1_bayesian(),
    }

    print("\n" + "="*60)
    print("  E1 SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")

    return all(results.values())


if __name__ == "__main__":
    e1_all()