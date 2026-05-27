"""
BCS Stage 1 Test Runner

Run all test suites and generate a summary report.
"""

import sys
import os

# Add E:\arc to path FIRST, before any other imports
bcs_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # E:\arc\bcs\tests -> E:\arc
if bcs_parent not in sys.path:
    sys.path.insert(0, bcs_parent)

# Now import test modules
import tests.e0_sanity, tests.e1_modality, tests.e2_structure, tests.e3_predictive, tests.e4_memory, tests.e5_variational


def run_all():
    """Run all test suites."""
    print("\n" + "="*70)
    print("  BCS STAGE 1 TESTS: INTERNAL UNDERSTANDING & PROCESSING")
    print("="*70)

    suites = [
        ("E0: Sanity Checks", tests.e0_sanity.e0_all),
        ("E1: Modality Detection", tests.e1_modality.e1_all),
        ("E2: Structure Detection", tests.e2_structure.e2_all),
        ("E3: Predictive Coding", tests.e3_predictive.e3_all),
        ("E4: Memory & Crystallization", tests.e4_memory.e4_all),
        ("E5: Variational Inference", tests.e5_variational.e5_all),
    ]

    results = {}
    for name, fn in suites:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n[FAIL] ERROR in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status}  {name}")

    print("\n" + "-"*70)
    print(f"  Total: {passed}/{total} suites passed")
    print("="*70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)