"""
BCS Stage 1: Run All 100 Capability Tests
"""
import sys
import os
import time
import json

# Add E:\arc to path for 'bcs' package
arc_dir = r"E:\arc"
if arc_dir not in sys.path:
    sys.path.insert(0, arc_dir)

os.chdir(r"E:\arc\bcs\tests\stage1_capabilities")

print("="*60)
print("BCS STAGE 1: RUNNING 100 CAPABILITY TESTS")
print("="*60)

passed = 0
failed = 0
errors = []

for i in range(1, 101):
    test_num = str(i).zfill(3)
    test_file = f"test_{test_num}.py"
    
    try:
        # Clear previous imports
        if f'test_{test_num}' in sys.modules:
            del sys.modules[f'test_{test_num}']
        
        # Import and run test
        exec(f"import test_{test_num} as t")
        result = eval(f"t.run_test()")
        
        if result:
            passed += 1
            print(f"  [{test_num}] PASS")
        else:
            failed += 1
            print(f"  [{test_num}] FAIL")
            
    except Exception as e:
        failed += 1
        errors.append(f"{test_num}: {str(e)[:80]}")
        print(f"  [{test_num}] ERROR: {str(e)[:60]}")

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Passed: {passed}/100")
print(f"Failed: {failed}/100")
print(f"Success rate: {passed}%")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors[:10]:
        print(f"  - {e}")

# Save summary
summary = {
    "total": 100,
    "passed": passed,
    "failed": failed,
    "success_rate": passed,
    "errors": errors
}

os.makedirs("test_results", exist_ok=True)
with open("test_results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to test_results/summary.json")
print("="*60)