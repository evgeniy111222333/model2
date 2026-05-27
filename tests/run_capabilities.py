"""
BCS Stage 1 Capabilities Test Suite Runner
Discovers and runs all 100 capability tests under tests/stage1_capabilities/
and compiles a detailed report.
"""

import sys
import os
import time
import json
import importlib.util
import numpy as np
from typing import Dict, List, Tuple

# Add bcs and bcs/tests directories to path for imports
bcs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # E:\arc\bcs
sys.path.insert(0, bcs_dir)
sys.path.insert(0, os.path.dirname(bcs_dir)) # E:\arc

def run_all_tests() -> Tuple[List[Dict], float]:
    print("\n" + "="*80)
    print("  BCS STAGE 1: SYSTEM CAPABILITIES RUNNER")
    print("="*80)
    
    start_time = time.time()
    
    # Generate tests first to ensure they exist
    print("Generating capability test files...")
    import tests.generate_all_tests
    
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stage1_capabilities')
    test_files = sorted([f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')])
    
    print(f"Found {len(test_files)} capability tests.")
    
    results = []
    
    for filename in test_files:
        test_path = os.path.join(tests_dir, filename)
        test_id = int(filename.split('_')[1].split('.')[0])
        
        # Load and run the test dynamically
        print(f"\n[{test_id:03d}/100] Loading {filename}...")
        try:
            spec = importlib.util.spec_from_file_location(f"test_{test_id:03d}", test_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            test_start = time.time()
            passed = module.run_test()
            test_duration = time.time() - test_start
            
            # Read the written JSON file for metrics
            json_path = f"test_results/cap_test_{test_id:03d}.json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    test_metrics = json.load(f)
            else:
                test_metrics = {
                    "test_id": test_id,
                    "test_name": filename,
                    "success": passed,
                    "warning": "JSON metrics file not found"
                }
            
            test_metrics["duration_seconds"] = test_duration
            results.append(test_metrics)
            
        except Exception as e:
            print(f"[FAIL] Error running {filename}: {e}")
            results.append({
                "test_id": test_id,
                "test_name": filename,
                "success": False,
                "error": str(e),
                "duration_seconds": 0.0
            })
            
    total_duration = time.time() - start_time
    return results, total_duration

def generate_report(results: List[Dict], duration: float):
    print("\n" + "="*80)
    print("  COMPILING DETAILED CAPABILITIES REPORT")
    print("="*80)
    
    passed_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0
    
    # Categorize results
    categories = {
        "Modality Detection": [],
        "Field Dynamics & Stability": [],
        "Predictive Coding & Errors": [],
        "Structure & Boundary Detection": [],
        "Clustering & Self-Organization": [],
        "Variational Inference & ELBO": [],
        "Information Bottleneck (IB)": [],
        "Memory Systems": [],
        "Context & Resonance": [],
        "Fisher Geometry & Optimization": []
    }
    
    for r in results:
        tid = r["test_id"]
        name = r.get("test_name", f"Test {tid:03d}")
        
        if 1 <= tid <= 10:
            categories["Modality Detection"].append(r)
        elif 11 <= tid <= 20:
            categories["Field Dynamics & Stability"].append(r)
        elif 21 <= tid <= 30:
            categories["Predictive Coding & Errors"].append(r)
        elif 31 <= tid <= 40:
            categories["Structure & Boundary Detection"].append(r)
        elif 41 <= tid <= 50:
            categories["Clustering & Self-Organization"].append(r)
        elif 51 <= tid <= 60:
            categories["Variational Inference & ELBO"].append(r)
        elif 61 <= tid <= 70:
            categories["Information Bottleneck (IB)"].append(r)
        elif 71 <= tid <= 80:
            categories["Memory Systems"].append(r)
        elif 81 <= tid <= 90:
            categories["Context & Resonance"].append(r)
        elif 91 <= tid <= 100:
            categories["Fisher Geometry & Optimization"].append(r)
            
    # Write summary markdown report
    report_path = "test_results/capabilities_summary_report.md"
    os.makedirs("test_results", exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BCS Stage 1 Capabilities Verification Report\n\n")
        f.write(f"- **Pass Rate**: {passed_count}/{total_count} ({pass_rate:.1f}%)\n")
        f.write(f"- **Total Execution Time**: {duration:.2f} seconds\n\n")
        
        f.write("## Category Summaries\n\n")
        f.write("| Category | Pass Rate | Avg Duration (s) |\n")
        f.write("| --- | --- | --- |\n")
        
        for cat_name, cat_tests in categories.items():
            cat_passed = sum(1 for r in cat_tests if r.get('success', False))
            cat_total = len(cat_tests)
            cat_rate = (cat_passed / cat_total * 100) if cat_total > 0 else 0.0
            cat_avg_dur = np.mean([r.get('duration_seconds', 0.0) for r in cat_tests]) if cat_tests else 0.0
            f.write(f"| {cat_name} | {cat_passed}/{cat_total} ({cat_rate:.1f}%) | {cat_avg_dur:.3f} |\n")
            
        f.write("\n## Detailed Capabilities Metrics\n\n")
        
        for cat_name, cat_tests in categories.items():
            f.write(f"### {cat_name}\n\n")
            f.write("| ID | Test Name | Status | Metrics / Error |\n")
            f.write("| --- | --- | --- | --- |\n")
            
            for r in cat_tests:
                status = "🟢 PASS" if r.get('success', False) else "🔴 FAIL"
                tid = r["test_id"]
                name = r.get("test_name", f"Test {tid:03d}")
                
                # Format metrics or error
                if "error" in r:
                    details = f"Error: {r['error']}"
                else:
                    exclude_keys = {"test_id", "test_name", "description", "success", "duration_seconds"}
                    metrics_subset = {k: v for k, v in r.items() if k not in exclude_keys}
                    details = ", ".join(f"{k}={v}" for k, v in metrics_subset.items())
                    
                f.write(f"| {tid:03d} | {name} | {status} | {details} |\n")
            f.write("\n")
            
    print(f"\nReport generated successfully at: {report_path}")
    
    # Save overall summary json
    summary_json = {
        "passed": passed_count,
        "total": total_count,
        "pass_rate": pass_rate,
        "duration_seconds": duration,
        "results": results
    }
    with open("test_results/capabilities_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)
        
    print("Summary JSON saved to test_results/capabilities_summary.json")

if __name__ == "__main__":
    results, duration = run_all_tests()
    generate_report(results, duration)
