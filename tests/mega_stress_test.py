"""
BCS Stage 1 Mega Capability Stress Test
Evaluates BCS V6/V7 on:
1. Volume (~50KB mixed data)
2. Depth (5-level nested hierarchy of patterns)
3. Speed (500 steps, Free Energy reduction, crystallization speed)
4. Multimodality (Text, Structured, Binary, Audio switching)
5. Anomalies (12 injected anomalies, detection rate)
6. Memory (Two-pass execution, recognition speed and recall)
"""

import os
import sys
import time
import json
import numpy as np

# Add E:\arc to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Monkey patch modules to support n_conversion_levels = 5 without IndexError
from bcs.information.variational import VariationalInference
from bcs.perception.predictive import HierarchicalPredictiveCoding

original_vi_init = VariationalInference.__init__
def patched_vi_init(self, n_levels=4, d_latent=None, d_observation=256):
    if d_latent is None:
        d_latent = [128, 64, 32, 16]
        while len(d_latent) < n_levels:
            d_latent.append(max(d_latent[-1] // 2, 8))
    else:
        d_latent = list(d_latent)
        while len(d_latent) < n_levels:
            d_latent.append(max(d_latent[-1] // 2, 8))
    original_vi_init(self, n_levels=n_levels, d_latent=d_latent, d_observation=d_observation)

VariationalInference.__init__ = patched_vi_init

original_hpc_init = HierarchicalPredictiveCoding.__init__
def patched_hpc_init(self, n_levels=4, d_representations=None):
    if d_representations is None:
        d_representations = [256, 128, 64, 32]
        while len(d_representations) < n_levels:
            d_representations.append(max(d_representations[-1] // 2, 16))
    else:
        d_representations = list(d_representations)
        while len(d_representations) < n_levels:
            d_representations.append(max(d_representations[-1] // 2, 16))
    original_hpc_init(self, n_levels=n_levels, d_representations=d_representations)

HierarchicalPredictiveCoding.__init__ = patched_hpc_init


from bcs.model import BCSModelV6

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_mixed_data() -> tuple:
    print("Generating mixed dataset (~51KB) with 4 modalities...")
    
    # 1. Text Segment (~12KB)
    # Try reading text.txt, or generate fallback
    text_txt_path = 'E:\\arc\\text.txt'
    if os.path.exists(text_txt_path):
        try:
            with open(text_txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read(12000).encode('utf-8')
            print(f"Loaded {len(text_content)} bytes of real text from text.txt")
        except Exception as e:
            print(f"Failed to read text.txt: {e}, using fallback.")
            text_content = b"BCS cognitive architecture text processing stream. " * 240
    else:
        text_content = b"BCS cognitive architecture text processing stream. " * 240
        print(f"text.txt not found, generated {len(text_content)} bytes fallback text.")
    
    # Ensure text is exactly 12000 bytes
    text_content = text_content[:12000]
    
    # 2. Structured Segment (~15KB)
    # Define 5-level nested hierarchy
    p1 = b"abc"
    p2 = p1 + b"_" + p1                    # abc_abc (len=7)
    p3 = p2 + b"==" + p2                   # abc_abc==abc_abc (len=16)
    p4 = p3 + b"####" + p3                 # abc_abc==abc_abc####abc_abc==abc_abc (len=36)
    p5 = p4 + b"$$$$" + p4                 # abc_abc==abc_abc####abc_abc==abc_abc$$$$abc_abc==abc_abc####abc_abc==abc_abc (len=76)
    
    json_records = []
    # Make total structured segment around 15KB
    for i in range(100):
        rec = f'{{"id":{i:03d},"val":{i*1.5:.2f},"pattern":"{p5.decode()}","valid":true}}'
        json_records.append(rec.encode('utf-8'))
    structured_content = b",\n".join(json_records)
    print(f"Generated {len(structured_content)} bytes of JSON structured content with nested patterns.")
    
    # 3. Binary Segment (~12KB)
    # Binary distribution signature: peaks at 0x00 and 0xFF, others low
    choices = [0x00, 0xFF, -1]
    probs = [0.4, 0.3, 0.3]
    np.random.seed(42)
    indices = np.random.choice(choices, size=12000, p=probs)
    binary_bytes = bytearray()
    for idx in indices:
        if idx == -1:
            binary_bytes.append(np.random.randint(0, 256))
        else:
            binary_bytes.append(idx)
    binary_content = bytes(binary_bytes)
    print(f"Generated {len(binary_content)} bytes of high-contrast binary data.")
    
    # 4. Audio Segment (~12KB)
    # Audio signature: Gaussian distribution centered around 128 (0x80)
    audio_vals = np.random.normal(128, 30, size=12000)
    audio_vals = np.clip(audio_vals, 0, 255).astype(np.uint8)
    audio_content = bytes(audio_vals)
    print(f"Generated {len(audio_content)} bytes of Gaussian centered audio data.")
    
    # Ingest limits: assemble full data
    full_data = text_content + structured_content + binary_content + audio_content
    
    # Calculate boundaries
    boundaries = {
        'text': (0, len(text_content)),
        'structured': (len(text_content), len(text_content) + len(structured_content)),
        'binary': (len(text_content) + len(structured_content), len(text_content) + len(structured_content) + len(binary_content)),
        'audio': (len(text_content) + len(structured_content) + len(binary_content), len(full_data))
    }
    
    return full_data, boundaries

def inject_anomalies(data: bytes, text_boundary: tuple) -> tuple:
    print("Injecting 12 anomalies at specific offsets...")
    # Inject anomalies only in the text segment (which has high predictability)
    start, end = text_boundary
    step = (end - start - 1000) // 12
    true_anomaly_positions = [start + 500 + i * step for i in range(12)]
    
    mutable_data = bytearray(data)
    for pos in true_anomaly_positions:
        # Replace normal text character with 0x00 (rare in text, highly anomalous)
        mutable_data[pos] = 0x00
        
    print(f"Injected anomalies at byte indices: {true_anomaly_positions}")
    return bytes(mutable_data), true_anomaly_positions

def run_modality_sliding_windows(model: BCSModelV6, data: bytes, boundaries: dict) -> list:
    print("\nRunning sliding-window modality detection...")
    window_size = 4000
    step_size = 2000
    classification_results = []
    
    detector = model.modality_detector
    if detector is None:
        print("Modality detector is disabled/missing.")
        return []
        
    for offset in range(0, len(data) - window_size + 1, step_size):
        window_data = data[offset : offset + window_size]
        center = offset + window_size // 2
        
        # Determine true modality
        true_mod = "unknown"
        for mod, (start, end) in boundaries.items():
            if start <= center < end:
                true_mod = mod
                break
        
        # If true_mod is text, map to text_ascii or text_utf8 signature
        if true_mod == 'text':
            true_mod = 'text_ascii' # simple mapping
            
        # Get local distribution
        counts = np.bincount(np.array(list(window_data), dtype=np.uint8), minlength=256)
        dist = counts.astype(np.float64) / len(window_data)
        
        detected_mod, posteriors = detector.detect(dist, N=len(window_data))
        
        # Check correctness (text matches either ascii or utf8)
        is_correct = False
        if true_mod == 'text_ascii' and detected_mod in ['text_ascii', 'text_utf8']:
            is_correct = True
        elif true_mod == 'structured' and detected_mod == 'structured':
            is_correct = True
        elif true_mod == 'binary' and detected_mod == 'binary':
            is_correct = True
        elif true_mod == 'audio' and detected_mod == 'audio':
            is_correct = True
            
        classification_results.append({
            'offset': offset,
            'window_center': center,
            'true_modality': true_mod,
            'detected_modality': detected_mod,
            'is_correct': is_correct,
            'posteriors': {k: float(v) for k, v in posteriors.items()}
        })
        
    correct_count = sum(1 for c in classification_results if c['is_correct'])
    accuracy = correct_count / len(classification_results) if classification_results else 0.0
    print(f"Modality Classification Accuracy: {accuracy*100:.1f}% ({correct_count}/{len(classification_results)})")
    
    return classification_results

def evaluate_anomalies(model: BCSModelV6, true_positions: list) -> tuple:
    print("\nEvaluating anomaly detection rate...")
    # Calculate adaptive threshold based on prediction errors
    errors, _ = model.pc.compute_prediction_error(model.field.u)
    std_err = np.std(errors)
    anomaly_threshold = model.numeric_policy.predictive_anomaly_threshold(errors, prior=1.5)
    
    detected_indices = model.pc.detect_anomalies(model.field.u, threshold=anomaly_threshold)
    
    # We count as detected if a detected index is within +/- 8 bytes of a true position
    detected_count = 0
    hits = []
    
    for pos in true_positions:
        found = False
        for det_idx in detected_indices:
            if abs(det_idx - pos) <= 8:
                found = True
                hits.append(int(det_idx))
                break
        if found:
            detected_count += 1
            
    rate = detected_count / len(true_positions)
    print(f"Detected {detected_count} out of {len(true_positions)} anomalies ({rate*100:.1f}%)")
    return rate, hits, detected_indices.tolist()

def main():
    print("="*80)
    print("  BCS STAGE 1: MEGA CAPABILITY STRESS TEST")
    print("="*80)
    
    # 1. Generate mixed data and inject anomalies
    raw_data, boundaries = generate_mixed_data()
    data, true_anomaly_positions = inject_anomalies(raw_data, boundaries['text'])
    
    # 2. Instantiate Model with n_conversion_levels=5
    print("\nInitializing BCS model...")
    model = BCSModelV6(
        n_conversion_levels=5,
        n_active_bytes=32,  # Optimized for CPU stress run speed
        use_variational=True,
        use_ib_optimizer=True,
        use_multiscale_opt=False,  # Keep offline
        use_hierarchical_pc=True,
        use_prediction_error_loop=True,
        use_crystallized_memory=True,
        use_working_memory=True,
        use_cluster_recognition=True,
        use_context_resonance=True,
        use_knowledge_transfer=True,
        use_level_splitting=True,
        use_sequence_memory=True,
        use_semantic_dynamics=True,
        use_semantic_readout=True,
        device='cpu'
    )
    
    # Ingest and initialize field
    model.ingest(data).build_tensors().init_field()
    
    # 3. PASS 1: Learning & Crystallization
    print("\n" + "-"*50)
    print("  PASS 1: LEARNING & CRYSTALLIZATION")
    print("-"*50)
    
    start_p1 = time.time()
    results_p1 = model.run(n_steps=500, record_every=50)
    dur_p1 = time.time() - start_p1
    
    print(f"Pass 1 finished in {dur_p1:.2f} seconds.")
    print(f"Pass 1 crystals consolidated: {len(model.crystal_memory.crystals)}")
    
    # 4. Sliding-window modality switching test
    window_classifications = run_modality_sliding_windows(model, data, boundaries)
    modality_accuracy = sum(1 for c in window_classifications if c['is_correct']) / len(window_classifications)
    
    # 5. Evaluate Anomalies
    anomaly_rate, anomaly_hits, all_detected_anomalies = evaluate_anomalies(model, true_anomaly_positions)
    
    # 6. PASS 2: Memory Recall & Convergence
    print("\n" + "-"*50)
    print("  PASS 2: MEMORY RECALL & CONVERGENCE")
    print("-"*50)
    
    # Reset field only (preserves crystal memory)
    model.init_field()
    
    start_p2 = time.time()
    results_p2 = model.run(n_steps=500, record_every=50)
    dur_p2 = time.time() - start_p2
    
    print(f"Pass 2 finished in {dur_p2:.2f} seconds.")
    
    # Recognition stats on Pass 2
    rec_history_p2 = results_p2.get('v7_recognition_history', [])
    rec_results = [r['result'] for r in rec_history_p2]
    n_rec = sum(1 for r in rec_results if r == 'recognized')
    n_novel = sum(1 for r in rec_results if r == 'novel')
    n_ambivalent = sum(1 for r in rec_results if r == 'ambivalent')
    rec_rate = n_rec / len(rec_results) if rec_results else 0.0
    
    print(f"Pass 2 recognition history: recognized={n_rec}, novel={n_novel}, ambivalent={n_ambivalent}")
    print(f"Recall Rate: {rec_rate*100:.1f}%")
    
    # Free Energy comparison
    fe_p1 = results_p1.get('free_energy_over_time', [])
    fe_p2 = results_p2.get('free_energy_over_time', [])
    
    # Check if Free Energy decreases stably
    # Compute slope of first pass FE
    fe_slope_p1 = 0.0
    if len(fe_p1) > 1:
        fe_slope_p1 = (fe_p1[-1] - fe_p1[0]) / len(fe_p1)
    
    # Phase analysis parameters (from last step of Pass 1)
    phase_p1 = results_p1.get('v5_phase_analysis', {})
    T_c = phase_p1.get('T_c', 0.0)
    psi = phase_p1.get('order_parameter', 0.0)
    xi = phase_p1.get('correlation_length', 0.0)
    
    # Check depth hierarchy
    conversion_levels = results_p1.get('conversion_levels', [])
    depth_reached = len(conversion_levels)
    print(f"\nHierarchy Depth Reached: {depth_reached} levels")
    for lvl in conversion_levels:
        print(f"  Level {lvl['level']}: {lvl['n_clusters']} clusters")
        
    # Compression Ratio
    total_crystals = len(model.crystal_memory.crystals)
    compression_ratio = len(data) / (total_crystals * 256 + 1e-10)
    
    # Compile Metrics
    metrics = {
        "volume_bytes": len(data),
        "boundaries": boundaries,
        "true_anomalies_count": len(true_anomaly_positions),
        "detected_anomalies_count": len(anomaly_hits),
        "anomaly_detection_rate": anomaly_rate,
        "anomaly_hits": anomaly_hits,
        "all_detected_anomalies": all_detected_anomalies,
        "modality_sliding_windows": window_classifications,
        "modality_accuracy": modality_accuracy,
        "pass_1_duration_seconds": dur_p1,
        "pass_2_duration_seconds": dur_p2,
        "speedup_factor": dur_p1 / (dur_p2 + 1e-10),
        "recall_rate": rec_rate,
        "recognition_count": n_rec,
        "novel_count": n_novel,
        "ambivalent_count": n_ambivalent,
        "free_energy_p1": fe_p1,
        "free_energy_p2": fe_p2,
        "fe_slope_p1": fe_slope_p1,
        "hierarchy_depth": depth_reached,
        "compression_ratio": compression_ratio,
        "phase_tc": T_c,
        "phase_psi": psi,
        "phase_xi": xi,
        "success": (depth_reached >= 4 and anomaly_rate >= 0.7 and fe_slope_p1 < 0)
    }
    
    # 7. Write JSON output
    os.makedirs("test_results", exist_ok=True)
    json_out_path = "test_results/mega_stress_test_results.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
    print(f"\nJSON metrics written to {json_out_path}")
    
    # 8. Write Markdown report
    md_out_path = "test_results/mega_stress_test_report.md"
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write("# BCS Stage 1 Mega Capability Stress Test Report\n\n")
        
        status = "🟢 PASS" if metrics["success"] else "🔴 FAIL / PARTIAL"
        f.write(f"### Status: {status}\n\n")
        
        f.write("## 1. Volume & Modality Composition\n")
        f.write(f"- **Total Substrate Size**: {metrics['volume_bytes']:,} bytes\n")
        f.write("- **Modalities Mixed**:\n")
        for k, (s, e) in boundaries.items():
            f.write(f"  - **{k.upper()}**: {s:,} to {e:,} ({e-s:,} bytes)\n")
            
        f.write("\n## 2. Depth: Pattern Hierarchy\n")
        f.write(f"- **Uncovered Levels**: {depth_reached} / 5 levels detected\n")
        f.write("- **Level breakdown**:\n")
        for lvl in conversion_levels:
            f.write(f"  - **Level {lvl['level']}**: {lvl['n_clusters']} clusters formed\n")
            
        f.write("\n## 3. Speed & Convergence\n")
        f.write(f"- **Pass 1 Duration**: {dur_p1:.3f} seconds\n")
        f.write(f"- **Pass 2 Duration**: {dur_p2:.3f} seconds\n")
        f.write(f"- **Speedup Factor (Recall Speed)**: {dur_p1 / (dur_p2 + 1e-10):.2f}x\n")
        f.write(f"- **Free Energy Slope (Pass 1)**: {fe_slope_p1:.6f} (negative means stable reduction)\n")
        
        f.write("\n## 4. Multimodality (Windowed Switching)\n")
        f.write(f"- **Sliding-window Classification Accuracy**: {modality_accuracy*100:.1f}%\n")
        f.write("- **Transitions Log**:\n")
        f.write("| Offset | Center | True Modality | Detected Modality | Correct? |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for c in window_classifications:
            corr_sym = "✓" if c['is_correct'] else "✗"
            f.write(f"| {c['offset']} | {c['window_center']} | {c['true_modality']} | {c['detected_modality']} | {corr_sym} |\n")
            
        f.write("\n## 5. Anomalies\n")
        f.write(f"- **Injected Anomalies**: {metrics['true_anomalies_count']}\n")
        f.write(f"- **Detected Anomalies**: {metrics['detected_anomalies_count']} ({anomaly_rate*100:.1f}%)\n")
        f.write(f"- **Detection threshold**: {anomaly_threshold:.4f}\n")
        f.write(f"- **Detected positions**: {anomaly_hits}\n")
        
        f.write("\n## 6. Memory & Recall\n")
        f.write(f"- **Pass 2 Recall Rate**: {rec_rate*100:.1f}%\n")
        f.write(f"- **Recognized Clusters (Pass 2)**: {n_rec}\n")
        f.write(f"- **Novel Clusters (Pass 2)**: {n_novel}\n")
        f.write(f"- **Compression Ratio**: {compression_ratio:.4f}\n")
        
        f.write("\n## 7. Stability Parameters\n")
        f.write(f"- **Critical Temperature (T_c)**: {T_c:.4f}\n")
        f.write(f"- **Order Parameter (ψ)**: {psi:.4f}\n")
        f.write(f"- **Correlation Length (ξ)**: {xi:.4f}\n")
        
    print(f"Markdown report written to {md_out_path}")
    print(f"Stress Test Execution Completed. Status: {'PASS' if metrics['success'] else 'FAIL'}")

if __name__ == "__main__":
    main()
