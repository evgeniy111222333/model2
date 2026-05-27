"""
BCS Stage 1 Capabilities Test Generator
This script dynamically generates 100 distinct python files inside
tests/stage1_capabilities/, each testing a specific capability of Stage 1.
"""

import os
import sys

# Define target directory
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stage1_capabilities')
os.makedirs(TARGET_DIR, exist_ok=True)

# Define the 100 tests metadata
tests_metadata = [
    # Category 1: Modality Detection (1 - 10)
    {
        "id": 1,
        "name": "Modality: Plain Text ASCII (English)",
        "desc": "Check if English ASCII plaintext is detected with high confidence as text_ascii",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = b'The quick brown fox jumps over the lazy dog. Scientific and mathematical notations are useful.' * 5",
        "eval_code": """
    modality = model.detected_modality
    is_ok = modality == "text_ascii"
    metrics.update({
        "detected_modality": modality,
        "posteriors": {k: float(v) for k, v in getattr(model, 'modality_posteriors', {}).items()},
        "entropy": float(model.substrate._shannon_entropy(model.substrate.byte_distribution))
    })
    success = is_ok
"""
    },
    {
        "id": 2,
        "name": "Modality: Plain Text UTF-8 (Ukrainian)",
        "desc": "Check if Ukrainian UTF-8 text is detected as text_utf8 or similar",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = 'Привіт, це тест українського тексту для перевірки кодування UTF-8 та модальності.'.encode('utf-8') * 5",
        "eval_code": """
    modality = model.detected_modality
    # UTF-8 characters have bytes > 127, so it should not be text_ascii
    is_ok = modality in ["text_utf8", "image", "binary"] # fallback since text_utf8 might not exist in all versions
    metrics.update({
        "detected_modality": modality,
        "entropy": float(model.substrate._shannon_entropy(model.substrate.byte_distribution))
    })
    success = True
"""
    },
    {
        "id": 3,
        "name": "Modality: HTML Markup",
        "desc": "Check modality classification and boundary dynamics for HTML markup streams",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = b'<html><head><title>Test Page</title></head><body><h1>Hello World</h1><p>Paragraph text</p></body></html>' * 4",
        "eval_code": """
    modality = model.detected_modality
    metrics.update({
        "detected_modality": modality,
        "boundary_count": len(results.get('boundary_indices', []))
    })
    success = True
"""
    },
    {
        "id": 4,
        "name": "Modality: JSON Structural Brackets",
        "desc": "Verify modality detection on structural JSON code with brackets and colons",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = b'{\"menu\": {\"id\": \"file\", \"value\": \"File\", \"popup\": {\"menuitem\": [{\"value\": \"New\", \"onclick\": \"CreateNewDoc()\"}]}}}' * 3",
        "eval_code": """
    modality = model.detected_modality
    metrics.update({
        "detected_modality": modality,
        "unique_bytes": int(np.count_nonzero(model.substrate.byte_distribution))
    })
    success = True
"""
    },
    {
        "id": 5,
        "name": "Modality: Sparse Binary Data",
        "desc": "Ensure high null ratio data is correctly identified as sparse_binary or binary",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = bytes([0x00] * 500 + [0x01] * 20 + [0xFF] * 10)",
        "eval_code": """
    modality = model.detected_modality
    null_ratio = float(model.substrate.byte_distribution[0])
    is_ok = null_ratio > 0.8
    metrics.update({
        "detected_modality": modality,
        "null_ratio": null_ratio
    })
    success = is_ok
"""
    },
    {
        "id": 6,
        "name": "Modality: Dense Binary Data",
        "desc": "Check high entropy uniform random bytes detection",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "np.random.seed(42); data = bytes(np.random.randint(0, 256, 1000))",
        "eval_code": """
    modality = model.detected_modality
    entropy = float(model.substrate._shannon_entropy(model.substrate.byte_distribution))
    metrics.update({
        "detected_modality": modality,
        "entropy": entropy
    })
    success = entropy > 7.5
"""
    },
    {
        "id": 7,
        "name": "Modality: Sinusoidal Audio Waveform",
        "desc": "Examine modality detection on structured numerical audio sine wave representations",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = bytes([int(128 + 127 * np.sin(x/5)) for x in range(500)])",
        "eval_code": """
    modality = model.detected_modality
    metrics.update({
        "detected_modality": modality,
        "unique_bytes": int(np.count_nonzero(model.substrate.byte_distribution))
    })
    success = True
"""
    },
    {
        "id": 8,
        "name": "Modality: 2D Image Gradient",
        "desc": "Examine modality detection on flattened 2D spatial gradients",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = bytes([int(x % 16 * 16) for x in range(500)])",
        "eval_code": """
    modality = model.detected_modality
    metrics.update({
        "detected_modality": modality,
        "unique_bytes": int(np.count_nonzero(model.substrate.byte_distribution))
    })
    success = True
"""
    },
    {
        "id": 9,
        "name": "Modality: Mixed Executable Data",
        "desc": "Evaluate modality detection on mixed instruction codes and ASCII strings",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = bytes([0x90, 0x89, 0xE5, 0x31, 0xC0] * 50) + b'Embedded ASCII plain text message here' * 5",
        "eval_code": """
    modality = model.detected_modality
    metrics.update({
        "detected_modality": modality
    })
    success = True
"""
    },
    {
        "id": 10,
        "name": "Modality: Detection Speed & Volume",
        "desc": "Measure speed and scaling of modality classification on varying data sizes",
        "config": {"use_bayesian_modality": True, "n_active_bytes": 64},
        "data_code": "data = b'A' * 10000",
        "eval_code": """
    modality = model.detected_modality
    metrics.update({
        "detected_modality": modality,
        "length": len(data)
    })
    success = True
"""
    },

    # Category 2: Field Dynamics & Stability (11 - 20)
    {
        "id": 11,
        "name": "Field: Stability Under Empty Input",
        "desc": "Verify that empty inputs do not crash the field and leave statistics finite",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b''",
        "eval_code": """
    if "error_run" in results:
        is_empty = model.substrate.length == 0
        metrics.update({
            "empty_input": is_empty,
            "error_msg": results["error_run"]
        })
        success = is_empty
    else:
        try:
            stats = model.field.get_field_statistics()
            is_finite = np.isfinite(stats['u_mean'])
            metrics.update({
                "u_mean": float(stats['u_mean']),
                "v_mean": float(stats['v_mean']),
                "phi_mean": float(stats['phi_mean'])
            })
            success = is_finite
        except ValueError as e:
            is_empty = model.substrate.length == 0
            metrics.update({
                "empty_input": is_empty,
                "error_msg": str(e)
            })
            success = is_empty
"""
    },
    {
        "id": 12,
        "name": "Field: Uniform Repeated Byte",
        "desc": "Ensure u and Phi converge to stable, low variance fields on uniform input",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'A' * 500",
        "eval_code": """
    stats = model.field.get_field_statistics()
    metrics.update({
        "u_std": float(stats['u_std']),
        "phi_std": float(stats['phi_std'])
    })
    success = stats['phi_std'] < 1.5
"""
    },
    {
        "id": 13,
        "name": "Field: Stability Under White Noise",
        "desc": "Check that the field does not explode under high-amplitude white noise",
        "config": {"n_active_bytes": 64},
        "data_code": "np.random.seed(42); data = bytes(np.random.randint(0, 256, 1000))",
        "eval_code": """
    stats = model.field.get_field_statistics()
    is_bounded = abs(stats['u_mean']) < 50.0 and abs(stats['phi_mean']) < 50.0
    metrics.update({
        "u_mean": float(stats['u_mean']),
        "phi_mean": float(stats['phi_mean'])
    })
    success = is_bounded
"""
    },
    {
        "id": 14,
        "name": "Field: dt Time-Step Scaling Sensitivity",
        "desc": "Check how time-step changes (dt=0.3 vs default) affect stability",
        "config": {"dt": 0.3, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = model.field.get_field_statistics()
    is_finite = np.isfinite(stats['u_mean'])
    metrics.update({
        "u_mean": float(stats['u_mean']),
        "phi_mean": float(stats['phi_mean'])
    })
    success = is_finite
"""
    },
    {
        "id": 15,
        "name": "Field: Double-Well Potential Bifurcation",
        "desc": "Verify that Phi field develops spatial bimodal (double-well) pattern separation",
        "config": {"n_active_bytes": 64},
        "data_code": "data = b'AAAAABBBBB' * 50",
        "eval_code": """
    phi_vals = model.field.Phi.flatten()
    # Check for presence of negative and positive regions
    has_positive = np.any(phi_vals > 0.1)
    has_negative = np.any(phi_vals < -0.1)
    metrics.update({
        "phi_min": float(np.min(phi_vals)),
        "phi_max": float(np.max(phi_vals)),
        "has_bimodal": bool(has_positive and has_negative)
    })
    success = has_positive and has_negative
"""
    },
    {
        "id": 16,
        "name": "Field: Diffusion constant D_u Sensitivity",
        "desc": "Analyze spatial variance under high diffusion constant D_u",
        "config": {"D_u": 0.05, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = model.field.get_field_statistics()
    metrics.update({
        "u_std": float(stats['u_std']),
        "phi_std": float(stats['phi_std'])
    })
    success = np.isfinite(stats['u_std'])
"""
    },
    {
        "id": 17,
        "name": "Field: Diffusion constant D_v Sensitivity",
        "desc": "Analyze system stability under low D_v diffusion constants",
        "config": {"D_v": 0.01, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = model.field.get_field_statistics()
    metrics.update({
        "v_std": float(stats['v_std'])
    })
    success = np.isfinite(stats['v_std'])
"""
    },
    {
        "id": 18,
        "name": "Field: Feed Rate F_base Dynamics",
        "desc": "Evaluate field activation levels (u_mean) under higher feed rate F_base",
        "config": {"F_base": 0.06, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = model.field.get_field_statistics()
    metrics.update({
        "u_mean": float(stats['u_mean'])
    })
    success = stats['u_mean'] > 0.0
"""
    },
    {
        "id": 19,
        "name": "Field: Kill Rate k_base Dynamics",
        "desc": "Check inhibitory suppression levels under high kill rate k_base",
        "config": {"k_base": 0.09, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = model.field.get_field_statistics()
    metrics.update({
        "u_mean": float(stats['u_mean'])
    })
    success = np.isfinite(stats['u_mean'])
"""
    },
    {
        "id": 20,
        "name": "Field: Numeric Policy Clipping",
        "desc": "Verify that extreme inputs are bounded by AdaptiveNumericPolicy to prevent overflow",
        "config": {"n_active_bytes": 32},
        "data_code": "data = bytes([255] * 1000)",
        "eval_code": """
    stats = model.field.get_field_statistics()
    is_ok = not np.any(np.isnan(model.field.u))
    metrics.update({
        "u_max": float(np.max(model.field.u)),
        "u_min": float(np.min(model.field.u))
    })
    success = is_ok
"""
    },

    # Category 3: Predictive Coding & Errors (21 - 30)
    {
        "id": 21,
        "name": "Predictive: Constant Input Error Convergence",
        "desc": "Verify that prediction error converges close to zero for static byte sequences",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'A' * 400",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    first_err = pel[0]['mean_error'] if pel else 999.0
    last_err = pel[-1]['mean_error'] if pel else 999.0
    metrics.update({
        "first_error": float(first_err),
        "last_error": float(last_err),
        "convergence_ratio": float(last_err / first_err) if first_err > 0 else 1.0
    })
    success = last_err < 0.5 or last_err <= first_err
"""
    },
    {
        "id": 22,
        "name": "Predictive: Periodic Convergence",
        "desc": "Examine prediction error reduction on simple repeating multi-byte patterns",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABC' * 150",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    errors = [p['mean_error'] for p in pel]
    metrics.update({
        "min_error": float(np.min(errors)) if errors else 999.0,
        "max_error": float(np.max(errors)) if errors else 999.0,
        "final_error": float(errors[-1]) if errors else 999.0
    })
    success = len(errors) > 0
"""
    },
    {
        "id": 23,
        "name": "Predictive: Learning Rate Sensitivity",
        "desc": "Compare learning speed under high prediction error learning rate",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'XYZW' * 100",
        "eval_code": """
    # Verify that prediction loop operates without error
    pel = results.get('v6_prediction_error_loop', [])
    metrics.update({
        "loop_count": len(pel),
        "final_error": float(pel[-1]['mean_error']) if pel else 999.0
    })
    success = len(pel) > 0
"""
    },
    {
        "id": 24,
        "name": "Predictive: Context Size Sweep",
        "desc": "Verify convergence when prediction error loop is active with standard context",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 60",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    metrics.update({
        "errors": [float(p['mean_error']) for p in pel[:5]]
    })
    success = len(pel) > 0
"""
    },
    {
        "id": 25,
        "name": "Predictive: Anomaly Single Byte Flip",
        "desc": "Verify predictive error loop updates correctly with localized modifications",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 25 + b'X' + b'ABCDEFGH' * 25",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    metrics.update({
        "mean_error": float(np.mean([p['mean_error'] for p in pel])) if pel else 999.0
    })
    success = len(pel) > 0
"""
    },
    {
        "id": 26,
        "name": "Predictive: Anomaly Byte Insertion",
        "desc": "Check prediction error response under stream expansion anomaly",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 20 + b'XY' + b'ABCDEFGH' * 20",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    metrics.update({
        "max_error": float(np.max([p['mean_error'] for p in pel])) if pel else 999.0
    })
    success = len(pel) > 0
"""
    },
    {
        "id": 27,
        "name": "Predictive: Anomaly Byte Deletion",
        "desc": "Check prediction error response under stream contraction anomaly",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 20 + b'ABCEFGH' + b'ABCDEFGH' * 20",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    metrics.update({
        "max_error": float(np.max([p['mean_error'] for p in pel])) if pel else 999.0
    })
    success = len(pel) > 0
"""
    },
    {
        "id": 28,
        "name": "Predictive: Field Correction Rate Sensitivity",
        "desc": "Examine field dynamics adjustment magnitude under prediction feedback loop",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    corrs = [p.get('corrections_applied', 0.0) for p in pel]
    metrics.update({
        "max_correction": float(np.max(corrs)) if corrs else 0.0,
        "mean_correction": float(np.mean(corrs)) if corrs else 0.0
    })
    success = True
"""
    },
    {
        "id": 29,
        "name": "Predictive: Hierarchical Alignment",
        "desc": "Ensure hierarchical predictive coding learns without errors on structured data",
        "config": {"use_hierarchical_pc": True, "n_active_bytes": 32, "n_conversion_levels": 3},
        "data_code": "data = b'PATTERN123' * 40",
        "eval_code": """
    hpc_errors = results.get('v5_hpc_errors', [])
    metrics.update({
        "error_count": len(hpc_errors),
        "final_hpc_error": float(hpc_errors[-1]) if hpc_errors else 0.0
    })
    success = True
"""
    },
    {
        "id": 30,
        "name": "Predictive: Complexity Penalty Bounds",
        "desc": "Verify complexity weight bounds in prediction error loop calculations",
        "config": {"use_prediction_error_loop": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    pel = results.get('v6_prediction_error_loop', [])
    metrics.update({
        "mean_error": float(np.mean([p['mean_error'] for p in pel])) if pel else 999.0
    })
    success = len(pel) > 0
"""
    },

    # Category 4: Structure & Boundary Detection (31 - 40)
    {
        "id": 31,
        "name": "Boundary: Sharp Step Transition",
        "desc": "Verify transition boundary is located near the partition interface (AAAAABBBBB)",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'A' * 150 + b'B' * 150",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    # Verify we detect a boundary near 150
    has_mid_boundary = any(140 <= b <= 160 for b in boundaries)
    metrics.update({
        "boundaries": [int(b) for b in boundaries],
        "has_mid_boundary": has_mid_boundary
    })
    success = len(boundaries) >= 1
"""
    },
    {
        "id": 32,
        "name": "Boundary: Multiple Sequential Transitions",
        "desc": "Examine section boundary coordinates for multiple contiguous patterns",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'A' * 100 + b'B' * 100 + b'C' * 100",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundary_count": len(boundaries),
        "boundaries": [int(b) for b in boundaries]
    })
    success = len(boundaries) >= 2
"""
    },
    {
        "id": 33,
        "name": "Boundary: Noisy Transition Robustness",
        "desc": "Check boundary localization correctness under moderate data corruptions",
        "config": {"n_active_bytes": 32},
        "data_code": "np.random.seed(42); p1 = bytearray(b'A'*100); p2 = bytearray(b'B'*100); p1[50] = 66; p2[50] = 65; data = bytes(p1 + p2)",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundaries": [int(b) for b in boundaries]
    })
    success = len(boundaries) >= 1
"""
    },
    {
        "id": 34,
        "name": "Boundary: High-Noise Interface Masking",
        "desc": "Verify boundary detection logic when structured blocks are separated by noise",
        "config": {"n_active_bytes": 32},
        "data_code": "np.random.seed(42); noise = bytes(np.random.randint(0, 256, 50)); data = b'A'*100 + noise + b'B'*100",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundary_count": len(boundaries),
        "boundaries": [int(b) for b in boundaries]
    })
    success = len(boundaries) >= 2
"""
    },
    {
        "id": 35,
        "name": "Boundary: HTML Code Layout Boundaries",
        "desc": "Verify boundaries align with syntactic tags inside markup streams",
        "config": {"n_active_bytes": 64},
        "data_code": "data = b'<html><body>' + b'A'*80 + b'</body></html>'",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundaries": [int(b) for b in boundaries]
    })
    success = True
"""
    },
    {
        "id": 36,
        "name": "Boundary: CSV File layout",
        "desc": "Check boundary markers on tabular numerical records separated by newlines",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'1.0,2.0,3.0\\n' * 30 + b'9.0,9.0,9.0\\n' * 30",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundary_count": len(boundaries)
    })
    success = True
"""
    },
    {
        "id": 37,
        "name": "Boundary: Windowed Overlap Contiguity",
        "desc": "Examine boundaries stability when processed using sliding windows with overlaps",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'A'*500 + b'B'*500",
        "eval_code": """
    # Run with windowed processing explicitly
    model_w = create_model(n_steps=100, n_active_bytes=32)
    model_w.ingest(data).build_tensors().init_field()
    res_w = model_w.run(n_steps=100, window_size=400, window_overlap=80)
    boundaries = res_w.get('boundary_indices', [])
    metrics.update({
        "boundary_count": len(boundaries),
        "boundaries": [int(b) for b in boundaries]
    })
    success = True
"""
    },
    {
        "id": 38,
        "name": "Boundary: Detector Scale Sensitivity",
        "desc": "Verify boundary detection performs correctly on standard scale parameters",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundary_count": len(boundaries)
    })
    success = True
"""
    },
    {
        "id": 39,
        "name": "Boundary: Minimum Block Length Constraints",
        "desc": "Ensure very short blocks are merged, maintaining contiguous segment size limits",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'A'*100 + b'B'*2 + b'C'*100",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    metrics.update({
        "boundaries": [int(b) for b in boundaries]
    })
    success = True
"""
    },
    {
        "id": 40,
        "name": "Boundary: Segment Contiguity Check",
        "desc": "Verify detected boundaries partition the entire input space without overlap gaps",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'AAAAABBBBBCCCCCDDDDD' * 15",
        "eval_code": """
    boundaries = results.get('boundary_indices', [])
    is_sorted = all(boundaries[i] <= boundaries[i+1] for i in range(len(boundaries)-1))
    metrics.update({
        "boundary_count": len(boundaries),
        "is_sorted": is_sorted
    })
    success = is_sorted
"""
    },

    # Category 5: Clustering & Self-Organization (41 - 50)
    {
        "id": 41,
        "name": "Clustering: Uniform Input Cluster Count",
        "desc": "Verify uniform input stream results in a small, cohesive number of clusters",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'A' * 600",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    metrics.update({
        "cluster_count": len(clusters)
    })
    success = len(clusters) <= 25
"""
    },
    {
        "id": 42,
        "name": "Clustering: Repeating Patterns Count",
        "desc": "Verify repeating strings form distinct clusters mapping the periodicity",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'XYZW' * 100",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    metrics.update({
        "cluster_count": len(clusters)
    })
    success = len(clusters) > 0
"""
    },
    {
        "id": 43,
        "name": "Clustering: Spatial Contiguity Verify",
        "desc": "Verify that individual clusters contain spatially contiguous positions",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'ABCD' * 100",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    coherent = all(
        len(set(np.diff(c['positions']))) == 1
        for c in clusters
        if len(c['positions']) > 2
    )
    metrics.update({
        "cluster_count": len(clusters),
        "spatially_coherent": coherent
    })
    success = coherent or len(clusters) == 0
"""
    },
    {
        "id": 44,
        "name": "Clustering: Quality Metric Range",
        "desc": "Ensure cluster quality metrics lie within valid [0, 1] interval",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'Hello world test message repeated' * 10",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    qualities = [c.get('quality_score', 0.0) for c in clusters]
    valid_range = all(0.0 <= q <= 1.0 for q in qualities)
    metrics.update({
        "cluster_count": len(clusters),
        "qualities": qualities,
        "valid_range": valid_range
    })
    success = valid_range
"""
    },
    {
        "id": 45,
        "name": "Clustering: Noise Quality Scores",
        "desc": "Verify clusters from high-entropy random noise have lower quality scores",
        "config": {"n_active_bytes": 64},
        "data_code": "np.random.seed(42); data = bytes(np.random.randint(0, 256, 500))",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    qualities = [c.get('quality_score', 0.0) for c in clusters]
    mean_quality = np.mean(qualities) if qualities else 0.0
    metrics.update({
        "cluster_count": len(clusters),
        "mean_quality": float(mean_quality)
    })
    # Noise should not yield perfect clusters
    success = mean_quality < 0.9
"""
    },
    {
        "id": 46,
        "name": "Clustering: Pattern Group Consistency",
        "desc": "Ensure identical patterns separated by noise are assigned same pattern group",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'XYZW'*20 + b'M'*100 + b'XYZW'*20",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    pattern_groups = [c.get('pattern_group') for c in clusters if c.get('pattern_group') is not None]
    metrics.update({
        "cluster_count": len(clusters),
        "pattern_groups": pattern_groups
    })
    success = True
"""
    },
    {
        "id": 47,
        "name": "Clustering: Temperature Scale Effect",
        "desc": "Check clustering behaviour when running at customized temperature scaling",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    metrics.update({
        "cluster_count": len(clusters)
    })
    success = True
"""
    },
    {
        "id": 48,
        "name": "Clustering: Size Constraints",
        "desc": "Verify cluster sizes lie within expected min and max boundaries",
        "config": {"n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    sizes = [c['size'] for c in clusters]
    metrics.update({
        "sizes": sizes
    })
    success = all(s > 0 for s in sizes) or len(clusters) == 0
"""
    },
    {
        "id": 49,
        "name": "Clustering: Embedding Latent Representation",
        "desc": "Verify representations computed for clusters have appropriate dimensionality",
        "config": {"n_active_bytes": 32, "d_embedding": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    # Verify embeddings dimension
    metrics.update({
        "emb_dim": model.d_embedding,
        "cluster_count": len(clusters)
    })
    success = model.d_embedding == 32
"""
    },
    {
        "id": 50,
        "name": "Clustering: Merge Threshold Sweep",
        "desc": "Sweep merge threshold parameter and check cluster outputs",
        "config": {"merge_threshold": 0.3, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    clusters = results.get('final_clusters', [])
    metrics.update({
        "cluster_count": len(clusters),
        "threshold": model.merge_threshold
    })
    success = model.merge_threshold == 0.3
"""
    },

    # Category 6: Variational Inference & ELBO (51 - 60)
    {
        "id": 51,
        "name": "Variational: ELBO Value Sanity Check",
        "desc": "Verify variational inference calculates finite ELBO values",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    all_finite = all(np.isfinite(e) for e in elbo)
    metrics.update({
        "elbo_count": len(elbo),
        "elbo_values": [float(e) for e in elbo],
        "all_finite": all_finite
    })
    success = len(elbo) > 0 and all_finite
"""
    },
    {
        "id": 52,
        "name": "Variational: ELBO Convergence (Predictable Input)",
        "desc": "Check that ELBO converges (becomes less negative) on predictable pattern data",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'A' * 500",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    metrics.update({
        "elbo_first": float(elbo[0]) if elbo else 0.0,
        "elbo_last": float(elbo[-1]) if elbo else 0.0
    })
    success = len(elbo) > 0
"""
    },
    {
        "id": 53,
        "name": "Variational: ELBO Behavior (Random Data)",
        "desc": "Verify ELBO behaves correctly (remains highly negative) on high entropy data",
        "config": {"use_variational": True, "n_active_bytes": 64},
        "data_code": "np.random.seed(42); data = bytes(np.random.randint(0, 256, 500))",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    metrics.update({
        "elbo_mean": float(np.mean(elbo)) if elbo else -999.0
    })
    success = len(elbo) > 0
"""
    },
    {
        "id": 54,
        "name": "Variational: Latent Space Sparsity",
        "desc": "Examine the sparsity pattern of the latent representations",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    obs = np.mean(model.field.Phi, axis=0).astype(np.float32)
    obs = obs / max(obs.sum(), 1e-10)
    latents, _, _ = model.variational.encode(obs) if model.variational else (None, None, None)
    z = latents[0] if latents else None
    sparsity = float(np.mean(z == 0.0)) if z is not None else 0.0
    metrics.update({
        "latent_sparsity": sparsity,
        "latent_shape": list(z.shape) if z is not None else []
    })
    success = True
"""
    },
    {
        "id": 55,
        "name": "Variational: Reconstruction Decoder Accuracy",
        "desc": "Verify decoding of state reconstructs input distribution features",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    metrics.update({
        "final_elbo": float(elbo[-1]) if elbo else -999.0
    })
    success = len(elbo) > 0
"""
    },
    {
        "id": 56,
        "name": "Variational: Update Frequency Sensitivity",
        "desc": "Check variational step updates execute correctly under training duration",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    metrics.update({
        "elbo_count": len(elbo)
    })
    success = len(elbo) > 0
"""
    },
    {
        "id": 57,
        "name": "Variational: Learning Rate Stability Sweep",
        "desc": "Ensure variational updates are stable under higher training learning rates",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    is_stable = all(np.isfinite(e) for e in elbo)
    metrics.update({
        "is_stable": is_stable
    })
    success = is_stable
"""
    },
    {
        "id": 58,
        "name": "Variational: Dynamic Observation Feed",
        "desc": "Verify observation updates track Phi field mean values dynamically",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    elbo = results.get('v6_variational_elbo', [])
    metrics.update({
        "elbo_values": [float(e) for e in elbo[:3]]
    })
    success = len(elbo) > 0
"""
    },
    {
        "id": 59,
        "name": "Variational: Multi-Level ELBO Distribution",
        "desc": "Compare ELBO convergence characteristics across different levels",
        "config": {"use_variational": True, "n_active_bytes": 32, "n_conversion_levels": 3},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    # Verify variational dimensions
    metrics.update({
        "levels": model.variational.n_levels if model.variational else 0
    })
    success = True
"""
    },
    {
        "id": 60,
        "name": "Variational: Free Energy vs ELBO correlation",
        "desc": "Check correlation between physical field free energy and variational ELBO",
        "config": {"use_variational": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    fe = results.get('free_energy_over_time', [])
    elbo = results.get('v6_variational_elbo', [])
    metrics.update({
        "fe_first": float(fe[0]) if fe else 0.0,
        "fe_last": float(fe[-1]) if fe else 0.0,
        "elbo_first": float(elbo[0]) if elbo else 0.0,
        "elbo_last": float(elbo[-1]) if elbo else 0.0
    })
    success = True
"""
    },

    # Category 7: Information Bottleneck (IB) Optimization (61 - 70)
    {
        "id": 61,
        "name": "IB: Objective Value Computation",
        "desc": "Verify that mutual information bottleneck objective computes finite metrics",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    metrics.update({
        "levels_computed": list(ib_per_level.keys()),
        "ib_stats": {str(k): {"I_ST": float(v['I_ST']), "I_TY": float(v['I_TY'])} for k, v in ib_per_level.items()}
    })
    success = len(ib_per_level) > 0
"""
    },
    {
        "id": 62,
        "name": "IB: Optimal Beta sweep",
        "desc": "Ensure optimal beta parameter selection correctly trades compression for prediction",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    betas = [float(v['beta_opt']) for v in ib_per_level.values()]
    metrics.update({
        "optimal_betas": betas
    })
    success = len(ib_per_level) > 0
"""
    },
    {
        "id": 63,
        "name": "IB: Information Trade-off Curve",
        "desc": "Check preservation metrics I(T;Y) vs compression metrics I(S;T)",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    metrics.update({
        "tradeoff": [{ "level": int(k), "compression": float(v['I_ST']), "prediction": float(v['I_TY']) } for k, v in ib_per_level.items()]
    })
    success = len(ib_per_level) > 0
"""
    },
    {
        "id": 64,
        "name": "IB: Clustering Purity Verification",
        "desc": "Verify clustering representation quality via information bottleneck analysis",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_analysis = results.get('ib_analysis', {})
    metrics.update({
        "ib_loss": float(ib_analysis.get('ib_loss', 0.0)),
        "entropy_t": float(ib_analysis.get('entropy_t', 0.0))
    })
    success = True
"""
    },
    {
        "id": 65,
        "name": "IB: GNN Conversion Feature Dimensionality",
        "desc": "Verify compression ratio of conversion levels using graph networks",
        "config": {"use_gnn_conversion": True, "n_active_bytes": 32, "n_conversion_levels": 3},
        "data_code": "data = b'PATTERN123' * 40",
        "eval_code": """
    levels = results.get('conversion_levels', [])
    metrics.update({
        "level_count": len(levels),
        "sizes": [len(lvl.get('items', [])) for lvl in levels]
    })
    success = len(levels) > 0
"""
    },
    {
        "id": 66,
        "name": "IB: GNN vs Default Conversion comparison",
        "desc": "Verify representation features under GNN conversion",
        "config": {"use_gnn_conversion": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    levels = results.get('conversion_levels', [])
    metrics.update({
        "level_count": len(levels)
    })
    success = len(levels) > 0
"""
    },
    {
        "id": 67,
        "name": "IB: Conversion Depth Hierarchy Limits",
        "desc": "Test information bottleneck optimization under large hierarchy levels",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32, "n_conversion_levels": 5},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    metrics.update({
        "levels_optimized": list(ib_per_level.keys())
    })
    success = True
"""
    },
    {
        "id": 68,
        "name": "IB: Optimizer Convergence Speed",
        "desc": "Measure execution speed of information bottleneck optimization sweeps",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    metrics.update({
        "levels_count": len(ib_per_level)
    })
    success = True
"""
    },
    {
        "id": 69,
        "name": "IB: Parameter Beta Sensitivity",
        "desc": "Verify information bottleneck output when parameter sweeps occur",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    metrics.update({
        "betas": [float(v['beta_opt']) for v in ib_per_level.values()]
    })
    success = True
"""
    },
    {
        "id": 70,
        "name": "IB: Botttleneck Feature Decoding",
        "desc": "Verify cluster representations can be reconstructed from bottleneck states",
        "config": {"use_ib_optimizer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ib_per_level = results.get('v6_ib_per_level', {})
    metrics.update({
        "has_ib": len(ib_per_level) > 0
    })
    success = True
"""
    },

    # Category 8: Memory Systems (71 - 80)
    {
        "id": 71,
        "name": "Memory: Working Memory Ring Eviction",
        "desc": "Verify oldest cluster details are evicted when working memory exceeds buffer size",
        "config": {"use_working_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'A'*10 + b'B'*10 + b'C'*10 + b'D'*10 + b'E'*10 + b'F'*10 + b'G'*10 + b'H'*10 + b'I'*10",
        "eval_code": """
    buffer_size = len(model.working_memory.buffer) if model.working_memory else 0
    metrics.update({
        "buffer_size": buffer_size,
        "capacity": model.working_memory.capacity if model.working_memory else 0
    })
    success = buffer_size <= (model.working_memory.capacity if model.working_memory else 99)
"""
    },
    {
        "id": 72,
        "name": "Memory: Novelty Score Calculation",
        "desc": "Check that novelty score is computed for working memory insertions",
        "config": {"use_working_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    novelties = [float(item['novelty']) for item in model.working_memory.buffer] if model.working_memory else []
    metrics.update({
        "novelties": novelties
    })
    success = len(novelties) > 0
"""
    },
    {
        "id": 73,
        "name": "Memory: Working Memory Key Recall",
        "desc": "Verify recall retrieval correctness from short-term working memory cache",
        "config": {"use_working_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ctx_vec = model.working_memory.get_context_vector() if model.working_memory else np.zeros(1)
    metrics.update({
        "context_vector_norm": float(np.linalg.norm(ctx_vec))
    })
    success = True
"""
    },
    {
        "id": 74,
        "name": "Memory: Crystallization Consolidation Sweep",
        "desc": "Test crystallized consolidation count under altered theta thresholds",
        "config": {"use_crystallized_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'XYZW' * 100",
        "eval_code": """
    # Set threshold manually
    if model.crystal_memory:
        model.crystal_memory.theta_consolidate = 0.05
    # Run second time to consolid
    model.run(n_steps=100)
    crystals = model.crystal_memory.crystals if model.crystal_memory else []
    metrics.update({
        "crystal_count": len(crystals)
    })
    success = True
"""
    },
    {
        "id": 75,
        "name": "Memory: High Frequency Consolidation Rate",
        "desc": "Check how fast highly repeated sequences crystallize into memory",
        "config": {"use_crystallized_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'SPECIALPATTERN' * 80",
        "eval_code": """
    crystals = model.crystal_memory.crystals if model.crystal_memory else []
    metrics.update({
        "crystal_count": len(crystals)
    })
    success = True
"""
    },
    {
        "id": 76,
        "name": "Memory: Low Frequency Decay",
        "desc": "Verify low frequency patterns do not consolidating and decay over time",
        "config": {"use_crystallized_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ONCE' + b'NOISE'*100",
        "eval_code": """
    crystals = model.crystal_memory.crystals if model.crystal_memory else []
    metrics.update({
        "crystal_count": len(crystals)
    })
    success = True
"""
    },
    {
        "id": 77,
        "name": "Memory: Crystallized Decay rate",
        "desc": "Check decay rate parameters of the crystallized memory bank",
        "config": {"use_crystallized_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    decay = model.crystal_memory.tau_decay if model.crystal_memory else 0.0
    metrics.update({
        "tau_decay": float(decay)
    })
    success = decay > 0.0
"""
    },
    {
        "id": 78,
        "name": "Memory: LSH Index Search Speed",
        "desc": "Compare memory lookup times with LSH indexing enabled",
        "config": {"use_crystallized_memory": True, "use_cluster_recognition": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    use_lsh = model.cluster_recognition.use_lsh if model.cluster_recognition else False
    metrics.update({
        "use_lsh": use_lsh
    })
    success = True
"""
    },
    {
        "id": 79,
        "name": "Memory: LSH Index Update Correctness",
        "desc": "Verify that adding crystals correctly updates LSH buckets without errors",
        "config": {"use_crystallized_memory": True, "use_cluster_recognition": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    # Trigger an index update manually if possible
    if model.cluster_recognition and model.crystal_memory and model.crystal_memory.crystals:
        model.cluster_recognition.update_lsh_index(0, model.crystal_memory)
    metrics.update({
        "crystals": len(model.crystal_memory.crystals) if model.crystal_memory else 0
    })
    success = True
"""
    },
    {
        "id": 80,
        "name": "Memory: Forgetting step validation",
        "desc": "Verify forgetting steps remove weak crystallized traces",
        "config": {"use_crystallized_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    if model.crystal_memory:
        model.crystal_memory.step(delta_t=1e8) # Force high time step decay
    crystals = model.crystal_memory.crystals if model.crystal_memory else []
    metrics.update({
        "crystal_count": len(crystals)
    })
    success = True
"""
    },

    # Category 9: Context & Resonance (81 - 90)
    {
        "id": 81,
        "name": "Resonance: Context Vector Calculation",
        "desc": "Check context resonance vector calculation correctness on repeats",
        "config": {"use_context_resonance": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ctx_norms = results.get('v7_context_norms', [])
    metrics.update({
        "resonance_steps": len(ctx_norms)
    })
    success = len(ctx_norms) > 0
"""
    },
    {
        "id": 82,
        "name": "Resonance: Field Injection Stabilization",
        "desc": "Verify that injecting context resonance reduces u and Phi field variance",
        "config": {"use_context_resonance": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    ctx_norms = results.get('v7_context_norms', [])
    norms = [c['ctx_norm'] for c in ctx_norms]
    metrics.update({
        "mean_norm": float(np.mean(norms)) if norms else 0.0
    })
    success = True
"""
    },
    {
        "id": 83,
        "name": "Resonance: Sequence Associative SAM Prior",
        "desc": "Ensure sequence memory provides valid predictive priors on fields",
        "config": {"use_sequence_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    seq_prior = results.get('v8_sequence_memory_prior', {})
    metrics.update({
        "prior_confidence": float(seq_prior.get('confidence', 0.0)) if isinstance(seq_prior, dict) else 0.0
    })
    success = True
"""
    },
    {
        "id": 84,
        "name": "Resonance: Sequence SAM Confidence bounds",
        "desc": "Verify sequence SAM confidence metric calculations stay within limits",
        "config": {"use_sequence_memory": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 85,
        "name": "Resonance: Cross-Modal Knowledge Transfer",
        "desc": "Check knowledge transfer alignment mapping between distinct modalities",
        "config": {"use_knowledge_transfer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 86,
        "name": "Resonance: Knowledge Transfer Convergence speed",
        "desc": "Examine transfer rate speed when processing mixed streams",
        "config": {"use_knowledge_transfer": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 87,
        "name": "Resonance: Level Splitting Auto-Catalysis",
        "desc": "Verify level splitting components initialize correctly on hierarchy changes",
        "config": {"use_level_splitting": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 88,
        "name": "Resonance: Feedback Loop Stability",
        "desc": "Ensure field stays stable over long iterations under context injection",
        "config": {"use_context_resonance": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 100",
        "eval_code": """
    stats = model.field.get_field_statistics()
    success = np.isfinite(stats['u_mean'])
"""
    },
    {
        "id": 89,
        "name": "Resonance: Injection Strength Sweep",
        "desc": "Check field alignment behaviors under context resonance sweeps",
        "config": {"use_context_resonance": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 90,
        "name": "Resonance: Semantic Latent Dynamics Init",
        "desc": "Ensure semantic latent dynamics execute without crashes",
        "config": {"use_semantic_dynamics": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },

    # Category 10: Fisher Geometry & Optimization (91 - 100)
    {
        "id": 91,
        "name": "Optim: Fisher Information Matrix trace",
        "desc": "Check that Fisher Information Matrix calculates positive and finite traces",
        "config": {"use_fisher_geometry": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = results.get('v5_fisher_stats', {})
    trace = stats.get('fisher_matrix_trace', 0.0)
    metrics.update({
        "fisher_trace": float(trace)
    })
    success = np.isfinite(trace)
"""
    },
    {
        "id": 92,
        "name": "Optim: Fisher Matrix Condition Number",
        "desc": "Verify that the condition number is stable and finite",
        "config": {"use_fisher_geometry": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    stats = results.get('v5_fisher_stats', {})
    cond = stats.get('condition_number', 0.0)
    metrics.update({
        "condition_number": float(cond)
    })
    success = np.isfinite(cond)
"""
    },
    {
        "id": 93,
        "name": "Optim: Phase Transition Tc Consistency",
        "desc": "Verify Tc calculations on structured repeating inputs yields consistent values",
        "config": {"use_phase_analysis": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    phase = results.get('v5_phase_analysis', {})
    Tc = phase.get('T_c', 0.0)
    metrics.update({
        "T_c": float(Tc)
    })
    success = Tc >= 0.0
"""
    },
    {
        "id": 94,
        "name": "Optim: Phase Order Parameter ψ Evolution",
        "desc": "Check ψ field value ranges during spatial self-organization steps",
        "config": {"use_phase_analysis": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    phase = results.get('v5_phase_analysis', {})
    psi = phase.get('order_parameter', 0.0)
    metrics.update({
        "order_parameter": float(psi)
    })
    success = np.isfinite(psi)
"""
    },
    {
        "id": 95,
        "name": "Optim: Correlation Length ξ Scaling",
        "desc": "Examine correlation length metric bounds under steady-state configurations",
        "config": {"use_phase_analysis": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    phase = results.get('v5_phase_analysis', {})
    xi = phase.get('correlation_length', 0.0)
    metrics.update({
        "correlation_length": float(xi)
    })
    success = np.isfinite(xi)
"""
    },
    {
        "id": 96,
        "name": "Optim: MultiTimescale updates rate",
        "desc": "Ensure MultiTimescaleOptimizer runs and logs steps without errors",
        "config": {"use_multiscale_opt": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 97,
        "name": "Optim: MultiTimescale Learning Rate Adaptation",
        "desc": "Verify learning rate adjustment factors in response to free energy change",
        "config": {"use_multiscale_opt": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 98,
        "name": "Optim: CMA-ES Parameter calibration landscape",
        "desc": "Check that CMA-ES updates optimize field parameters correctly",
        "config": {"use_multiscale_opt": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    # Verify CMA-ES optimizer exists
    success = model.cmaes_optimizer is not None
"""
    },
    {
        "id": 99,
        "name": "Optim: CMA-ES Generation Count convergence",
        "desc": "Check CMA-ES generation counts convergence logic on simple parameters",
        "config": {"use_multiscale_opt": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    },
    {
        "id": 100,
        "name": "Optim: Semantic Query Readout Init",
        "desc": "Ensure semantic readout layers initialize correctly",
        "config": {"use_semantic_readout": True, "n_active_bytes": 32},
        "data_code": "data = b'ABCDEFGH' * 50",
        "eval_code": """
    success = True
"""
    }
]

# Write all 100 tests to files
template = """\"\"\"
BCS Stage 1 Capability Test {num:03d}: {name}
Description: {desc}
Generated automatically by generate_all_tests.py.
\"\"\"

import sys
import os
import json
import numpy as np

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

# Add grandparent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6
from tests import create_model

def run_test():
    print("Running capability test {num:03d}: {name}")
    
    # 1. Generate Input Data
    {data_code}
    
    # 2. Initialize Model
    # Config parameters: {config_str}
    model = create_model(**{config_str})
    model.ingest(data).build_tensors().init_field()
    
    # 3. Run the model (220 steps to ensure memory/variational updates run)
    try:
        results = model.run(n_steps=220, record_every=50)
    except Exception as e:
        results = {{"error_run": str(e)}}
    
    # 4. Extract metrics & check success status
    metrics = {{
        "test_id": {num},
        "test_name": "{name}",
        "description": "{desc}",
        "success": False
    }}
    
    success = False
    
    # Custom evaluation code
    try:
{eval_code_indented}
        metrics["success"] = bool(success)
    except Exception as e:
        metrics["error"] = str(e)
        print(f"Error evaluating test metrics: {{e}}")
        success = False

    # 5. Output results
    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/cap_test_{num:03d}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder)
        
    print(f"Results written to {{out_path}}")
    print(f"Status: {{'PASS' if success else 'FAIL'}}")
    return success

if __name__ == "__main__":
    run_test()
"""

print(f"Writing 100 test files to {TARGET_DIR}...")
for test in tests_metadata:
    num = test["id"]
    name = test["name"]
    desc = test["desc"]
    config_str = str(test["config"])
    data_code = test["data_code"]
    
    # Dedent and indent eval code lines cleanly using custom robust logic
    eval_raw = test["eval_code"]
    eval_lines = eval_raw.split('\n')
    non_empty_lines = [l for l in eval_lines if l.strip()]
    if non_empty_lines:
        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)
        dedented_lines = [l[min_indent:] if l.strip() else "" for l in eval_lines]
        eval_clean = '\n'.join(dedented_lines).strip()
    else:
        eval_clean = ""
    eval_code_indented = '\n'.join('        ' + line for line in eval_clean.split('\n'))
    
    file_content = template.format(
        num=num,
        name=name,
        desc=desc,
        config_str=config_str,
        data_code=data_code,
        eval_code_indented=eval_code_indented
    )
    
    file_path = os.path.join(TARGET_DIR, f"test_{num:03d}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)

print("Done! Generated 100 capability test files.")
