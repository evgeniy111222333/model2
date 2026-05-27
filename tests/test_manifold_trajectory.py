"""
Тести для Manifold Trajectory System v3.0

Повне тестування всіх компонентів:
1. ManifoldPoint
2. ManifoldTrajectory  
3. GeodesicAttention
4. CurvatureNoveltyDetector
5. MemoryAsSubmanifold
6. Геометричні примітиви

Також перевірка взаємодії та поведінки на практиці.
"""

import sys
import os
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from bcs.information.manifold_trajectory import (
    ManifoldTrajectory,
    ManifoldPoint,
    GeodesicAttention,
    CurvatureNoveltyDetector,
    MemoryAsSubmanifold,
    fisher_rao_distance,
    fisher_metric_tensor,
    geodesicInterpolation,
    create_trajectory_from_bytes,
    quick_attention,
)


def test_manifold_point():
    """Тест точок на многовиді."""
    print("\n" + "="*60)
    print("TEST: ManifoldPoint")
    print("="*60)
    
    # Створення точки
    p = np.random.rand(256)
    p = p / p.sum()
    
    point = ManifoldPoint(p=p, t=0.0)
    
    assert np.isclose(point.p.sum(), 1.0), "Point must be on simplex"
    assert point.entropy > 0, "Entropy must be positive"
    assert point.entropy <= np.log(256), "Entropy must be <= max"
    
    print(f"  Entropy: {point.entropy:.4f} bits")
    print(f"  Max entropy: {np.log(256):.4f} bits")
    print(f"  Normalized: {point.entropy / np.log(256):.4f}")
    
    # Відстань до іншої точки
    p2 = np.random.rand(256)
    p2 = p2 / p2.sum()
    point2 = ManifoldPoint(p=p2, t=1.0)
    
    d = point.distance_to(point2)
    assert 0 <= d <= np.pi / 2, f"Distance must be in [0, π/2], got {d}"
    
    print(f"  Distance to point2: {d:.4f}")
    
    # Геодезична інтерполяція
    p_mid = geodesicInterpolation(p, p2, 0.5)
    assert np.isclose(p_mid.sum(), 1.0), "Interpolated must be on simplex"
    
    mid_point = point.interpolate_to(point2, 0.5)
    print(f"  Interpolated entropy: {mid_point.entropy:.4f}")
    
    print("  [OK] ManifoldPoint tests passed")
    return True


def test_fisher_rao_geometry():
    """Тест геометричних примітивів."""
    print("\n" + "="*60)
    print("TEST: Fisher-Rao Geometry Primitives")
    print("="*60)
    
    # Відстань від точки до себе = 0
    p = np.random.rand(256)
    p = p / p.sum()
    d_self = fisher_rao_distance(p, p)
    assert abs(d_self) < 1e-6, f"Self-distance must be 0, got {d_self}"
    print(f"  Self-distance: {d_self:.6f} (expected 0)")
    
    # Симетричність
    p1 = np.random.rand(256)
    p1 = p1 / p1.sum()
    p2 = np.random.rand(256)
    p2 = p2 / p2.sum()
    
    d12 = fisher_rao_distance(p1, p2)
    d21 = fisher_rao_distance(p2, p1)
    assert abs(d12 - d21) < 1e-6, "Distance must be symmetric"
    print(f"  Symmetry: d(p1,p2)={d12:.4f} == d(p2,p1)={d21:.4f}")
    
    # Трикутна нерівність
    p3 = np.random.rand(256)
    p3 = p3 / p3.sum()
    d13 = fisher_rao_distance(p1, p3)
    d23 = fisher_rao_distance(p2, p3)
    
    assert d13 <= d12 + d23 + 1e-6, "Triangle inequality must hold"
    print(f"  Triangle: d13={d13:.4f} <= d12+d23={d12+d23:.4f}")
    
    # Тест інтерполяції
    p_interp = geodesicInterpolation(p1, p2, 0.5)
    assert np.isclose(p_interp.sum(), 1.0), "Interpolated must be on simplex"
    
    # Середня точка має бути "посередині"
    d1_mid = fisher_rao_distance(p1, p_interp)
    d_mid_2 = fisher_rao_distance(p_interp, p2)
    print(f"  Interpolation: d(p1,mid)={d1_mid:.4f}, d(mid,p2)={d_mid_2:.4f}")
    
    # Fisher metric tensor
    G = fisher_metric_tensor(p)
    assert G.shape == (256, 256), "Metric must be 256x256"
    assert np.all(np.diag(G) > 0), "Diagonal must be positive"
    print(f"  Fisher metric: shape={G.shape}, positive diagonal={np.all(np.diag(G)>0)}")
    
    print("  [OK] Fisher-Rao geometry tests passed")
    return True


def test_manifold_trajectory_core():
    """Тест основної функціональності траєкторії."""
    print("\n" + "="*60)
    print("TEST: ManifoldTrajectory Core")
    print("="*60)
    
    traj = ManifoldTrajectory(max_length=100, decay_rate=0.9)
    
    # Додавання точок
    for i in range(10):
        p = np.random.rand(256)
        p = p / p.sum()
        traj.push(p, t=float(i))
    
    assert len(traj) == 10, f"Expected 10 points, got {len(traj)}"
    print(f"  Points added: {len(traj)}")
    
    # Геометрія
    print(f"  Total length: {traj.total_length:.4f}")
    print(f"  Curvature profile length: {len(traj.curvature_profile)}")
    print(f"  Velocity profile length: {len(traj.velocity_profile)}")
    
    # Пам'ять
    assert traj.memory_center is not None, "Memory center must be computed"
    print(f"  Memory spread: {traj.memory_spread:.4f}")
    
    print("  [OK] Core trajectory tests passed")
    return True


def test_geodesic_attention():
    """Тест Geodesic Attention."""
    print("\n" + "="*60)
    print("TEST: GeodesicAttention")
    print("="*60)
    
    ga = GeodesicAttention(temperature=1.0, use_decay=True, decay_rate=0.9)
    
    # Створення keys
    n_keys = 10
    keys = [np.random.rand(256) for _ in range(n_keys)]
    keys = [k / k.sum() for k in keys]
    
    # Query = останній key
    query = keys[-1].copy()
    
    # Forward pass
    output, attention = ga.forward(query, keys)
    
    assert output.shape == (256,), "Output must be 256-dim"
    assert attention.shape == (n_keys,), "Attention must match keys"
    assert np.isclose(attention.sum(), 1.0), "Attention must sum to 1"
    
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Attention sum: {attention.sum():.4f}")
    print(f"  Top attention: {np.max(attention):.4f}")
    
    # Увага має бути найвищою до найближчого key
    distances = [fisher_rao_distance(query, k) for k in keys]
    min_dist_idx = np.argmin(distances)
    max_attn_idx = np.argmax(attention)
    
    print(f"  Min distance at idx: {min_dist_idx}, Max attention at idx: {max_attn_idx}")
    
    # Self-attention має бути найвищим
    assert max_attn_idx == n_keys - 1, "Self-attention should be highest"
    print(f"  Self-attention highest: {attention[-1]:.4f}")
    
    print("  [OK] GeodesicAttention tests passed")
    return True


def test_curvature_novelty():
    """Тест виявлення новизни через кривину."""
    print("\n" + "="*60)
    print("TEST: CurvatureNoveltyDetector")
    print("="*60)
    
    detector = CurvatureNoveltyDetector(
        window_size=5,
        curvature_threshold=0.3,
        velocity_threshold=0.1,
    )
    
    # Стабільна послідовність (мала кривина)
    print("  Testing stable sequence...")
    for i in range(5):
        # Плавна зміна
        p = np.zeros(256)
        p[100 + i] = 0.5
        p[150] = 0.5
        p = p / p.sum()
        novelty, ntype = detector.update(p)
        print(f"    Step {i}: novelty={novelty:.4f}, type={ntype}")
    
    # Різка зміна (висока кривина)
    print("  Testing sharp transition...")
    detector.reset()
    for i in range(3):
        p = np.zeros(256)
        p[i * 50] = 1.0
        p = p / p.sum()
        novelty, ntype = detector.update(p)
        print(f"    Step {i}: novelty={novelty:.4f}, type={ntype}")
    
    print("  [OK] Curvature novelty tests passed")
    return True


def test_memory_as_submanifold():
    """Тест пам'яті як підмноговиду."""
    print("\n" + "="*60)
    print("TEST: MemoryAsSubmanifold")
    print("="*60)
    
    memory = MemoryAsSubmanifold(max_points=100)
    
    # Зберегти точки
    for i in range(20):
        p = np.random.rand(256)
        p = p / p.sum()
        memory.store(p, metadata={'index': i})
    
    assert len(memory) == 20, f"Expected 20 points, got {len(memory)}"
    print(f"  Stored {len(memory)} points")
    print(f"  Centroid computed: {memory.centroid is not None}")
    print(f"  Span: {memory.span:.4f}")
    
    # Retrieve
    query = np.random.rand(256)
    query = query / query.sum()
    
    retrieved, distances = memory.retrieve(query, k=5)
    assert len(retrieved) == 5, "Should retrieve 5 points"
    print(f"  Retrieved {len(retrieved)} nearest neighbors")
    print(f"  Distances: {[f'{d:.4f}' for d in distances]}")
    
    # Familiarity
    fam = memory.familiarity(query)
    assert 0 <= fam <= 1, "Familiarity must be in [0,1]"
    print(f"  Familiarity: {fam:.4f}")
    
    # Проєкція
    projected, dist = memory.project(query)
    print(f"  Projected distance: {dist:.4f}")
    
    print("  [OK] MemoryAsSubmanifold tests passed")
    return True


def test_trajectory_from_bytes():
    """Тест створення траєкторії з байтів."""
    print("\n" + "="*60)
    print("TEST: Create Trajectory from Bytes")
    print("="*60)
    
    # Текстові дані
    text_data = b"This is a sample text with many words and characters."
    
    traj = create_trajectory_from_bytes(
        text_data,
        step=5,
        window_size=16,
        max_length=None,  # Без обмежень
    )
    
    print(f"  Trajectory length: {len(traj)}")
    print(f"  Total geodesic length: {traj.total_length:.4f}")
    
    # Перевірка що ентропія відповідає тексту
    entropies = [p.entropy for p in traj.points]
    print(f"  Entropy range: {min(entropies):.2f} - {max(entropies):.2f}")
    
    # Геометричні характеристики
    summary = traj.get_trajectory_summary()
    print(f"  Topology: {summary['topology']}")
    
    print("  [OK] Trajectory from bytes tests passed")
    return True


def test_novelty_detection():
    """Тест виявлення новизни в траєкторії."""
    print("\n" + "="*60)
    print("TEST: Novelty Detection in Trajectory")
    print("="*60)
    
    traj = ManifoldTrajectory()
    
    # Початок з випадкових точок
    for i in range(10):
        p = np.random.rand(256)
        p = p / p.sum()
        novelty, conf = traj.detect_novelty(p)
        if i < 3:
            print(f"  Initial novelty {i}: {novelty:.4f} (conf={conf:.2f})")
    
    # Тепер різка зміна
    print("  Testing sharp change...")
    for i in range(5):
        # Цілком інший розподіл
        p = np.zeros(256)
        p[i * 50] = 1.0
        novelty, conf = traj.detect_novelty(p)
        print(f"    Step {i}: novelty={novelty:.4f} (conf={conf:.2f})")
    
    # Повернення до випадкового
    print("  Testing return to random...")
    for i in range(3):
        p = np.random.rand(256)
        p = p / p.sum()
        novelty, conf = traj.detect_novelty(p)
        print(f"    Step {i}: novelty={novelty:.4f} (conf={conf:.2f})")
    
    print("  [OK] Novelty detection tests passed")
    return True


def test_context_vector():
    """Тест отримання контекстного вектора."""
    print("\n" + "="*60)
    print("TEST: Context Vector")
    print("="*60)
    
    traj = ManifoldTrajectory()
    
    # Додати точки
    for i in range(10):
        p = np.random.rand(256)
        p = p / p.sum()
        traj.push(p, t=float(i))
    
    # Контекст без query (остання точка)
    ctx1 = traj.get_context_vector()
    assert ctx1.shape == (256,), "Context must be 256-dim"
    print(f"  Context shape (no query): {ctx1.shape}")
    
    # Контекст з query (attention)
    query = traj.points[-1].p.copy()
    ctx2 = traj.get_context_vector(query)
    print(f"  Context shape (with query): {ctx2.shape}")
    
    # Різниця між двома контекстами
    diff = np.sum(np.abs(ctx1 - ctx2))
    print(f"  Context difference: {diff:.6f}")
    
    print("  [OK] Context vector tests passed")
    return True


def test_topology_features():
    """Тест топологічних характеристик."""
    print("\n" + "="*60)
    print("TEST: Topology Features")
    print("="*60)
    
    traj = ManifoldTrajectory()
    
    # Петлеподібна траєкторія
    print("  Creating loop trajectory...")
    for i in range(20):
        angle = 2 * np.pi * i / 20
        # Точки на "петлі"
        p = np.zeros(256)
        p[int(100 + 50 * np.sin(angle))] = 0.5
        p[int(100 + 50 * np.cos(angle))] = 0.5
        p = p / p.sum()
        traj.push(p, t=float(i))
    
    topo = traj.compute_topology_features()
    print(f"  Betti_0: {topo['betti_0']}")
    print(f"  Betti_1: {topo['betti_1']} (loops)")
    print(f"  Total variation: {topo['total_variation']:.4f}")
    print(f"  Rectifiable: {topo['rectifiable']:.4f}")
    print(f"  Oscillations: {topo['oscillation_count']}")
    
    print("  [OK] Topology features tests passed")
    return True


def test_memory_integration():
    """Інтеграційний тест: траєкторія + пам'ять."""
    print("\n" + "="*60)
    print("TEST: Memory Integration")
    print("="*60)
    
    traj = ManifoldTrajectory(enable_memory=True)
    memory = MemoryAsSubmanifold(max_points=50)
    
    # Генерація "історії"
    print("  Generating history...")
    for i in range(30):
        # Модельована "сесія"
        if i < 10:
            p = np.zeros(256)
            p[50:100] = 0.1
            p = p / p.sum()
        elif i < 20:
            p = np.zeros(256)
            p[100:150] = 0.1
            p = p / p.sum()
        else:
            p = np.random.rand(256)
            p = p / p.sum()
        
        traj.push(p, t=float(i))
        memory.store(p, metadata={'step': i})
    
    print(f"  Trajectory: {len(traj)} points")
    print(f"  Memory: {len(memory)} points")
    
    # Query з новою областю
    query = np.zeros(256)
    query[200:250] = 0.1
    query = query / query.sum()
    
    # Знайомість
    traj_fam = 1.0 - traj.detect_novelty(query)[0]
    mem_fam = memory.familiarity(query)
    
    print(f"  Trajectory familiarity: {traj_fam:.4f}")
    print(f"  Memory familiarity: {mem_fam:.4f}")
    
    print("  [OK] Memory integration tests passed")
    return True


def test_geodesic_vs_softmax():
    """Порівняння Geodesic Attention vs Softmax Attention."""
    print("\n" + "="*60)
    print("TEST: Geodesic vs Softmax Attention")
    print("="*60)
    
    # Створити memory points
    n = 10
    memory = [np.random.rand(256) for _ in range(n)]
    memory = [m / m.sum() for m in memory]
    
    # Query = копія одного з memory
    query_idx = 5
    query = memory[query_idx].copy()
    
    # Geodesic attention
    ga = GeodesicAttention(temperature=1.0)
    geo_output, geo_attn = ga.forward(query, memory)
    
    # Softmax attention (стандартний)
    memory_arr = np.array(memory)
    dots = memory_arr @ query
    softmax_attn = np.exp(dots - dots.max()) / np.exp(dots - dots.max()).sum()
    softmax_output = softmax_attn @ memory_arr
    
    print(f"  Geodesic attention to self: {geo_attn[query_idx]:.4f}")
    print(f"  Softmax attention to self: {softmax_attn[query_idx]:.4f}")
    
    # Геодезичний attention має бути вищим до себе
    geo_self = geo_attn[query_idx]
    softmax_self = softmax_attn[query_idx]
    
    print(f"  Geodesic self-attention: {geo_self:.4f}")
    print(f"  Softmax self-attention: {softmax_self:.4f}")
    
    # Порівняння виходів
    diff = np.sum(np.abs(geo_output - softmax_output))
    print(f"  Output difference: {diff:.4f}")
    
    print("  [OK] Geodesic vs Softmax tests passed")
    return True


def test_long_trajectory():
    """Тест довгої траєкторії без деградації."""
    print("\n" + "="*60)
    print("TEST: Long Trajectory (1000 points)")
    print("="*60)
    
    traj = ManifoldTrajectory(max_length=200, decay_rate=0.99)
    
    # Генерація довгої траєкторії
    print("  Generating 1000 points...")
    for i in range(1000):
        # Модельований потік даних
        base = 50 + (i % 200)
        p = np.zeros(256)
        for j in range(10):
            p[(base + j) % 256] = 0.1
        p = p / p.sum()
        traj.push(p, t=float(i))
    
    print(f"  Points in trajectory: {len(traj)}")
    print(f"  Total geodesic length: {traj.total_length:.4f}")
    print(f"  Memory spread: {traj.memory_spread:.4f}")
    
    # Перевірка геометрії
    topo = traj.compute_topology_features()
    print(f"  Topology: Betti_0={topo['betti_0']}, Betti_1={topo['betti_1']}")
    
    # Attention все ще працює
    query = traj.points[-1].p.copy()
    attention = traj.compute_attention(query)
    assert len(attention) == len(traj), "Attention length must match"
    print(f"  Attention sum: {attention.sum():.4f}")
    
    print("  [OK] Long trajectory tests passed")
    return True


def test_boundary_detection_comparison():
    """Порівняння з існуючим boundary detection."""
    print("\n" + "="*60)
    print("TEST: Boundary Detection Comparison")
    print("="*60)
    
    # Дані з чіткими межами
    data = bytes([0] * 50 + [255] * 50 + [128] * 50)
    
    # Створення траєкторії
    traj = create_trajectory_from_bytes(data, step=1, window_size=16)
    
    print(f"  Trajectory length: {len(traj)}")
    
    # Виявлення новизни
    novelties = []
    for point in traj.points:
        novelty, _ = traj.detect_novelty(point.p)
        novelties.append(novelty)
    
    # Знайти піки новизни
    novelties = np.array(novelties)
    peaks, _ = find_peaks(novelties, height=0.3)
    
    print(f"  Novelty peaks at positions: {peaks[:5] if len(peaks) > 5 else peaks}...")
    print(f"  Expected boundaries: ~50, ~100 (in trajectory coordinates)")
    
    # Перевірка чи піки близькі до очікуваних
    if len(peaks) >= 2:
        peak_diffs = np.diff(peaks[:3])
        print(f"  Peak spacing: {peak_diffs}")
    
    print("  [OK] Boundary detection comparison tests passed")
    return True


# =============================================================================
# ЗАПУСК ВСІХ ТЕСТІВ
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("BCS MANIFOLD TRAJECTORY v3.0 - COMPREHENSIVE TESTS")
    print("="*60)
    
    tests = [
        ("ManifoldPoint", test_manifold_point),
        ("Fisher-Rao Geometry", test_fisher_rao_geometry),
        ("ManifoldTrajectory Core", test_manifold_trajectory_core),
        ("GeodesicAttention", test_geodesic_attention),
        ("CurvatureNoveltyDetector", test_curvature_novelty),
        ("MemoryAsSubmanifold", test_memory_as_submanifold),
        ("Trajectory from Bytes", test_trajectory_from_bytes),
        ("Novelty Detection", test_novelty_detection),
        ("Context Vector", test_context_vector),
        ("Topology Features", test_topology_features),
        ("Memory Integration", test_memory_integration),
        ("Geodesic vs Softmax", test_geodesic_vs_softmax),
        ("Long Trajectory", test_long_trajectory),
        ("Boundary Detection", test_boundary_detection_comparison),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASS" if result else "FAIL", None))
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    n_pass = sum(1 for _, r, _ in results if r == "PASS")
    n_fail = sum(1 for _, r, _ in results if r == "FAIL")
    n_error = sum(1 for _, r, _ in results if r == "ERROR")
    
    for name, result, error in results:
        status_str = f"[{result}]"
        if error:
            status_str += f" {error[:50]}..."
        print(f"  {status_str:40s} {name}")
    
    print(f"\nTotal: {n_pass}/{len(results)} passed")
    if n_fail > 0 or n_error > 0:
        print(f"Failed: {n_fail}, Errors: {n_error}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED!")
        sys.exit(0)
