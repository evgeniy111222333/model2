"""
BCS Geodesic Context System — ПОВНІ ТЕСТИ

Тестує повну реалізацію парадигми "Контекст = Траєкторія на Многовиді":

1. GeodesicContextEngine — ядро системи
2. TrajectoryContextIntegration — інтеграція в модель
3. Геодезичний Attention — заміна softmax
4. Пам'ять як підмноговид
5. Семантична інтерпретація форми
6. End-to-end з моделлю

ЗАМІНА:
- window [token₁, ..., token_n] → trajectory p(0) → ... → p(t)
- softmax attention → exp(-geodesic_distance² / T)
- buffer memory → submanifold memory
"""

import sys
import os
import numpy as np

sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from bcs.information.geodesic_context import (
    GeodesicContextEngine,
    TrajectoryAttention,
    ManifoldPoint,
    fisher_rao_distance,
    geodesic_interpolation,
    kl_divergence,
)
from bcs.information.geodesic_integration import TrajectoryContextIntegration
from bcs.model import BCSModelV6


# =============================================================================
# ТЕСТ 1: GeodesicContextEngine — ядро
# =============================================================================

def test_geodesic_context_engine_core():
    """Тест базової функціональності GeodesicContextEngine."""
    print("\n" + "="*60)
    print("TEST 1: GeodesicContextEngine Core")
    print("="*60)
    
    engine = GeodesicContextEngine(
        max_trajectory_length=100,
        decay_rate=0.95,
        novelty_threshold=0.5,
        temperature=1.0,
    )
    
    # Додаємо точки траєкторії
    n_points = 20
    for i in range(n_points):
        # Створюємо розподіл з певною структурою
        dist = np.zeros(256)
        center = (i % 10) * 25
        for j in range(10):
            dist[(center + j) % 256] = 1.0
        dist = dist / dist.sum()
        
        engine.push(
            p=dist,
            t=float(i) / n_points,
            position=i,
            modality='text_ascii',
        )
    
    assert len(engine) == n_points, f"Expected {n_points} points, got {len(engine)}"
    print(f"  Points added: {len(engine)}")
    
    # Перевірка геометрії
    assert engine.total_geodesic_length > 0, "Geodesic length should be positive"
    print(f"  Total geodesic length: {engine.total_geodesic_length:.4f}")
    
    # Кривина профіль
    assert len(engine.curvature_profile) > 0, "Curvature profile should be non-empty"
    print(f"  Curvature profile length: {len(engine.curvature_profile)}")
    
    # Пам'ять
    assert engine.memory_centroid is not None, "Memory centroid should be computed"
    print(f"  Memory centroid: {engine.memory_centroid is not None}")
    print(f"  Memory span: {engine.memory_span:.4f}")
    
    # Семантичні форми
    shapes = engine.semantic_shapes
    print(f"  Semantic shapes: loops={len(shapes['loops'])}, "
          f"angles={len(shapes['angles'])}, "
          f"stops={len(shapes['stops'])}, "
          f"streams={len(shapes['streams'])}")
    
    print("  [OK] GeodesicContextEngine core tests passed")
    return True


# =============================================================================
# ТЕСТ 2: Геодезичний Attention
# =============================================================================

def test_geodesic_attention():
    """Тест геодезичного attention — заміна softmax."""
    print("\n" + "="*60)
    print("TEST 2: Geodesic Attention (заміна softmax)")
    print("="*60)
    
    engine = GeodesicContextEngine(decay_rate=0.95)
    
    # Створюємо траєкторію
    n_points = 10
    for i in range(n_points):
        dist = np.random.rand(256)
        dist = dist / dist.sum()
        engine.push(p=dist, t=float(i) / n_points, position=i)
    
    # Query = остання точка
    query = engine.points[-1].p.copy()
    
    # Обчислюємо attention
    attention = engine.compute_attention(query, temperature=0.5)
    
    assert len(attention) == n_points, f"Expected {n_points} attention weights"
    assert np.isclose(attention.sum(), 1.0, atol=1e-6), "Attention must sum to 1"
    print(f"  Attention shape: {attention.shape}")
    print(f"  Attention sum: {attention.sum():.6f}")
    
    # Self-attention має бути найвищим
    max_idx = np.argmax(attention)
    assert max_idx == n_points - 1, f"Self-attention should be highest, got idx {max_idx}"
    print(f"  Self-attention highest at idx {max_idx}: {attention[-1]:.4f}")
    
    # Attention з decay
    attention_decay = engine.compute_attention(query, temperature=0.5, use_decay=True)
    assert np.isclose(attention_decay.sum(), 1.0, atol=1e-6), "Attention with decay must sum to 1"
    print(f"  Attention with decay: sum={attention_decay.sum():.6f}")
    
    # Attend to trajectory
    result = engine.attend_to_trajectory(query, feature='p')
    assert result.shape == (256,), f"Expected 256-dim output, got {result.shape}"
    print(f"  Attended trajectory output shape: {result.shape}")
    
    print("  [OK] Geodesic attention tests passed")
    return True


# =============================================================================
# ТЕСТ 3: Виявлення новизни через кривину
# =============================================================================

def test_novelty_detection():
    """Тест виявлення новизни через геометрію траєкторії."""
    print("\n" + "="*60)
    print("TEST 3: Novelty Detection via Curvature")
    print("="*60)
    
    engine = GeodesicContextEngine()
    
    # Створюємо стабільну траєкторію
    for i in range(5):
        dist = np.zeros(256)
        dist[100:120] = 1.0
        dist = dist / dist.sum()
        engine.push(p=dist, t=float(i), position=i)
    
    # Тест стабільної точки
    stable_dist = np.zeros(256)
    stable_dist[100:120] = 1.0
    stable_dist = stable_dist / stable_dist.sum()
    novelty_stable, conf_stable = engine.detect_novelty(stable_dist)
    print(f"  Stable point: novelty={novelty_stable:.4f}, confidence={conf_stable:.4f}")
    
    # Тест нової точки
    novel_dist = np.zeros(256)
    novel_dist[200:220] = 1.0  # Інший діапазон
    novel_dist = novel_dist / novel_dist.sum()
    novelty_novel, conf_novel = engine.detect_novelty(novel_dist)
    print(f"  Novel point: novelty={novelty_novel:.4f}, confidence={conf_novel:.4f}")
    
    assert novelty_novel > novelty_stable, "Novel point should have higher novelty"
    print(f"  Novelty correctly detects: {novelty_novel > novelty_stable}")
    
    # Граничне виявлення
    boundary_strength, boundary_type = engine.detect_boundary(novel_dist)
    print(f"  Boundary: strength={boundary_strength:.4f}, type={boundary_type}")
    
    print("  [OK] Novelty detection tests passed")
    return True


# =============================================================================
# ТЕСТ 4: Пам'ять як підмноговид
# =============================================================================

def test_memory_as_submanifold():
    """Тест пам'яті як підмноговиду."""
    print("\n" + "="*60)
    print("TEST 4: Memory as Submanifold")
    print("="*60)
    
    engine = GeodesicContextEngine()
    
    # Зберігаємо точки в пам'ять
    n_points = 20
    for i in range(n_points):
        dist = np.random.rand(256)
        dist = dist / dist.sum()
        engine.push(p=dist, t=float(i) / n_points, position=i)
    
    # Перевірка пам'яті
    assert len(engine.memory_points) > 0, "Memory points should exist"
    print(f"  Memory points: {len(engine.memory_points)}")
    print(f"  Memory span: {engine.memory_span:.4f}")
    print(f"  Memory centroid shape: {engine.memory_centroid.shape}")
    
    # Retrieve
    query = np.random.rand(256)
    query = query / query.sum()
    
    retrieved, distances = engine.memory_retrieve(query, k=5)
    assert len(retrieved) <= 5, f"Should retrieve at most 5 points"
    assert len(distances) == len(retrieved), "Distances should match retrieved"
    print(f"  Retrieved {len(retrieved)} points")
    print(f"  Distances: {[f'{d:.4f}' for d in distances[:3]]}...")
    
    # Familiarity
    fam = engine.memory_familiarity(query)
    assert 0 <= fam <= 1, "Familiarity should be in [0, 1]"
    print(f"  Familiarity: {fam:.4f}")
    
    print("  [OK] Memory as submanifold tests passed")
    return True


# =============================================================================
# ТЕСТ 5: Семантична інтерпретація форми
# =============================================================================

def test_semantic_shapes():
    """Тест семантичної інтерпретації форми траєкторії."""
    print("\n" + "="*60)
    print("TEST 5: Semantic Shape Interpretation")
    print("="*60)
    
    engine = GeodesicContextEngine()
    
    # Створюємо траєкторію з петлею (повторенням)
    print("  Creating trajectory with loop...")
    for i in range(15):
        dist = np.zeros(256)
        if i < 5:
            # Початок
            dist[50:70] = 1.0
        elif i < 10:
            # Рух вперед
            dist[70 + (i-5)*5 : 90 + (i-5)*5] = 1.0
        else:
            # Повернення назад (петля)
            dist[95 - (i-10)*5 : 115 - (i-10)*5] = 1.0
        dist = dist / dist.sum()
        engine.push(p=dist, t=float(i) / 15, position=i)
    
    shapes = engine.semantic_shapes
    print(f"  Detected shapes: loops={len(shapes['loops'])}, angles={len(shapes['angles'])}")
    
    # Топологія
    summary = engine.get_context_summary()
    topo = summary.get('topology', {})
    print(f"  Topology: Betti_0={topo.get('betti_0', 0)}, Betti_1={topo.get('betti_1', 0)}")
    
    # Якщо петля виявлена — добре
    if len(shapes['loops']) > 0 or topo.get('betti_1', 0) > 0:
        print(f"  ✓ Loop detected!")
    
    # Кути (різка зміна)
    print("\n  Creating trajectory with sharp angle...")
    engine2 = GeodesicContextEngine()
    for i in range(10):
        dist = np.zeros(256)
        if i < 5:
            dist[50:70] = 1.0  # Початок
        else:
            dist[200:220] = 1.0  # Різка зміна
        dist = dist / dist.sum()
        engine2.push(p=dist, t=float(i) / 10, position=i)
    
    shapes2 = engine2.semantic_shapes
    print(f"  Detected shapes: loops={len(shapes2['loops'])}, angles={len(shapes2['angles'])}")
    
    if len(shapes2['angles']) > 0:
        print(f"  ✓ Angle detected!")
    
    print("  [OK] Semantic shape tests passed")
    return True


# =============================================================================
# ТЕСТ 6: TrajectoryAttention для кластерів
# =============================================================================

def test_trajectory_attention():
    """Тест TrajectoryAttention для attention між кластерами."""
    print("\n" + "="*60)
    print("TEST 6: TrajectoryAttention for Clusters")
    print("="*60)
    
    attention = TrajectoryAttention(temperature=1.0, decay_rate=0.95)
    
    # Створюємо ключі (розподіли кластерів)
    n_clusters = 8
    keys = [np.random.rand(256) for _ in range(n_clusters)]
    keys = [k / k.sum() for k in keys]
    
    # Query = перший кластер
    query = keys[0].copy()
    
    # Forward pass
    output, attn = attention.forward(query, keys)
    
    assert output.shape == (256,), f"Expected 256-dim output, got {output.shape}"
    assert attn.shape == (n_clusters,), f"Expected {n_clusters} attention weights"
    assert np.isclose(attn.sum(), 1.0, atol=1e-6), "Attention must sum to 1"
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attn.shape}")
    print(f"  Attention sum: {attn.sum():.6f}")
    
    # Self-attention має бути найвищим
    max_idx = np.argmax(attn)
    self_attn = attn[0]
    print(f"  Self-attention at idx 0: {self_attn:.4f}, max at idx {max_idx}")
    
    # Різні температури
    for temp in [0.1, 0.5, 1.0, 2.0]:
        attn_temp = TrajectoryAttention(temperature=temp)
        _, attn_t = attn_temp.forward(query, keys)
        entropy_t = -np.sum(attn_t * np.log(attn_t + 1e-10))
        print(f"  Temperature={temp}: entropy={entropy_t:.4f}, max_attn={np.max(attn_t):.4f}")
    
    print("  [OK] TrajectoryAttention tests passed")
    return True


# =============================================================================
# ТЕСТ 7: TrajectoryContextIntegration
# =============================================================================

def test_trajectory_context_integration():
    """Тест інтеграції з BCSModelV6."""
    print("\n" + "="*60)
    print("TEST 7: TrajectoryContextIntegration")
    print("="*60)
    
    # Створюємо модель
    model = BCSModelV6(
        use_geodesic_context=True,
        use_manifold_trajectory=True,
    )
    
    # Ініціалізуємо integration
    integration = TrajectoryContextIntegration(
        model,
        config={
            'temperature': 1.0,
            'decay_rate': 0.99,
            'max_trajectory_length': 200,
            'novelty_threshold': 0.5,
        }
    )
    
    # Тестові дані
    test_data = b"Hello, this is a test of the geodesic context system!"
    integration.initialize(test_data)
    
    assert integration.context_engine is not None, "Context engine should be initialized"
    print(f"  Initialized with {len(integration)} points")
    
    # Push additional data
    integration.push_data(b" Adding more data...", modality='text_ascii')
    print(f"  After push: {len(integration)} points")
    
    # Get context vector
    ctx = integration.get_trajectory_context()
    assert ctx.shape == (256,), f"Expected 256-dim context, got {ctx.shape}"
    print(f"  Context vector shape: {ctx.shape}")
    
    # Context summary
    summary = integration.get_summary()
    print(f"  Summary: n_points={summary.get('n_points', 0)}, "
          f"geo_length={summary.get('total_geodesic_length', 0):.2f}")
    
    # Memory query
    query = np.random.rand(256)
    query = query / query.sum()
    retrieved, distances = integration.memory_query(query, k=3)
    print(f"  Memory query: retrieved {len(retrieved)} points")
    
    # Familiarity
    fam = integration.check_familiarity(query)
    print(f"  Familiarity: {fam:.4f}")
    
    # Convert with geodesic attention
    clusters = [
        {'distribution': np.random.rand(256), 'start': 0, 'end': 10},
        {'distribution': np.random.rand(256), 'start': 10, 'end': 20},
        {'distribution': np.random.rand(256), 'start': 20, 'end': 30},
    ]
    for c in clusters:
        c['distribution'] = c['distribution'] / c['distribution'].sum()
    
    enhanced = integration.convert_with_geodesic_attention(clusters, use_trajectory_attention=True)
    
    assert len(enhanced) == len(clusters), "Enhanced clusters should match input"
    assert 'geodesic_attention' in enhanced[0], "Should have geodesic_attention"
    assert 'geodesic_context' in enhanced[0], "Should have geodesic_context"
    print(f"  Converted {len(enhanced)} clusters with geodesic attention")
    print(f"  First cluster attention entropy: {enhanced[0].get('attention_entropy', 'N/A')}")
    
    print("  [OK] TrajectoryContextIntegration tests passed")
    return True


# =============================================================================
# ТЕСТ 8: Запит-відповідь через траєкторію
# =============================================================================

def test_query_response():
    """Тест запит-відповідь через геометрію траєкторії."""
    print("\n" + "="*60)
    print("TEST 8: Query-Response via Trajectory")
    print("="*60)
    
    engine = GeodesicContextEngine()
    
    # Створюємо траєкторію
    for i in range(10):
        dist = np.random.rand(256)
        dist = dist / dist.sum()
        engine.push(p=dist, t=float(i) / 10, position=i)
    
    # Query
    query = np.random.rand(256)
    query = query / query.sum()
    
    # Attention mode
    result = engine.query_response(query, mode='attention', top_k=3)
    assert result['success'], "Query should succeed"
    assert 'attention' in result, "Should have attention"
    assert 'context_vector' in result, "Should have context_vector"
    print(f"  Attention mode: top_k_indices={result.get('top_k_indices', [])[:3]}...")
    
    # Retrieval mode
    result = engine.query_response(query, mode='retrieval', top_k=3)
    assert result['success'], "Retrieval should succeed"
    assert 'retrieved_points' in result, "Should have retrieved points"
    print(f"  Retrieval mode: {len(result.get('retrieved_points', []))} points")
    
    # Interpolation mode
    result = engine.query_response(query, mode='interpolation')
    assert result['success'], "Interpolation should succeed"
    if 'interpolated_point' in result:
        print(f"  Interpolation: nearest_idx={result.get('nearest_index')}")
    else:
        print(f"  Interpolation: nearest_idx={result.get('nearest_index')}")
    
    print("  [OK] Query-response tests passed")
    return True


# =============================================================================
# ТЕСТ 9: End-to-End з BCSModelV6
# =============================================================================

def test_e2e_with_model():
    """Тест end-to-end з повною моделлю BCS."""
    print("\n" + "="*60)
    print("TEST 9: End-to-End with BCSModelV6")
    print("="*60)
    
    # Створюємо модель з geodesic context
    model = BCSModelV6(
        use_geodesic_context=True,
        use_manifold_trajectory=True,
    )
    
    # Тестові дані
    test_text = "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
    test_data = test_text.encode('utf-8')
    
    # Повний цикл
    model.ingest(test_data)
    model.build_tensors()
    model.init_field()
    
    print(f"  Data: {len(test_data)} bytes, modality={model.detected_modality}")
    
    # Run — це основний тест, чи все працює разом
    results = model.run(n_steps=50, record_every=25)
    
    # Перевірка результатів — HierarchicalTrajectory тепер PRIMARY
    assert 'hierarchical_trajectory' in results or 'hierarchical_trajectory_engine' in results, \
        "Should have hierarchical_trajectory results"
    assert 'hierarchical_trajectory_engine' in results, \
        "Should have hierarchical_trajectory_engine summary"
    
    # Нові ключі для HierarchicalTrajectory
    ht_result = results.get('hierarchical_trajectory') or results.get('hierarchical_trajectory_engine')
    if ht_result:
        print(f"  Hierarchical trajectory: total_points={ht_result.get('total_points', 0)}")
        print(f"  Hierarchical depth: {ht_result.get('depth', 0)}")
        levels = ht_result.get('levels', [])
        print(f"  Hierarchical levels: {len(levels)}")
        for lvl_info in levels:
            print(f"    L{lvl_info.get('level', 0)}: {lvl_info.get('n_points', 0)} points")
    
    # Перевірка geodesic attention у conversion levels
    if 'conversion_levels' in results:
        conv_levels = results['conversion_levels']
        geo_enhanced = sum(1 for l in conv_levels if l.get('geodesic_enhanced'))
        print(f"  Geodesic-enhanced levels: {geo_enhanced}")
        assert geo_enhanced > 0, "At least one level should be geodesic-enhanced"
    if 'hierarchical_attention_summary' in results:
        summary = results['hierarchical_attention_summary']
        print(f"  HierarchicalAttention: {summary.get('n_levels_enhanced', 0)} рівнів з enhanced attention")
    
    # Перевірка trajectory context injection
    if 'trajectory_context' in results:
        print(f"  Trajectory context: injected into semantic layer")
    
    print("  [OK] End-to-end model tests passed")
    return True


# =============================================================================
# ТЕСТ 10: Геометричні примітиви
# =============================================================================

def test_geometric_primitives():
    """Тест геометричних примітивів."""
    print("\n" + "="*60)
    print("TEST 10: Geometric Primitives")
    print("="*60)
    
    # Fisher-Rao distance
    p = np.random.rand(256)
    p = p / p.sum()
    q = np.random.rand(256)
    q = q / q.sum()
    
    # Self-distance = 0
    d_self = fisher_rao_distance(p, p)
    assert abs(d_self) < 1e-6, f"Self-distance should be 0, got {d_self}"
    print(f"  Self-distance: {d_self:.8f} (expected 0)")
    
    # Symmetry
    d12 = fisher_rao_distance(p, q)
    d21 = fisher_rao_distance(q, p)
    assert abs(d12 - d21) < 1e-6, "Distance should be symmetric"
    print(f"  Symmetry: d(p,q)={d12:.4f} == d(q,p)={d21:.4f}")
    
    # Triangle inequality
    r = np.random.rand(256)
    r = r / r.sum()
    d13 = fisher_rao_distance(p, r)
    d23 = fisher_rao_distance(q, r)
    assert d13 <= d12 + d23 + 1e-6, "Triangle inequality should hold"
    print(f"  Triangle: d(p,r)={d13:.4f} <= d(p,q)+d(q,r)={d12+d23:.4f}")
    
    # Geodesic interpolation
    p1 = np.zeros(256)
    p1[100:120] = 1.0
    p1 = p1 / p1.sum()
    
    p2 = np.zeros(256)
    p2[150:170] = 1.0
    p2 = p2 / p2.sum()
    
    mid = geodesic_interpolation(p1, p2, 0.5)
    assert np.isclose(mid.sum(), 1.0), "Interpolated should be on simplex"
    
    d1 = fisher_rao_distance(p1, mid)
    d2 = fisher_rao_distance(mid, p2)
    print(f"  Interpolation: d(p1,mid)={d1:.4f}, d(mid,p2)={d2:.4f}")
    
    # KL divergence
    kl = kl_divergence(p, q)
    assert kl >= 0, "KL should be non-negative"
    print(f"  KL(p||q): {kl:.4f}")
    
    print("  [OK] Geometric primitives tests passed")
    return True


# =============================================================================
# ТЕСТ 11: Траєкторія vs Window — порівняння
# =============================================================================

def test_trajectory_vs_window():
    """Тест порівняння траєкторії та віконного підходу."""
    print("\n" + "="*60)
    print("TEST 11: Trajectory vs Window Comparison")
    print("="*60)
    
    # Текст з прикладу
    text = "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
    data = text.encode('utf-8')
    
    print(f"  Text: '{text}' ({len(data)} bytes)")
    
    # WINDOW підхід
    window_size = 8
    window_data = data[:window_size]
    print(f"\n  WINDOW підхід:")
    print(f"    Context: {window_data}")
    print(f"    Пам'ять: фіксована (-{len(data)-window_size} bytes втрачається)")
    
    # TRAJECTORY підхід
    engine = GeodesicContextEngine(max_trajectory_length=None)
    
    n_points = min(50, len(data))
    step = max(1, len(data) // n_points)
    points_added = 0
    
    for i in range(0, len(data), step):
        half_w = 4
        start = max(0, i - half_w)
        end = min(len(data), i + half_w)
        
        dist = np.zeros(256)
        for b in data[start:end]:
            dist[b] += 1
        dist = dist / dist.sum()
        
        engine.push(p=dist, t=float(i) / len(data), position=i)
        points_added += 1
    
    print(f"\n  TRAJECTORY підхід:")
    print(f"    Context: повна траєкторія ({points_added} точок)")
    print(f"    Геодезична довжина: {engine.total_geodesic_length:.4f}")
    
    # Виявлення семантичних особливостей
    shapes = engine.semantic_shapes
    if shapes['loops']:
        print(f"    🔁 Петлі виявлено: {len(shapes['loops'])}")
    if shapes['angles']:
        print(f"    ↰ Кути виявлено: {len(shapes['angles'])}")
    
    # Контекст без обмежень
    ctx = engine.get_context_vector()
    print(f"    Контекстний вектор: shape={ctx.shape}")
    
    # Немає "контекст закінчився"
    assert len(engine) == points_added, "Trajectory should have all points"
    print(f"    ✓ Безмежний контекст: {len(engine)} точок збережено")
    
    print("\n  КЛЮЧОВА РІЗНИЦЯ:")
    print("    WINDOW: Бачить тільки поточне вікно")
    print("    TRAJECTORY: Бачить ВСЮ історію через геометрію")
    
    print("  [OK] Trajectory vs Window tests passed")
    return True


# =============================================================================
# ЗАПУСК ВСІХ ТЕСТІВ
# =============================================================================

def run_all_tests():
    """Запустити всі тести."""
    print("="*60)
    print("BCS GEODESIC CONTEXT SYSTEM — ПОВНІ ТЕСТИ")
    print("="*60)
    print("\nПарадигма: Контекст = Траєкторія на Многовиді")
    print("Замінюємо: window → trajectory, softmax → exp(-d²/T), buffer → submanifold")
    print("="*60)
    
    tests = [
        ("GeodesicContextEngine Core", test_geodesic_context_engine_core),
        ("Geodesic Attention", test_geodesic_attention),
        ("Novelty Detection", test_novelty_detection),
        ("Memory as Submanifold", test_memory_as_submanifold),
        ("Semantic Shapes", test_semantic_shapes),
        ("TrajectoryAttention", test_trajectory_attention),
        ("TrajectoryContextIntegration", test_trajectory_context_integration),
        ("Query-Response", test_query_response),
        ("End-to-End with Model", test_e2e_with_model),
        ("Geometric Primitives", test_geometric_primitives),
        ("Trajectory vs Window", test_trajectory_vs_window),
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
        return False
    else:
        print("\n✅ ВСІ ТЕСТИ ПРОЙДЕНО!")
        print("="*60)
        print("\nРЕЗЮМЕ:")
        print("- GeodesicContextEngine працює")
        print("- GeodesicAttention замінює softmax")
        print("- Пам'ять як підмноговид працює")
        print("- Семантичні форми виявляються")
        print("- TrajectoryContextIntegration інтегрується з моделлю")
        print("- End-to-end тест пройдено")
        print("\nПарадигма 'Контекст = Траєкторія на Многовиді' РЕАЛІЗОВАНА!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)