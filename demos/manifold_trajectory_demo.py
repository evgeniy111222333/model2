"""
Демонстрація: Контекст = Траєкторія на Многовиді

ПОКАЗУЄ:
1. "Форма траєкторії = семантика"
2. Петлі = повторення
3. Кут = різка зміна теми
4. Геометрична пам'ять

Текст: "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
"""

import sys
import numpy as np
sys.path.insert(0, r"E:\arc")
sys.path.insert(0, r"E:\arc\bcs")

from bcs.information.manifold_trajectory import (
    ManifoldTrajectory,
    GeodesicAttention,
    create_trajectory_from_bytes,
)


def analyze_text_trajectory():
    """
    Аналіз тексту як траєкторії на многовиді.
    
    Показує:
    - Як форма траєкторії відображає семантику
    - Де знаходяться петлі (повторення)
    - Де знаходяться кути (зміни теми)
    """
    print("="*70)
    print("ДЕМО: Контекст = Траєкторія на Многовиді")
    print("="*70)
    
    # Текст з прикладу
    text = "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
    data = text.encode('utf-8')
    
    print(f"\n📝 Текст: '{text}'")
    print(f"   Байти: {len(data)}")
    
    # Створення траєкторії з window=8, step=1
    # Кожна точка = розподіл байтів у вікні
    traj = create_trajectory_from_bytes(
        data, 
        step=2,      # Кожні 2 байти
        window_size=6  # Вікно з 6 байт
    )
    
    print(f"\n📊 Траєкторія:")
    print(f"   Точок: {len(traj)}")
    print(f"   Сумарна довжина (геодезична): {traj.total_length:.4f}")
    
    # Аналіз форми траєкторії
    print(f"\n📈 Геометричні характеристики:")
    
    # Кривина — показує ЗМІНИ напрямку
    if traj.curvature_profile:
        max_curv_idx = np.argmax(traj.curvature_profile)
        max_curv = traj.curvature_profile[max_curv_idx]
        print(f"   Макс. кривина: {max_curv:.4f} (позиція {max_curv_idx})")
        print(f"   Середня кривина: {np.mean(traj.curvature_profile):.4f}")
    
    # Швидкість — показує ІНТЕНСИВНІСТЬ змін
    if traj.velocity_profile:
        max_vel_idx = np.argmax(traj.velocity_profile)
        max_vel = traj.velocity_profile[max_vel_idx]
        print(f"   Макс. швидкість: {max_vel:.4f} (позиція {max_vel_idx})")
        print(f"   Середня швидкість: {np.mean(traj.velocity_profile):.4f}")
    
    # Топологія
    topo = traj.compute_topology_features()
    print(f"\n🔺 Топологія:")
    print(f"   Betti_0 (компоненти): {topo['betti_0']}")
    print(f"   Betti_1 (петлі): {topo['betti_1']}")
    print(f"   Осциляції: {topo['oscillation_count']}")
    
    # Покроковий аналіз
    print(f"\n📍 Покроковий аналіз траєкторії:")
    print("-" * 70)
    
    for i, point in enumerate(traj.points):
        # Отримати байти цього вікна
        start = i * 2
        end = start + 6
        window_bytes = data[start:end] if start < len(data) else b''
        
        # Форма символу для відображення
        try:
            window_text = window_bytes.decode('utf-8', errors='replace')
        except:
            window_text = repr(window_bytes)
        
        # Геометричні ознаки цієї точки
        curvature = traj.curvature_profile[i] if i < len(traj.curvature_profile) else 0
        velocity = traj.velocity_profile[i] if i < len(traj.velocity_profile) else 0
        
        # Індикатор форми
        if curvature > 0.3:
            form_marker = "↰"  # Петля (різкий поворот)
        elif velocity > np.mean(traj.velocity_profile) * 1.5 if traj.velocity_profile else False:
            form_marker = "→"  # Швидкий рух
        else:
            form_marker = "·"  # Стабільно
        
        print(f"  p({i:2d}) {form_marker} | bytes: {window_bytes.hex()} | '{window_text:10s}' | curv={curvature:.3f} vel={velocity:.3f}")
    
    print("-" * 70)
    
    # Знайти семантично важливі точки
    print(f"\n🎯 Семантично важливі точки:")
    
    # Знайти точку з МАКСИМАЛЬНОЮ кривиною (різка зміна)
    if traj.curvature_profile:
        max_curv_idx = np.argmax(traj.curvature_profile)
        if max_curv_idx < len(traj.points):
            p = traj.points[max_curv_idx]
            start = max_curv_idx * 2
            end = start + 6
            window = data[start:end] if start < len(data) else b''
            print(f"\n  🔄 КУТ (різка зміна теми):")
            print(f"     Позиція: {max_curv_idx}")
            print(f"     Кривина: {traj.curvature_profile[max_curv_idx]:.4f}")
            print(f"     Вікно: '{window.decode('utf-8', errors='replace')}'")
            print(f"     Що сталось: ЗМІНА напрямку траєкторії")
    
    # Знайти точку з МАКСИМАЛЬНОЮ швидкістю (важлива подія)
    if traj.velocity_profile:
        max_vel_idx = np.argmax(traj.velocity_profile)
        if max_vel_idx < len(traj.points):
            p = traj.points[max_vel_idx]
            start = max_vel_idx * 2
            end = start + 6
            window = data[start:end] if start < len(data) else b''
            print(f"\n  ⚡ ШВИДКИЙ РУХ (важлива подія):")
            print(f"     Позиція: {max_vel_idx}")
            print(f"     Швидкість: {traj.velocity_profile[max_vel_idx]:.4f}")
            print(f"     Вікно: '{window.decode('utf-8', errors='replace')}'")
    
    # Знайти ПЕТЛЮ (якщо Betti_1 > 0)
    if topo['betti_1'] > 0:
        print(f"\n  🔁 ПЕТЛЯ (повторення теми):")
        print(f"     Betti_1: {topo['betti_1']}")
        print(f"     Означає: Траєкторія повертає назад")
    
    return traj


def compare_window_vs_trajectory():
    """
    Порівняння: Window vs Trajectory
    
    ПОКАЗУЄ чому траєкторія КРАЩА.
    """
    print("\n" + "="*70)
    print("ПОРІВНЯННЯ: Window vs Trajectory")
    print("="*70)
    
    text = "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
    data = text.encode('utf-8')
    
    print(f"\n📝 Текст: '{text}'")
    
    # WINDOW підхід
    window_size = 6
    window_data = data[:window_size]
    
    print(f"\n📦 WINDOW підхід:")
    print(f"   Контекст: {window_data}")
    print(f"   Пам'ять: фіксована (-{window_size} байт втрачається)")
    print(f"   Інформація про історію: НЕМАЄ")
    print(f"   Семантика: окремі слова")
    
    # TRAJECTORY підхід
    traj = create_trajectory_from_bytes(data, step=2, window_size=6)
    
    print(f"\n🌀 TRAJECTORY підхід:")
    print(f"   Контекст: повна траєкторія ({len(traj)} точок)")
    print(f"   Пам'ять: геометрична (форма траєкторії)")
    print(f"   Інформація про історію: ПОВНА")
    print(f"   Семантика: форма = значення")
    
    # Ключова різниця
    print(f"\n⚡ КЛЮЧОВА РІЗНИЦЯ:")
    print(f"")
    print(f"   WINDOW: Бачить тільки ПОТОЧНЕ вікно")
    print(f"   TRAJECTORY: Бачить ВСЮ ТРАЄКТОРІЮ")
    print(f"")
    print(f"   Приклад:")
    print(f"   - Коли я сказав 'яблука' → потім 'з'їв'")
    print(f"   - WINDOW: не знає що це пов'язано")
    print(f"   - TRAJECTORY: бачить ПЕТЛЮ в траєкторії")
    print(f"               і розуміє що це ПОВТОРЕННЯ")


def geodesic_attention_demo():
    """
    Демо: Geodesic Attention
    
    ПОКАЗУЄ: як attention працює через геометрію.
    """
    print("\n" + "="*70)
    print("GEOAttention: Attention через Геометрію")
    print("="*70)
    
    text = "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
    data = text.encode('utf-8')
    
    # Створити траєкторію
    traj = create_trajectory_from_bytes(data, step=2, window_size=6)
    
    print(f"\n📍 Geodesic Attention:")
    
    # Query = остання точка (поточний контекст)
    query = traj.points[-1].p.copy()
    query_window = data[-6:].decode('utf-8', errors='replace')
    
    print(f"   Query: '{query_window}'")
    
    # Compute attention
    attention = traj.compute_attention(query, temperature=0.5)
    
    print(f"\n   Attention до історії:")
    for i, (point, attn) in enumerate(zip(traj.points, attention)):
        start = i * 2
        end = start + 6
        window = data[start:end] if start < len(data) else b''
        
        # Bar visualization
        bar_len = int(attn * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        
        print(f"   p({i}) {bar} {attn:.3f} | '{window.decode('utf-8', errors='replace')}'")
    
    # Знайти найбільшу увагу
    max_attn_idx = np.argmax(attention)
    start = max_attn_idx * 2
    end = start + 6
    max_window = data[start:end] if start < len(data) else b''
    
    print(f"\n🎯 Максимальна увага: '{max_window.decode('utf-8', errors='replace')}'")
    print(f"   Це НАЙБЛИЖЧЕ за геометрією до поточної точки")
    print(f"   Тобто: семантично найближче, не лексично")


def memory_as_geometry_demo():
    """
    Демо: Пам'ять = Геометрія
    
    ПОКАЗУЄ: як геометрія замінює пам'ять-буфер.
    """
    print("\n" + "="*70)
    print("ПАМ'ЯТЬ = Геометрія")
    print("="*70)
    
    # Симуляція "сесії"
    session_text = """
    Користувач: Хочу замовити піцу
    Асістент: Яка піца вам подобається?
    Користувач: Пепероні
    Асістент: Велику чи маленьку?
    Користувач: Велику
    """
    
    data = session_text.encode('utf-8')
    traj = create_trajectory_from_bytes(data, step=5, window_size=10)
    
    print(f"\n📝 Сесія ({len(data)} байт):")
    print(f"   Траєкторія: {len(traj)} точок")
    print(f"   Сумарна довжина: {traj.total_length:.4f}")
    
    print(f"\n🧠 Геометрична пам'ять:")
    print(f"   Центроїд: {traj.memory_center is not None}")
    print(f"   Span (розкид): {traj.memory_spread:.4f}")
    
    # Query: нова тема
    new_topic = "Хочу ще десерт"
    new_data = new_topic.encode('utf-8')
    
    # Розподіл нової теми
    new_dist = np.zeros(256)
    for b in new_data:
        new_dist[b] += 1
    new_dist = new_dist / new_dist.sum()
    
    # Знайомість
    novelty, confidence = traj.detect_novelty(new_dist)
    
    print(f"\n🔍 Запит: '{new_topic}'")
    print(f"   Новизна: {novelty:.4f}")
    print(f"   Впевненість: {confidence:.4f}")
    
    if novelty > 0.5:
        print(f"   → Це НОВА тема для траєкторії")
    else:
        print(f"   → Це в межах існуючого контексту")


def streaming_demo():
    """
    Демо: Streaming mode
    
    ПОКАЗУЄ: як працює неперервний потік.
    """
    print("\n" + "="*70)
    print("STREAMING: Неперервний потік")
    print("="*70)
    
    traj = ManifoldTrajectory(max_length=50)
    
    text = "Сьогодні я купив яблука. Вони були солодкі. З'їв їх."
    data = text.encode('utf-8')
    
    print(f"\n📝 Обробка тексту по 2 байти:")
    
    for i in range(0, len(data) - 2, 2):
        window = data[i:i+4]
        
        # Розподіл байтів
        dist = np.zeros(256)
        for b in window:
            dist[b] += 1
        dist = dist / dist.sum()
        
        # Додати до траєкторії
        traj.push(dist, t=i)
        
        # Геометрична характеристика
        novelty, _ = traj.detect_novelty(dist)
        
        # Візуалізація
        if novelty > 0.5:
            marker = "⚡"
        elif len(traj) > 3 and traj.velocity_profile:
            if traj.velocity_profile[-1] > np.mean(traj.velocity_profile):
                marker = "→"
            else:
                marker = "·"
        else:
            marker = "·"
        
        print(f"  byte {i:2d}-{i+3:2d}: {marker} '{window.decode('utf-8', errors='replace')}' | novelty={novelty:.2f}")
    
    print(f"\n📊 Результат:")
    print(f"   Точок: {len(traj)}")
    print(f"   Довжина: {traj.total_length:.4f}")
    print(f"   Петлі: {traj.compute_topology_features()['betti_1']}")


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main():
    print("╔" + "═"*68 + "╗")
    print("║" + " " * 15 + "MANIFOLD TRAJECTORY DEMO" + " " * 24 + "║")
    print("║" + " " * 10 + "Контекст = Траєкторія на Многовиді" + " " * 19 + "║")
    print("╚" + "═"*68 + "╝")
    
    # 1. Основна демонстрація
    traj = analyze_text_trajectory()
    
    # 2. Порівняння
    compare_window_vs_trajectory()
    
    # 3. Geodesic Attention
    geodesic_attention_demo()
    
    # 4. Пам'ять як геометрія
    memory_as_geometry_demo()
    
    # 5. Streaming
    streaming_demo()
    
    print("\n" + "="*70)
    print("✅ ДЕМО ЗАВЕРШЕНО")
    print("="*70)
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  КЛЮЧОВА ІДЕЯ:                                                      ║
║                                                                      ║
║  WINDOW:  контекст = фіксований буфер даних                          ║
║           память    = фіксований розмір                              ║
║           час       = токени                                         ║
║                                                                      ║
║  TRAJECTORY: контекст = траєкторія на многовиді                      ║
║             память    = форма траєкторії                             ║
║             час       = геометрія траєкторії                        ║
║                                                                      ║
║  Форма = Семантика                                                  ║
║  - Петля  = Повторення                                              ║
║  - Кут    = Зміна теми                                              ║
║  - Швидк. = Важлива подія                                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
