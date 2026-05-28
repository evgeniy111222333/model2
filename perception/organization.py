import numpy as np
import warnings
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _kl_divergence, _js_divergence, _safe_normalize
from bcs.perception.boundary import MultiScaleBoundaryDetector
from bcs.core.field import PredictionErrorLoop
from bcs.information.conversion import ConversionLayersV3
from bcs.perception.predictive import PredictiveCoding


def _is_utf8_lead_byte(b: int) -> bool:
    """Check if byte is a valid UTF-8 lead byte (start of sequence)."""
    return (b >= 0xC0 and b <= 0xDF) or (b >= 0xE0 and b <= 0xEF) or (b >= 0xF0 and b <= 0xF4) or b < 0x80


def _is_utf8_continuation_byte(b: int) -> bool:
    """Check if byte is a UTF-8 continuation byte."""
    return 0x80 <= b <= 0xBF


def _is_utf8_boundary(pos: int, data: bytes) -> bool:
    """Return True if pos is a valid split point between UTF-8 codepoints."""
    N = len(data)
    return 0 <= pos <= N and (pos == 0 or pos == N or not _is_utf8_continuation_byte(data[pos]))


def _find_utf8_boundary(near: int, direction: int, data: bytes, max_search: int = 10) -> int:
    """
    Find nearest UTF-8 safe boundary from position `near` in given direction.

    Args:
        near: position to search from
        direction: -1 (left) or +1 (right)
        data: raw bytes
        max_search: max bytes to search

    Returns:
        Position that is a valid UTF-8 boundary (start or end of complete sequence)
    """
    N = len(data)
    near = max(0, min(int(near), N))
    if _is_utf8_boundary(near, data):
        return near

    # Try to move toward a safe split point.
    step = -1 if direction < 0 else 1
    for delta in range(1, max_search + 1):
        candidate = near + delta * step
        if candidate < 0 or candidate > N:
            break
        if _is_utf8_boundary(candidate, data):
            return candidate

    # Fallback: return original position
    return near


def _nearest_utf8_boundary(pos: int, data: bytes, max_search: int = 10) -> int:
    """Find the closest valid UTF-8 boundary to pos, preferring the right side on ties."""
    N = len(data)
    pos = max(0, min(int(pos), N))
    if _is_utf8_boundary(pos, data):
        return pos
    for delta in range(1, max_search + 1):
        right = pos + delta
        if right <= N and _is_utf8_boundary(right, data):
            return right
        left = pos - delta
        if left >= 0 and _is_utf8_boundary(left, data):
            return left
    return pos


def _snap_cluster_boundaries_to_utf8(clusters: List[Dict], data: bytes) -> List[Dict]:
    """
    FINAL FIX: Snap all cluster boundaries to UTF-8 safe positions.

    This is called as the LAST step of detect_clusters(), after ALL clustering
    operations (initial segmentation, merging, _ensure_min_clusters, _ensure_non_overlapping).

    Some operations like _ensure_non_overlapping can create boundaries at arbitrary
    byte positions (e.g., at index 4 in "Привіт"). This function ensures every
    cluster boundary lands on a valid UTF-8 sequence boundary.

    UTF-8 byte roles:
    - Lead byte (0xC0-0xDF, 0xE0-0xEF, 0xF0-0xF4): starts a sequence
    - Continuation byte (0x80-0xBF): continues a sequence
    - ASCII (0x00-0x7F): standalone, always safe

    A cluster end is safe if the byte BEFORE it is NOT a continuation byte.
    A cluster start is safe if the byte AT it is NOT a continuation byte.
    """
    N = len(data)
    if len(clusters) == 0 or N == 0:
        return clusters

    def find_safe_start(start: int) -> int:
        """Find a safe inclusive start position."""
        start = max(0, min(int(start), N))
        if _is_utf8_boundary(start, data):
            return start
        return _find_utf8_boundary(start, +1, data, max_search=10)

    def find_safe_end(end: int) -> int:
        """Find a safe exclusive end position."""
        return _nearest_utf8_boundary(end, data, max_search=10)

    # Sort clusters by start position
    sorted_clusters = sorted(clusters, key=lambda c: c['start'])

    result = []
    expected_start = 0  # Track expected next start position to avoid overlaps

    for c in sorted_clusters:
        start = c['start']
        end = c['end']

        # Don't go backward - ensure start >= expected_start
        if start < expected_start:
            start = expected_start

        # Ensure start is safe
        start = find_safe_start(start)
        # Ensure end is safe
        end = find_safe_end(end)
        # Ensure end > start
        if end <= start:
            # Very rare edge case - at least give 1 byte
            if start < N:
                end = start + 1
            else:
                continue

        # Rebuild positions
        positions = np.arange(start, end)

        # Rebuild cluster dict
        new_cluster = dict(c)
        new_cluster['start'] = start
        new_cluster['end'] = end
        new_cluster['size'] = len(positions)
        new_cluster['positions'] = positions
        result.append(new_cluster)

        expected_start = end

    # Final verification
    for c in result:
        start = c['start']
        end = c['end']
        assert _is_utf8_boundary(start, data), \
            f"Cluster [{start}:{end}] starts inside a UTF-8 sequence"
        assert _is_utf8_boundary(end, data), \
            f"Cluster [{start}:{end}] ends inside a UTF-8 sequence"

    return result


def _snap_boundaries_to_utf8(boundaries: np.ndarray, data: bytes) -> np.ndarray:
    """
    Post-process cluster boundaries to ensure they don't cut through UTF-8 sequences.
    
    For each boundary, search outward until finding a safe UTF-8 boundary:
    - Left boundaries: snap to start of UTF-8 sequence (lead byte or ASCII)
    - Right boundaries: snap to end of UTF-8 sequence (next char's start)
    
    Args:
        boundaries: initial boundaries from field-based detection
        data: raw bytes
        
    Returns:
        UTF-8-safe boundaries
    """
    N = len(data)
    safe_boundaries = []
    
    # First boundary (0) is always safe
    if len(boundaries) == 0:
        return np.array([], dtype=int)
    
    # Start with 0
    safe_boundaries.append(0)
    
    # Process each internal boundary
    for b in boundaries:
        if b <= 0 or b >= N:
            continue
        
        new_b = _nearest_utf8_boundary(b, data, max_search=5)
        
        # Ensure we don't go backwards past previous boundary
        if safe_boundaries and new_b <= safe_boundaries[-1]:
            new_b = safe_boundaries[-1] + 1
            # Snap to next UTF-8 boundary
            new_b = _find_utf8_boundary(new_b, +1, data, max_search=5)
        
        if new_b < N and new_b > safe_boundaries[-1]:
            safe_boundaries.append(int(new_b))
    
    return np.array(sorted(set(safe_boundaries)), dtype=int)


def _js_divergence_many(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized Jensen-Shannon divergence from one distribution to many."""
    if q.size == 0:
        return np.zeros(0, dtype=np.float64)
    p_safe = np.maximum(np.asarray(p, dtype=np.float64), 1e-10)
    q_safe = np.maximum(np.asarray(q, dtype=np.float64), 1e-10)
    m = 0.5 * (q_safe + p_safe[None, :])
    m_safe = np.maximum(m, 1e-10)
    kl_pm = np.sum(p_safe[None, :] * np.log(p_safe[None, :] / m_safe), axis=1)
    kl_qm = np.sum(q_safe * np.log(q_safe / m_safe), axis=1)
    return 0.5 * (kl_pm + kl_qm)


class SelfOrganizerV4:
    """
    Механізм самоорганізації V4 з мультимасштабним аналізом границь.

    Відповідність формалізму:
    - Визначення 4.7: Вільна енергія F_free = E_total - T·S
    - Теорема 4.2: Фазовий перехід при T < T_c
    - Визначення 4.8: Когнітивний кластер C = (I, p)

    V4 покращення:
    - MultiScaleBoundaryDetector замість одинарного масштабу
    - Оцінка якості кластерів (когерентність, відмінність, компактність)
    - Злиття несуміжних кластерів зі схожими розподілами
    """

    def __init__(
        self,
        field_system,
        predictive_coding: Optional[PredictiveCoding] = None,
        temperature: float = 1.0,
        boundary_detector: Optional[MultiScaleBoundaryDetector] = None,
    ):
        self.field = field_system
        self.pc = predictive_coding
        self.T = temperature
        self.clusters = []

        # Мультимасштабний виявлювач границь
        self.boundary_detector = boundary_detector or MultiScaleBoundaryDetector()
        self.numeric_policy = self.boundary_detector.numeric_policy
        self.last_boundary_policy = {}
        self.last_cluster_policy = {}

    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Рівняння (14): KL-дивергенція."""
        return _kl_divergence(p, q)

    def compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon дивергенція (симетрична)."""
        return _js_divergence(p, q)

    def _distribution_cumsum(self) -> np.ndarray:
        substrate = self.field.substrate
        cum = getattr(substrate, '_one_hot_cumsum', None)
        if cum is None:
            cum = np.vstack([
                np.zeros((1, 256), dtype=np.float32),
                np.cumsum(substrate.one_hot, axis=0),
            ])
            if hasattr(substrate, '_one_hot_cumsum'):
                substrate._one_hot_cumsum = cum
        return cum

    def _segment_distribution(self, start: int, end: int) -> np.ndarray:
        start = max(0, min(int(start), self.field.N))
        end = max(start, min(int(end), self.field.N))
        if end <= start:
            return np.ones(256, dtype=np.float32) / 256.0
        cum = self._distribution_cumsum()
        counts = (cum[end] - cum[start]).astype(np.float32)
        total = float(counts.sum())
        if total <= 1e-10:
            return np.ones(256, dtype=np.float32) / 256.0
        return counts / total

    def _local_distribution_at(self, pos: int, window: int) -> np.ndarray:
        half = max(1, int(window) // 2)
        start = max(0, int(pos) - half)
        end = min(self.field.N, int(pos) + half + 1)
        return self._segment_distribution(start, end)

    def detect_boundaries(self) -> np.ndarray:
        """
        Виявлення границь кластерів через мультимасштабний аналіз.

        V6 FIX #2: Адаптивний percentile та min_gap. Попередній фіксований
        percentile=55.0 для однорідних даних давав занадто мало границь
        (сигнал дуже слабкий → навіть 55-й перцентиль ≈ 0). Тепер: якщо
        сигнал слабкий (σ < 0.01), підвищуємо чутливість (percentile=40),
        щоб виявляти хоча б мінімальну структуру.
        """
        N = self.field.N

        # Аномалії предиктивного кодування
        pc_anomalies = None
        if self.pc is not None:
            errors, _ = self.pc.compute_prediction_error(self.field.u)
            anomaly_threshold = self.numeric_policy.predictive_anomaly_threshold(errors, prior=1.5)
            pc_anomalies = self.pc.detect_anomalies(self.field.u, threshold=anomaly_threshold)

        # V6 FIX #2: Адаптивний percentile на основі сили сигналу
        # Спочатку обчислюємо confidence щоб оцінити якість сигналу
        confidence = self.boundary_detector.detect(
            self.field.substrate, self.field.v, pc_anomalies
        )
        selection_policy = self.numeric_policy.boundary_selection_policy(confidence, N)
        percentile = float(selection_policy['percentile'])
        min_gap = int(selection_policy['min_gap'])
        self.last_boundary_policy = dict(selection_policy)

        # Використовуємо вже обчислений confidence
        threshold_val = np.percentile(confidence, percentile)
        
        # Знаходження локальних піків вище порогу
        candidate_peaks = []
        for i in range(N):
            if confidence[i] < threshold_val:
                continue
            is_peak = True
            if i > 0 and confidence[i] < confidence[i - 1]:
                is_peak = False
            if i < N - 1 and confidence[i] < confidence[i + 1]:
                is_peak = False
            if is_peak:
                candidate_peaks.append((i, confidence[i]))

        if len(candidate_peaks) == 0:
            # V6 FIX #2: Якщо жодного піку не знайдено, використовуємо
            # рівномірний поділ щоб гарантувати мінімум 2 кластери
            n_min_boundaries = int(selection_policy['fallback_boundaries'])
            boundaries = np.linspace(0, N, n_min_boundaries + 2, dtype=int)[1:-1]
            return boundaries

        # Сортування за зменшенням впевненості
        candidate_peaks.sort(key=lambda x: x[1], reverse=True)

        # Жадібний відбір з урахуванням мінімальної відстані min_gap
        boundaries_list = []
        for idx, conf in candidate_peaks:
            too_close = False
            for pb in boundaries_list:
                if abs(idx - pb) < min_gap:
                    too_close = True
                    break
            if not too_close:
                boundaries_list.append(idx)

        boundaries_list.sort()
        return np.array(boundaries_list, dtype=int)

    def detect_clusters(self) -> List[Dict]:
        """
        Виявлення просторово-когерентних когнітивних кластерів.

        V4 покращення:
        1. Мультимасштабний аналіз границь
        2. Оцінка якості кластерів
        3. Злиття несуміжних кластерів зі схожими розподілами

        V6 FIX #2: Гарантуємо мінімальну кількість кластерів.
        Для однорідних даних злиття може колапсувати всі сегменти
        в один гігантський кластер. Мінімум: max(2, N//200) кластерів.
        """
        N = self.field.N
        boundaries = self.detect_boundaries()
        self.last_boundaries = boundaries  # зберігаємо для зовнішнього доступу
        
        # FIX: Snap boundaries to UTF-8 safe positions
        # This ensures clusters don't cut through multi-byte UTF-8 sequences
        data = self.field.substrate.raw_data
        if len(boundaries) > 0:
            boundaries = _snap_boundaries_to_utf8(boundaries, data)
            self.last_boundaries = boundaries
        
        cluster_policy = self.numeric_policy.cluster_policy(N)
        self.last_cluster_policy = dict(cluster_policy)

        # Створюємо початкові сегменти
        segments = []
        prev = 0
        for b in boundaries:
            if b > prev:
                segments.append((prev, b))
            prev = b
        if prev < N:
            segments.append((prev, N))

        # Обчислюємо розподіли сегментів
        local_window = max(N // 20, 4)

        # Об'єднуємо схожі суміжні сегменти
        # CONCEPT FIX: Передаємо boundary_positions щоб зберегти границі
        # знайдені детектором (не зливати великі сегменти через них)
        merged_segments = self._merge_similar_segments(
            segments, local_window, boundary_positions=boundaries
        )

        # V6 FIX #2: Гарантуємо мінімальну кількість кластерів
        # Якщо злиття колапсувало все в занадто мало кластерів,
        # рівномірно розбиваємо найбільші кластери
        min_clusters = int(cluster_policy['min_clusters'])
        if len(merged_segments) < min_clusters and N > 50:
            merged_segments = self._ensure_min_clusters(merged_segments, min_clusters, N)

        # Формуємо кластери з оцінкою якості
        clusters = []
        for seg_start, seg_end in merged_segments:
            positions = np.arange(seg_start, seg_end)
            if len(positions) == 0:
                continue

            # Розподіл кластера
            cluster_dist = self._segment_distribution(seg_start, seg_end)

            # Статистика поля
            u_seg = self.field.u[positions]
            v_seg = self.field.v[positions]

            cluster = {
                'positions': positions,
                'start': seg_start,
                'end': seg_end,
                'size': len(positions),
                'distribution': cluster_dist,
                'mean_u': float(np.mean(u_seg)),
                'std_u': float(np.std(u_seg)),
                'mean_v': float(np.mean(v_seg)),
                'std_v': float(np.std(v_seg)),
                'dominant_bytes': self._get_dominant_bytes(cluster_dist, top_n=5),
            }

            # Оцінка якості кластера
            cluster['quality_score'] = self._compute_cluster_quality(
                cluster, clusters, local_window
            )

            clusters.append(cluster)

        # Злиття несуміжних кластерів зі схожими розподілами
        clusters = self._merge_non_adjacent(clusters, N)

        # Усунення перекраттів (гарантуємо просторову когерентність)
        clusters = self._ensure_non_overlapping(clusters, local_window)

        # V6 FIX #2: Фінальна перевірка мінімуму кластерів.
        # _merge_non_adjacent може знову колапсувати все в 1 кластер
        # для однорідних даних. Гарантуємо мінімум після ВСІХ злиттів.
        min_clusters_final = int(cluster_policy['min_clusters'])
        if len(clusters) < min_clusters_final and N > 50:
            # Рівномірно розбиваємо на min_clusters_final кластерів
            chunk_size = max(N // min_clusters_final, 20)
            final_segments = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                if end > start:
                    final_segments.append((start, end))
            clusters = []
            for seg_start, seg_end in final_segments:
                positions = np.arange(seg_start, seg_end)
                if len(positions) == 0:
                    continue
                cluster_dist = self._segment_distribution(seg_start, seg_end)
                u_seg = self.field.u[positions]
                v_seg = self.field.v[positions]
                cluster = {
                    'positions': positions,
                    'start': seg_start,
                    'end': seg_end,
                    'size': len(positions),
                    'distribution': cluster_dist,
                    'mean_u': float(np.mean(u_seg)),
                    'std_u': float(np.std(u_seg)),
                    'mean_v': float(np.mean(v_seg)),
                    'std_v': float(np.std(v_seg)),
                    'dominant_bytes': self._get_dominant_bytes(cluster_dist, top_n=5),
                    'quality_score': 0.5,  # Помірна якість для рівномірного поділу
                }
                clusters.append(cluster)

        # FINAL FIX: Snap ALL cluster boundaries to UTF-8 safe positions
        # This must be done LAST, after ALL clustering operations (merge, split, etc.)
        # to catch any boundaries created by _ensure_non_overlapping splitting
        clusters = _snap_cluster_boundaries_to_utf8(clusters, data)

        self.clusters = clusters
        return clusters

    def _ensure_min_clusters(
        self,
        segments: List[Tuple[int, int]],
        min_clusters: int,
        N: int,
    ) -> List[Tuple[int, int]]:
        """
        V6 FIX #2: Гарантувати мінімальну кількість сегментів шляхом
        розбиття найбільших сегментів. Це запобігає колапсу в 1 кластер
        для однорідних даних.
        """
        while len(segments) < min_clusters:
            # Знаходимо найбільший сегмент
            max_idx = max(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
            s, e = segments[max_idx]
            size = e - s
            if size < 20:
                break  # Сегменти занадто малі для розбиття

            # Розбиваємо навпіл
            mid = (s + e) // 2
            new_segments = segments[:max_idx] + [(s, mid), (mid, e)] + segments[max_idx + 1:]
            segments = new_segments

        return segments

    def _compute_cluster_quality(
        self,
        cluster: Dict,
        existing_clusters: List[Dict],
        local_window: int,
    ) -> float:
        """
        Обчислення оцінки якості кластера.

        Компоненти:
        1. Внутрішня когерентність: середня JS-дивергенція всередині кластера (нижче = краще)
        2. Зовнішня відмінність: мінімальна JS-дивергенція до сусідніх кластерів (вище = краще)
        3. Просторова компактність: відношення розміру до розмаху
        """
        positions = cluster['positions']
        cluster_dist = cluster['distribution']
        size = cluster['size']

        # 1. Внутрішня когерентність
        if size > 1:
            # Порівнюємо кожну позицію з середнім розподілом кластера
            coherence_values = []
            for pos in positions[::max(1, size // 10)]:  # Вибірка для ефективності
                js = self.compute_js_divergence(
                    self._local_distribution_at(int(pos), local_window),
                    cluster_dist,
                )
                coherence_values.append(js)
            coherence = np.mean(coherence_values) if coherence_values else 0.0
        else:
            coherence = 0.0

        # 2. Зовнішня відмінність
        distinctness = 0.0
        if len(existing_clusters) > 0:
            existing_dists = np.vstack([other['distribution'] for other in existing_clusters])
            js_values = _js_divergence_many(cluster_dist, existing_dists)
            distinctness = float(np.min(js_values)) if js_values.size else 0.0

        # 3. Просторова компактність
        span = cluster['end'] - cluster['start']
        compactness = size / max(span, 1)

        # Комбінована оцінка якості (нормалізована до [0, 1])
        # Нижча когерентність → краще, вища відмінність → краще, вища компактність → краще
        quality = (
            -coherence * 2.0   # Штраф за внутрішню розбіжність
            + distinctness * 3.0  # Бонус за відмінність від сусідів
            + compactness * 1.0   # Бонус за компактність
        )

        # Нормалізація до [0, 1] через sigmoid
        quality_normalized = 1.0 / (1.0 + np.exp(-quality))
        return float(quality_normalized)

    def _merge_similar_segments(
        self, segments: List[Tuple[int, int]], local_window: int,
        js_threshold: float = 0.15,
        boundary_positions: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        """Об'єднання суміжних сегментів з схожими розподілами.

        V4 fix: адаптивний поріг злиття. Малі сегменти зливаються
        вільніше, але великі сегменти (потенційні самостійні секції)
        зливаються лише при дуже низькій JS-дивергенції. Це запобігає
        злиттю всього потоку в один гігантський кластер для
        однорідних даних (наприклад, UTF-8 текст).

        CONCEPT FIX: Якщо boundary_positions передані, границі знайдені
        детектором ЗАВЖДИ зберігаються — вони є "джерелом істини" про
        структурні переходи в даних. Злиття дозволене лише для дуже
        малих сегментів (менших за min_autonomous_size). Концепція каже:
        "на першій стадії виникають локальні кластери, що відповідають
        найпростішим патернам (повторення, межі)" — отже границі між
        повтореннями МАЮТЬ зберігатися як окремі кластери.
        """
        if len(segments) <= 1:
            return segments

        N = self.field.N
        merge_policy = self.numeric_policy.cluster_policy(N)
        js_threshold = float(merge_policy['adjacent_js_threshold'])
        # V6 FIX #2: Мінімальний розмір самостійного сегмента адаптований
        # до N, але з обмеженням знизу. Попередній N//20 при N=2000→100
        # був занадто великим — будь-який сегмент < 100 вважався "малим"
        # і зливався з сусідом. Тепер: min_autonomous_size ≤ 40.
        min_autonomous_size = int(merge_policy['min_autonomous_size'])

        # CONCEPT FIX: Набір границь для збереження
        boundary_set = set()
        if boundary_positions is not None:
            boundary_set = set(int(b) for b in boundary_positions)

        merged = [segments[0]]

        for seg_start, seg_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            prev_size = prev_end - prev_start
            curr_size = seg_end - seg_start

            # CONCEPT FIX: Якщо границя між сегментами була знайдена
            # детектором — ЗБЕРІГАЄМО її (не зливаємо). Це гарантує,
            # що структурні переходи не знищуються злиттям.
            boundary_at_junction = prev_end in boundary_set

            # Завжди зберігаємо границю, якщо обидва сегменти достатньо
            # великі для самостійності (навіть без boundary_set)
            both_large = (prev_size >= min_autonomous_size and
                         curr_size >= min_autonomous_size)

            if boundary_at_junction and both_large:
                # Границя між двома великими сегментами — зберігаємо
                merged.append((seg_start, seg_end))
                continue

            prev_dist = self._segment_distribution(prev_start, prev_end)
            curr_dist = self._segment_distribution(seg_start, seg_end)

            js = self.compute_js_divergence(prev_dist, curr_dist)

            # Адаптивний поріг: якщо обидва сегменти великі,
            # потрібна нижча JS для злиття (вони можуть бути різними секціями)
            adaptive_threshold = (
                js_threshold * merge_policy['large_segment_js_factor']
                if both_large else js_threshold
            )

            # V6 FIX #2: Зменшений поріг злиття великих кластерів.
            # Попередній 0.6*N дозволяв зливати занадто великі блоки.
            # Тепер: 0.4*N + мінімум 2 кластери, або 0.3*N для дуже великих.
            combined_size = prev_size + curr_size
            too_large_ratio = float(merge_policy['too_large_ratio'])
            too_large = combined_size > N * too_large_ratio and len(merged) > 1

            # CONCEPT FIX: Якщо границя знайдена детектором, зливаємо
            # лише якщо обидва сегменти малі (менші за min_autonomous_size)
            if boundary_at_junction:
                # Зберігаємо границю, якщо хоча б один сегмент великий
                if prev_size >= min_autonomous_size or curr_size >= min_autonomous_size:
                    merged.append((seg_start, seg_end))
                    continue
                # Обидва малі — дозволяємо злиття з підвищеним порогом
                adaptive_threshold = js_threshold * merge_policy['boundary_small_js_factor']

            if js < adaptive_threshold and not too_large:
                merged[-1] = (prev_start, seg_end)
            else:
                merged.append((seg_start, seg_end))

        return merged

    def _merge_non_adjacent(
        self,
        clusters: List[Dict],
        data_length: int,
        js_threshold: float = 0.10,
    ) -> List[Dict]:
        """
        Групування несуміжних кластерів зі схожими розподілами.

        CONCEPT FIX: За Визначенням 4.8 (умова iii), кластер має бути
        ЗВ'ЯЗНИМ (суміжним). Попередній код ЗЛИВАВ несуміжні кластери
        в один великий кластер, що порушувало цю умову. Для повторюваних
        даних (наприклад, "Hello World!..." × 50) усі 47 сегментів
        зливалися в 1 гігантський кластер [0:2350].

        Новий підхід: замість злиття, додаємо `pattern_group` мітку
        до кожного кластера. Кластери з однаковою міткою розпізнаються
        як екземпляри одного патерну (концепція Розділ 13: "повторення
        як найпростіший патерн"). Кластери залишаються окремими
        (просторова структура зберігається), але пов'язуються через
        pattern_group для вищих рівнів конвертації.

        Це відповідає концепції: "на першій стадії виникають локальні
        кластери, що відповідають найпростішим патернам (повторення,
        межі)" — кожен повтор є окремий кластер, але вони групуються
        як один патерн.
        """
        n = len(clusters)
        if n <= 2:
            # Додаємо pattern_group = 0 всім
            for i, c in enumerate(clusters):
                c['pattern_group'] = i
            return clusters

        # Обчислюємо всі попарні JS дивергенції (не-сусідні)
        cluster_dists = np.vstack([c['distribution'] for c in clusters])
        pair_js = []
        for i in range(n):
            if i + 2 >= n:
                continue
            js_row = _js_divergence_many(cluster_dists[i], cluster_dists[i + 2:])
            for j_offset, js in enumerate(js_row, start=i + 2):
                pair_js.append((i, j_offset, float(js)))

        if not pair_js:
            for i, c in enumerate(clusters):
                c['pattern_group'] = i
            return clusters

        # Адаптивний поріг: мінімум з (фіксованого, 25-й перцентиль JS)
        js_values = [js for _, _, js in pair_js]
        js_array = np.array(js_values)
        merge_policy = self.numeric_policy.cluster_policy(data_length, js_values)
        adaptive_threshold = min(
            float(merge_policy['non_adjacent_js_threshold']),
            float(np.percentile(js_array, 25)),
        )
        adaptive_threshold = max(adaptive_threshold, 0.01)
        self.last_cluster_policy.update({
            'non_adjacent_js_threshold_effective': float(adaptive_threshold),
            'non_adjacent_js_pairs': int(len(pair_js)),
        })

        # Побудова міток груп через Union-Find
        group = list(range(n))

        def find(x):
            while group[x] != x:
                group[x] = group[group[x]]
                x = group[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                group[rx] = ry

        # Групування кластерів зі схожими розподілами
        for i, j, js in pair_js:
            if js < adaptive_threshold:
                union(i, j)

        # Призначаємо pattern_group мітки (але НЕ зливаємо кластери!)
        group_map = {}
        next_group_id = 0
        for i in range(n):
            root = find(i)
            if root not in group_map:
                group_map[root] = next_group_id
                next_group_id += 1
            clusters[i]['pattern_group'] = group_map[root]
            clusters[i]['linked_clusters'] = sum(1 for j in range(n) if find(j) == root)

        return clusters

    def _split_into_contiguous_blocks(
        self, sorted_positions: List[int],
    ) -> List[List[int]]:
        """Розділення відсортованих позицій на контігуальні блоки."""
        if not sorted_positions:
            return []
        blocks = []
        current_block = [sorted_positions[0]]
        for i in range(1, len(sorted_positions)):
            if sorted_positions[i] == sorted_positions[i - 1] + 1:
                current_block.append(sorted_positions[i])
            else:
                blocks.append(current_block)
                current_block = [sorted_positions[i]]
        blocks.append(current_block)
        return blocks

    def _ensure_non_overlapping(
        self, clusters: List[Dict], local_window: int,
    ) -> List[Dict]:
        """
        Перевірка та усунення перекриттів між кластерами.
        Кожна позиція належить щонайбільше одному кластеру.
        Пріоритет: кластер з вищим quality_score отримує позицію.
        """
        if len(clusters) <= 1:
            return clusters

        N = self.field.N
        # Призначення позицій: кожна позиція — лише одному кластеру
        assignment = np.full(N, -1, dtype=int)

        # Сортуємо кластери за якістю (вищий пріоритет першим)
        sorted_indices = sorted(
            range(len(clusters)),
            key=lambda i: clusters[i].get('quality_score', 0.5),
            reverse=True,
        )

        for i in sorted_indices:
            c = clusters[i]
            for pos in c['positions']:
                if 0 <= pos < N and assignment[pos] == -1:
                    assignment[pos] = i

        # Перебудова кластерів з призначених позицій
        result = []
        for i in sorted_indices:
            c = clusters[i]
            assigned_mask = assignment == i
            assigned_positions = np.where(assigned_mask)[0]
            if len(assigned_positions) == 0:
                continue

            # Розділяємо на контігуальні блоки
            blocks = self._split_into_contiguous_blocks(assigned_positions.tolist())
            for block_positions in blocks:
                block_arr = np.array(block_positions)
                start = int(block_arr[0])
                end = int(block_arr[-1]) + 1

                u_seg = self.field.u[block_arr]
                v_seg = self.field.v[block_arr]

                block_dist = self._segment_distribution(start, end)

                block_cluster = {
                    'positions': block_arr,
                    'start': start,
                    'end': end,
                    'size': len(block_arr),
                    'distribution': block_dist,
                    'mean_u': float(np.mean(u_seg)),
                    'std_u': float(np.std(u_seg)),
                    'mean_v': float(np.mean(v_seg)),
                    'std_v': float(np.std(v_seg)),
                    'dominant_bytes': self._get_dominant_bytes(block_dist, top_n=5),
                    'quality_score': c.get('quality_score', 0.5),
                    'pattern_group': c.get('pattern_group'),  # FIX: preserve pattern_group
                }
                result.append(block_cluster)

        result.sort(key=lambda c: c['start'])
        return result

    def _get_dominant_bytes(self, dist: np.ndarray, top_n: int = 5) -> List[Tuple[int, float]]:
        """Отримати top-N домінуючих байтів кластера."""
        top_indices = np.argsort(dist)[-top_n:][::-1]
        return [(int(idx), float(dist[idx])) for idx in top_indices if dist[idx] > 0.01]

    def find_critical_temperature(self, T_range: np.ndarray = None) -> float:
        """
        Обчислення T_c через аналіз вільної енергії.
        T_c — точка фазового переходу (максимальна кривина F(T)).
        """
        if T_range is None:
            T_range = np.linspace(0.01, 10.0, 200)

        free_energies = []
        for T in T_range:
            F = self.field.compute_free_energy(T)
            free_energies.append(F)

        F_arr = np.array(free_energies)

        if len(F_arr) > 5:
            dF = np.gradient(F_arr, T_range)
            d2F = np.gradient(dF, T_range)
            kernel = np.ones(5) / 5
            d2F = np.convolve(d2F, kernel, mode='same')
            T_c = T_range[np.argmax(np.abs(d2F))]
        else:
            T_c = 1.0

        return float(T_c)



class LevelSplitting:
    """
    Автокаталітичне розщеплення рівнів абстракції.

    Коли на рівні ℓ KL-дивергенція між кластерами стає занадто великою,
    система розщеплює рівень на два. Аналог: клітина → тканина → орган.

    Крок 0: Детекція потреби — коефіцієнт бімодальності Сарлса b = (κ+1)/g
            Якщо b > 0.555 → розподіл бімодальний → запуск розщеплення
    Крок 1: Побудова графа подібності G_ℓ = (V, E, W), W_ij = exp(-KL(p_i||p_j))
    Крок 2: Спектральна кластеризація: L_norm = I - D^{-1/2}WD^{-1/2},
            eigengap для кількості груп, k-means на власних векторах
    Крок 3: Створення нового рівня ℓ': h_m = centroid, W_ℓ' = I
    Крок 4: Калібрація: η_ℓ' = η_base · (1 + α_calib · e^{-t/τ_calib})
    Крок 5: Валідація: ΔF = F_after - F_before, ΔF < -ε → keep, else merge_back
    """

    def __init__(
        self,
        bimodality_threshold: float = 0.555,
        min_clusters_for_split: int = 6,
        calibration_alpha: float = 10.0,
        calibration_tau: float = 500.0,
        validation_epsilon: float = 0.05,  # Збільшено для стабільніших метрик
        validation_steps: int = 20,  # Більше кроків для стабілізації
        min_stability_ratio: float = 0.7,  # Енергія має знижуватися у 70%+ вимірів
        base_lr: float = 0.001,
    ):
        self.bimodality_threshold = bimodality_threshold
        self.min_clusters_for_split = min_clusters_for_split
        self.calibration_alpha = calibration_alpha
        self.calibration_tau = calibration_tau
        self.validation_epsilon = validation_epsilon
        self.validation_steps = validation_steps
        self.min_stability_ratio = min_stability_ratio
        self.base_lr = base_lr

        self.split_history = []
        self.levels = []
        self.calibration_state = {}

    def set_levels(self, conversion_results: List[Dict]):
        """Встановити поточні рівні з результатів конвертації."""
        self.levels = []
        for level_data in conversion_results:
            level = {
                'level_idx': level_data['level'],
                'clusters': [item['cluster'] for item in level_data.get('items', [])],
                'representations': [
                    r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r
                    for r in [item['representation'] for item in level_data.get('items', [])]
                ],
                'n_clusters': level_data.get('n_clusters', 0),
                'W_transform': None,
            }
            self.levels.append(level)

    def detect_bimodality(self, level_idx: int) -> Dict:
        """
        Крок 0: Детекція потреби розщеплення.
        Коефіцієнт бімодальності Сарлса (Рівняння 44):
            b = (g² + 1) / (excess_kurtosis + 3)
        де g — skewness, excess_kurtosis + 3 — абсолютний четвертий момент
        (kurtosis), який завжди ≥ g² + 1, що гарантує b ≤ 1.
        b > 0.555 → бімодальний → запуск розщеплення.
        """
        if level_idx >= len(self.levels):
            return {'bimodal': False, 'coefficient': 0.0, 'reason': 'level_out_of_range'}
        clusters = self.levels[level_idx]['clusters']
        if len(clusters) < self.min_clusters_for_split:
            return {'bimodal': False, 'coefficient': 0.0, 'reason': 'too_few_clusters'}

        distances = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                kl = _kl_divergence(clusters[i]['distribution'],
                                    clusters[j]['distribution'])
                distances.append(kl)
        if len(distances) < 3:
            return {'bimodal': False, 'coefficient': 0.0, 'reason': 'too_few_distances'}

        D = np.array(distances)
        mean_d = np.mean(D)
        std_d = np.std(D)
        if std_d < 1e-10:
            return {'bimodal': False, 'coefficient': 0.0, 'reason': 'zero_variance'}

        # Skewness та excess kurtosis
        g = float(np.mean(((D - mean_d) / std_d) ** 3))
        kappa_raw = float(np.mean(((D - mean_d) / std_d) ** 4))
        excess_kurtosis = kappa_raw - 3.0

        # Формула Сарлса (Рівняння 44):
        # b = (g² + 1) / (excess_kurtosis + 3)
        # Де excess_kurtosis + 3 = κ_raw — абсолютний четвертий момент
        # Властивість: κ_raw ≥ g² + 1, тому b ≤ 1
        denominator = kappa_raw  # = excess_kurtosis + 3
        if denominator < 1e-10:
            # Якщо kurtosis ≈ 0 — невизначеність, не бімодальний
            return {'bimodal': False, 'coefficient': 0.0, 'reason': 'zero_kurtosis'}

        b = (g ** 2 + 1.0) / denominator

        # b ≤ 1 за означенням (kurtosis ≥ g² + 1), але через числові
        # похибки обмежуємо зверху
        b_clamped = min(max(b, 0.0), 1.0)

        return {
            'bimodal': b_clamped > self.bimodality_threshold,
            'coefficient': float(b_clamped),
            'skewness': float(g),
            'kurtosis': float(kappa_raw),
            'excess_kurtosis': float(excess_kurtosis),
            'distances_mean': float(mean_d),
            'distances_std': float(std_d),
            'reason': 'checked',
        }

    def build_similarity_graph(self, level_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Крок 1: Побудова графа подібності G_ℓ = (V, E, W).
        W_ij = exp(-KL(p_i || p_j))
        """
        clusters = self.levels[level_idx]['clusters']
        n = len(clusters)
        if n == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))

        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    W[i, j] = 1.0
                else:
                    kl = _kl_divergence(clusters[i]['distribution'],
                                        clusters[j]['distribution'])
                    W[i, j] = np.exp(-kl)
        adjacency = (W > 0.01).astype(np.float64)
        return W, adjacency

    def spectral_clustering(self, W: np.ndarray, max_groups: int = 4) -> List[List[int]]:
        """
        Крок 2: Спектральна кластеризація.
        L_norm = I - D^{-1/2} W D^{-1/2}
        Eigengap: λ_{k+1} - λ_k визначає кількість груп.
        K-means на власних векторах.
        """
        n = W.shape[0]
        if n <= 1:
            return [[0]]

        degree = W.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
        L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        L_norm = (L_norm + L_norm.T) / 2.0

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError:
            return [list(range(n))]

        # Eigengap heuristic
        n_groups = 2
        max_gap = 0.0
        for k in range(1, min(len(eigenvalues) - 1, max_groups)):
            gap = eigenvalues[k + 1] - eigenvalues[k]
            if gap > max_gap:
                max_gap = gap
                n_groups = k + 1

        Y = eigenvectors[:, :n_groups]
        return self._kmeans(Y, n_groups)

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 50) -> List[List[int]]:
        """K-means для спектральної кластеризації (k-means++ init)."""
        n = X.shape[0]
        if k >= n:
            return [[i] for i in range(n)]

        # k-means++ ініціалізація
        centroids = [X[np.random.randint(n)].copy()]
        for _ in range(1, k):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
            total = dists.sum()
            if total < 1e-10:
                centroids.append(X[np.random.randint(n)].copy())
            else:
                probs = dists / total
                idx = np.random.choice(n, p=probs)
                centroids.append(X[idx].copy())
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            for i in range(n):
                dists = [np.sum((X[i] - c) ** 2) for c in centroids]
                labels[i] = np.argmin(dists)
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if np.sum(mask) > 0:
                    new_centroids[c] = X[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        groups = [[] for _ in range(k)]
        for i in range(n):
            groups[labels[i]].append(i)
        return [g for g in groups if g]

    @staticmethod
    def _project_repr(vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Project a representation to target_dim by truncation/padding."""
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if len(arr) >= target_dim:
            return arr[:target_dim].copy()
        return np.pad(arr, (0, target_dim - len(arr))).astype(np.float32)

    def create_new_level(self, level_idx: int, groups: List[List[int]]) -> Dict:
        """
        Крок 3: Створення нового рівня ℓ' між ℓ та ℓ+1.
        h_m^(ℓ') = (1/|G_m|) Σ_{i∈G_m} h_i^(ℓ)
        W_ℓ'^(1) = I
        """
        if level_idx >= len(self.levels):
            return {}
        clusters = self.levels[level_idx]['clusters']
        representations = self.levels[level_idx]['representations']

        new_clusters = []
        new_reprs = []
        for group in groups:
            if not group:
                continue
            valid_repr_dims = [
                len(np.asarray(representations[idx]).reshape(-1))
                for idx in group if idx < len(representations)
            ]
            repr_dim = max(valid_repr_dims) if valid_repr_dims else 256
            combined_dist = np.zeros(256, dtype=np.float32)
            combined_repr = np.zeros(repr_dim, dtype=np.float32)
            repr_count = 0
            total_size = 0
            all_positions = []
            min_start, max_end = float('inf'), 0
            for idx in group:
                c = clusters[idx]
                w = c['size']
                combined_dist += w * c['distribution']
                if idx < len(representations):
                    combined_repr += self._project_repr(representations[idx], repr_dim)
                    repr_count += 1
                total_size += w
                all_positions.extend(c['positions'].tolist())
                min_start = min(min_start, c['start'])
                max_end = max(max_end, c['end'])
            combined_dist = _safe_normalize(combined_dist)
            combined_repr /= max(repr_count, 1)

            new_cluster = {
                'positions': np.array(sorted(set(all_positions))),
                'start': int(min_start),
                'end': int(max_end),
                'size': total_size,
                'distribution': combined_dist,
                'mean_u': float(np.mean([clusters[i]['mean_u'] for i in group])),
                'std_u': float(np.mean([clusters[i]['std_u'] for i in group])),
                'mean_v': float(np.mean([clusters[i]['mean_v'] for i in group])),
                'std_v': float(np.mean([clusters[i]['std_v'] for i in group])),
                'dominant_bytes': clusters[group[0]].get('dominant_bytes', []),
                'quality_score': float(np.mean([clusters[i].get('quality_score', 0.5) for i in group])),
                'child_group': group,
            }
            new_clusters.append(new_cluster)
            new_reprs.append(combined_repr)

        d_repr = len(new_reprs[0]) if new_reprs else 256
        return {
            'level_idx': -1,
            'clusters': new_clusters,
            'representations': new_reprs,
            'n_clusters': len(new_clusters),
            'W_transform': np.eye(d_repr, dtype=np.float32),
        }

    def calibrate(self, level_idx: int, current_time: float) -> Dict:
        """
        Крок 4: Калібрація нового рівня.
        η_ℓ' = η_base · (1 + α_calib · e^{-t/τ_calib}), α_calib ≈ 10.
        Висока пластичність → градієнти течуть через новий рівень →
        кластери можуть мігрувати між групами.
        """
        if level_idx not in self.calibration_state:
            self.calibration_state[level_idx] = {
                'start_time': current_time,
                'plasticity': 1.0,
            }
        state = self.calibration_state[level_idx]
        t_since_start = current_time - state['start_time']

        plasticity = 1.0 + self.calibration_alpha * np.exp(-t_since_start / self.calibration_tau)
        state['plasticity'] = plasticity
        effective_lr = self.base_lr * plasticity

        if level_idx < len(self.levels) and self.levels[level_idx].get('W_transform') is not None:
            W = self.levels[level_idx]['W_transform']
            clusters = self.levels[level_idx]['clusters']
            new_reprs = self.levels[level_idx]['representations']

            # Lower level (we need lower level's representations)
            lower_level_idx = int(level_idx - 1)
            if 0 <= lower_level_idx < len(self.levels):
                lower_reprs = self.levels[lower_level_idx]['representations']

                # V7: PyTorch autograd замість ручного dtanh = 1 - pred²
                W_t = torch.from_numpy(W.astype(np.float32)).requires_grad_(True)
                total_loss = torch.tensor(0.0)
                n_valid = 0

                for m, cluster in enumerate(clusters):
                    group = cluster.get('child_group', [])
                    if not group:
                        continue

                    # Aggregated lower representations for this group. Lower
                    # levels may mix 64/128/256-dimensional representations,
                    # so align them before averaging.
                    valid_lower = [
                        lower_reprs[idx] for idx in group
                        if idx < len(lower_reprs)
                    ]
                    if not valid_lower:
                        continue
                    lower_dim = max(len(np.asarray(v).reshape(-1)) for v in valid_lower)
                    agg_lower = np.mean(
                        [self._project_repr(v, lower_dim) for v in valid_lower],
                        axis=0,
                    )
                    agg_lower_proj = agg_lower[:W.shape[1]] if len(agg_lower) > W.shape[1] else np.pad(agg_lower, (0, W.shape[1] - len(agg_lower)))

                    target = new_reprs[m]
                    target_proj = target[:W.shape[0]] if len(target) > W.shape[0] else np.pad(target, (0, W.shape[0] - len(target)))

                    # Torch forward pass — autograd traces through tanh
                    agg_t = torch.from_numpy(agg_lower_proj.astype(np.float32))
                    target_t = torch.from_numpy(target_proj.astype(np.float32))
                    pred_t = torch.tanh(W_t @ agg_t)
                    total_loss = total_loss + torch.sum((pred_t - target_t) ** 2)
                    n_valid += 1

                if n_valid > 0:
                    total_loss = total_loss / n_valid
                    total_loss.backward()
                    with torch.no_grad():
                        W_t.data -= effective_lr * W_t.grad
                        W_t.data.clamp_(-2.0, 2.0)
                    W = W_t.detach().cpu().numpy()
                    self.levels[level_idx]['W_transform'] = W

        return {
            'level_idx': level_idx,
            'plasticity': float(plasticity),
            'effective_lr': float(effective_lr),
            'time_since_start': float(t_since_start),
        }

    def validate_split(self, F_before: float, F_after: float, 
                        stability_trajectory: Optional[List[float]] = None) -> Dict:
        """
        Крок 5: Покращена валідація.
        
        Використовує три критерії для надійної перевірки:
        1. ΔF < -ε: чи енергія знизилася більше ніж на ε
        2. Stability ratio: у скількох % вимірів енергія знижувалась
        3. Consecutive improvement: чи є послідовні покращення в кінці
        
        Повертає:
        - success: True тільки якщо всі критерії виконані
        - delta_F: загальна зміна енергії
        - stability_ratio: частка вимірів де енергія знижувалась
        - details: розгорнута інформація для аналізу
        """
        delta_F = F_after - F_before
        primary_success = delta_F < -self.validation_epsilon
        
        # Критерій стабільності через траєкторію
        stability_ratio = 1.0  # 默认 100%
        consecutive_improvements = 0
        last_n_improvements = 0
        
        if stability_trajectory is not None and len(stability_trajectory) > 1:
            n_steps = len(stability_trajectory) - 1
            decreases = sum(1 for i in range(n_steps) 
                           if stability_trajectory[i+1] < stability_trajectory[i])
            stability_ratio = decreases / n_steps if n_steps > 0 else 1.0
            
            # Перевіряємо останні 5 кроків — чи є покращення
            recent = stability_trajectory[-5:] if len(stability_trajectory) >= 5 else stability_trajectory
            for i in range(len(recent) - 1):
                if recent[i+1] < recent[i]:
                    consecutive_improvements += 1
            last_n_improvements = consecutive_improvements
        
        # Комбінований критерій: primary + stability
        # Перші 3 split'и — lenient (система ще вчиться)
        is_lenient = len(self.split_history) < 3
        success = primary_success and (stability_ratio >= self.min_stability_ratio)
        if is_lenient:
            success = success or (delta_F < 0 and stability_ratio >= 0.5)
        
        return {
            'delta_F': float(delta_F),
            'success': success,
            'action': 'keep' if success else 'merge_back',
            'primary_success': primary_success,
            'stability_ratio': float(stability_ratio),
            'consecutive_improvements': int(consecutive_improvements),
            'is_lenient': is_lenient,
            'primary_delta': float(delta_F),
            'epsilon_used': float(self.validation_epsilon),
            'stability_threshold': float(self.min_stability_ratio),
        }

    def attempt_split(self, level_idx: int, model) -> Dict:
        """Повний цикл розщеплення: 0→1→2→3→4→5."""
        field_system = model.field
        detection = self.detect_bimodality(level_idx)
        if not detection['bimodal']:
            return {'split_attempted': False, 'reason': 'not_bimodal', 'detection': detection}

        W, adjacency = self.build_similarity_graph(level_idx)
        groups = self.spectral_clustering(W)
        if len(groups) < 2:
            return {'split_attempted': False, 'reason': 'single_group', 'detection': detection}

        # Зберігаємо оригінальний стан поля та рівнів для можливого відновлення
        phi_orig = field_system.Phi.copy()
        u_orig = field_system.u.copy()
        v_orig = field_system.v.copy()
        W_field_orig = model.W_field.copy() if model.W_field is not None else None

        F_before = field_system.compute_free_energy(1.0)
        new_level = self.create_new_level(level_idx, groups)
        if not new_level.get('clusters'):
            return {'split_attempted': False, 'reason': 'empty_new_level', 'detection': detection}

        new_level['level_idx'] = level_idx + 0.5
        self.levels.insert(level_idx + 1, new_level)

        # Крок 4: Калібрація нового рівня
        calibration = self.calibrate(level_idx + 1, float(field_system.step_count))

        # Оновлюємо GNN / конвертацію та поле взаємодії
        # Щоб новий рівень вплинув на систему, ми маємо перерахувати конвертацію
        # та оновити W_field польової системи.
        if model.use_gnn_conversion and model.gnn_conversion is not None:
            model.gnn_conversion.n_levels = len(self.levels)
            conv = model.gnn_conversion.convert(model.organizer.clusters, model.substrate)
        else:
            conv_layers = ConversionLayersV3(n_levels=len(self.levels))
            conv = conv_layers.convert(model.organizer.clusters, model.substrate)

        if conv is not None and model.W_field is not None:
            modified_field, attention = model.feedback.apply(model.W_field, conv)
            model.W_field = modified_field
            field_system.update_feed_rate(modified_field)

        # Запускаємо коротку релаксацію поля під впливом нового зворотного зв'язку
        # Збираємо траєкторію енергії для перевірки стабільності
        stability_trajectory = [F_before]
        for _ in range(self.validation_steps):
            field_system.step()
            current_fe = field_system.compute_free_energy(1.0)
            stability_trajectory.append(current_fe)
        
        F_after = stability_trajectory[-1]
        validation = self.validate_split(F_before, F_after, stability_trajectory)

        if not validation['success']:
            # Відкочуємо зміни
            self.levels.pop(level_idx + 1)
            field_system.Phi = phi_orig
            field_system.u = u_orig
            field_system.v = v_orig
            if W_field_orig is not None:
                model.W_field = W_field_orig
                field_system.update_feed_rate(W_field_orig)
            if model.use_gnn_conversion and model.gnn_conversion is not None:
                model.gnn_conversion.n_levels = len(self.levels)
        else:
            # Спліт успішний, фіксуємо зміни в ієрархічній моделі
            if model.use_gnn_conversion and model.gnn_conversion is not None:
                model.gnn_conversion.n_levels = len(self.levels)
            model.n_conversion_levels = len(self.levels)

        self.split_history.append({
            'level_idx': level_idx,
            'detection': detection,
            'n_groups': len(groups),
            'validation': validation,
            'calibration': calibration,
        })

        return {
            'split_attempted': True,
            'split_successful': validation['success'],
            'detection': detection,
            'groups': groups,
            'validation': validation,
            'new_n_clusters': new_level['n_clusters'],
        }

    def attempt_merge(self, level_idx: int, model) -> Dict:
        """
        Зрощення рівнів: якщо P(C_i^(ℓ)|C_j^(ℓ+1)) > 1 - ε
        для більшості кластерів → проміжний рівень зникає.
        """
        if level_idx >= len(self.levels) - 1:
            return {'merge_attempted': False, 'reason': 'no_upper_level'}
        lower = self.levels[level_idx]
        upper = self.levels[level_idx + 1]
        if not lower['clusters'] or not upper['clusters']:
            return {'merge_attempted': False, 'reason': 'empty_level'}

        n_conditional = 0
        for lc in lower['clusters']:
            l_start, l_end = lc['start'], lc['end']
            for uc in upper['clusters']:
                u_start, u_end = uc['start'], uc['end']
                if u_start <= l_start and l_end <= u_end:
                    n_conditional += 1
                    break

        ratio = n_conditional / max(len(lower['clusters']), 1)
        if ratio > 0.9:
            merged_clusters = lower['clusters'] + upper['clusters']
            merged_reprs = lower['representations'] + upper['representations']
            self.levels[level_idx] = {
                'level_idx': level_idx,
                'clusters': merged_clusters,
                'representations': merged_reprs,
                'n_clusters': len(merged_clusters),
                'W_transform': lower.get('W_transform'),
            }
            self.levels.pop(level_idx + 1)

            # Оновлюємо модель
            if model.use_gnn_conversion and model.gnn_conversion is not None:
                model.gnn_conversion.n_levels = len(self.levels)
            model.n_conversion_levels = len(self.levels)

            return {'merge_attempted': True, 'merge_successful': True, 'conditional_ratio': ratio}
        return {'merge_attempted': True, 'merge_successful': False, 'conditional_ratio': ratio}


# =============================================================================
# 8. Повна Модель БКС V6
# =============================================================================


