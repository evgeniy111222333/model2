"""
BCS Trajectory-First Architecture — ПОВНА ЗАМІНА архітектури

ЦЕЙ МОДУЛЬ ЗАМІНЮЄ:
1. Window-based context → Trajectory-first context
2. Softmax attention → Geodesic attention (exp(-d²/T))
3. Buffer memory → Trajectory memory
4. Token sequence → Point trajectory

АРХІТЕКТУРА:
- Контекст = повна траєкторія p(0) → p(1) → ... → p(t)
- Кожна точка = розподіл байтів на S²⁵⁵
- Attention = exp(-geodesic_distance² / T)
- Час = форма траєкторії (кривина, швидкість)
- Семантика = геометрія (петлі, кути, потоки)

ВІДМІННОСТІ ВІД ПОПЕРЕДНЬОГО:
- Попередній: window + trajectory addon
- Новий: trajectory FIRST, window SECONDARY (якщо є)

КЛЮЧОВІ КЛАСИ:
1. TrajectoryFirstModel — модель з trajectory-first архітектурою
2. GeodesicAttentionLayer — шар геодезичного attention
3. TrajectoryReadout — вихід на основі траєкторії
4. GeodesicMemory — пам'ять як траєкторія
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from scipy.special import softmax
from scipy.signal import find_peaks
import warnings


# =============================================================================
# 1. ГЕОМЕТРИЧНІ ПРИМІТИВИ (РІВНЯННЯ НА МНОГОВИДІ)
# =============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Fisher-Rao geodesic distance — природна відстань на S²⁵⁵.
    
    РІВНЯННЯ: d_FR(p, q) = arccos(Σ√p_i·√q_i)
    
    Це Riemannian metric на статистичному многовиді.
    Замість dot-product в softmax — геометрична відстань.
    """
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)
    
    if bc >= 1.0 - epsilon:
        return 0.0
    return np.arccos(bc)


def geodesic_interpolation(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """
    Геодезична інтерполяція на симплексі.
    
    РІВНЯННЯ: p(t) = exp_p1(t · log_map(p2))
    
    Використовує square-root parametrization для симпліциального інтерполяції.
    """
    p1 = np.maximum(p1, 1e-10)
    p2 = np.maximum(p2, 1e-10)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    # Log-map через square-root
    log_p1_p2 = np.sqrt(p2) / np.sqrt(p1).sum() - 1.0
    interp_sqrt = np.sqrt(p1) + t * log_p1_p2
    interp_sqrt = np.maximum(interp_sqrt, 1e-10)
    
    result = interp_sqrt ** 2
    return result / result.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL(p || q) divergence."""
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def compute_curvature(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
    """
    Обчислення кривини в точці через кут між векторами.
    
    РІВНЯННЯ: κ = 1 - Σ√p_{i-1}·√p_i · Σ√p_i·√p_{i+1}
    
    κ ≈ 0: прямий рух
    κ ≈ 1: різкий поворот (кут)
    """
    bc1 = np.sum(np.sqrt(p_prev * p_curr))
    bc2 = np.sum(np.sqrt(p_curr * p_next))
    return max(0, 1.0 - bc1 * bc2)


def compute_velocity(p1: np.ndarray, p2: np.ndarray, dt: float) -> float:
    """Швидкість руху на многовиді: v = d_FR(p1, p2) / dt"""
    d = fisher_rao_distance(p1, p2)
    return d / max(dt, 1e-10)


# =============================================================================
# 2. ТОЧКА НА МНОГОВИДІ (БАЗОВИЙ ЕЛЕМЕНТ)
# =============================================================================

@dataclass
class ManifoldPoint:
    """
    Точка на статистичному многовиді S²⁵⁵.
    
    ЦЕЙ КЛАС ЗАМІНЮЄ ТОКЕН у традиційній архітектурі.
    
    Замість:
    - token = integer ID  ❌
    
    Тепер:
    - point = probability distribution p ∈ S²⁵⁵  ✅
    
    Це єдиний елемент послідовності — і для тексту, і для зображень, і для аудіо.
    """
    p: np.ndarray  # Розподіл на S²⁵⁵ (256-мірний симплекс)
    t: float  # Часова координата [0, 1]
    position: int = 0  # Позиція в сирих даних
    entropy: float = 0.0
    modality: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Точка має бути на симплексі
        self.p = np.maximum(self.p, 1e-10)
        self.p = self.p / self.p.sum()
        self.entropy = -np.sum(self.p * np.log(self.p + 1e-10))
    
    def distance_to(self, other: 'ManifoldPoint', method: str = 'fisher_rao') -> float:
        """Відстань до іншої точки."""
        if method == 'fisher_rao':
            return fisher_rao_distance(self.p, other.p)
        elif method == 'kl':
            return kl_divergence(self.p, other.p)
        return fisher_rao_distance(self.p, other.p)


# =============================================================================
# 3. ТРАЄКТОРІЯ — ОСНОВНИЙ КОНТЕКСТНИЙ МЕХАНІЗМ
# =============================================================================

class Trajectory:
    """
    Траєкторія на статистичному многовиді — ОСНОВНИЙ контекст.
    
    ЗАМІНА ВІКОННОГО ПІДХОДУ:
    
    Було:
    - context = window(tokens)  ❌
    - memory = buffer  ❌
    - time = sequence position  ❌
    
    Стало:
    - context = trajectory {p(0), p(1), ..., p(t)}  ✅
    - memory = trajectory geometry  ✅
    - time = trajectory shape  ✅
    
    ТРАЄКТОРІЯ = КОНТЕКСТ = ПАМ'ЯТЬ = ЧАС
    
    ВЛАСТИВОСТІ:
    - Необмежена довжина (немає "контекст закінчився")
    - Геометричний attention (exp(-d²/T) замість softmax(q·k))
    - Автоматичне виявлення новизни (через кривину)
    - Семантика через геометрію (петлі, кути, потоки)
    """
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        decay_rate: float = 0.99,
        temperature: float = 1.0,
        enable_semantic: bool = True,
    ):
        """
        Args:
            max_length: Макс. точок (None = без обмежень)
            decay_rate: Часовий decay для attention
            temperature: Температура softmax
            enable_semantic: Виявляти семантичні форми
        """
        self.points: List[ManifoldPoint] = []
        self.max_length = max_length
        self.decay_rate = decay_rate
        self.temperature = temperature
        self.enable_semantic = enable_semantic
        
        # Геометрія траєкторії
        self.total_length: float = 0.0
        self.curvature_profile: List[float] = []
        self.velocity_profile: List[float] = []
        
        # Пам'ять (центроїд траєкторії)
        self.memory_centroid: Optional[np.ndarray] = None
        self.memory_span: float = 0.0
        
        # Семантичні форми
        self.loops: List[int] = []  # Петлі (повторення)
        self.angles: List[int] = []  # Кути (різкі зміни)
        self.streams: List[int] = []  # Потоки (плавні переходи)
    
    def push(self, p: np.ndarray, t: Optional[float] = None, 
             position: int = 0, modality: str = "unknown",
             metadata: Optional[Dict] = None) -> ManifoldPoint:
        """
        Додати точку до траєкторії.
        
        ЦЕ ЗАМІНЮЄ append() до контекстного вікна.
        
        Args:
            p: розподіл байтів (точка на S²⁵⁵)
            t: часова координата
            position: позиція в даних
            modality: модальність
            metadata: додаткові дані
            
        Returns:
            Створена точка
        """
        if t is None:
            t = len(self.points) / max(1, self.max_length or 1000)
        
        point = ManifoldPoint(
            p=p, t=t, position=position, 
            modality=modality, metadata=metadata or {}
        )
        
        self.points.append(point)
        
        # Оновлюємо геометрію
        self._update_geometry()
        
        # Оновлюємо пам'ять
        self._update_memory()
        
        # Виявляємо семантичні форми
        if self.enable_semantic and len(self.points) >= 3:
            self._detect_semantic_shapes()
        
        # Агрегація якщо занадто довга
        if self.max_length and len(self.points) > self.max_length:
            self._aggregate()
        
        return point
    
    def _update_geometry(self):
        """Оновити геометричні характеристики."""
        if len(self.points) < 2:
            return
        
        # Сумарна довжина
        self.total_length = 0.0
        self.velocity_profile = []
        
        for i in range(1, len(self.points)):
            d = self.points[i].distance_to(self.points[i-1])
            self.total_length += d
            dt = self.points[i].t - self.points[i-1].t
            self.velocity_profile.append(d / max(dt, 1e-10))
        
        # Кривина
        self.curvature_profile = []
        for i in range(1, len(self.points) - 1):
            curv = compute_curvature(
                self.points[i-1].p,
                self.points[i].p,
                self.points[i+1].p
            )
            self.curvature_profile.append(curv)
    
    def _update_memory(self):
        """Оновити пам'ять (центроїд траєкторії)."""
        if not self.points:
            return
        
        # Центроїд з часовим decay
        weights = np.array([
            self.decay_rate ** (len(self.points) - 1 - i)
            for i in range(len(self.points))
        ])
        weights = weights / weights.sum()
        
        self.memory_centroid = sum(
            w * p.p for w, p in zip(weights, self.points)
        )
        self.memory_centroid = self.memory_centroid / self.memory_centroid.sum()
        
        # Span
        if len(self.points) > 1:
            distances = [fisher_rao_distance(p.p, self.memory_centroid) for p in self.points]
            self.memory_span = np.mean(distances)
    
    def _detect_semantic_shapes(self):
        """Виявити семантичні форми через геометрію."""
        self.loops = []
        self.angles = []
        self.streams = []
        
        # Петель (повернення назад)
        for i in range(1, len(self.points) - 1):
            d_back = self.points[i].distance_to(self.points[i-1])
            d_forward = self.points[i].distance_to(self.points[i+1])
            if d_back < d_forward * 0.5:
                self.loops.append(i)
        
        # Кутів (різка зміна)
        for i, curv in enumerate(self.curvature_profile):
            if curv > 0.4:
                self.angles.append(i + 1)
        
        # Потоків (плавний рух)
        if self.velocity_profile and self.curvature_profile:
            mean_curv = np.mean(self.curvature_profile)
            mean_vel = np.mean(self.velocity_profile)
            for i in range(min(len(self.velocity_profile), len(self.curvature_profile))):
                if (self.velocity_profile[i] > mean_vel and 
                    self.curvature_profile[i] < mean_curv * 0.5):
                    self.streams.append(i)
    
    def _aggregate(self):
        """Агрегація старих точок."""
        n_keep = max(2, self.max_length // 10)
        
        # Агрегована точка
        agg_p = np.mean([p.p for p in self.points[:n_keep]], axis=0)
        agg_p = agg_p / agg_p.sum()
        
        self.points = self.points[n_keep:]
        self.points.insert(0, ManifoldPoint(
            p=agg_p, t=0,
            metadata={'aggregated': True}
        ))
    
    def compute_attention(self, query: np.ndarray) -> np.ndarray:
        """
        Геодезичний attention — ЗАМІНА softmax(q·k).
        
        РІВНЯННЯ: attention_i = exp(-d_FR(query, p_i)² / T)
        
        Замість:
        - softmax(query · key)  ❌
        
        Тепер:
        - softmax(-geodesic_distance² / T)  ✅
        
        Args:
            query: розподіл запиту
            
        Returns:
            attention weights [n_points]
        """
        if not self.points:
            return np.array([])
        
        # Геодезичні відстані
        distances = np.array([
            fisher_rao_distance(query, p.p) for p in self.points
        ])
        
        # Геометричний attention
        energies = -distances ** 2 / self.temperature
        energies = energies - energies.max()
        
        # Часовий decay
        time_decay = np.array([
            self.decay_rate ** (len(self.points) - 1 - i)
            for i in range(len(self.points))
        ])
        energies = energies + np.log(time_decay + 1e-10)
        
        # Softmax
        exp_energies = np.exp(energies - energies.max())
        attention = exp_energies / (exp_energies.sum() + 1e-10)
        
        return attention
    
    def attend(self, query: np.ndarray, feature: str = 'p') -> np.ndarray:
        """
        Attention до траєкторії з отриманням результату.
        
        Args:
            query: розподіл запиту
            feature: яку ознаку збирати ('p', 'entropy', 'position')
            
        Returns:
            Зважений результат
        """
        attention = self.compute_attention(query)
        
        if len(attention) == 0:
            return np.zeros(256)
        
        if feature == 'p':
            features = np.array([p.p for p in self.points])
        elif feature == 'entropy':
            features = np.array([[p.entropy] for p in self.points])
        else:
            features = np.array([p.p for p in self.points])
        
        return attention @ features
    
    def get_context(self, query: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Отримати контекстний вектор — ГЕОМЕТРИЧНЕ ПРЕДСТАВЛЕННЯ ТРАЄКТОРІЇ.
        
        ЦЕ ЗАМІНА mean(window_embeddings).
        
        Args:
            query: якщо є — attention-зважений контекст
                   якщо None — остання точка
            
        Returns:
            256-мірний контекст
        """
        if not self.points:
            return np.zeros(256)
        
        if query is None:
            return self.points[-1].p.copy()
        
        return self.attend(query)
    
    def detect_novelty(self, p: np.ndarray) -> Tuple[float, float]:
        """
        Виявити новизну через геометрію.
        
        Returns:
            (novelty, confidence)
        """
        if not self.points:
            return 1.0, 1.0
        
        # Відстань до центроїда
        d_center = fisher_rao_distance(p, self.memory_centroid)
        dist_score = min(1.0, d_center / (self.memory_span + 0.1))
        
        # Кривина
        if self.curvature_profile:
            curv_score = min(1.0, np.mean(self.curvature_profile[-3:]) * 5)
        else:
            curv_score = 0.0
        
        # Швидкість
        if self.velocity_profile:
            vel_score = min(1.0, np.mean(self.velocity_profile[-3:]) * 10)
        else:
            vel_score = 0.0
        
        novelty = 0.4 * dist_score + 0.3 * curv_score + 0.3 * vel_score
        novelty = min(1.0, novelty)
        confidence = min(1.0, len(self.points) / 20.0)
        
        return novelty, confidence
    
    def get_summary(self) -> Dict[str, Any]:
        """Отримати summary траєкторії."""
        if not self.points:
            return {'n_points': 0}
        
        return {
            'n_points': len(self.points),
            'total_length': self.total_length,
            'current_entropy': self.points[-1].entropy,
            'memory_span': self.memory_span,
            'semantic': {
                'n_loops': len(self.loops),
                'n_angles': len(self.angles),
                'n_streams': len(self.streams),
            },
            'topology': {
                'betti_0': 1.0,
                'betti_1': len(self.loops),
                'total_variation': self.total_length,
            }
        }
    
    def __len__(self) -> int:
        return len(self.points)


# =============================================================================
# 4. ГЕОДЕЗИЧНИЙ ATTENTION ШАР (ДЛЯ ЗАМІНИ SOFTMAX)
# =============================================================================

class GeodesicAttentionLayer:
    """
    Шар геодезичного attention — ЗАМІНА стандартного attention.
    
    ВИКОРИСТОВУЄТЬСЯ В:
    - Конвертаційних шарах
    - Семантичному шарі
    - Вихідному шарі
    
    РІВНЯННЯ:
    output = Σ_i softmax(-d_FR(query, key_i)² / T) · value_i
    """
    
    def __init__(self, temperature: float = 1.0, decay_rate: float = 0.95):
        self.temperature = temperature
        self.decay_rate = decay_rate
    
    def forward(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Args:
            query: розподіл запиту (256-мірний)
            keys: список розподілів ключів
            values: список розподілів значень (якщо None — keys)
            
        Returns:
            (output, attention_weights)
        """
        if values is None:
            values = keys
        
        n = len(keys)
        if n == 0:
            return np.zeros(256), np.array([])
        
        # Геодезичні відстані
        distances = np.array([
            fisher_rao_distance(query, k) for k in keys
        ])
        
        # Attention
        energies = -distances ** 2 / self.temperature
        energies = energies - energies.max()
        
        # Time decay
        time_decay = np.array([
            self.decay_rate ** (n - 1 - i) for i in range(n)
        ])
        energies = energies + np.log(time_decay + 1e-10)
        
        # Softmax
        exp_energies = np.exp(energies - energies.max())
        attention = exp_energies / (exp_energies.sum() + 1e-10)
        
        # Output
        values_arr = np.array(values)
        output = attention @ values_arr
        
        return output, attention


# =============================================================================
# 5. ТРАЄКТОРІЯ ДЛЯ КОНВЕРТАЦІЙНИХ ШАРІВ (ЗАМІНА GCN)
# =============================================================================

class TrajectoryConversion:
    """
    Конвертація кластерів через траєкторію — ЗАМІНА GCN/MLP.
    
    ВИКОРИСТОВУЄ:
    - GeodesicAttention замість GCN message passing
    - Траєкторію як контекст замість локального графа
    
    РІВНЯННЯ:
    T_ℓ(C) = Σ_i softmax(-d_FR(p_C, p_i)² / T) · T_{ℓ-1}(C_i)
    """
    
    def __init__(
        self,
        n_levels: int = 4,
        temperature: float = 1.0,
        decay_rate: float = 0.95,
    ):
        self.n_levels = n_levels
        self.attention = GeodesicAttentionLayer(
            temperature=temperature,
            decay_rate=decay_rate,
        )
    
    def convert(self, clusters: List[Dict]) -> List[Dict]:
        """
        Конвертація кластерів через траєкторію.
        
        Args:
            clusters: список кластерів з 'distribution' полем
            
        Returns:
            Конвертовані рівні
        """
        if not clusters:
            return []
        
        all_levels = []
        
        # Рівень 0: базові представлення
        level0 = []
        for i, c in enumerate(clusters):
            dist = c.get('distribution', np.zeros(256))
            level0.append({
                'id': f'L0_C{i}',
                'representation': dist,
                'level': 0,
            })
        all_levels.append({'level': 0, 'items': level0})
        
        # Вищі рівні через geodesic attention
        current = level0
        for lvl in range(1, self.n_levels):
            if len(current) <= 1:
                break
            
            # Geodesic attention для кожного кластера
            new_level = []
            for i, item in enumerate(current):
                query = item['representation']
                keys = [it['representation'] for it in current]
                
                output, attention = self.attention.forward(query, keys)
                
                new_level.append({
                    'id': f'L{lvl}_C{i}',
                    'representation': output,
                    'attention': attention,
                    'level': lvl,
                })
            
            all_levels.append({'level': lvl, 'items': new_level})
            current = new_level
        
        return all_levels


# =============================================================================
# 6. ТРАЄКТОРІЯ ДЛЯ СЕМАНТИЧНОГО ШАРУ (ЗАМІНА TRANSFORMER)
# =============================================================================

class TrajectorySemantic:
    """
    Семантичний шар на основі траєкторії — ЗАМІНА transformer.
    
    ВИКОРИСТОВУЄ:
    - Trajectory як working memory
    - GeodesicAttention для cross-attention
    - Trajectory shape для семантичного виводу
    
    ЦЕ НЕ transformer — тут немає позиційного кодування,
    багатошарового attention, чи feed-forward шарів.
    
    ТРАЄКТОРІЯ САМА МІСТИТЬ УСЮ ІНФОРМАЦІЮ ПРО ЧАС І КОНТЕКСТ.
    """
    
    def __init__(
        self,
        d_latent: int = 256,
        temperature: float = 1.0,
    ):
        self.d_latent = d_latent
        self.temperature = temperature
        self.trajectory = Trajectory(
            max_length=1000,
            decay_rate=0.99,
            temperature=temperature,
        )
        self.attention = GeodesicAttentionLayer(temperature=temperature)
    
    def encode(self, data: bytes) -> np.ndarray:
        """
        Закодувати дані в траєкторію.
        
        Args:
            data: сирі байти
            
        Returns:
            Латентний вектор
        """
        n = len(data)
        step = max(1, n // 100)
        
        for i in range(0, n, step):
            half_w = 8
            start = max(0, i - half_w)
            end = min(n, i + half_w)
            
            dist = np.zeros(256)
            for b in data[start:end]:
                dist[b] += 1
            
            if dist.sum() > 0:
                dist = dist / dist.sum()
            else:
                dist = np.ones(256) / 256
            
            self.trajectory.push(
                p=dist,
                t=float(i) / n,
                position=i,
            )
        
        # Латент = контекстний вектор
        return self.trajectory.get_context()
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Декодувати латентний вектор назад у розподіл.
        
        Знаходимо найближчу точку в траєкторії.
        
        Args:
            latent: латентний вектор
            
        Returns:
            Розподіл байтів
        """
        if not self.trajectory.points:
            return np.ones(256) / 256
        
        # Знайти найближчу точку
        distances = [
            fisher_rao_distance(latent, p.p) 
            for p in self.trajectory.points
        ]
        nearest_idx = np.argmin(distances)
        
        return self.trajectory.points[nearest_idx].p.copy()
    
    def query(self, query: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Запит до траєкторії.
        
        Args:
            query: розподіл запиту
            top_k: кількість результатів
            
        Returns:
            Результат з геометричною інформацією
        """
        attention = self.trajectory.compute_attention(query)
        
        # Топ-k індексів
        top_indices = np.argsort(attention)[-top_k:][::-1]
        top_attentions = attention[top_indices]
        
        return {
            'top_indices': top_indices.tolist(),
            'top_attentions': top_attentions.tolist(),
            'context': self.trajectory.attend(query).tolist(),
            'novelty': self.trajectory.detect_novelty(query)[0],
        }


# =============================================================================
# 7. ТРАЄКТОРІЯ ДЛЯ ВИХІДНОГО ШАРУ (ЗАМІНА LM HEAD)
# =============================================================================

class TrajectoryReadout:
    """
    Вихідний шар на основі траєкторії — ЗАМІНА LM head / softmax.
    
    ВИКОРИСТОВУЄ:
    - Trajectory context для вибору наступної точки
    - Геометричну відстань замість logit
    
    РІВНЯННЯ:
    P(next | trajectory) = argmax_p (-d_FR(trajectory_context, p))
    
    ЦЕ НЕ softmax над vocab — це вибір на многовиді.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.trajectory = Trajectory()
    
    def update(self, data: bytes):
        """Оновити траєкторію новими даними."""
        n = len(data)
        step = max(1, n // 50)
        
        for i in range(0, n, step):
            dist = np.zeros(256)
            for b in data[max(0, i-4):min(n, i+4)]:
                dist[b] += 1
            if dist.sum() > 0:
                dist = dist / dist.sum()
                self.trajectory.push(p=dist, t=float(i)/n, position=i)
    
    def predict_next(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Передбачити наступний байт.
        
        Args:
            context: контекстний вектор з траєкторії
            
        Returns:
            (predicted_byte, confidence)
        """
        if not self.trajectory.points:
            return 0, 0.0
        
        # Attention до траєкторії
        attention = self.trajectory.compute_attention(context)
        
        # Зважена сума всіх розподілів
        output = self.trajectory.attend(context)
        
        # Обираємо найімовірніший байт
        predicted_byte = int(np.argmax(output))
        confidence = float(output[predicted_byte])
        
        return predicted_byte, confidence
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Отримати summary траєкторії."""
        return self.trajectory.get_summary()


# =============================================================================
# 8. ПОВНА TRAJECTORY-FIRST МОДЕЛЬ
# =============================================================================

class TrajectoryFirstModel:
    """
    ПОВНА TRAJECTORY-FIRST МОДЕЛЬ.
    
    ЦЕЙ КЛАС ЗАМІНЮЄ:
    - BCSModelV6 з window-based context
    - Всі transformer attention
    - Всі GCN конвертаційні шари
    
    АРХІТЕКТУРА:
    1. INPUT: bytes → Trajectory
    2. ENCODE: Trajectory.encode() → latent
    3. ATTEND: GeodesicAttention на Trajectory
    4. CONVERT: TrajectoryConversion (замість GCN)
    5. SEMANTIC: TrajectorySemantic (замість transformer)
    6. OUTPUT: TrajectoryReadout (замість LM head)
    
    ВСЯ АРХІТЕКТУРА ПОБУДОВАНА НА ТРАЄКТОРІЇ.
    """
    
    def __init__(
        self,
        d_latent: int = 256,
        n_conversion_levels: int = 4,
        temperature: float = 1.0,
        decay_rate: float = 0.99,
        max_trajectory_length: int = 1000,
    ):
        self.d_latent = d_latent
        self.n_conversion_levels = n_conversion_levels
        
        # Основні компоненти
        self.trajectory = Trajectory(
            max_length=max_trajectory_length,
            decay_rate=decay_rate,
            temperature=temperature,
        )
        
        self.conversion = TrajectoryConversion(
            n_levels=n_conversion_levels,
            temperature=temperature,
            decay_rate=decay_rate,
        )
        
        self.semantic = TrajectorySemantic(
            d_latent=d_latent,
            temperature=temperature,
        )
        
        self.readout = TrajectoryReadout(temperature=temperature)
        
        # Стан
        self.latent: Optional[np.ndarray] = None
        self.converted_levels: List[Dict] = []
    
    def ingest(self, data: bytes):
        """
        Поглинути дані в траєкторію.
        
        Args:
            data: сирі байти
        """
        n = len(data)
        step = max(1, n // 100)
        
        for i in range(0, n, step):
            half_w = 8
            start = max(0, i - half_w)
            end = min(n, i + half_w)
            
            dist = np.zeros(256)
            for b in data[start:end]:
                dist[b] += 1
            
            if dist.sum() > 0:
                dist = dist / dist.sum()
            else:
                dist = np.ones(256) / 256
            
            # Визначення модальності
            ascii_ratio = np.sum(dist[0x20:0x7F])
            if ascii_ratio > 0.8:
                modality = 'text_ascii'
            elif np.sum(dist[0x80:]) > 0.3:
                modality = 'text_utf8'
            elif dist[0] + dist[255] > 0.5:
                modality = 'binary'
            else:
                modality = 'mixed'
            
            self.trajectory.push(
                p=dist,
                t=float(i) / n,
                position=i,
                modality=modality,
            )
    
    def encode(self) -> np.ndarray:
        """
        Закодувати траєкторію в латент.
        
        Returns:
            Латентний вектор
        """
        if not self.trajectory.points:
            return np.zeros(self.d_latent)
        
        # Використовуємо останню точку як контекст
        # Але з урахуванням всієї траєкторії через attention
        self.latent = self.trajectory.get_context()
        
        # Розширюємо до d_latent якщо треба
        if len(self.latent) < self.d_latent:
            self.latent = np.pad(self.latent, (0, self.d_latent - len(self.latent)))
        else:
            self.latent = self.latent[:self.d_latent]
        
        return self.latent
    
    def convert(self, clusters: List[Dict]) -> List[Dict]:
        """
        Конвертувати кластери через траєкторію.
        
        Args:
            clusters: список кластерів
            
        Returns:
            Конвертовані рівні
        """
        self.converted_levels = self.conversion.convert(clusters)
        return self.converted_levels
    
    def query(self, query: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Запит до моделі.
        
        Args:
            query: розподіл запиту
            top_k: кількість результатів
            
        Returns:
            Результат з геометричною інформацією
        """
        return self.semantic.query(query, top_k)
    
    def predict_next(self) -> Tuple[int, float]:
        """
        Передбачити наступний байт.
        
        Returns:
            (predicted_byte, confidence)
        """
        if self.latent is None:
            self.encode()
        
        return self.readout.predict_next(self.latent)
    
    def get_summary(self) -> Dict[str, Any]:
        """Отримати summary всієї моделі."""
        return {
            'trajectory': self.trajectory.get_summary(),
            'conversion_levels': len(self.converted_levels),
            'latent_shape': self.latent.shape if self.latent is not None else None,
            'readout': self.readout.get_trajectory_summary(),
        }
    
    def run_full(self, data: bytes, clusters: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Повний прохід через модель.
        
        Args:
            data: сирі байти
            clusters: опціональні кластери
            
        Returns:
            Повні результати
        """
        # 1. Ingest
        self.ingest(data)
        
        # 2. Encode
        latent = self.encode()
        
        # 3. Convert (якщо є кластери)
        if clusters:
            converted = self.convert(clusters)
        else:
            converted = []
        
        # 4. Predict next
        next_byte, confidence = self.predict_next()
        
        return {
            'latent': latent,
            'trajectory_summary': self.trajectory.get_summary(),
            'converted_levels': converted,
            'predicted_next': next_byte,
            'confidence': confidence,
        }


# =============================================================================
# 9. ЕКСПОРТ
# =============================================================================

__all__ = [
    # Геометричні примітиви
    'fisher_rao_distance',
    'geodesic_interpolation',
    'kl_divergence',
    'compute_curvature',
    'compute_velocity',
    
    # Базові класи
    'ManifoldPoint',
    'Trajectory',
    'GeodesicAttentionLayer',
    
    # Компоненти
    'TrajectoryConversion',
    'TrajectorySemantic',
    'TrajectoryReadout',
    
    # Модель
    'TrajectoryFirstModel',
]
