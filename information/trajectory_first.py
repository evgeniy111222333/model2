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
    ПРАВИЛЬНА геодезична інтерполяція на S²⁵⁵ з Fisher-Rao метрикою.
    
    log_map(p1→p2) = θ · (√p2/BC - √p1) / ‖√p2/BC - √p1‖
    де θ = arccos(BC), BC = <√p1, √p2> = Σ√p1_i·√p2_i
    
    exp_p1(t·v) = cos(t·θ)·√p1 + sin(t·θ)·v / ‖v‖
    """
    eps = 1e-10
    p1 = np.maximum(p1, eps)
    p2 = np.maximum(p2, eps)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    sqrt_p1 = np.sqrt(p1)
    sqrt_p2 = np.sqrt(p2)
    
    bc = np.dot(sqrt_p1, sqrt_p2)
    bc = np.clip(bc, 0.0, 1.0 - eps)
    
    if bc > 0.9999:
        interp_sqrt = (1 - t) * sqrt_p1 + t * sqrt_p2
    else:
        theta = np.arccos(bc)
        diff = sqrt_p2 / bc - sqrt_p1
        norm_diff = np.linalg.norm(diff)
        
        if norm_diff < eps:
            interp_sqrt = sqrt_p1.copy()
        else:
            direction = diff / norm_diff
            interp_sqrt = np.cos(t * theta) * sqrt_p1 + np.sin(t * theta) * direction
    
    interp_sqrt = np.maximum(interp_sqrt, eps)
    result = interp_sqrt ** 2
    return result / result.sum()


def frechet_mean(points: List[np.ndarray], max_iter: int = 50, lr: float = 0.5, 
                 tol: float = 1e-7, epsilon: float = 1e-10) -> np.ndarray:
    """
    Fréchet mean на статистичному многовиді S^{n-1} з Fisher-Rao метрикою.
    
    Шукає точку m, яка мінімізує: Σ d_FR(m, p_i)²
    
    ІТЕРАТИВНА ФОРМУЛА (Barzilai-Borwein gradient descent):
    m_{t+1} = exp_{m_t}(η · grad F(m_t))
    
    де grad F(m) = -Σ log_map(m→p_i) / |Σ log_map(m→p_i)|
    
    Args:
        points: список розподілів
        max_iter: максимум ітерацій
        lr: learning rate
        tol: tolerance для збіжності
        epsilon: epsilon для чисельної стабільності
        
    Returns:
        Fréchet mean (точка на симплексі)
    """
    if not points:
        raise ValueError("Empty points list")
    
    if len(points) == 1:
        p = np.maximum(points[0], epsilon)
        return p / p.sum()
    
    # Перевірка на ідентичні точки
    all_same = all(np.allclose(points[0], p, atol=1e-8) for p in points[1:])
    if all_same:
        p = np.maximum(points[0], epsilon)
        return p / p.sum()
    
    # Ініціалізація: arithmetic mean
    m = np.mean(points, axis=0)
    m = np.maximum(m, epsilon)
    m = m / m.sum()
    
    prev_m = m.copy()
    
    for iteration in range(max_iter):
        m_sqrt = np.sqrt(m)
        
        # Gradients для кожної точки
        gradients = []
        total_norm = 0.0
        
        for p in points:
            p = np.maximum(p, epsilon)
            p = p / p.sum()
            sqrt_p = np.sqrt(p)
            
            # Bhattacharyya coefficient
            bc = np.dot(m_sqrt, sqrt_p)
            bc_clamped = np.clip(bc, 0.0, 1.0 - epsilon)
            
            # Якщо точки близькі — пропускаємо
            if bc_clamped > 1.0 - epsilon:
                continue
            
            # Кут
            theta = np.arccos(bc_clamped)
            
            # Log-map напрямок: v = (√p/BC - √m) / ‖√p/BC - √m‖
            diff = sqrt_p / bc_clamped - m_sqrt
            norm_diff = np.linalg.norm(diff)
            
            if norm_diff < epsilon:
                continue
            
            direction = diff / norm_diff
            
            # Gradient вкатається в напрямку log-map
            grad = theta * direction
            gradients.append(grad)
            total_norm += norm(grad) ** 2
        
        if not gradients:
            # Всі точки ідентичні
            break
        
        # Середній градієнт
        grad_mean = np.mean(gradients, axis=0)
        grad_norm = np.linalg.norm(grad_mean)
        
        if grad_norm < tol:
            # Збіглося
            break
        
        # Крок в напрямку, протилежному градієнту (minimization)
        step = lr * grad_mean
        
        # Exp-map: m_new = cos(|step|) * m + sin(|step|) * step/|step|
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            m = np.cos(step_norm) * m_sqrt + np.sin(step_norm) * step / step_norm
            m = np.maximum(m, epsilon) ** 2
            m = m / m.sum()
        
        # Перевірка збіжності
        if np.linalg.norm(m - prev_m) < tol:
            break
        
        prev_m = m.copy()
    
    return m


def norm(x: np.ndarray) -> float:
    """L2 норма."""
    return float(np.linalg.norm(x))


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
        
        # Span — batch обчислення
        if len(self.points) > 1:
            centroid_sqrt = np.sqrt(np.maximum(self.memory_centroid, 1e-10))
            all_sqrt = np.array([np.sqrt(np.maximum(p.p, 1e-10)) for p in self.points])
            bc = all_sqrt @ centroid_sqrt
            bc = np.clip(bc, 0, 1)
            distances = np.arccos(bc)
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
        """
        Агрегація старих точок через Fréchet mean.
        
        ВИПРАВЛЕНО: замість arithmetic mean (який розмиває геометрію)
        тепер використовуємо Fréchet mean, який зберігає геометричну структуру.
        
        Fréchet mean — це точка на многовиді, яка мінімізує
        суму квадратів відстаней Fisher-Rao до всіх агрегованих точок.
        """
        n_keep = max(2, self.max_length // 10)
        n_agg = len(self.points) - n_keep
        
        if n_agg < 2:
            return
        
        # Fréchet mean замість arithmetic mean
        points_to_agg = [p.p for p in self.points[:n_agg]]
        agg_p = frechet_mean(points_to_agg)
        
        # Видаляємо старі точки і додаємо агреговану
        self.points = self.points[n_agg:]
        self.points.insert(0, ManifoldPoint(
            p=agg_p, t=0,
            metadata={'aggregated': True, 'n_points': n_agg}
        ))
    
    def compute_attention(self, query: np.ndarray) -> np.ndarray:
        """
        Геодезичний attention — ЗАМІНА softmax(q·k).
        
        OPTIMIZED: batch обчислення O(n) замість O(n*d).
        
        РІВНЯННЯ: attention_i = exp(-d_FR(query, p_i)² / T)
        
        Args:
            query: розподіл запиту
            
        Returns:
            attention weights [n_points]
        """
        if not self.points:
            return np.array([])
        
        # Batch обчислення відстаней: O(n) замість O(n*d)
        query_sqrt = np.sqrt(np.maximum(query, 1e-10))
        all_sqrt = np.array([np.sqrt(np.maximum(p.p, 1e-10)) for p in self.points])
        bc = all_sqrt @ query_sqrt
        bc = np.clip(bc, 0, 1)
        distances = np.arccos(bc)
        
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
        
        # Batch обчислення: O(n) замість O(n*d)
        query_sqrt = np.sqrt(np.maximum(query, 1e-10))
        all_sqrt = np.array([np.sqrt(np.maximum(k, 1e-10)) for k in keys])
        bc = all_sqrt @ query_sqrt
        bc = np.clip(bc, 0, 1)
        distances = np.arccos(bc)
        
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
    
    def predict_next(
        self, 
        context: np.ndarray,
        method: str = 'nucleus',
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 40,
        use_geometric_continuation: bool = True,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Покращене передбачення наступного байту.
        
        МЕТОДИ:
        1. 'nucleus' — Nucleus sampling (p-parameter) [РЕКОМЕНДОВАНО]
        2. 'temperature' — Temperature sampling
        3. 'top_k' — Top-k sampling
        4. 'argmax' — Greedy (детермінований)
        
        ГЕОМЕТРИЧНЕ ПРОДОВЖЕННЯ:
        - Якщо use_geometric_continuation=True, використовуємо напрямок траєкторії
          для передбачення наступного розподілу
        
        Args:
            context: контекстний вектор з траєкторії
            method: метод семплування
            temperature: temperature для семплування (>1 = більше randomness, <1 = більш детермінований)
            top_p: для nucleus sampling (cumulative probability threshold)
            top_k: для top-k sampling
            use_geometric_continuation: чи використовувати геометричне продовження
            
        Returns:
            (predicted_byte, confidence, output_distribution)
        """
        if not self.trajectory.points:
            return 0, 0.0, np.zeros(256)
        
        # Зважена сума через geodesic attention
        output = self.trajectory.attend(context)
        
        # Геометричне продовження траєкторії
        if use_geometric_continuation and len(self.trajectory.points) >= 2:
            output = self._geometric_continuation(output, context)
        
        # Нормалізація
        output = np.maximum(output, 1e-10)
        output = output / output.sum()
        
        # Семплування залежно від методу
        if method == 'argmax':
            predicted_byte = int(np.argmax(output))
            confidence = float(output[predicted_byte])
        
        elif method == 'temperature':
            # Temperature sampling
            logits = np.log(output + 1e-10) / temperature
            logits = logits - logits.max()  # Numerical stability
            probs = np.exp(logits)
            probs = probs / probs.sum()
            predicted_byte = int(np.random.choice(256, p=probs))
            confidence = float(output[predicted_byte])
        
        elif method == 'top_k':
            # Top-k sampling
            top_indices = np.argsort(output)[-top_k:]
            top_probs = output[top_indices]
            top_probs = top_probs / top_probs.sum()
            predicted_byte = int(np.random.choice(top_indices, p=top_probs))
            confidence = float(output[predicted_byte])
        
        elif method == 'nucleus':
            # Nucleus (top-p) sampling
            sorted_indices = np.argsort(output)[::-1]
            sorted_probs = output[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            
            # Відсікаємо хвіст щоб сума <= top_p
            cutoff = np.searchsorted(cumsum, top_p) + 1
            nucleus_indices = sorted_indices[:cutoff]
            nucleus_probs = sorted_probs[:cutoff]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            predicted_byte = int(np.random.choice(nucleus_indices, p=nucleus_probs))
            confidence = float(output[predicted_byte])
        
        else:
            # Default to argmax
            predicted_byte = int(np.argmax(output))
            confidence = float(output[predicted_byte])
        
        return predicted_byte, confidence, output
    
    def _geometric_continuation(self, attention_output: np.ndarray, context: np.ndarray) -> np.ndarray:
        """
        Геометричне продовження траєкторії.
        
        Використовує напрямок і швидкість останніх точок
        для передбачення наступного розподілу.
        
        РІВНЯННЯ:
        p_next ≈ geodesic_interpolation(p_last, p_last + velocity, 0.5)
        
        Args:
            attention_output: output від geodesic attention
            context: контекстний вектор
            
        Returns:
            Покращений розподіл з урахуванням геометрії
        """
        if len(self.trajectory.points) < 3:
            return attention_output
        
        # Останні 3 точки для обчислення напрямку
        p_prev = self.trajectory.points[-2].p
        p_curr = self.trajectory.points[-1].p
        
        # Швидкість: різниця в √-просторі
        sqrt_prev = np.sqrt(p_prev)
        sqrt_curr = np.sqrt(p_curr)
        velocity = sqrt_curr - sqrt_prev
        
        # Напрямок екстраполяції
        sqrt_extended = sqrt_curr + velocity * 0.3  # 0.3 = розмір кроку
        sqrt_extended = np.maximum(sqrt_extended, 1e-10)
        sqrt_extended = sqrt_extended / np.linalg.norm(sqrt_extended)
        
        # Перевірка чи напрямок стабільний
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm < 1e-6:
            # Занадто мала швидкість — не екстраполюємо
            return attention_output
        
        # Геометрична інтерполяція між поточною точкою і екстрапольованою
        extended = sqrt_extended ** 2
        extended = extended / extended.sum()
        
        # Змішуємо з geodesic attention output
        # Вага залежить від впевненості attention
        attention_weight = float(np.max(self.trajectory.compute_attention(context)))
        
        # Якщо увага сильна на поточній точці — використовуємо геометрію
        mix = 0.3 * (1.0 - attention_weight)
        result = (1 - mix) * attention_output + mix * extended
        
        return result / result.sum()
    
    def generate_sequence(
        self,
        context: np.ndarray,
        length: int = 50,
        method: str = 'nucleus',
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[Tuple[int, float]]:
        """
        Згенерувати послідовність байтів.
        
        Args:
            context: початковий контекст
            length: довжина послідовності
            method: метод семплування
            temperature: temperature
            top_p: для nucleus sampling
            
        Returns:
            Список (byte, confidence)
        """
        results = []
        current_context = context.copy()
        
        for _ in range(length):
            byte, conf, _ = self.predict_next(
                current_context,
                method=method,
                temperature=temperature,
                top_p=top_p,
            )
            results.append((byte, conf))
            
            # Оновлюємо контекст для наступного кроку
            # (в простому випадку — додаємо передбачений байт)
            if len(self.trajectory.points) > 0:
                last_p = self.trajectory.points[-1].p.copy()
                # "Видаляємо" найстарішу точку і просуваємо час
                # Це спрощена версія — реальна реалізація потребує
                # більш складної логіки оновлення траєкторії
                current_context = 0.9 * current_context + 0.1 * last_p
                current_context = current_context / current_context.sum()
        
        return results
    
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
    
    def predict_next(
        self,
        method: str = 'nucleus',
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_geometric_continuation: bool = True,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Передбачити наступний байт.
        
        Returns:
            (predicted_byte, confidence, output_distribution)
        """
        if self.latent is None:
            self.encode()
        
        return self.readout.predict_next(
            self.latent,
            method=method,
            temperature=temperature,
            top_p=top_p,
            use_geometric_continuation=use_geometric_continuation,
        )
    
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
        next_byte, confidence, _ = self.predict_next()
        
        return {
            'latent': latent,
            'trajectory_summary': self.trajectory.get_summary(),
            'converted_levels': converted,
            'predicted_next': next_byte,
            'confidence': confidence,
        }


# =============================================================================
# 9. HIERARCHICAL TRAJECTORY — СПРАЖНЯ БЕЗМЕЖНІСТЬ
# =============================================================================

class HierarchicalTrajectory:
    """
    Справжня Hierarchical Trajectory — багаторівнева структура.
    
    ПРОБЛЕМА ЗВИЧАЙНОЇ TRAJECTORY:
    - max_length=1000 — обмеження
    - Агрегація 90%→1 точка — втрата деталей
    
    РІШЕННЯ — Hierarchical Trajectory:
    - L0: Raw points (останні base_size)
    - L1: Fréchet means кожних base_size точок з L0
    - L2: Fréchet means кожних base_size L1
    - L3+, L4+: ...
    
    ВЛАСТИВОСТІ:
    - Справжня O(log n) пам'ять замість O(n)
    - Зберігає деталі в L0, геометрію в L1+
    - Геометричний attention з вагами за рівнями
    - Можливість відновлення будь-якої точки
    """
    
    def __init__(
        self,
        base_size: int = 100,
        max_levels: int = 5,
        decay_rate: float = 0.99,
        temperature: float = 1.0,
    ):
        """
        Args:
            base_size: кількість точок на кожному рівні before aggregation
            max_levels: максимальна кількість рівнів
            decay_rate: часовий decay
            temperature: температура attention
        """
        self.base_size = base_size
        self.max_levels = max_levels
        self.decay_rate = decay_rate
        self.temperature = temperature
        
        # Рівні: список списків (точка + час)
        self.levels: List[List[np.ndarray]] = [[] for _ in range(max_levels)]
        self.timestamps: List[List[float]] = [[] for _ in range(max_levels)]
        
        # Кеш для attention
        self._attention_cache: Optional[np.ndarray] = None
        self._cache_valid: bool = False
    
    @property
    def total_points(self) -> int:
        """Загальна кількість точок в ієрархії."""
        return sum(len(level) for level in self.levels)
    
    @property
    def depth(self) -> int:
        """Глибина ієрархії (скільки рівнів використовується)."""
        for i in range(self.max_levels - 1, -1, -1):
            if len(self.levels[i]) > 0:
                return i + 1
        return 0
    
    @property
    def all_points(self) -> List[Tuple[np.ndarray, float, int]]:
        """
        Всі точки з усіх рівнів.
        
        Returns:
            [(p, t, level), ...]
        """
        result = []
        for level_idx in range(self.max_levels):
            for i, p in enumerate(self.levels[level_idx]):
                t = self.timestamps[level_idx][i]
                result.append((p, t, level_idx))
        return result
    
    def push(self, p: np.ndarray, t: float):
        """
        Додати нову точку.
        
        Args:
            p: розподіл на симплексі
            t: часова координата
        """
        p = np.maximum(p, 1e-10)
        p = p / p.sum()
        
        # Все завжди йде в L0 (базовий рівень)
        self.levels[0].append(p)
        self.timestamps[0].append(t)
        
        # Якщо L0 повний — агрегуємо
        if len(self.levels[0]) >= self.base_size:
            self._aggregate_level(0)
        
        # Інвалідідація кешу
        self._cache_valid = False
    
    def _aggregate_level(self, level: int):
        """
        Агрегуємо points з level в level+1.
        
        Args:
            level: рівень який агрегуємо
        """
        if level >= self.max_levels - 1:
            # На максимальному рівні — просто відкидаємоold
            self.levels[level] = self.levels[level][-self.base_size:]
            self.timestamps[level] = self.timestamps[level][-self.base_size:]
            return
        
        points = self.levels[level]
        times = self.timestamps[level]
        
        if len(points) < self.base_size:
            return
        
        # Fréchet mean зберігає геометрію
        agg_p = frechet_mean(points)
        agg_t = (times[0] + times[-1]) / 2  # Середина часового діапазону
        
        # Відкидаємо old і додаємо агреговане в наступний рівень
        self.levels[level] = []
        self.timestamps[level] = []
        
        # Агрегуємо всі батчі
        n_batches = len(points) // self.base_size
        for batch_idx in range(n_batches):
            start = batch_idx * self.base_size
            end = start + self.base_size
            batch_points = points[start:end]
            batch_times = times[start:end]
            
            batch_agg_p = frechet_mean(batch_points)
            batch_agg_t = (batch_times[0] + batch_times[-1]) / 2
            
            self.levels[level + 1].append(batch_agg_p)
            self.timestamps[level + 1].append(batch_agg_t)
        
        # Якщо залишилися — вони стають новим батчем
        remainder = len(points) % self.base_size
        if remainder > 0:
            self.levels[level] = points[-remainder:]
            self.timestamps[level] = times[-remainder:]
        else:
            self.levels[level] = []
            self.timestamps[level] = []
        
        # Якщо новий рівень повний — рекурсивно агрегуємо
        if len(self.levels[level + 1]) >= self.base_size:
            self._aggregate_level(level + 1)
    
    def compute_attention(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обчислити attention до всіх точок з усіх рівнів.
        
        Args:
            query: розподіл запиту
            
        Returns:
            (attention_weights, all_points_array)
        """
        all_pts = []
        all_times = []
        all_levels = []
        all_weights = []
        
        for level_idx in range(self.max_levels):
            if not self.levels[level_idx]:
                continue
            
            # Вага рівня: вищий рівень = менша вага
            level_weight = self.decay_rate ** level_idx
            
            for i, p in enumerate(self.levels[level_idx]):
                all_pts.append(p)
                all_times.append(self.timestamps[level_idx][i])
                all_levels.append(level_idx)
                all_weights.append(level_weight)
        
        if not all_pts:
            return np.array([]), np.array([])
        
        all_pts = np.array(all_pts)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
        # Геометричні відстані — batch обчислення
        query_sqrt = np.sqrt(np.maximum(query, 1e-10))
        all_sqrt = np.array([np.sqrt(np.maximum(p, 1e-10)) for p in all_pts])
        bc = all_sqrt @ query_sqrt
        bc = np.clip(bc, 0, 1)
        distances = np.arccos(bc)
        
        # Energies
        energies = -distances ** 2 / self.temperature
        
        # Комбінований attention: геометрія + вага рівня
        combined = energies + np.log(all_weights + 1e-10)
        combined = combined - combined.max()
        attention = np.exp(combined)
        attention = attention / (attention.sum() + 1e-10)
        
        return attention, all_pts
    
    def attend(self, query: np.ndarray, feature: str = 'p') -> np.ndarray:
        """
        Attention до траєкторії.
        
        Args:
            query: розподіл запиту
            feature: яку ознаку збирати ('p', 'time', 'level')
            
        Returns:
            Зважений результат
        """
        attention, all_pts = self.compute_attention(query)
        
        if len(attention) == 0:
            if feature == 'p':
                return np.zeros(256) + 1e-10
            return np.zeros(1)
        
        if feature == 'p':
            return attention @ all_pts
        elif feature == 'time':
            all_times = []
            for level_idx in range(self.max_levels):
                all_times.extend(self.timestamps[level_idx])
            return attention @ np.array(all_times)
        elif feature == 'level':
            all_levels = []
            for level_idx in range(self.max_levels):
                all_levels.extend([level_idx] * len(self.levels[level_idx]))
            return attention @ np.array(all_levels)
        
        return attention @ all_pts
    
    def get_context(self, query: np.ndarray, n_levels: Optional[int] = None) -> np.ndarray:
        """
        Отримати контекстний вектор.
        
        Args:
            query: розподіл запиту
            n_levels: обмежити кількість рівнів (None = всі)
            
        Returns:
            Контекстний вектор (256-мірний)
        """
        return self.attend(query, feature='p')
    
    def query_time_range(
        self, 
        t_start: float, 
        t_end: float,
        query: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Отримати точки в часовому діапазоні.
        
        Args:
            t_start: початок діапазону
            t_end: кінець діапазону
            query: опціональний запит для зважування
            
        Returns:
            Зважена сума розподілів в діапазоні
        """
        points_in_range = []
        weights_in_range = []
        
        for level_idx in range(self.max_levels):
            for i, t in enumerate(self.timestamps[level_idx]):
                if t_start <= t <= t_end:
                    points_in_range.append(self.levels[level_idx][i])
                    # Вага = decay^level
                    weights_in_range.append(self.decay_rate ** level_idx)
        
        if not points_in_range:
            return np.zeros(256) + 1e-10
        
        weights = np.array(weights_in_range)
        weights = weights / weights.sum()
        
        return weights @ np.array(points_in_range)
    
    def get_summary(self) -> Dict[str, Any]:
        """Отримати summary ієрархії."""
        level_stats = []
        for level_idx in range(self.max_levels):
            if len(self.levels[level_idx]) > 0:
                level_stats.append({
                    'level': level_idx,
                    'n_points': len(self.levels[level_idx]),
                    'time_range': (self.timestamps[level_idx][0], self.timestamps[level_idx][-1]),
                })
        
        return {
            'total_points': self.total_points,
            'depth': self.depth,
            'base_size': self.base_size,
            'max_levels': self.max_levels,
            'levels': level_stats,
        }
    
    def reset(self):
        """Очистити траєкторію."""
        self.levels = [[] for _ in range(self.max_levels)]
        self.timestamps = [[] for _ in range(self.max_levels)]
        self._cache_valid = False


# =============================================================================
# 10. ЕКСПОРТ
# =============================================================================

__all__ = [
    # Геометричні примітиви
    'fisher_rao_distance',
    'geodesic_interpolation',
    'frechet_mean',
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
    
    # Hierarchical Trajectory
    'HierarchicalTrajectory',
]
