"""
BCS Manifold Trajectory Module - v3.0

РЕВОЛЮЦІЙНА СИСТЕМА: Контекст як Траєкторія на Многовиді

Замість вікна - траєкторія p(0) → p(1) → ... → p(t) через симплекс S²⁵⁵

Ключові компоненти:
1. ManifoldTrajectory - зберігає траєкторію точок на многовиді
2. GeodesicAttention - attention через геодезичні відстані
3. CurvatureNoveltyDetector - виявляє новизну через кривину многовиду
4. MemoryAsSubmanifold - пам'ять як підмноговид

Це не покращення window. Це нова парадигма: ШІ з геометричною пам'яттю.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from scipy.special import softmax
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import warnings

# =============================================================================
# 1. ІНФОРМАЦІЙНА ГЕОМЕТРІЯ ПРИМІТИВИ
# =============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Fisher-Rao distance: geodesic distance on statistical manifold.
    
    d_FR(p, q) = arccos(Σ√p_i·√q_i)
    
    Це природна відстань на многовиді розподілів.
    """
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)
    
    if bc >= 1.0 - epsilon:
        return 0.0
    return np.arccos(bc)


def fisher_metric_tensor(p: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Fisher Information Matrix для categorical distribution.
    
    g_ij(p) = 1/p_i if i==j, else 0
    
    Це Riemanniana метрика на симплексі.
    """
    p = np.maximum(p, epsilon)
    p = p / p.sum()
    return np.diag(1.0 / p)


def geodesicInterpolation(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
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
    """
    if not points:
        raise ValueError("Empty points list")
    
    if len(points) == 1:
        p = np.maximum(points[0], epsilon)
        return p / p.sum()
    
    first = points[0]
    all_same = all(np.allclose(first, p, atol=1e-8) for p in points[1:])
    if all_same:
        p = np.maximum(first, epsilon)
        return p / p.sum()
    
    m = np.mean(points, axis=0)
    m = np.maximum(m, epsilon)
    m = m / m.sum()
    
    prev_m = m.copy()
    
    for iteration in range(max_iter):
        m_sqrt = np.sqrt(m)
        gradients = []
        
        for p in points:
            p = np.maximum(p, epsilon)
            p = p / p.sum()
            sqrt_p = np.sqrt(p)
            
            bc = np.dot(m_sqrt, sqrt_p)
            bc_clamped = np.clip(bc, 0.0, 1.0 - epsilon)
            
            if bc_clamped > 1.0 - epsilon:
                continue
            
            theta = np.arccos(bc_clamped)
            diff = sqrt_p / bc_clamped - m_sqrt
            norm_diff = np.linalg.norm(diff)
            
            if norm_diff < epsilon:
                continue
            
            direction = diff / norm_diff
            gradients.append(theta * direction)
        
        if not gradients:
            break
        
        grad_mean = np.mean(gradients, axis=0)
        grad_norm = np.linalg.norm(grad_mean)
        
        if grad_norm < tol:
            break
        
        step = lr * grad_mean
        step_norm = np.linalg.norm(step)
        
        if step_norm > 0:
            m = np.cos(step_norm) * m_sqrt + np.sin(step_norm) * step / step_norm
            m = np.maximum(m, epsilon) ** 2
            m = m / m.sum()
        
        if np.linalg.norm(m - prev_m) < tol:
            break
        
        prev_m = m.copy()
    
    return m


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL(p || q) divergence."""
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def wasserstein_distance_1d(p: np.ndarray, q: np.ndarray) -> float:
    """
    1-D Wasserstein distance (Earth Mover's Distance).
    """
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.sum(np.abs(cdf_p - cdf_q))


# =============================================================================
# 2. ТОЧКА НА МНОГОВИДІ
# =============================================================================

@dataclass
class ManifoldPoint:
    """
    Точка на статистичному многовиді S²⁵⁵.
    
    Атрибути:
    - p: probability distribution (розподіл байтів)
    - t: час/позиція в траєкторії
    - metadata: додаткові дані (семантика, модальність, etc.)
    """
    p: np.ndarray  # Розподіл на S²⁵⁵
    t: float  # Часова координата
    entropy: float = 0.0
    modality: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Нормалізація
        self.p = np.maximum(self.p, 1e-10)
        self.p = self.p / self.p.sum()
        
        # Обчислення ентропії
        self.entropy = -np.sum(self.p * np.log(self.p + 1e-10))
    
    def distance_to(self, other: 'ManifoldPoint', method: str = 'fisher_rao') -> float:
        """
        Відстань до іншої точки на многовиді.
        """
        if method == 'fisher_rao':
            return fisher_rao_distance(self.p, other.p)
        elif method == 'kl':
            return kl_divergence(self.p, other.p)
        elif method == 'wasserstein':
            return wasserstein_distance_1d(self.p, other.p)
        else:
            return fisher_rao_distance(self.p, other.p)
    
    def interpolate_to(self, other: 'ManifoldPoint', t: float) -> 'ManifoldPoint':
        """
        Геодезична інтерполяція до іншої точки.
        """
        p_interp = geodesicInterpolation(self.p, other.p, t)
        return ManifoldPoint(
            p=p_interp,
            t=self.t + t * (other.t - self.t),
            entropy=-np.sum(p_interp * np.log(p_interp + 1e-10)),
            metadata={'interpolated': True}
        )


# =============================================================================
# 3. ЯДРО СИСТЕМИ: ТРАЄКТОРІЯ НА МНОГОВИДІ
# =============================================================================

class ManifoldTrajectory:
    """
    Траєкторія на статистичному многовиді.
    
    Контекст = траєкторія через многовид, не вікно.
    
    Форма траєкторії визначає семантику:
    - Петля = повторення
    - Кут = різка зміна теми
    - Пряма = плавний перехід
    - Зупинка = стабільний контекст
    
    Властивості:
    - Необмежена довжина (пам'ять як геометрія)
    - Геометричний attention (не softmax)
    - Автоматичне виявлення новизни (кривина)
    """
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        decay_rate: float = 0.95,
        novelty_threshold: float = 0.5,
        enable_curvature: bool = True,
        enable_memory: bool = True,
    ):
        """
        Args:
            max_length: Максимальна довжина траєкторії (None = без обмежень)
            decay_rate: Знецінювання старих точок (для attention)
            novelty_threshold: Поріг для виявлення новизни
            enable_curvature: Увімкнути виявлення кривини
            enable_memory: Увімкнути пам'ять як підмноговид
        """
        self.points: List[ManifoldPoint] = []
        self.max_length = max_length
        self.decay_rate = decay_rate
        self.novelty_threshold = novelty_threshold
        self.enable_curvature = enable_curvature
        self.enable_memory = enable_memory
        
        # Геометричні характеристики траєкторії
        self.total_length: float = 0.0
        self.curvature_profile: List[float] = []
        self.velocity_profile: List[float] = []  # Швидкість зміни розподілу
        self.acceleration_profile: List[float] = []  # Прискорення
        
        # Пам'ять як підмноговид
        self.memory_center: Optional[np.ndarray] = None
        self.memory_spread: float = 0.0
        
        # Кеш для attention
        self._attention_cache: Optional[np.ndarray] = None
        self._attention_valid: bool = False
    
    # =========================================================================
    # ОСНОВНІ ОПЕРАЦІЇ
    # =========================================================================
    
    def push(self, p: np.ndarray, t: Optional[float] = None, metadata: Optional[Dict] = None) -> ManifoldPoint:
        """
        Додати нову точку до траєкторії.
        
        Автоматично обчислює геометричні характеристики.
        """
        if t is None:
            t = len(self.points)
        
        # Створення нової точки
        point = ManifoldPoint(
            p=p,
            t=t,
            metadata=metadata or {}
        )
        
        # Додавання до траєкторії
        self.points.append(point)
        
        # Оновлення геометрії
        self._update_geometry()
        
        # Оновлення пам'яті
        if self.enable_memory:
            self._update_memory()
        
        # Необмежена пам'ять: якщо max_length, агрегуємо
        if self.max_length is not None and len(self.points) > self.max_length:
            self._aggregate_old_points()
        
        # Інвалідідація кешу attention
        self._attention_valid = False
        
        return point
    
    def _update_geometry(self):
        """
        Оновлення геометричних характеристик траєкторії.
        """
        if len(self.points) < 2:
            return
        
        # Сумарна довжина (сума геодезичних відстаней)
        self.total_length = 0.0
        self.velocity_profile = []
        
        for i in range(1, len(self.points)):
            d = self.points[i].distance_to(self.points[i-1])
            self.total_length += d
            
            # Швидкість = відстань / час
            dt = max(self.points[i].t - self.points[i-1].t, 1e-10)
            v = d / dt
            self.velocity_profile.append(v)
        
        # Кривина: зміна напрямку (кут між послідовними відрізками)
        self.curvature_profile = []
        for i in range(1, len(self.points) - 1):
            if i == 0:
                continue
            
            p1 = self.points[i-1].p
            p2 = self.points[i].p
            p3 = self.points[i+1].p
            
            # Кут між (p2-p1) і (p3-p2) в термінах косинусної відстані
            bc1 = np.sum(np.sqrt(p1 * p2))
            bc2 = np.sum(np.sqrt(p2 * p3))
            
            # Кривина = зміна напрямку
            curvature = 1.0 - bc1 * bc2  # ~0 = прямий, ~1 = різкий поворот
            self.curvature_profile.append(max(0, curvature))
        
        # Прискорення: друга похідна швидкості
        self.acceleration_profile = []
        if len(self.velocity_profile) >= 2:
            for i in range(1, len(self.velocity_profile)):
                a = self.velocity_profile[i] - self.velocity_profile[i-1]
                self.acceleration_profile.append(abs(a))
    
    def _update_memory(self):
        """
        Оновлення пам'яті як центроїда підмноговиду.
        """
        if len(self.points) == 0:
            return
        
        # Центроїд: зважена сума точок
        if len(self.points) <= 10:
            # Для маленької траєкторії - просте середнє
            self.memory_center = np.mean([p.p for p in self.points], axis=0)
        else:
            # Для великої - експоненційне затухання
            weights = np.array([self.decay_rate ** (len(self.points) - 1 - i) 
                              for i in range(len(self.points))])
            weights = weights / weights.sum()
            self.memory_center = sum(w * p.p for w, p in zip(weights, self.points))
        
        # Розкид: середня відстань до центроїда — batch обчислення
        if len(self.points) > 1:
            center_sqrt = np.sqrt(np.maximum(self.memory_center, 1e-10))
            all_sqrt = np.array([np.sqrt(np.maximum(p.p, 1e-10)) for p in self.points])
            bc = all_sqrt @ center_sqrt
            bc = np.clip(bc, 0, 1)
            distances = np.arccos(bc)
            self.memory_spread = np.mean(distances)
        else:
            self.memory_spread = 0.0
    
    def _aggregate_old_points(self):
        """
        Агрегація старих точок в підмноговид для економії пам'яті.
        
        ВИПРАВЛЕНО: Fréchet mean замість arithmetic mean.
        """
        n_keep = self.max_length // 10  # Залишаємо 10%
        
        if n_keep < 2:
            return
        
        n_agg = len(self.points) - n_keep
        if n_agg < 2:
            return
        
        # Fréchet mean замість arithmetic mean
        points_to_agg = [p.p for p in self.points[:n_agg]]
        agg_p = frechet_mean(points_to_agg)
        
        # Зберігаємо останні точки
        self.points = self.points[n_agg:]
        
        # Вставляємо агреговану точку на початок
        self.points.insert(0, ManifoldPoint(
            p=agg_p,
            t=0,
            metadata={'aggregated': True, 'n_points': n_agg}
        ))
    
    # =========================================================================
    # GEODESIC ATTENTION (увага через геометрію)
    # =========================================================================
    
    def compute_attention(
        self, 
        query: np.ndarray, 
        temperature: float = 1.0,
        decay: bool = True
    ) -> np.ndarray:
        """
        Geodesic Attention: attention до всіх точок траєкторії.
        
        Увага пропорційна exp(-geodesic_distance² / T), не dot-product.
        
        Args:
            query: розподіл для attention (поточна точка)
            temperature: температура softmax
            decay: чи застосовувати часовий decay
            
        Returns:
            attention weights (normalized)
        """
        if len(self.points) == 0:
            return np.array([])
        
        # Обчислення геодезичних відстаней
        distances = []
        for point in self.points:
            d = fisher_rao_distance(query, point.p)
            distances.append(d)
        
        distances = np.array(distances)
        
        # Геометричний attention: exp(-d²/T)
        # Замість softmax на dot-product - відстань на многовиді
        energies = -distances ** 2 / temperature
        energies = energies - energies.max()  # Numerical stability
        
        # Часовий decay
        if decay:
            time_decay = np.array([self.decay_rate ** (len(self.points) - 1 - i) 
                                  for i in range(len(self.points))])
            energies = energies + np.log(time_decay + 1e-10)
        
        # Softmax
        exp_energies = np.exp(energies - energies.max())
        attention = exp_energies / (exp_energies.sum() + 1e-10)
        
        return attention
    
    def attend(
        self, 
        query: np.ndarray, 
        feature: str = 'p',
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Геометричний attention з поверненням зваженого результату.
        """
        attention = self.compute_attention(query, temperature)
        
        if len(attention) == 0:
            return np.zeros(256)
        
        # Збирання відповідної ознаки
        if feature == 'p':
            features = np.array([p.p for p in self.points])
        elif feature == 'entropy':
            features = np.array([[p.entropy] for p in self.points])
        else:
            features = np.array([p.p for p in self.points])
        
        # Зважена сума
        result = attention @ features
        
        return result
    
    # =========================================================================
    # ВИЯВЛЕННЯ НОВИЗНИ ЧЕРЕЗ КРИВИНУ
    # =========================================================================
    
    def detect_novelty(self, p: np.ndarray) -> Tuple[float, float]:
        """
        Виявлення новизни нової точки.
        
        Повертає:
        - novelty_score: наскільки точка "новина" для траєкторії
        - confidence: впевненість у оцінці
        
        Використовує:
        1. Відстань до траєкторії
        2. Кривина в поточній точці
        3. Швидкість зміни
        """
        if len(self.points) == 0:
            return 1.0, 1.0  # Перша точка - максимальна новизна
        
        # 1. Відстань до центроїда траєкторії
        if self.memory_center is not None:
            dist_to_center = fisher_rao_distance(p, self.memory_center)
            dist_score = min(1.0, dist_to_center / (self.memory_spread + 0.1))
        else:
            dist_to_center = fisher_rao_distance(p, self.points[-1].p)
            dist_score = min(1.0, dist_to_center / 1.0)
        
        # 2. Кривина (якщо є достатньо точок)
        if len(self.curvature_profile) > 0 and self.enable_curvature:
            recent_curvature = np.mean(self.curvature_profile[-3:])
            curvature_score = min(1.0, recent_curvature * 5)
        else:
            curvature_score = 0.0
        
        # 3. Швидкість зміни
        if len(self.velocity_profile) > 0:
            recent_velocity = np.mean(self.velocity_profile[-3:])
            velocity_score = min(1.0, recent_velocity * 10)
        else:
            velocity_score = 0.0
        
        # Комбінована оцінка новизни
        novelty = 0.4 * dist_score + 0.3 * curvature_score + 0.3 * velocity_score
        novelty = min(1.0, novelty)
        
        # Впевненість: чим більше історії, тим вища впевненість
        confidence = min(1.0, len(self.points) / 20.0)
        
        return novelty, confidence
    
    def get_context_vector(self, query: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Отримати "контекстний вектор" - геометричне представлення траєкторії.
        
        Якщо query=None, повертає позицію останньої точки.
        Якщо query є, повертає результат геометричного attention.
        """
        if len(self.points) == 0:
            return np.zeros(256) + 1e-10
        
        if query is None:
            # Просто остання точка
            return self.points[-1].p.copy()
        else:
            # Attention до траєкторії
            return self.attend(query, feature='p')
    
    # =========================================================================
    # ТОПОЛОГІЧНІ ХАРАКТЕРИСТИКИ
    # =========================================================================
    
    def compute_topology_features(self) -> Dict[str, float]:
        """
        Обчислення топологічних характеристик траєкторії.
        
        Повертає Betti-like числа та інші топологічні ознаки.
        """
        if len(self.points) < 3:
            return {
                'betti_0': 1.0,  # Зв'язність
                'betti_1': 0.0,  # Петлі
                'total_variation': 0.0,
                'rectifiable': 1.0,
                'oscillation_count': 0,
            }
        
        # Betti_0: кількість компонент ( завжди 1 для траєкторії)
        betti_0 = 1.0
        
        # Betti_1: кількість петель (різкі повороти назад)
        loops = 0
        for i in range(1, len(self.points) - 1):
            # Перевірка чи повертаємо назад
            d_back = self.points[i].distance_to(self.points[i-1])
            d_forward = self.points[i].distance_to(self.points[i+1])
            if d_back < d_forward * 0.5:  # Напрямок змінився
                loops += 1
        betti_1 = float(loops)
        
        # Total variation
        tv = self.total_length
        
        # Rectifiable: чи можна апроксимувати плавною кривою
        rectifiable = 1.0 - min(1.0, np.std(self.curvature_profile) * 2) if self.curvature_profile else 1.0
        
        # Oscillation count: кількість осциляцій
        if len(self.velocity_profile) > 2:
            velocity = np.array(self.velocity_profile)
            peaks, _ = find_peaks(velocity)
            oscillation_count = len(peaks)
        else:
            oscillation_count = 0
        
        return {
            'betti_0': betti_0,
            'betti_1': betti_1,
            'total_variation': tv,
            'rectifiable': rectifiable,
            'oscillation_count': oscillation_count,
        }
    
    # =========================================================================
    # ЕКСПОРТ ТА ВІЗУАЛІЗАЦІЯ
    # =========================================================================
    
    def get_trajectory_summary(self) -> Dict:
        """
        Повертає summary траєкторії для аналізу.
        """
        return {
            'n_points': len(self.points),
            'total_length': self.total_length,
            'current_entropy': self.points[-1].entropy if self.points else 0,
            'current_modality': self.points[-1].modality if self.points else 'unknown',
            'memory_spread': self.memory_spread,
            'topology': self.compute_topology_features(),
            'curvature_stats': {
                'mean': np.mean(self.curvature_profile) if self.curvature_profile else 0,
                'max': np.max(self.curvature_profile) if self.curvature_profile else 0,
            },
            'velocity_stats': {
                'mean': np.mean(self.velocity_profile) if self.velocity_profile else 0,
                'max': np.max(self.velocity_profile) if self.velocity_profile else 0,
            },
        }
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __repr__(self) -> str:
        return f"ManifoldTrajectory(n={len(self.points)}, length={self.total_length:.3f})"


# =============================================================================
# 4. ГЕОДЕЗИЧНИЙ ATTENTION МЕХАНІЗМ (Standalone)
# =============================================================================

class GeodesicAttention:
    """
    Geodesic Attention Mechanism.
    
    Замість softmax на dot-product, використовує геодезичні відстані
    на статистичному многовиді.
    
    Переваги:
    - Геометрично коректний attention
    - Автоматично враховує "відстань у сенсі"
    - Не потребує learnable проекцій
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        use_decay: bool = True,
        decay_rate: float = 0.95,
        metric: str = 'fisher_rao',
    ):
        self.temperature = temperature
        self.use_decay = use_decay
        self.decay_rate = decay_rate
        self.metric = metric
    
    def forward(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Geodesic attention forward pass.
        
        Args:
            query: розподіл (поточна точка)
            keys: список розподілів (ключі)
            values: список розподілів (значення), якщо None - keys
            
        Returns:
            (weighted_output, attention_weights)
        """
        if values is None:
            values = keys
        
        n = len(keys)
        if n == 0:
            return np.zeros(256), np.array([])
        
        # Обчислення геодезичних відстаней
        distances = []
        for key in keys:
            if self.metric == 'fisher_rao':
                d = fisher_rao_distance(query, key)
            elif self.metric == 'kl':
                d = kl_divergence(query, key)
            elif self.metric == 'wasserstein':
                d = wasserstein_distance_1d(query, key)
            else:
                d = fisher_rao_distance(query, key)
            distances.append(d)
        
        distances = np.array(distances)
        
        # Енергії: exp(-d²/T)
        energies = -distances ** 2 / self.temperature
        energies = energies - energies.max()
        
        # Часовий decay
        if self.use_decay:
            time_decay = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
            energies = energies + np.log(time_decay + 1e-10)
        
        # Softmax
        exp_energies = np.exp(energies - energies.max())
        attention = exp_energies / (exp_energies.sum() + 1e-10)
        
        # Зважена сума
        values_arr = np.array(values)
        output = attention @ values_arr
        
        return output, attention


# =============================================================================
# 5. ВИЯВЛЕННЯ НОВИЗНИ ЧЕРЕЗ КРИВИНУ (Standalone)
# =============================================================================

class CurvatureNoveltyDetector:
    """
    Curvature-based Novelty Detection on Statistical Manifold.
    
    Ідея: новизна = висока кривина многовиду в поточній точці.
    
    Коли траєкторія "різко повертає" - це означає:
    - Нова тема
    - Зміна контексту
    - Important event
    """
    
    def __init__(
        self,
        window_size: int = 5,
        curvature_threshold: float = 0.3,
        velocity_threshold: float = 0.1,
        enable_acceleration: bool = True,
    ):
        self.window_size = window_size
        self.curvature_threshold = curvature_threshold
        self.velocity_threshold = velocity_threshold
        self.enable_acceleration = enable_acceleration
        
        self.history: List[np.ndarray] = []
    
    def update(self, p: np.ndarray) -> Tuple[float, str]:
        """
        Оновлення детектора нової точкою.
        
        Returns:
            (novelty_score, novelty_type)
            
        novelty_type:
            - 'novel': нова точка
            - 'familiar': відома точка
            - 'transition': перехід
            - 'stable': стабільний стан
        """
        self.history.append(p)
        
        if len(self.history) < 3:
            return 1.0, 'novel'
        
        # Обмежити історію
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]
        
        # Обчислення кривини
        curvatures = []
        for i in range(2, len(self.history)):
            p1 = self.history[i-2]
            p2 = self.history[i-1]
            p3 = self.history[i]
            
            # Кривина = 1 - косинус кута
            bc1 = np.sum(np.sqrt(p1 * p2))
            bc2 = np.sum(np.sqrt(p2 * p3))
            c = 1.0 - bc1 * bc2
            curvatures.append(max(0, c))
        
        current_curvature = curvatures[-1] if curvatures else 0
        
        # Обчислення швидкості
        if len(self.history) >= 2:
            velocity = fisher_rao_distance(self.history[-2], self.history[-1])
        else:
            velocity = 0
        
        # Обчислення прискорення
        if self.enable_acceleration and len(curvatures) >= 2:
            acceleration = abs(curvatures[-1] - curvatures[-2])
        else:
            acceleration = 0
        
        # Класифікація
        novelty_score = (
            0.4 * min(1.0, current_curvature * 5) +
            0.3 * min(1.0, velocity * 10) +
            0.3 * min(1.0, acceleration * 20)
        )
        
        if novelty_score > 0.7:
            novelty_type = 'novel'
        elif novelty_score > 0.4:
            novelty_type = 'transition'
        elif velocity < self.velocity_threshold:
            novelty_type = 'stable'
        else:
            novelty_type = 'familiar'
        
        return novelty_score, novelty_type
    
    def reset(self):
        """Очистити історію."""
        self.history = []


# =============================================================================
# 6. ПАМ'ЯТЬ ЯК ПІДМНОГОВИД (Standalone)
# =============================================================================

class MemoryAsSubmanifold:
    """
    Пам'ять як підмноговид статистичного многовиду.
    
    Замість буфера - підмноговид який представляє "простір відомого".
    
    Операції:
    - Store: додати точку до підмноговиду
    - Retrieve: знайти найближчу точку
    - Interpolate: інтерполяція між точками пам'яті
    - Project: спроеціювати нову точку на підмноговид
    """
    
    def __init__(
        self,
        max_points: int = 1000,
        compression_threshold: float = 0.1,
        retrieval_threshold: float = 0.2,
    ):
        self.max_points = max_points
        self.compression_threshold = compression_threshold
        self.retrieval_threshold = retrieval_threshold
        
        # Точки пам'яті
        self.points: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        
        # Статистика
        self.centroid: Optional[np.ndarray] = None
        self.span: float = 0.0  # "Діаметр" пам'яті
    
    def store(self, p: np.ndarray, metadata: Optional[Dict] = None):
        """
        Зберегти точку в пам'ять.
        """
        p = np.maximum(p, 1e-10)
        p = p / p.sum()
        
        self.points.append(p)
        self.metadata.append(metadata or {})
        
        # Оновлення статистики
        self._update_stats()
        
        # Компресія якщо занадто багато
        if len(self.points) > self.max_points:
            self._compress()
    
    def _update_stats(self):
        """Оновлення статистики пам'яті."""
        if not self.points:
            return
        
        # Центроїд
        self.centroid = np.mean(self.points, axis=0)
        self.centroid = self.centroid / self.centroid.sum()
        
        # Span — batch обчислення
        if len(self.points) > 1:
            centroid_sqrt = np.sqrt(np.maximum(self.centroid, 1e-10))
            all_sqrt = np.array([np.sqrt(np.maximum(p, 1e-10)) for p in self.points])
            bc = all_sqrt @ centroid_sqrt
            bc = np.clip(bc, 0, 1)
            distances = np.arccos(bc)
            self.span = np.max(distances)
        else:
            self.span = 0.0
    
    def _compress(self):
        """
        Компресія пам'яті через агрегацію близьких точок.
        """
        # Знайти пари близьких точок і об'єднати їх
        to_remove = set()
        
        for i in range(len(self.points)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self.points)):
                if j in to_remove:
                    continue
                d = fisher_rao_distance(self.points[i], self.points[j])
                if d < self.compression_threshold:
                    # Об'єднати в одну точку (середнє)
                    new_p = (self.points[i] + self.points[j]) / 2
                    new_p = new_p / new_p.sum()
                    self.points[i] = new_p
                    to_remove.add(j)
        
        # Видалити об'єднані
        self.points = [p for i, p in enumerate(self.points) if i not in to_remove]
        self.metadata = [m for i, m in enumerate(self.metadata) if i not in to_remove]
        
        self._update_stats()
    
    def retrieve(
        self, 
        query: np.ndarray, 
        k: int = 5,
        include_metadata: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Знайти k найближчих точок в пам'яті.
        """
        if len(self.points) == 0:
            return [], []
        
        # Batch обчислення: O(n) замість O(n*d)
        query_sqrt = np.sqrt(np.maximum(query, 1e-10))
        all_sqrt = np.array([np.sqrt(np.maximum(p, 1e-10)) for p in self.points])
        bc = all_sqrt @ query_sqrt
        bc = np.clip(bc, 0, 1)
        distances = np.arccos(bc).tolist()
        
        # Топ-k
        k = min(k, len(distances))
        top_k_idx = np.argsort(distances)[:k]
        
        retrieved_points = [self.points[i] for i in top_k_idx]
        retrieved_distances = [distances[i] for i in top_k_idx]
        
        if include_metadata:
            retrieved_metadata = [self.metadata[i] for i in top_k_idx]
            return retrieved_points, retrieved_distances, retrieved_metadata
        
        return retrieved_points, retrieved_distances
    
    def interpolate(
        self, 
        p1_idx: int, 
        p2_idx: int, 
        t: float
    ) -> np.ndarray:
        """
        Геодезична інтерполяція між двома точками пам'яті.
        """
        p1 = self.points[p1_idx]
        p2 = self.points[p2_idx]
        return geodesicInterpolation(p1, p2, t)
    
    def project(self, p: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Проєкція точки на підмноговид (найближча точка).
        """
        if len(self.points) == 0:
            return p, 0.0
        
        distances = [fisher_rao_distance(p, mem_p) for mem_p in self.points]
        min_idx = np.argmin(distances)
        
        return self.points[min_idx], distances[min_idx]
    
    def familiarity(self, p: np.ndarray) -> float:
        """
        Оцінка "знайомості" точки (1 = відомо, 0 = невідомо).
        """
        if len(self.points) == 0:
            return 0.0
        
        distance, _ = self.project(p)
        d = fisher_rao_distance(p, distance)
        
        # Перетворення відстані в знайомість
        # Якщо d=0 -> familiarity=1
        # Якщо d>=span -> familiarity=0
        if self.span > 0:
            fam = max(0, 1.0 - d / (self.span + 1e-10))
        else:
            fam = 1.0 if d < 0.1 else 0.0
        
        return fam
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __repr__(self) -> str:
        return f"MemoryAsSubmanifold(n={len(self)}, span={self.span:.3f})"


# =============================================================================
# 7. ЗРУЧНІ ФУНКЦІЇ
# =============================================================================

def create_trajectory_from_bytes(
    data: bytes,
    step: int = 1,
    window_size: int = 16,
    **kwargs
) -> ManifoldTrajectory:
    """
    Створити траєкторію з байтових даних.
    
    Кожна точка = розподіл байтів у вікні.
    """
    trajectory = ManifoldTrajectory(**kwargs)
    
    n = len(data)
    for i in range(0, n - window_size + 1, step):
        window = data[i:i + window_size]
        
        # Розподіл байтів у вікні
        dist = np.zeros(256)
        for b in window:
            dist[b] += 1
        dist = dist / dist.sum()
        
        trajectory.push(dist, t=i / n)  # Нормалізований час
    
    return trajectory


def quick_attention(
    query: np.ndarray,
    memory: List[np.ndarray],
    temperature: float = 1.0
) -> np.ndarray:
    """
    Швидкий геодезичний attention.
    """
    ga = GeodesicAttention(temperature=temperature)
    result, _ = ga.forward(query, memory)
    return result


# =============================================================================
# 8. ЕКСПОРТ ДЛЯ СУМІСНОСТІ
# =============================================================================

__all__ = [
    'ManifoldTrajectory',
    'ManifoldPoint',
    'GeodesicAttention',
    'CurvatureNoveltyDetector',
    'MemoryAsSubmanifold',
    'fisher_rao_distance',
    'fisher_metric_tensor',
    'geodesicInterpolation',
    'create_trajectory_from_bytes',
    'quick_attention',
]
