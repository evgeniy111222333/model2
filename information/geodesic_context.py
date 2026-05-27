"""
BCS Geodesic Context Engine — Повна реалізація парадигми "Контекст = Траєкторія на Многовиді"

ЦЕЙ ФАЙЛ ЗАМІНЮЄ:
1. Window-based context → Trajectory-based context
2. Softmax attention → Geodesic attention (exp(-d²/T))
3. Buffer memory → Geometry memory (submanifold)
4. Token time → Trajectory shape time

ОСНОВНІ КЛАСИ:
1. GeodesicContextEngine — головний двигун контексту
2. TrajectoryAttention — геодезичний attention для кластерів
3. GeodesicQueryResponse — запит-відповідь через траєкторію

АРХІТЕКТУРА:
- Контекст = повна траєкторія p(0) → p(1) → ... → p(t)
- p(t) = розподіл байтів до моменту t (точка на S²⁵⁵)
- Attention = exp(-geodesic_distance² / T)
- Пам'ять = підмноговид траєкторії
- Час = форма траєкторії (кривина, швидкість)

"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy.special import softmax
from scipy.signal import find_peaks

# =============================================================================
# 1. ГЕОМЕТРИЧНІ ПРИМІТИВИ (скопійовані з manifold_trajectory.py для автономності)
# =============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Fisher-Rao geodesic distance on statistical manifold S²⁵⁵.
    
    d_FR(p, q) = arccos(Σ√p_i·√q_i) = arccos(Bhattacharyya(p,q))
    
    Це природна відстань на многовиді розподілів — замість dot-product.
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


def fisher_rao_distance_batch(query_sqrt: np.ndarray, targets: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Batch Fisher-Rao distance: O(n) замість O(n*d).
    
    query_sqrt: √p_query shape (256,)
    targets: √p_i для всіх точок shape (n, 256)
    
    Returns: відстані shape (n,)
    """
    # Bhattacharyya coefficients: Σ√p_query·√p_i для всіх i = dot product!
    bc = targets @ query_sqrt  # shape (n,)
    bc = np.clip(bc, 0, 1)
    return np.arccos(bc)


def geodesic_interpolation(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """
    ПРАВИЛЬНА геодезична інтерполяція на S²⁵⁵ з Fisher-Rao метрикою.
    
    log_map(p1→p2) = θ · (√p2/BC - √p1) / ‖√p2/BC - √p1‖
    де θ = arccos(BC), BC = <√p1, √p2> = Σ√p1_i·√p2_i
    
    exp_p1(t·v) = cos(t·θ)·√p1 + sin(t·θ)·v / ‖v‖
    
    Це справжня геодезична на сфері одиничних векторів √p.
    """
    eps = 1e-10
    p1 = np.maximum(p1, eps)
    p2 = np.maximum(p2, eps)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    sqrt_p1 = np.sqrt(p1)
    sqrt_p2 = np.sqrt(p2)
    
    # Bhattacharyya coefficient (скалярний добуток на сфері)
    bc = np.dot(sqrt_p1, sqrt_p2)
    bc = np.clip(bc, 0.0, 1.0 - eps)
    
    # Випадок майже ідентичних точок → евклідова інтерполяція
    if bc > 0.9999:
        interp_sqrt = (1 - t) * sqrt_p1 + t * sqrt_p2
    else:
        # Кут геодезичної
        theta = np.arccos(bc)
        
        # Напрямок log-map: v = (√p2/BC - √p1) / ‖√p2/BC - √p1‖
        diff = sqrt_p2 / bc - sqrt_p1
        norm_diff = np.linalg.norm(diff)
        
        if norm_diff < eps:
            interp_sqrt = sqrt_p1.copy()
        else:
            direction = diff / norm_diff
            # Exp-map: cos(t·θ)·p1 + sin(t·θ)·direction
            interp_sqrt = np.cos(t * theta) * sqrt_p1 + np.sin(t * theta) * direction
    
    interp_sqrt = np.maximum(interp_sqrt, eps)
    result = interp_sqrt ** 2
    return result / result.sum()


def frechet_mean(points: List[np.ndarray], max_iter: int = 50, lr: float = 0.5,
                 tol: float = 1e-7, epsilon: float = 1e-10) -> np.ndarray:
    """
    Fréchet mean на статистичному многовиді S^{n-1} з Fisher-Rao метрикою.
    
    Шукає точку m, яка мінімізує: Σ d_FR(m, p_i)²
    
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
    first = points[0]
    all_same = all(np.allclose(first, p, atol=1e-8) for p in points[1:])
    if all_same:
        p = np.maximum(first, epsilon)
        return p / p.sum()
    
    # Ініціалізація: arithmetic mean
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


# =============================================================================
# 2. ТОЧКА НА МНОГОВИДІ
# =============================================================================

@dataclass
class ManifoldPoint:
    """
    Точка на статистичному многовиді S²⁵⁵.
    
    Кожна точка = розподіл байтів у момент t.
    Це заміна токена в традиційній архітектурі.
    """
    p: np.ndarray  # Розподіл на S²⁵⁵
    t: float  # Часова координата
    position: int = 0  # Позиція в даних
    entropy: float = 0.0
    modality: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.p = np.maximum(self.p, 1e-10)
        self.p = self.p / self.p.sum()
        self.entropy = -np.sum(self.p * np.log(self.p + 1e-10))
    
    def distance_to(self, other: 'ManifoldPoint', method: str = 'fisher_rao') -> float:
        """Відстань до іншої точки на многовиді."""
        if method == 'fisher_rao':
            return fisher_rao_distance(self.p, other.p)
        elif method == 'kl':
            return kl_divergence(self.p, other.p)
        return fisher_rao_distance(self.p, other.p)


# =============================================================================
# 3. ЯДРО: ГЕОДЕЗИЧНИЙ КОНТЕКСТНИЙ ДВИГУН
# =============================================================================

class GeodesicContextEngine:
    """
    ГЕОДЕЗИЧНИЙ КОНТЕКСТНИЙ ДВИГУН — заміна window-based context.
    
    Парадигма: Контекст = Траєкторія на Многовиді
    
    Замість:
    - context = window [token₁, token₂, ..., token_n]  ❌
    
    Тепер:
    - context = trajectory p(0) → p(1) → ... → p(t)  ✅
    - p(t) = розподіл байтів до моменту t (точка на S²⁵⁵)
    
    ПЕРЕВАГИ:
    1. Безмежний контекст (немає "контекст закінчився")
    2. Attention = геометрія (exp(-geodesic_distance² / T))
    3. Пам'ять = геометрія (форма траєкторії)
    4. Час = форма (кривина = зміна теми, петля = повторення)
    
    КОМПОНЕНТИ:
    - ManifoldTrajectory: траєкторія як контекст
    - GeodesicAttention: attention через геометрію
    - MemorySubmanifold: пам'ять як підмноговид
    - CurvatureDetector: виявлення новизни через кривину
    """
    
    def __init__(
        self,
        max_trajectory_length: Optional[int] = None,
        decay_rate: float = 0.99,
        novelty_threshold: float = 0.5,
        temperature: float = 1.0,
        enable_curvature: bool = True,
        enable_memory: bool = True,
        enable_semantic: bool = True,
    ):
        """
        Args:
            max_trajectory_length: Макс. довжина траєкторії (None = без обмежень)
            decay_rate: Знецінювання старих точок
            novelty_threshold: Поріг для виявлення новизни
            temperature: Температура для attention softmax
            enable_curvature: Увімкнути виявлення кривини
            enable_memory: Увімкнути пам'ять як підмноговид
            enable_semantic: Увімкнути семантичну інтерпретацію
        """
        # Траєкторія — основний контекст
        self.points: List[ManifoldPoint] = []
        self.max_trajectory_length = max_trajectory_length
        
        # Параметри
        self.decay_rate = decay_rate
        self.novelty_threshold = novelty_threshold
        self.temperature = temperature
        self.enable_curvature = enable_curvature
        self.enable_memory = enable_memory
        self.enable_semantic = enable_semantic
        
        # Геометричні характеристики траєкторії
        self.total_geodesic_length: float = 0.0
        self.curvature_profile: List[float] = []
        self.velocity_profile: List[float] = []
        self.acceleration_profile: List[float] = []
        
        # Пам'ять як підмноговид
        self.memory_centroid: Optional[np.ndarray] = None
        self.memory_span: float = 0.0
        self.memory_points: List[np.ndarray] = []  # Точки пам'яті
        
        # Семантична інтерпретація
        self.semantic_shapes: Dict[str, List[int]] = {
            'loops': [],      # Петлі (повторення)
            'angles': [],     # Кути (різкі зміни теми)
            'stops': [],      # Зупинки (стабільний контекст)
            'streams': [],    # Потоки (плавні переходи)
        }
        
        # Кеш для attention
        self._attention_cache: Optional[np.ndarray] = None
        self._attention_valid: bool = False
        
        # Статистика
        self.total_bytes_processed: int = 0
    
    # =========================================================================
    # ОСНОВНІ ОПЕРАЦІЇ
    # =========================================================================
    
    def push(self, p: np.ndarray, t: Optional[float] = None, 
             position: int = 0, modality: str = "unknown",
             metadata: Optional[Dict] = None) -> ManifoldPoint:
        """
        Додати нову точку до траєкторії контексту.
        
        Це заміна append() до контекстного вікна.
        Кожна нова точка — новий момент часу на многовиді.
        
        Args:
            p: розподіл байтів (точка на S²⁵⁵)
            t: часова координата (якщо None — автоінкремент)
            position: позиція в даних
            modality: модальність даних
            metadata: додаткові дані
            
        Returns:
            Створена точка
        """
        if t is None:
            t = len(self.points)
        
        # Створення точки на многовиді
        point = ManifoldPoint(
            p=p,
            t=t,
            position=position,
            modality=modality,
            metadata=metadata or {}
        )
        
        # Додавання до траєкторії
        self.points.append(point)
        self.total_bytes_processed += position
        
        # Оновлення геометрії траєкторії
        self._update_geometry()
        
        # Оновлення пам'яті як підмноговиду
        if self.enable_memory:
            self._update_memory_submanifold()
        
        # Семантична інтерпретація форми
        if self.enable_semantic:
            self._update_semantic_shapes()
        
        # Необмежена пам'ять: агрегація старих точок
        if (self.max_trajectory_length is not None and 
            len(self.points) > self.max_trajectory_length):
            self._aggregate_trajectory()
        
        # Інвалідідація кешу attention
        self._attention_valid = False
        
        return point
    
    def _update_geometry(self):
        """Оновлення геометричних характеристик траєкторії."""
        if len(self.points) < 2:
            return
        
        # Сумарна геодезична довжина
        self.total_geodesic_length = 0.0
        self.velocity_profile = []
        
        for i in range(1, len(self.points)):
            d = self.points[i].distance_to(self.points[i-1])
            self.total_geodesic_length += d
            
            # Швидкість = відстань / час
            dt = max(self.points[i].t - self.points[i-1].t, 1e-10)
            v = d / dt
            self.velocity_profile.append(v)
        
        # Кривина: зміна напрямку
        self.curvature_profile = []
        for i in range(1, len(self.points) - 1):
            p1 = self.points[i-1].p
            p2 = self.points[i].p
            p3 = self.points[i+1].p
            
            # Кут між послідовними відрізками
            bc1 = np.sum(np.sqrt(p1 * p2))
            bc2 = np.sum(np.sqrt(p2 * p3))
            
            # Кривина = різкість повороту
            curvature = 1.0 - bc1 * bc2
            self.curvature_profile.append(max(0, curvature))
        
        # Прискорення
        self.acceleration_profile = []
        if len(self.velocity_profile) >= 2:
            for i in range(1, len(self.velocity_profile)):
                a = self.velocity_profile[i] - self.velocity_profile[i-1]
                self.acceleration_profile.append(abs(a))
    
    def _update_memory_submanifold(self):
        """
        Оновлення пам'яті як підмноговиду.
        
        Пам'ять = геометрична структура траєкторії,
        не буфер фіксованого розміру.
        """
        if len(self.points) == 0:
            return
        
        # Центроїд траєкторії
        if len(self.points) <= 10:
            self.memory_centroid = np.mean([p.p for p in self.points], axis=0)
        else:
            weights = np.array([
                self.decay_rate ** (len(self.points) - 1 - i) 
                for i in range(len(self.points))
            ])
            weights = weights / weights.sum()
            self.memory_centroid = sum(w * p.p for w, p in zip(weights, self.points))
        
        # Розкид пам'яті — batch обчислення
        if len(self.points) > 1:
            centroid_sqrt = np.sqrt(np.maximum(self.memory_centroid, 1e-10))
            all_sqrt = np.array([np.sqrt(np.maximum(p.p, 1e-10)) for p in self.points])
            bc = all_sqrt @ centroid_sqrt
            bc = np.clip(bc, 0, 1)
            distances = np.arccos(bc)
            self.memory_span = np.mean(distances)
        else:
            self.memory_span = 0.0
        
        # Зберігаємо точки пам'яті (для retrieval)
        self.memory_points = [p.p.copy() for p in self.points[-100:]]
    
    def _update_semantic_shapes(self):
        """
        Інтерпретація форми траєкторії як семантики.
        
        Форма = Семантика:
        - Петля = повторення (повернення назад)
        - Кут = різка зміна теми
        - Зупинка = стабільний контекст
        - Потік = плавний перехід
        """
        self.semantic_shapes = {
            'loops': [],
            'angles': [],
            'stops': [],
            'streams': [],
        }
        
        if len(self.curvature_profile) < 2:
            return
        
        # Знаходження петель (повернення назад)
        for i in range(1, len(self.points) - 1):
            d_back = self.points[i].distance_to(self.points[i-1])
            d_forward = self.points[i].distance_to(self.points[i+1])
            
            # Петля: повертаємо назад
            if d_back < d_forward * 0.5:
                self.semantic_shapes['loops'].append(i)
            
            # Кут: різка зміна напрямку
            elif self.curvature_profile[i-1] > 0.5:
                self.semantic_shapes['angles'].append(i)
        
        # Знаходження зупинок (низька швидкість)
        if self.velocity_profile:
            mean_vel = np.mean(self.velocity_profile)
            std_vel = np.std(self.velocity_profile)
            
            for i, v in enumerate(self.velocity_profile):
                if v < mean_vel - std_vel:
                    self.semantic_shapes['stops'].append(i)
        
        # Знаходження потоків (висока швидкість, низька кривина)
        if self.velocity_profile and self.curvature_profile:
            mean_curv = np.mean(self.curvature_profile)
            mean_vel = np.mean(self.velocity_profile)
            
            for i in range(min(len(self.velocity_profile), len(self.curvature_profile))):
                if (self.velocity_profile[i] > mean_vel and 
                    self.curvature_profile[i] < mean_curv):
                    self.semantic_shapes['streams'].append(i)
    
    def _aggregate_trajectory(self):
        """
        Агрегація старої частини траєкторії в підмноговид.
        
        ВИПРАВЛЕНО: використовує Fréchet mean замість arithmetic mean.
        
        Fréchet mean зберігає геометричну структуру траєкторії,
        тоді як arithmetic mean розмиває геометрію многовиду.
        """
        if len(self.points) < 2:
            return
        
        n_keep = self.max_trajectory_length // 10  # Зберігаємо 10%
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
            position=0,
            metadata={'aggregated': True, 'n_points': n_agg}
        ))
    
    # =========================================================================
    # ГЕОДЕЗИЧНИЙ ATTENTION (заміна softmax attention)
    # =========================================================================
    
    def compute_attention(
        self, 
        query: np.ndarray,
        temperature: Optional[float] = None,
        use_decay: bool = True,
        return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Геодезичний Attention — заміна softmax attention.
        
        Замість:
        - attention = softmax(query @ keys.T)  ❌
        
        Тепер:
        - attention = exp(-geodesic_distance² / T)  ✅
        
        OPTIMIZED: batch обчислення відстаней — O(n) замість O(n*d).
        
        Args:
            query: розподіл для attention (поточна точка)
            temperature: температура softmax
            use_decay: чи застосовувати часовий decay
            return_all: повернути все (attention, distances, energies)
            
        Returns:
            attention weights [n_points]
            Якщо return_all: (attention, distances, energies)
        """
        if len(self.points) == 0:
            if return_all:
                return np.array([]), np.array([]), np.array([])
            return np.array([])
        
        if temperature is None:
            temperature = self.temperature
        
        # Batch обчислення відстаней: O(n) замість O(n*d)
        # Всі sqrt(p) в одному масиві
        query_sqrt = np.sqrt(np.maximum(query, 1e-10))
        all_sqrt = np.array([np.sqrt(np.maximum(point.p, 1e-10)) for point in self.points])
        
        # Bhattacharyya coefficients: dot products
        bc = all_sqrt @ query_sqrt  # shape (n,)
        bc = np.clip(bc, 0, 1)
        distances = np.arccos(bc)
        
        # Геометричний attention: exp(-d²/T)
        energies = -distances ** 2 / temperature
        energies = energies - energies.max()  # Numerical stability
        
        # Часовий decay
        if use_decay:
            time_decay = np.array([
                self.decay_rate ** (len(self.points) - 1 - i) 
                for i in range(len(self.points))
            ])
            energies = energies + np.log(time_decay + 1e-10)
        
        # Softmax
        exp_energies = np.exp(energies - energies.max())
        attention = exp_energies / (exp_energies.sum() + 1e-10)
        
        if return_all:
            return attention, distances, energies
        return attention
    
    def attend_to_trajectory(
        self,
        query: np.ndarray,
        feature: str = 'p',
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """
        Звернути attention до траєкторії і отримати зважений результат.
        
        Це заміна: output = attention @ values
        
        Args:
            query: розподіл запиту
            feature: яку ознаку збирати ('p', 'entropy', 'position')
            temperature: температура
            
        Returns:
            Зважений результат (256-мірний вектор)
        """
        attention = self.compute_attention(query, temperature)
        
        if len(attention) == 0:
            return np.zeros(256) + 1e-10
        
        # Збір ознак
        if feature == 'p':
            features = np.array([p.p for p in self.points])
        elif feature == 'entropy':
            features = np.array([[p.entropy] for p in self.points])
        elif feature == 'position':
            features = np.array([[p.position] for p in self.points])
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
        Виявлення новизни через геометрію траєкторії.
        
        Замість порогових евристик — геометрична новизна:
        1. Відстань до центроїда траєкторії
        2. Кривина в поточній точці
        3. Швидкість зміни
        
        Returns:
            (novelty_score, confidence)
        """
        if len(self.points) == 0:
            return 1.0, 1.0  # Перша точка = максимальна новизна
        
        # 1. Відстань до центроїда
        if self.memory_centroid is not None:
            dist_to_center = fisher_rao_distance(p, self.memory_centroid)
            dist_score = min(1.0, dist_to_center / (self.memory_span + 0.1))
        else:
            dist_score = 0.5
        
        # 2. Кривина
        if len(self.curvature_profile) > 0 and self.enable_curvature:
            recent_curvatures = self.curvature_profile[-3:] if len(self.curvature_profile) >= 3 else self.curvature_profile
            curvature_score = min(1.0, np.mean(recent_curvatures) * 5)
        else:
            curvature_score = 0.0
        
        # 3. Швидкість
        if len(self.velocity_profile) > 0:
            recent_velocities = self.velocity_profile[-3:] if len(self.velocity_profile) >= 3 else self.velocity_profile
            velocity_score = min(1.0, np.mean(recent_velocities) * 10)
        else:
            velocity_score = 0.0
        
        # Комбінована оцінка
        novelty = 0.4 * dist_score + 0.3 * curvature_score + 0.3 * velocity_score
        novelty = min(1.0, novelty)
        
        # Впевненість: чим більше історії, тим вища
        confidence = min(1.0, len(self.points) / 20.0)
        
        return novelty, confidence
    
    def detect_boundary(self, p: np.ndarray) -> Tuple[float, str]:
        """
        Виявлення границі через кривину многовиду.
        
        Returns:
            (boundary_strength, boundary_type)
            
        boundary_type:
            - 'novel': нова тема
            - 'transition': перехід
            - 'stable': стабільний стан
        """
        novelty, confidence = self.detect_novelty(p)
        
        if novelty > 0.7:
            return novelty, 'novel'
        elif novelty > 0.4:
            return novelty, 'transition'
        else:
            return novelty, 'stable'
    
    # =========================================================================
    # КОНТЕКСТНИЙ ВЕКТОР (заміна window embedding)
    # =========================================================================
    
    def get_context_vector(self, query: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Отримати контекстний вектор — геометричне представлення траєкторії.
        
        Замість:
        - context = mean(window embeddings)  ❌
        
        Тепер:
        - Якщо query=None: позиція останньої точки
        - Якщо query є: результат геометричного attention
        
        Args:
            query: розподіл запиту (якщо None — остання точка)
            
        Returns:
            256-мірний контекстний вектор (розподіл на S²⁵⁵)
        """
        if len(self.points) == 0:
            return np.zeros(256) + 1e-10
        
        if query is None:
            # Просто остання точка
            return self.points[-1].p.copy()
        else:
            # Attention до траєкторії
            return self.attend_to_trajectory(query, feature='p')
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Отримати summary контексту для аналізу.
        
        Returns:
            Словник з характеристиками траєкторії
        """
        if len(self.points) == 0:
            return {
                'n_points': 0,
                'total_geodesic_length': 0.0,
                'current_entropy': 0.0,
                'current_modality': 'unknown',
                'memory_span': 0.0,
                'semantic_shapes': {},
                'topology': {},
            }
        
        # Топологічні характеристики
        topo = self._compute_topology()
        
        return {
            'n_points': len(self.points),
            'total_geodesic_length': self.total_geodesic_length,
            'current_entropy': self.points[-1].entropy,
            'current_modality': self.points[-1].modality,
            'memory_span': self.memory_span,
            'semantic_shapes': {
                'n_loops': len(self.semantic_shapes['loops']),
                'n_angles': len(self.semantic_shapes['angles']),
                'n_stops': len(self.semantic_shapes['stops']),
                'n_streams': len(self.semantic_shapes['streams']),
            },
            'topology': topo,
            'curvature_stats': {
                'mean': np.mean(self.curvature_profile) if self.curvature_profile else 0.0,
                'max': np.max(self.curvature_profile) if self.curvature_profile else 0.0,
            },
            'velocity_stats': {
                'mean': np.mean(self.velocity_profile) if self.velocity_profile else 0.0,
                'max': np.max(self.velocity_profile) if self.velocity_profile else 0.0,
            },
        }
    
    def _compute_topology(self) -> Dict[str, float]:
        """
        Обчислення топологічних характеристик траєкторії.
        
        Returns:
            Betti-like числа та інші топологічні ознаки
        """
        if len(self.points) < 3:
            return {
                'betti_0': 1.0,
                'betti_1': 0.0,
                'total_variation': 0.0,
                'rectifiable': 1.0,
                'oscillation_count': 0,
            }
        
        # Betti_0: зв'язність (завжди 1 для траєкторії)
        betti_0 = 1.0
        
        # Betti_1: кількість петель
        betti_1 = len(self.semantic_shapes['loops'])
        
        # Total variation
        tv = self.total_geodesic_length
        
        # Rectifiable: чи можна апроксимувати плавною кривою
        if self.curvature_profile:
            rectifiable = 1.0 - min(1.0, np.std(self.curvature_profile) * 2)
        else:
            rectifiable = 1.0
        
        # Oscillation count
        if len(self.velocity_profile) > 2:
            peaks, _ = find_peaks(np.array(self.velocity_profile))
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
    # ПАМ'ЯТЬ ЯК ПІДМНОГОВИД
    # =========================================================================
    
    def memory_retrieve(
        self, 
        query: np.ndarray, 
        k: int = 5,
        include_metadata: bool = False
    ) -> Union[Tuple[List[np.ndarray], List[float]], Tuple[List[np.ndarray], List[float], List[Dict]]]:
        """
        Знайти k найближчих точок в пам'яті (підмноговиді).
        
        Args:
            query: розподіл запиту
            k: кількість результатів
            include_metadata: включити метадані
            
        Returns:
            Якщо include_metadata: (retrieved_points, distances, metadata)
            Інакше: (retrieved_points, distances)
        """
        if len(self.memory_points) == 0:
            if include_metadata:
                return [], [], []
            return [], []
        
        # Batch обчислення відстаней: O(n) замість O(n*d)
        query_sqrt = np.sqrt(np.maximum(query, 1e-10))
        all_sqrt = np.array([np.sqrt(np.maximum(mp, 1e-10)) for mp in self.memory_points])
        bc = all_sqrt @ query_sqrt
        bc = np.clip(bc, 0, 1)
        distances = np.arccos(bc).tolist()
        
        # Топ-k
        k = min(k, len(distances))
        top_k_idx = np.argsort(distances)[:k]
        
        retrieved = [self.memory_points[i] for i in top_k_idx]
        dists = [distances[i] for i in top_k_idx]
        
        if include_metadata:
            metas = []
            for i in top_k_idx:
                if i < len(self.points):
                    metas.append(self.points[i].metadata)
                else:
                    metas.append({})
            return retrieved, dists, metas
        
        return retrieved, dists
    
    def memory_familiarity(self, p: np.ndarray) -> float:
        """
        Оцінка "знайомості" точки (1 = відомо, 0 = невідомо).
        
        Замість точного match — геометрична відстань до підмноговиду.
        """
        if len(self.memory_points) == 0:
            return 0.0
        
        distances = [fisher_rao_distance(p, mp) for mp in self.memory_points]
        min_dist = min(distances)
        
        # Перетворення відстані в знайомість
        if self.memory_span > 0:
            fam = max(0, 1.0 - min_dist / (self.memory_span + 1e-10))
        else:
            fam = 1.0 if min_dist < 0.1 else 0.0
        
        return fam
    
    # =========================================================================
    # ЗАПИТ-ВІДПОВІДЬ ЧЕРЕЗ ТРАЄКТОРІЮ
    # =========================================================================
    
    def query_response(
        self,
        query: np.ndarray,
        mode: str = 'attention',
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Запит-відповідь через геометрію траєкторії.
        
        Args:
            query: розподіл запиту
            mode: режим ('attention', 'retrieval', 'interpolation')
            top_k: кількість результатів
            
        Returns:
            Результат запиту з геометричною інтерпретацією
        """
        if len(self.points) == 0:
            return {
                'mode': mode,
                'success': False,
                'message': 'Empty trajectory',
            }
        
        if mode == 'attention':
            attention, distances, energies = self.compute_attention(
                query, return_all=True
            )
            
            return {
                'mode': mode,
                'success': True,
                'attention': attention,
                'distances': distances,
                'energies': energies,
                'top_k_indices': np.argsort(attention)[-top_k:][::-1].tolist(),
                'top_k_attention': attention[np.argsort(attention)[-top_k:][::-1]].tolist(),
                'context_vector': self.attend_to_trajectory(query).tolist(),
            }
        
        elif mode == 'retrieval':
            retrieved, dists, metas = self.memory_retrieve(
                query, k=top_k, include_metadata=True
            )
            
            return {
                'mode': mode,
                'success': True,
                'retrieved_points': [p.tolist() for p in retrieved],
                'distances': dists,
                'metadata': metas,
            }
        
        elif mode == 'interpolation':
            # Batch обчислення відстаней
            query_sqrt = np.sqrt(np.maximum(query, 1e-10))
            all_sqrt = np.array([np.sqrt(np.maximum(p.p, 1e-10)) for p in self.points])
            bc = all_sqrt @ query_sqrt
            bc = np.clip(bc, 0, 1)
            distances = np.arccos(bc)
            nearest_idx = np.argmin(distances)
            
            if nearest_idx < len(self.points) - 1:
                t = 0.5  # Середня точка
                interpolated = geodesic_interpolation(
                    self.points[nearest_idx].p,
                    self.points[nearest_idx + 1].p,
                    t
                )
                
                return {
                    'mode': mode,
                    'success': True,
                    'nearest_index': int(nearest_idx),
                    'nearest_distance': float(distances[nearest_idx]),
                    'interpolated_point': interpolated.tolist(),
                    'interpolation_t': t,
                }
            else:
                return {
                    'mode': mode,
                    'success': True,
                    'nearest_index': int(nearest_idx),
                    'nearest_distance': float(distances[nearest_idx]),
                    'nearest_point': self.points[nearest_idx].p.tolist(),
                }
        
        return {'mode': mode, 'success': False, 'message': 'Unknown mode'}
    
    # =========================================================================
    # УТИЛІТИ
    # =========================================================================
    
    def reset(self):
        """Очистити траєкторію."""
        self.points = []
        self.total_geodesic_length = 0.0
        self.curvature_profile = []
        self.velocity_profile = []
        self.acceleration_profile = []
        self.memory_centroid = None
        self.memory_span = 0.0
        self.memory_points = []
        self.semantic_shapes = {
            'loops': [],
            'angles': [],
            'stops': [],
            'streams': [],
        }
        self._attention_cache = None
        self._attention_valid = False
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __repr__(self) -> str:
        return (f"GeodesicContextEngine(n={len(self)}, "
                f"length={self.total_geodesic_length:.2f}, "
                f"span={self.memory_span:.3f})")


# =============================================================================
# 4. TRAJECTORY ATTENTION ДЛЯ КЛАСТЕРІВ
# =============================================================================

class TrajectoryAttention:
    """
    Attention для кластерів через геометрію траєкторії.
    
    Використовується в конвертаційних шарах замість стандартного attention.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        decay_rate: float = 0.95,
        metric: str = 'fisher_rao',
    ):
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.metric = metric
    
    def forward(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Геодезичний attention forward pass.
        
        Args:
            query: розподіл запиту
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
        distances = []
        for key in keys:
            if self.metric == 'fisher_rao':
                d = fisher_rao_distance(query, key)
            elif self.metric == 'kl':
                d = kl_divergence(query, key)
            else:
                d = fisher_rao_distance(query, key)
            distances.append(d)
        
        distances = np.array(distances)
        
        # Геометричний attention: exp(-d²/T)
        energies = -distances ** 2 / self.temperature
        energies = energies - energies.max()
        
        # Часовий decay
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
# 5. ЕКСПОРТ
# =============================================================================

__all__ = [
    'GeodesicContextEngine',
    'TrajectoryAttention',
    'ManifoldPoint',
    'fisher_rao_distance',
    'geodesic_interpolation',
    'kl_divergence',
]