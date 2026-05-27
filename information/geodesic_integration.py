"""
BCS Trajectory Context Integration — Інтеграція GeodesicContextEngine в BCSModelV6

ЗАМІНЮЄ:
1. Window-based context → Trajectory context
2. Standard attention → Geodesic attention
3. Buffer memory → Memory submanifold

ВИКОРИСТАННЯ:
1. Під час ініціалізації моделі: створити GeodesicContextEngine
2. Під час обробки даних: накопичувати траєкторію
3. Під час конвертації: використовувати GeodesicAttention замість стандартного
4. Під час семантичного зчитування: використовувати trajectory context

ПРИКЛАД:
    from bcs.information.geodesic_integration import TrajectoryContextIntegration
    
    integration = TrajectoryContextIntegration(model)
    integration.initialize()
    
    # Під час обробки:
    integration.push_data(data_chunk)
    
    # Під час конвертації:
    conversion_result = integration.convert_with_geodesic_attention(clusters)
    
    # Під час семантичного зчитування:
    context = integration.get_trajectory_context()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.information.geodesic_context import (
    GeodesicContextEngine,
    TrajectoryAttention,
    fisher_rao_distance,
)


class TrajectoryContextIntegration:
    """
    Інтеграція геодезичного контексту в BCSModelV6.
    
    Підключає:
    - GeodesicContextEngine → як основний контекстний механізм
    - TrajectoryAttention → для attention в конвертаційних шарах
    - Memory submanifold → для пам'яті
    
    ЗАМІНЮЄ:
    - window-based context
    - standard softmax attention
    - buffer-based memory
    """
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        Args:
            model: BCSModelV6
            config: конфігурація (temperature, decay_rate, etc.)
        """
        self.model = model
        
        # Конфігурація
        self.config = config or {}
        self.temperature = self.config.get('temperature', 1.0)
        self.decay_rate = self.config.get('decay_rate', 0.99)
        self.max_trajectory = self.config.get('max_trajectory_length', 1000)
        self.novelty_threshold = self.config.get('novelty_threshold', 0.5)
        
        # Геодезичний контекстний двигун
        self.context_engine: Optional[GeodesicContextEngine] = None
        
        # Trajectory attention для кластерів
        self.trajectory_attention = TrajectoryAttention(
            temperature=self.temperature,
            decay_rate=self.decay_rate,
        )
        
        # Чи ініціалізовано
        self._initialized = False
    
    def initialize(self, data: Optional[bytes] = None):
        """
        Ініціалізувати геодезичний контекст.
        
        Args:
            data: опціональні початкові дані
        """
        self.context_engine = GeodesicContextEngine(
            max_trajectory_length=self.max_trajectory,
            decay_rate=self.decay_rate,
            novelty_threshold=self.novelty_threshold,
            temperature=self.temperature,
            enable_curvature=True,
            enable_memory=True,
            enable_semantic=True,
        )
        
        # Якщо є початкові дані — ініціалізуємо траєкторію
        if data is not None:
            self._initialize_trajectory_from_data(data)
        
        self._initialized = True
        return self
    
    def _initialize_trajectory_from_data(self, data: bytes):
        """
        Ініціалізувати траєкторію з байтових даних.
        
        Для кожної позиції створюємо точку на многовиді.
        """
        n = len(data)
        step = max(1, n // 100)  # Не більше 100 точок
        
        for i in range(0, n, step):
            # Локальний розподіл навколо позиції
            half_w = 8
            start = max(0, i - half_w)
            end = min(n, i + half_w)
            
            # Розподіл байтів
            dist = np.zeros(256)
            for b in data[start:end]:
                dist[b] += 1
            
            if dist.sum() > 0:
                dist = dist / dist.sum()
            else:
                dist = np.ones(256) / 256
            
            # Визначення модальності
            modality = self._detect_local_modality(dist)
            
            # Додаємо точку
            self.context_engine.push(
                p=dist,
                t=float(i) / n,
                position=i,
                modality=modality,
                metadata={'data_range': (start, end)}
            )
    
    def _detect_local_modality(self, dist: np.ndarray) -> str:
        """Визначити модальність локального розподілу."""
        ascii_range = np.sum(dist[0x20:0x7F])
        high_bytes = np.sum(dist[0x80:])
        
        if ascii_range > 0.8:
            return 'text_ascii'
        elif high_bytes > 0.3:
            return 'text_utf8'
        elif dist[0] + dist[255] > 0.5:
            return 'binary'
        elif np.max(dist) > 0.1:
            return 'structured'
        else:
            return 'mixed'
    
    def push_data(self, data_chunk: bytes, modality: str = "unknown"):
        """
        Додати новий чанк даних до траєкторії.
        
        Args:
            data_chunk: байти для додавання
            modality: модальність даних
        """
        if self.context_engine is None:
            self.initialize()
        
        n = len(data_chunk)
        if n == 0:
            return
        
        # Створюємо розподіл
        dist = np.zeros(256)
        for b in data_chunk:
            dist[b] += 1
        dist = dist / dist.sum()
        
        # Додаємо точку
        current_t = len(self.context_engine.points) / max(1, self.max_trajectory)
        self.context_engine.push(
            p=dist,
            t=current_t,
            position=len(data_chunk),
            modality=modality,
        )
    
    def get_trajectory_context(
        self, 
        query: Optional[np.ndarray] = None,
        mode: str = 'attention'
    ) -> np.ndarray:
        """
        Отримати контекстний вектор з траєкторії.
        
        Args:
            query: опціональний запит для attention
            mode: 'attention' або 'last_point'
            
        Returns:
            256-мірний контекстний вектор
        """
        if self.context_engine is None:
            return np.zeros(256) + 1e-10
        
        return self.context_engine.get_context_vector(query)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Отримати summary траєкторії."""
        if self.context_engine is None:
            return {}
        return self.context_engine.get_context_summary()
    
    # =========================================================================
    # ГЕОДЕЗИЧНИЙ ATTENTION ДЛЯ КОНВЕРТАЦІЙНИХ ШАРІВ
    # =========================================================================
    
    def convert_with_geodesic_attention(
        self,
        clusters: List[Dict],
        use_trajectory_attention: bool = True,
        use_decay: bool = True,
    ) -> List[Dict]:
        """
        Конвертація кластерів з геодезичним attention.
        
        ЗАМІНЮЄ стандартний attention в ConversionLayersV3.
        
        Args:
            clusters: список кластерів
            use_trajectory_attention: використовувати TrajectoryAttention
            use_decay: використовувати часовий decay
            
        Returns:
            Модифіковані кластери з геодезичним attention
        """
        if not clusters:
            return clusters
        
        # Отримуємо розподіли кластерів
        cluster_dists = [c.get('distribution', np.zeros(256)) for c in clusters]
        
        if use_trajectory_attention:
            # Використовуємо TrajectoryAttention для кожного кластера
            result_clusters = []
            
            for i, cluster in enumerate(clusters):
                query = cluster_dists[i]
                
                # Geodesic attention
                output, attention = self.trajectory_attention.forward(
                    query=query,
                    keys=cluster_dists,
                    values=cluster_dists,
                )
                
                # Додаємо геометричну інформацію до кластера
                cluster = cluster.copy()
                cluster['geodesic_attention'] = attention
                cluster['geodesic_context'] = output
                cluster['attention_entropy'] = -np.sum(attention * np.log(attention + 1e-10))
                
                result_clusters.append(cluster)
            
            return result_clusters
        else:
            # Стандартний attention
            return clusters
    
    def geodesic_attention_layer(
        self,
        representations: List[np.ndarray],
        query_repr: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Шар геодезичного attention для представлень.
        
        Args:
            representations: список представлень (256-мірних)
            query_repr: запит (якщо None — використовує останній)
            temperature: температура softmax
            
        Returns:
            (output, attention_weights)
        """
        if not representations:
            return np.zeros(256), np.array([])
        
        repr_arr = np.array(representations)
        
        if query_repr is None:
            query_repr = repr_arr[-1]
        
        # Геодезичні відстані
        distances = np.array([
            fisher_rao_distance(query_repr, r) for r in representations
        ])
        
        # Attention
        energies = -distances ** 2 / temperature
        energies = energies - energies.max()
        exp_energies = np.exp(energies - energies.max())
        attention = exp_energies / (exp_energies.sum() + 1e-10)
        
        # Output
        output = attention @ repr_arr
        
        return output, attention
    
    # =========================================================================
    # ІНТЕГРАЦІЯ З СЕМАНТИЧНИМ ШАРОМ
    # =========================================================================
    
    def inject_trajectory_into_semantic(
        self,
        semantic_layer,
        context_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Впровадити траєкторію в семантичний шар.
        
        Args:
            semantic_layer: SemanticLatentDynamics
            context_vector: опціональний контекстний вектор
            
        Returns:
            Результат ін'єкції
        """
        if self.context_engine is None:
            return {'success': False, 'message': 'Context not initialized'}
        
        if context_vector is None:
            context_vector = self.context_engine.get_context_vector()
        
        # Отримуємо summary
        summary = self.context_engine.get_context_summary()
        
        # Створюємо додаткові ознаки для семантичного шару
        trajectory_features = {
            'trajectory_length': summary.get('n_points', 0),
            'geodesic_length': summary.get('total_geodesic_length', 0.0),
            'memory_span': summary.get('memory_span', 0.0),
            'n_loops': summary.get('semantic_shapes', {}).get('n_loops', 0),
            'n_angles': summary.get('semantic_shapes', {}).get('n_angles', 0),
            'current_entropy': summary.get('current_entropy', 0.0),
            'curvature_mean': summary.get('curvature_stats', {}).get('mean', 0.0),
            'velocity_mean': summary.get('velocity_stats', {}).get('mean', 0.0),
        }
        
        return {
            'success': True,
            'context_vector': context_vector,
            'trajectory_features': trajectory_features,
            'summary': summary,
        }
    
    # =========================================================================
    # ВИЯВЛЕННЯ НОВИЗНИ ТА ГРАНИЦЬ
    # =========================================================================
    
    def detect_novelty(self, p: np.ndarray) -> Tuple[float, float]:
        """
        Виявити новизну через геометрію траєкторії.
        
        Returns:
            (novelty_score, confidence)
        """
        if self.context_engine is None:
            return 1.0, 1.0
        
        return self.context_engine.detect_novelty(p)
    
    def detect_boundary(self, p: np.ndarray) -> Tuple[float, str]:
        """
        Виявити границю через кривину многовиду.
        
        Returns:
            (boundary_strength, boundary_type)
        """
        if self.context_engine is None:
            return 1.0, 'novel'
        
        return self.context_engine.detect_boundary(p)
    
    def detect_context_boundary(self, data: bytes) -> List[Dict]:
        """
        Виявити границі контексту в даних.
        
        Returns:
            Список границь з їх характеристиками
        """
        if self.context_engine is None or len(self.context_engine.points) < 3:
            return []
        
        boundaries = []
        
        for i in range(1, len(self.context_engine.points) - 1):
            p_prev = self.context_engine.points[i-1].p
            p_curr = self.context_engine.points[i].p
            p_next = self.context_engine.points[i+1].p
            
            # Кривина в точці
            bc1 = np.sum(np.sqrt(p_prev * p_curr))
            bc2 = np.sum(np.sqrt(p_curr * p_next))
            curvature = 1.0 - bc1 * bc2
            
            if curvature > 0.5:  # Різка зміна
                boundaries.append({
                    'position': self.context_engine.points[i].position,
                    'time': self.context_engine.points[i].t,
                    'curvature': curvature,
                    'type': 'angle',
                    'entropy': self.context_engine.points[i].entropy,
                })
        
        return boundaries
    
    # =========================================================================
    # ЗАПИТ-ВІДПОВІДЬ ЧЕРЕЗ ТРАЄКТОРІЮ
    # =========================================================================
    
    def query_trajectory(
        self,
        query: np.ndarray,
        mode: str = 'attention',
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Запит до траєкторії.
        
        Args:
            query: розподіл запиту
            mode: 'attention', 'retrieval', 'interpolation'
            top_k: кількість результатів
            
        Returns:
            Результат запиту
        """
        if self.context_engine is None:
            return {'success': False, 'message': 'Context not initialized'}
        
        return self.context_engine.query_response(query, mode, top_k)
    
    def memory_query(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Запит до пам'яті (підмноговиду).
        
        Returns:
            (retrieved_points, distances)
        """
        if self.context_engine is None:
            return [], []
        
        return self.context_engine.memory_retrieve(query, k=k)
    
    def check_familiarity(self, p: np.ndarray) -> float:
        """
        Перевірити знайомість точки через пам'ять.
        
        Returns:
            familiarity score [0, 1]
        """
        if self.context_engine is None:
            return 0.0
        
        return self.context_engine.memory_familiarity(p)
    
    # =========================================================================
    # УТИЛІТИ
    # =========================================================================
    
    def reset(self):
        """Скинути траєкторію."""
        if self.context_engine is not None:
            self.context_engine.reset()
    
    def get_summary(self) -> Dict[str, Any]:
        """Отримати повний summary."""
        if self.context_engine is None:
            return {
                'initialized': False,
                'n_points': 0,
            }
        
        summary = self.context_engine.get_context_summary()
        summary['initialized'] = True
        
        return summary
    
    def __len__(self) -> int:
        if self.context_engine is None:
            return 0
        return len(self.context_engine)


# =============================================================================
# ЕКСПОРТ
# =============================================================================

__all__ = [
    'TrajectoryContextIntegration',
    'GeodesicContextEngine',
    'TrajectoryAttention',
]