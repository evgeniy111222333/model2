import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

class FeedbackMechanism:
    """
    Механізм зворотного зв'язку від вищих шарів до субстрату.

    Розділ 5.2 концепції: "зворотний зв'язок від вищих шарів до субстрату"

    Отримує представлення кластерів від ConversionLayers, обчислює
    "сигнал уваги" A(i) для кожної позиції субстрату, та модулює
    поле взаємодії: W'(i) = W(i) · (1 + α·A(i)).

    Це змушує субстрат "фокусуватися" на регіонах, ідентифікованих
    як важливі вищими шарами.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Сила зворотного зв'язку. Більше α → сильніша модуляція.
        """
        self.alpha = alpha

    def compute_attention(
        self,
        conversion_results: List[Dict],
        N: int,
    ) -> np.ndarray:
        """
        Обчислення сигналу уваги A(i) на основі результатів конвертації.

        Позиції, що входять до кластерів з високою якістю на будь-якому
        рівні конвертації, отримують вищу увагу. Вищі рівні мають меншу
        вагу (вони більш абстрактні).
        """
        attention = np.zeros(N, dtype=np.float32)

        for level_data in conversion_results:
            level = level_data['level']
            # Вага рівня: нижчі рівні (більш деталізовані) мають більшу вагу
            level_weight = 1.0 / (1.0 + level)

            for item in level_data['items']:
                cluster = item['cluster']
                start = cluster['start']
                end = cluster['end']

                # Сигнал якості кластера
                quality = cluster.get('quality_score', 0.5)

                # Кластери з більшою кількістю об'єднаних частин отримують бонус
                linked = cluster.get('linked_clusters', 1)
                link_bonus = min(1.0 + 0.2 * (linked - 1), 2.0)

                # Увага пропорційна якості та вазі рівня
                attention[start:end] += level_weight * quality * link_bonus

        # Нормалізація до [0, 1]
        max_att = attention.max()
        if max_att > 0:
            attention /= max_att

        return attention

    def apply(
        self,
        interaction_field: np.ndarray,
        conversion_results: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Застосування зворотного зв'язку до поля взаємодії.

        W'(i) = W(i) · (1 + α·A(i))

        Args:
            interaction_field: Поточне поле взаємодії W(i)
            conversion_results: Результати ієрархічної конвертації

        Returns:
            modified_field: Модульоване поле взаємодії
            attention: Сигнал уваги A(i)
        """
        N = len(interaction_field)
        attention = self.compute_attention(conversion_results, N)

        # Модуляція поля взаємодії (підтримка 1D та 2D)
        if interaction_field.ndim == 1:
            modified_field = interaction_field * (1.0 + self.alpha * attention)
        else:
            modified_field = interaction_field * (1.0 + self.alpha * attention[:, None])

        # Перенормалізація до [0, 1]
        max_W = modified_field.max()
        if max_W > 0:
            modified_field /= max_W

        return modified_field.astype(np.float32), attention


# =============================================================================
# INLINE CLASSES FROM bcs_core_v5 (вбудовані для автономності V6)
# =============================================================================


