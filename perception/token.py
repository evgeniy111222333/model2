import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _kl_divergence

class EmergentTokenDiscovery:
    """
    Виявлення емерджентних багатобайтових токенів.

    Кластери байтового субстрату об'єднуються в токени на основі:
    1. Частотної стабільності (токен з'являється у різних контекстах)
    2. Інформаційної цінності (токен несе більше інформації ніж сума байтів)
    3. Предиктивної сили (токен покращує передбачення)

    Метрика: mutual information gain
    MI(token; context) > Σ MI(byte_i; context)

    Відповідність концепції:
    - Розділ 7: Емерджентні абстракції
    - Визначення 7.1: Когнітивний токен τ = (s, p, c)
    - Рівняння (25): Умова емерджентності: I(τ; C) > Σ I(b_i; C)
    """

    def __init__(
        self,
        min_frequency: int = 2,
        max_token_length: int = 8,
        min_info_gain: float = 0.1,
    ):
        self.min_frequency = min_frequency
        self.max_token_length = max_token_length
        self.min_info_gain = min_info_gain
        self.discovered_tokens = []

    def discover_candidates(
        self,
        substrate,
        clusters: List[Dict],
    ) -> Dict[bytes, Dict]:
        """
        Виявити кандидатів у токени з кластерів.

        Для кожного кластера витягуємо байтові послідовності
        та будуємо таблицю n-gram частот.
        """
        data = substrate.raw_data
        N = len(data)
        ngram_table = {}  # bytes → {count, positions, in_clusters}

        for cluster in clusters:
            start = cluster['start']
            end = cluster['end']

            # Витягуємо послідовність байтів кластера
            cluster_bytes = data[start:end]

            # Будуємо n-gram для довжин 2..max_token_length
            for length in range(2, min(self.max_token_length + 1, len(cluster_bytes) + 1)):
                for i in range(len(cluster_bytes) - length + 1):
                    token = cluster_bytes[i:i + length]

                    if token not in ngram_table:
                        ngram_table[token] = {
                            'count': 0,
                            'positions': [],
                            'in_clusters': set(),
                            'length': length,
                        }

                    ngram_table[token]['count'] += 1
                    ngram_table[token]['positions'].append(start + i)
                    ngram_table[token]['in_clusters'].add(
                        next((j for j, c in enumerate(clusters)
                              if c['start'] <= start + i < c['end']), -1)
                    )

        # Фільтрація за мінімальною частотою
        candidates = {
            token: info for token, info in ngram_table.items()
            if info['count'] >= self.min_frequency
        }

        return candidates

    def compute_token_information_gain(
        self,
        token: bytes,
        substrate,
        clusters: List[Dict],
        transitions: Optional[np.ndarray] = None,
    ) -> float:
        """
        Обчислити information gain токена відносно окремих байтів із Laplace smoothing.

        IG(τ) = MI(τ; context) - (1/L) Σ_i MI(b_i; context)
        """
        data = substrate.raw_data
        N = len(data)
        L = len(token)

        if N < L + 1:
            return 0.0

        epsilon = 1e-5

        # === Глобальний розподіл із Laplace smoothing ===
        global_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64) + epsilon
        global_dist = global_counts / global_counts.sum()

        # Розподіл байтів після токена за допомогою C-level find
        token_positions = []
        pos = 0
        limit = N - L
        while pos < limit:
            pos = data.find(token, pos)
            if pos == -1 or pos >= limit:
                break
            token_positions.append(pos)
            pos += 1

        if len(token_positions) < self.min_frequency:
            return 0.0

        # Context distribution after token з Laplace smoothing
        context_counts = np.zeros(256, dtype=np.float64) + epsilon
        n_context = epsilon * 256.0
        for pos in token_positions:
            if pos + L < N:
                context_counts[data[pos + L]] += 1.0
                n_context += 1.0

        context_dist = context_counts / n_context

        # MI(token; context) = KL(context || global)
        mi_token = _kl_divergence(context_dist, global_dist)

        # === Σ MI(b_i; context) для кожного байта токена ===
        mi_bytes_sum = 0.0
        if transitions is not None:
            for byte_val in token:
                byte_context_counts = transitions[byte_val] + epsilon
                n_byte_context = byte_context_counts.sum()
                byte_context_dist = byte_context_counts / n_byte_context
                mi_byte = _kl_divergence(byte_context_dist, global_dist)
                mi_bytes_sum += mi_byte
        else:
            data_arr = np.frombuffer(data, dtype=np.uint8)
            transitions_local = np.zeros((256, 256), dtype=np.float64)
            if N > 1:
                np.add.at(transitions_local, (data_arr[:-1], data_arr[1:]), 1.0)
            for byte_val in token:
                byte_context_counts = transitions_local[byte_val] + epsilon
                n_byte_context = byte_context_counts.sum()
                byte_context_dist = byte_context_counts / n_byte_context
                mi_byte = _kl_divergence(byte_context_dist, global_dist)
                mi_bytes_sum += mi_byte

        # Information gain
        avg_mi_bytes = mi_bytes_sum / L
        info_gain = mi_token - avg_mi_bytes

        return float(info_gain)

    def compute_predictive_power(
        self,
        token: bytes,
        substrate,
        pc=None,
        u_field: Optional[np.ndarray] = None,
    ) -> float:
        """
        Обчислити предиктивну силу токена відносно реального стану поля u_field.
        """
        if pc is None:
            return 0.0

        data = substrate.raw_data
        N = len(data)
        L = len(token)

        if N < L + 1:
            return 0.0

        # Знаходимо всі позиції токена за допомогою C-level find
        token_positions = []
        pos = 0
        limit = N - L
        while pos < limit:
            pos = data.find(token, pos)
            if pos == -1 or pos >= limit:
                break
            token_positions.append(pos)
            pos += 1

        if len(token_positions) < self.min_frequency:
            return 0.0

        # Використовуємо реальний field_state u_field якщо він наданий, інакше плоске поле 0.5
        u_state = u_field if u_field is not None else (np.ones(N, dtype=np.float32) * 0.5)
        # Приведення довжини до розміру субстрату N
        if len(u_state) < N:
            u_state = np.pad(u_state, (0, N - len(u_state)), mode='edge')
        elif len(u_state) > N:
            u_state = u_state[:N]

        errors, _ = pc.compute_prediction_error(u_state)

        # Середня помилка на позиціях після токена
        after_token_errors = []
        for pos in token_positions:
            if pos + L < N and pos + L < len(errors):
                after_token_errors.append(abs(errors[pos + L]))

        if len(after_token_errors) == 0:
            return 0.0

        mean_error_at_token = np.mean(after_token_errors)
        global_mean_error = np.mean(np.abs(errors)) if len(errors) > 0 else 1.0

        if global_mean_error < 1e-10:
            return 0.0

        # Відносне зменшення помилки передбачення
        predictive_power = max(0.0, (global_mean_error - mean_error_at_token) / global_mean_error)

        return float(predictive_power)

    def compute_permutation_p_value(
        self,
        token: bytes,
        substrate,
        observed_ig: float,
        n_permutations: int = 50,
    ) -> float:
        """
        Обчислити p-value для токена за допомогою Permutation test.
        Перемішуємо вхідну послідовність байтів n_permutations разів
        та обчислюємо випадковий information gain (нульова гіпотеза).
        """
        if observed_ig <= 0.0:
            return 1.0

        data = substrate.raw_data
        N = len(data)
        L = len(token)

        # Швидкий MockSubstrate для уникнення накладних витрат
        from collections import namedtuple
        MockSubstrate = namedtuple('MockSubstrate', ['raw_data'])

        # Копіюємо для безпечного перемішування
        data_arr = np.frombuffer(data, dtype=np.uint8).copy()

        better_count = 0
        for _ in range(n_permutations):
            np.random.shuffle(data_arr)
            shuffled_bytes = bytes(data_arr)
            mock_sub = MockSubstrate(raw_data=shuffled_bytes)

            shuffled_transitions = np.zeros((256, 256), dtype=np.float64)
            if N > 1:
                np.add.at(shuffled_transitions, (data_arr[:-1], data_arr[1:]), 1.0)

            shuffled_ig = self.compute_token_information_gain(
                token, mock_sub, [], transitions=shuffled_transitions
            )
            if shuffled_ig >= observed_ig:
                better_count += 1

        p_val = (better_count + 1) / (n_permutations + 1)
        return float(p_val)

    def discover(
        self,
        substrate,
        clusters: List[Dict],
        pc=None,
        u_field: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Повний цикл виявлення емерджентних токенів зі статистикою Bonferroni.
        """
        # 1. Виявлення кандидатів
        candidates = self.discover_candidates(substrate, clusters)

        # Precompute transitions for the original substrate
        data = substrate.raw_data
        N = len(data)
        data_arr = np.frombuffer(data, dtype=np.uint8)
        transitions = np.zeros((256, 256), dtype=np.float64)
        if N > 1:
            np.add.at(transitions, (data_arr[:-1], data_arr[1:]), 1.0)

        # 2. Попередня фільтрація за IG
        prefiltered_candidates = []
        for token_bytes, info in candidates.items():
            info_gain = self.compute_token_information_gain(
                token_bytes, substrate, clusters, transitions=transitions
            )

            if info_gain < self.min_info_gain:
                continue

            prefiltered_candidates.append((token_bytes, info, info_gain))

        m = len(prefiltered_candidates)  # Кількість тестів для Bonferroni correction

        # 3. Оцінка та статистичний аналіз кожного кандидата
        tokens = []
        for token_bytes, info, info_gain in prefiltered_candidates:
            # Предиктивна сила на реальному полі u_field
            pred_power = self.compute_predictive_power(token_bytes, substrate, pc, u_field=u_field)

            # Частотна стабільність
            cluster_diversity = len(info['in_clusters'])

            # Статистичний тест (Permutation Test)
            p_value = self.compute_permutation_p_value(token_bytes, substrate, info_gain, n_permutations=50)
            
            # Bonferroni Correction
            p_value_adjusted = min(p_value * m, 1.0)
            is_significant = p_value_adjusted < 0.05

            # Композитна оцінка якості з урахуванням статистичної значущості
            sig_multiplier = 1.0 if is_significant else 0.5
            quality = (
                0.4 * min(info_gain / 2.0, 1.0) +  # Нормалізований info gain
                0.3 * min(pred_power, 1.0) +         # Предиктивна сила
                0.3 * min(cluster_diversity / 3.0, 1.0)  # Частотна стабільність
            ) * sig_multiplier

            token_entry = {
                'token': token_bytes,
                'token_hex': token_bytes.hex(),
                'length': info['length'],
                'frequency': info['count'],
                'positions': info['positions'][:10],
                'cluster_diversity': cluster_diversity,
                'info_gain': float(info_gain),
                'predictive_power': float(pred_power),
                'quality': float(quality),
                'p_value': float(p_value),
                'p_value_adjusted': float(p_value_adjusted),
                'is_statistically_significant': bool(is_significant),
            }

            try:
                token_entry['token_str'] = token_bytes.decode('utf-8', errors='replace')
            except Exception:
                token_entry['token_str'] = f"<hex:{token_bytes.hex()}>"

            tokens.append(token_entry)

        # 4. Сортування за якістю
        tokens.sort(key=lambda t: t['quality'], reverse=True)

        # 5. Видалення підтокенів
        filtered_tokens = self._remove_subtokens(tokens)

        self.discovered_tokens = filtered_tokens
        return filtered_tokens

    def _remove_subtokens(self, tokens: List[Dict]) -> List[Dict]:
        """Видалити подтокени якщо батьківський токен має вищу якість."""
        if len(tokens) <= 1:
            return tokens

        result = []
        used = set()

        for i, token_i in enumerate(tokens):
            if i in used:
                continue

            token_bytes_i = token_i['token']
            is_subtoken = False

            for j, token_j in enumerate(tokens):
                if i == j or j in used:
                    continue
                token_bytes_j = token_j['token']

                # Чи є token_i подтокеном token_j?
                if token_bytes_i in token_bytes_j and len(token_bytes_i) < len(token_bytes_j):
                    # Так, але перевіряємо якість
                    if token_j['quality'] >= token_i['quality'] * 0.8:
                        is_subtoken = True
                        break

            if not is_subtoken:
                result.append(token_i)
                used.add(i)

        return result


