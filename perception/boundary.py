import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _js_divergence
from bcs.core.policy import AdaptiveNumericPolicy

class MultiScaleBoundaryDetector:
    """
    Мультимасштабний аналіз границь — Розділ 4.5 концепції.

    Аналіз границь на кількох масштабах: [4, 8, 16, 32, 64].
    Для кожного масштабу обчислюються:
    1. Градієнт ентропії локального розподілу
    2. Щільність байтових переходів
    3. Комбінація з довірчою вагою масштабу

    Границі виявлені на кількох масштабах отримують вищу довіру.
    """

    def __init__(
        self,
        scales: Optional[List[int]] = None,
        numeric_policy: Optional[AdaptiveNumericPolicy] = None,
    ):
        self.scales = scales if scales is not None else [4, 8, 16, 32, 64]
        self.numeric_policy = numeric_policy or AdaptiveNumericPolicy()
        self.last_policy = {}

    def detect(
        self,
        substrate,
        v_field: Optional[np.ndarray] = None,
        pc_anomalies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Виявлення границь з довірчістю на кожній позиції.

        Returns:
            boundary_confidence: (N,) масив довіри границі ∈ [0, 1]
        """
        N = substrate.length
        if N == 0:
            return np.zeros(0, dtype=np.float32)

        boundary_confidence = np.zeros(N, dtype=np.float32)
        bv = substrate.byte_values
        signal_policy = self.numeric_policy.boundary_signal_policy(N)
        self.last_policy['boundary_signal'] = dict(signal_policy)

        for scale in self.scales:
            if scale >= N:
                continue

            # Локальні розподіли на цьому масштабі
            local_dist = substrate.compute_local_distributions(window=scale)

            # 1. Градієнт ентропії
            p_safe = np.maximum(local_dist, 1e-10)
            entropy = -np.sum(local_dist * np.log2(p_safe), axis=1).astype(np.float32)
            entropy_grad = np.abs(np.gradient(entropy))

            # 2. Градієнт щільності байтових переходів
            #    (векторизовано через згортку; використовуємо градієнт замість
            #    сирової щільності, бо для UTF-8 тексту щільність переходів
            #    постійно висока і не дає сигналу про границі)
            transitions = np.zeros(N, dtype=np.float32)
            transitions[1:] = (bv[1:] != bv[:-1]).astype(np.float32)
            half = scale // 2
            kernel = np.ones(scale, dtype=np.float32) / scale
            trans_density = np.convolve(transitions, kernel, mode='same')
            trans_grad = np.abs(np.gradient(trans_density))

            # 3. Distribution shift: L2 distance between adjacent local distributions
            dist_shift = np.zeros(N, dtype=np.float32)
            if N > 1:
                diff = local_dist[1:] - local_dist[:-1]
                dist_shift[1:] = np.sqrt(np.sum(diff ** 2, axis=1)).astype(np.float32)
            dist_shift_smooth = np.convolve(dist_shift, kernel, mode='same')

            # Нормалізуємо кожен сигнал незалежно перед комбінацією,
            # щоб жоден сигнал не домінував над іншими
            def _norm_sig(s):
                m = s.max()
                return s / m if m > 0 else s

            eg_norm = _norm_sig(entropy_grad)
            tg_norm = _norm_sig(trans_grad)
            ds_norm = _norm_sig(dist_shift_smooth)

            # Комбінація сигналів на цьому масштабі
            scale_signal = (
                eg_norm * signal_policy['entropy_grad_weight']
                + tg_norm * signal_policy['transition_grad_weight']
                + ds_norm * signal_policy['distribution_shift_weight']
            )

            # Вага масштабу: менший масштаб → вища точність → більша вага
            weight = 1.0 / (1.0 + np.log2(max(scale, 2)))
            boundary_confidence += weight * scale_signal

        # === Макро-масштабний сигнал зміни розподілу ===
        # V6 FIX #2: Багатороздільний макро-аналіз з обмеженням вікна.
        # Попередній macro_window = N//5 зростав без обмежень (N=2000→400),
        # що для однорідних даних усереднювало JS-дивергенцію до нуля.
        # Фікс: використовуємо КІЛЬКА вікон різного розміру (кожне ≤ 150),
        # і беремо МАКСИМАЛЬНИЙ сигнал. Це дозволяє виявляти границі
        # навіть на довгих однорідних ділянках.
        macro_windows = [int(signal_policy['max_macro_window'])]
        if N > 200:
            macro_windows.append(int(signal_policy['mid_macro_window']))
        if N > 500:
            macro_windows.append(int(signal_policy['small_macro_window']))

        one_hot = substrate.one_hot
        cum = np.vstack(
            [np.zeros((1, 256), dtype=np.float32),
             np.cumsum(one_hot, axis=0)]
        )

        for macro_window in macro_windows:
            if macro_window >= N:
                continue

            macro_half = macro_window // 2
            indices = np.arange(N)
            left_starts = np.maximum(0, indices - macro_half)
            right_ends = np.minimum(N, indices + macro_half)

            left_sums = cum[indices] - cum[left_starts]
            right_sums = cum[right_ends] - cum[indices]

            # Обчислення розмірів вікон для виявлення країв
            left_counts = left_sums.sum(axis=1, keepdims=True)
            right_counts = right_sums.sum(axis=1, keepdims=True)

            left_sizes = np.maximum(left_counts, 1.0)
            right_sizes = np.maximum(right_counts, 1.0)

            left_dist = left_sums / left_sizes
            right_dist = right_sums / right_sizes

            # JS-дивергенція між лівим і правим розподілами
            m = 0.5 * (left_dist + right_dist)
            p = np.maximum(left_dist, 1e-10)
            q = np.maximum(right_dist, 1e-10)
            m_safe = np.maximum(m, 1e-10)
            kl_pm = np.sum(p * np.log(p / m_safe), axis=1)
            kl_qm = np.sum(q * np.log(q / m_safe), axis=1)
            macro_signal = (0.5 * kl_pm + 0.5 * kl_qm).astype(np.float32)

            # Придушення країв: позиції з малим вікном зліва чи справа
            # мають штучно високий JS через порожнє вікно
            min_window_size = max(macro_half // 2, 5)
            left_too_small = (left_counts.ravel() < min_window_size).astype(np.float32)
            right_too_small = (right_counts.ravel() < min_window_size).astype(np.float32)
            edge_mask = 1.0 - np.maximum(left_too_small, right_too_small)
            macro_signal *= edge_mask

            # Згладжування макро-сигналу
            macro_kernel_size = max(macro_window // 3, 3)
            macro_kernel = np.ones(macro_kernel_size, dtype=np.float32) / macro_kernel_size
            macro_signal = np.convolve(macro_signal, macro_kernel, mode='same')

            # Нормалізація та додавання з вагою, обернено пропорційною вікну
            # Менші вікна отримують вищу вагу (вони точніші для однорідних даних)
            macro_max = macro_signal.max()
            if macro_max > 0:
                macro_signal /= macro_max
            weight = signal_policy['macro_weight_base'] / (1.0 + np.log2(max(macro_window, 2)))
            boundary_confidence += macro_signal * weight

        # Додатковий сигнал від v-поля (якщо доступне)
        if v_field is not None:
            dv = np.abs(np.gradient(v_field))
            max_dv = dv.max()
            if max_dv > 0:
                boundary_confidence += dv / max_dv

        # Додатковий сигнал від предиктивного кодування
        if pc_anomalies is not None and len(pc_anomalies) > 0:
            boundary_confidence[pc_anomalies] += signal_policy['pc_anomaly_boost']

        # Нормалізація до [0, 1]
        max_conf = boundary_confidence.max()
        if max_conf > 0:
            boundary_confidence /= max_conf

        return boundary_confidence

    def detect_boundary_positions(
        self,
        substrate,
        v_field: Optional[np.ndarray] = None,
        pc_anomalies: Optional[np.ndarray] = None,
        percentile: Optional[float] = None,
        min_gap: Optional[int] = None,
    ) -> np.ndarray:
        """
        Виявлення позицій границь з мультимасштабним аналізом.

        Returns:
            boundaries: відсортований масив позицій границь
        """
        N = substrate.length
        if N == 0:
            return np.array([], dtype=int)

        confidence = self.detect(substrate, v_field, pc_anomalies)
        selection_policy = self.numeric_policy.boundary_selection_policy(confidence, N)
        self.last_policy['boundary_selection'] = dict(selection_policy)
        if percentile is None:
            percentile = float(selection_policy['percentile'])
        if min_gap is None:
            min_gap = int(selection_policy['min_gap'])

        # Поріг на основі перцентиля
        threshold = np.percentile(confidence, percentile)
        
        # Знаходження локальних піків вище порогу
        candidate_peaks = []
        for i in range(N):
            if confidence[i] < threshold:
                continue
            is_peak = True
            if i > 0 and confidence[i] < confidence[i - 1]:
                is_peak = False
            if i < N - 1 and confidence[i] < confidence[i + 1]:
                is_peak = False
            if is_peak:
                candidate_peaks.append((i, confidence[i]))

        if len(candidate_peaks) == 0:
            return np.array([], dtype=int)

        # Сортування за зменшенням впевненості
        candidate_peaks.sort(key=lambda x: x[1], reverse=True)

        # Жадібний відбір з урахуванням мінімальної відстані min_gap
        boundaries = []
        for idx, conf in candidate_peaks:
            too_close = False
            for pb in boundaries:
                if abs(idx - pb) < min_gap:
                    too_close = True
                    break
            if not too_close:
                boundaries.append(idx)

        boundaries.sort()
        return np.array(boundaries, dtype=int)


