import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

def _make_bytesubstrate():
    """
    Створює клас ByteSubstrate inline (V6 автономний, без зовнішніх імпортів).
    Мінімальна реалізація, необхідна для роботи V6.
    """
    class ByteSubstrate:
        BYTE_ALPHABET_SIZE = 256

        def __init__(self, data, max_length=None):
            if isinstance(data, str):
                data = data.encode('utf-8')
            if max_length is not None and len(data) > max_length:
                data = data[:max_length]
            self.raw_data = data
            self.length = len(data)
            self.byte_values = np.array(list(data), dtype=np.uint8)
            self.byte_distribution = self._compute_distribution()
            # One-hot encoding (Рівняння 1): s ↦ (e_1, ..., e_N)
            self.one_hot = np.zeros((self.length, 256), dtype=np.float32)
            if self.length > 0:
                self.one_hot[np.arange(self.length), self.byte_values] = 1.0

        def _compute_distribution(self):
            counts = np.bincount(self.byte_values, minlength=256)
            return counts.astype(np.float64) / max(self.length, 1)

        def detect_modality(self):
            dist = self.byte_distribution
            ascii_range = np.sum(dist[0x20:0x7F])
            null_ratio = dist[0x00]
            entropy = self._shannon_entropy(dist)
            unique_bytes = int(np.count_nonzero(dist))
            if ascii_range > 0.85 and entropy < 5.0:
                return "text_ascii"
            elif null_ratio > 0.3:
                return "sparse_binary"
            elif unique_bytes < 32 and entropy < 3.0:
                return "structured_data"
            else:
                return "mixed"

        def compute_local_distributions(self, window=16):
            N = self.length
            if N == 0:
                return np.zeros((0, 256), dtype=np.float32)
            one_hot = np.zeros((N, 256), dtype=np.float32)
            one_hot[np.arange(N), self.byte_values] = 1.0
            half = window // 2
            cum = np.vstack([np.zeros((1, 256), dtype=np.float32), np.cumsum(one_hot, axis=0)])
            indices = np.arange(N, dtype=np.intp)
            starts = np.maximum(0, indices - half)
            ends = np.minimum(N, indices + half + 1)
            window_sums = cum[ends] - cum[starts]
            window_sizes = (ends - starts).astype(np.float32)[:, None]
            return (window_sums / np.maximum(window_sizes, 1.0)).astype(np.float32)

        @staticmethod
        def _shannon_entropy(dist):
            p = dist[dist > 0]
            return float(-np.sum(p * np.log2(p)))

        def compute_byte_transitions(self):
            """Матриця переходів: T[k1, k2] = count(b_i=k1, b_{i+1}=k2)"""
            T = np.zeros((256, 256), dtype=np.float32)
            if self.length > 1:
                np.add.at(T, (self.byte_values[:-1], self.byte_values[1:]), 1.0)
            row_sums = T.sum(axis=1, keepdims=True)
            T = T / np.maximum(row_sums, 1.0)
            return T

        def segment(self, window_size, overlap=0):
            step = window_size - overlap
            segments = []
            for start in range(0, self.length - window_size + 1, step):
                segments.append(ByteSubstrate(self.raw_data[start:start + window_size]))
            return segments

        def windowed_process(
            self,
            window_size: int,
            overlap: int = 0,
            max_windows: Optional[int] = None,
        ) -> List[Tuple[int, int]]:
            """
            CONCEPT FIX (Розділ 8.3): Віконна обробка з перетинанням.

            Для великих потоків (N > 10⁶) концепція описує обробку вікнами
            розміром W з перетином δW ≥ λ_max. Це забезпечує:
            1. Обмежене споживання пам'яті: O(W·256) замість O(N·256)
            2. Неперервність контексту: перетин гарантує, що кластери,
               що знаходяться на межі вікон, будуть виявлені обома вікнами
            3. Ієрархічність: кристали з попередніх вікон впливають на нове
               вікно через механізм контекстного резонансу

            Концепція: "поле активації обчислюється лише для поточного вікна
            розміром W, причому перетин між сусідніми вікнами δW ≥ λ_max
            забезпечує неперервність контексту."

            Args:
                window_size: Розмір вікна W
                overlap: Перетин між сусідніми вікнами δW (має бути ≥ λ_max)
                max_windows: Максимум вікон (None = всі)

            Returns:
                Список кортежів (start, end) для кожного вікна
            """
            if window_size <= 0 or window_size > self.length:
                return [(0, self.length)]

            if overlap < 0:
                overlap = 0
            if overlap >= window_size:
                overlap = window_size - 1

            step = window_size - overlap
            windows = []
            start = 0
            while start < self.length:
                end = min(start + window_size, self.length)
                windows.append((start, end))
                if max_windows is not None and len(windows) >= max_windows:
                    break
                start += step
                # Якщо останнє вікно неповне і ми вже маємо хоча б одне — зупиняємось
                if end == self.length:
                    break

            return windows

        def __len__(self):
            return self.length

    return ByteSubstrate


# =============================================================================
# INLINE CLASSES FROM bcs_core_v4 (вбудовані для автономності V6)
# =============================================================================



ByteSubstrate = _make_bytesubstrate()
