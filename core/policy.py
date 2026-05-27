import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

class AdaptiveNumericPolicy:
    """Empirical calibration for early perception and field numeric gates."""

    def __init__(self, max_history: int = 256):
        self.max_history = int(max(32, max_history))
        self.history = {}

    @staticmethod
    def _finite(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(float(value), float(low), float(high)))

    def observe(self, name: str, value):
        try:
            v = float(value)
        except Exception:
            return
        if not np.isfinite(v):
            return
        vals = self.history.setdefault(str(name), [])
        vals.append(v)
        if len(vals) > self.max_history:
            del vals[:len(vals) - self.max_history]

    def _quantile(self, values: np.ndarray, q: float, fallback: float) -> float:
        vals = self._finite(values)
        if vals.size == 0:
            return float(fallback)
        return float(np.percentile(vals, float(np.clip(q, 0.0, 100.0))))

    def predictive_anomaly_threshold(self, errors: np.ndarray, prior: float = 1.5) -> float:
        vals = np.abs(self._finite(errors))
        if vals.size < 4:
            value = float(prior)
        else:
            std = max(float(np.std(vals)), 1e-10)
            value = float(np.percentile(vals, 92.0) / std)
        value = self._clip(value, 0.75, 4.0)
        self.observe('predictive_anomaly_threshold', value)
        return value

    def boundary_signal_policy(self, N: int) -> Dict[str, float]:
        n = max(int(N), 1)
        return {
            'entropy_grad_weight': 1.0,
            'transition_grad_weight': self._clip(np.log1p(n) / np.log(256.0), 1.0, 3.0),
            'distribution_shift_weight': self._clip(np.sqrt(np.log1p(n)), 2.0, 5.0),
            'macro_weight_base': self._clip(np.log1p(n) / 2.0, 1.4, 3.2),
            'pc_anomaly_boost': self._clip(1.0 + 1.0 / np.sqrt(max(n, 1)), 0.6, 1.2),
            'max_macro_window': int(self._clip(max(30, n // 5), 30, 150)),
            'mid_macro_window': int(self._clip(max(30, n // 10), 30, 100)),
            'small_macro_window': int(self._clip(max(20, n // 20), 20, 60)),
        }

    def boundary_selection_policy(self, confidence: np.ndarray, N: int) -> Dict[str, float]:
        vals = self._finite(confidence)
        n = max(int(N), 1)
        if vals.size == 0:
            signal_strength = 0.0
            percentile = 55.0
        else:
            signal_strength = float(np.std(vals))
            q25 = self._quantile(vals, 25.0, 0.0)
            q75 = self._quantile(vals, 75.0, 0.0)
            spread = max(q75 - q25, 1e-8)
            percentile = 100.0 * (1.0 - self._clip(0.5 * spread + signal_strength, 0.35, 0.65))
        percentile = self._clip(percentile, 35.0, 60.0)
        min_gap = int(self._clip(max(n // 30, 10), 4, min(50, max(4, n))))
        fallback_boundaries = max(1, n // max(80, int(np.sqrt(max(n, 1)))))
        policy = {
            'signal_strength': signal_strength,
            'percentile': percentile,
            'min_gap': min_gap,
            'fallback_boundaries': int(fallback_boundaries),
        }
        self.observe('boundary_signal_strength', signal_strength)
        self.observe('boundary_percentile', percentile)
        return policy

    def cluster_policy(self, N: int, js_values: Optional[List[float]] = None) -> Dict[str, float]:
        n = max(int(N), 1)
        js = np.asarray(js_values or [], dtype=np.float32)
        if js.size:
            base_js = self._clip(float(np.percentile(js, 25.0)), 0.01, 0.18)
            non_adjacent_js = self._clip(float(np.percentile(js, 20.0)), 0.01, 0.12)
        else:
            base_js = 0.15
            non_adjacent_js = 0.10
        autonomous_size = int(self._clip(n // 20, 12, 48))
        too_large_ratio = self._clip(0.45 - 0.05 * np.log10(max(n, 10)) / 3.0, 0.28, 0.45)
        min_clusters = max(2, n // max(160, int(np.sqrt(max(n, 1)))))
        policy = {
            'adjacent_js_threshold': base_js,
            'large_segment_js_factor': self._clip(autonomous_size / max(n, 1), 0.20, 0.45),
            'boundary_small_js_factor': 0.20,
            'non_adjacent_js_threshold': non_adjacent_js,
            'min_autonomous_size': autonomous_size,
            'too_large_ratio': too_large_ratio,
            'min_clusters': int(min_clusters),
        }
        self.observe('cluster_adjacent_js_threshold', base_js)
        self.observe('cluster_non_adjacent_js_threshold', non_adjacent_js)
        return policy

    def field_policy(self, byte_freq_norm: np.ndarray, N: int) -> Dict[str, float]:
        freq = self._finite(byte_freq_norm)
        nonzero = freq[freq > 0]
        if nonzero.size:
            high = self._clip(float(np.percentile(nonzero, 80.0)), 1.0 / max(int(N), 1), 0.20)
            mid = self._clip(float(np.percentile(nonzero, 35.0)), 1.0 / max(int(N) * 4, 1), high)
            entropy = -float(np.sum(nonzero * np.log2(np.maximum(nonzero, 1e-12))))
            entropy_norm = self._clip(entropy / np.log2(max(nonzero.size, 2)), 0.0, 1.0)
        else:
            high = 0.01
            mid = 0.001
            entropy_norm = 0.0
        policy = {
            'freq_high_threshold': high,
            'freq_mid_threshold': mid,
            'diffusion_frequent_scale': self._clip(0.45 + 0.10 * entropy_norm, 0.40, 0.65),
            'diffusion_mid_scale': self._clip(0.75 + 0.10 * entropy_norm, 0.65, 0.95),
            'diffusion_rare_scale': self._clip(1.15 + 0.15 * (1.0 - entropy_norm), 1.00, 1.35),
            'theta_base': self._clip(0.22 + 0.06 * entropy_norm, 0.18, 0.32),
            'theta_freq_gain': self._clip(0.08 + 0.05 * (1.0 - entropy_norm), 0.05, 0.16),
            'interaction_modulation_scale': self._clip(0.35 + 0.30 * entropy_norm, 0.30, 0.70),
            'kinetic_energy_weight': self._clip(0.06 + 0.08 * entropy_norm, 0.04, 0.16),
        }
        for key, value in policy.items():
            self.observe(f'field_{key}', value)
        return policy

    def report(self) -> Dict:
        stats = {}
        for key, vals in sorted(self.history.items()):
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size == 0:
                continue
            stats[key] = {
                'n': int(arr.size),
                'mean': float(np.mean(arr)),
                'q10': float(np.percentile(arr, 10)),
                'q50': float(np.percentile(arr, 50)),
                'q90': float(np.percentile(arr, 90)),
            }
        return {'type': 'adaptive_numeric_policy', 'stats': stats}


# =============================================================================
# V6 FIX #6: Inline ByteSubstrate (V6 повністю автономний)
# =============================================================================


