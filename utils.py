import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    m = 0.5 * (p + q)
    kl_pm = p * np.log(p / m)
    kl_qm = q * np.log(q / m)
    return float(0.5 * (np.sum(kl_pm) + np.sum(kl_qm)))

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    return float(np.sum(p * np.log(p / q)))
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    res = np.where(sum_exp > 1e-10, exp_x / sum_exp, 0.0)
    if np.any(sum_exp <= 1e-10):
        dim_size = x.shape[axis]
        uniform = np.ones_like(x) / max(dim_size, 1)
        res = np.where(sum_exp > 1e-10, res, uniform)
    return res

def _safe_normalize(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 1e-10)
    return p / np.sum(p)
