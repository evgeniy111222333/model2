import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union

class DynamicByteEmbedding(nn.Module):
    """
    Динамічний ембединг: h_i = φ(b_i, C_i) ∈ ℝ^d

    На відміну від трансформерів, де ембединг статичний,
    у БКС ембединг залежить від стану польової системи.

    C_i — контекстна матриця, яка залежить оточуючих байтів
    та стану поля активації.

    Реалізація:
    h_i = u_{b_i} + ContextProjection(field_state_i)
    де field_state_i = [u_i, v_i, local_entropy_i]

    V7: Конвертовано в torch.nn.Module з nn.Parameter.
    Замість ручного update() SGD → optimizer.step() з autograd.
    """

    def __init__(self, d_embedding: int = 64, n_bytes: int = 256):
        super().__init__()
        self.d_embedding = d_embedding
        self.n_bytes = n_bytes

        # Use NumPy randn to preserve seed reproducibility
        u_init = np.random.randn(n_bytes, d_embedding).astype(np.float32) * 0.05
        self.u = nn.Parameter(torch.from_numpy(u_init))

        # Контекстна проєкція: [u, v, local_entropy, local_mean, local_std] → d_embedding
        W_context_init = np.random.randn(d_embedding, 5).astype(np.float32) * 0.05
        self.W_context = nn.Parameter(torch.from_numpy(W_context_init))
        self.b_context = nn.Parameter(
            torch.zeros(d_embedding, dtype=torch.float32)
        )

        # Learned projection of the local Phi-context matrix C_i into embedding space.
        # This keeps the embedding dynamic without materializing an N x N context tensor.
        W_phi_init = np.random.randn(d_embedding, n_bytes).astype(np.float32) * 0.02
        self.W_phi_context = nn.Parameter(torch.from_numpy(W_phi_init))

        # Параметр масштабування контексту
        self.context_scale = 0.1
        self.phi_context_scale = 0.15
        self.max_context_window = 128

    def compute_embeddings(
        self,
        data: bytes,
        field_u: np.ndarray,
        field_v: np.ndarray,
        field_phi: Optional[np.ndarray] = None,
        active_byte_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Обчислити динамічні ембединги для всіх позицій.

        Args:
            data: сирі байтові дані
            field_u: u-поле (N,)
            field_v: v-поле (N,)

        Returns:
            embeddings: (N, d_embedding) матриця ембедингів
        """
        N = len(data)
        if N == 0:
            return np.zeros((0, self.d_embedding), dtype=np.float32)
        byte_values = np.frombuffer(data, dtype=np.uint8)

        # Статична частина: u_{b_i} — через torch embedding lookup
        byte_idx = torch.from_numpy(byte_values.astype(np.int64))
        static_part = self.u[byte_idx]  # (N, d_embedding) — torch tensor

        # Контекстна частина: локальна статистика поля без Python-loop по N.
        window = max(4, min(self.max_context_window, N // 50 if N > 0 else 4))
        padded_v = np.pad(field_v.astype(np.float32), (window, window), mode='edge')
        csum_v = np.concatenate([[0.0], np.cumsum(padded_v, dtype=np.float64)])
        starts = np.arange(N)
        ends = starts + 2 * window + 1
        local_sum = csum_v[ends] - csum_v[starts]
        local_mean = (local_sum / max(2 * window + 1, 1)).astype(np.float32)

        csum_v2 = np.concatenate([[0.0], np.cumsum(padded_v * padded_v, dtype=np.float64)])
        local_sum2 = csum_v2[ends] - csum_v2[starts]
        local_var = np.maximum(local_sum2 / max(2 * window + 1, 1) - local_mean.astype(np.float64) ** 2, 0.0)
        local_std = np.sqrt(local_var).astype(np.float32)

        # Binary approximation of local uncertainty. It is stable for signed u/v fields.
        local_signal = np.maximum(local_mean, 1e-6)
        local_bg = np.maximum(1.0 - np.clip(local_mean, 0.0, 1.0), 1e-6)
        p_local = (local_signal / (local_signal + local_bg)).astype(np.float32)
        one_minus_p = (np.float32(1.0) - p_local).astype(np.float32)

        log_p = np.zeros_like(p_local)
        np.log(p_local, where=(p_local > 0), out=log_p)

        log_one_minus_p = np.zeros_like(one_minus_p)
        np.log(one_minus_p, where=(one_minus_p > 0), out=log_one_minus_p)

        local_entropy = (-(p_local * log_p + one_minus_p * log_one_minus_p)).astype(np.float32)

        # Context input → torch for differentiable projection
        context_input = torch.from_numpy(
            np.stack([field_u, field_v, local_entropy, local_mean, local_std], axis=1).astype(np.float32)
        )  # (N, 5)
        dynamic_part = (context_input @ self.W_context.T + self.b_context) * self.context_scale

        phi_part = torch.zeros_like(static_part)
        if field_phi is not None and active_byte_indices is not None and len(active_byte_indices) > 0:
            phi_positive = np.maximum(field_phi.astype(np.float32), 0.0)
            phi_padded = np.pad(phi_positive, ((window, window), (0, 0)), mode='edge')
            csum_phi = np.vstack([
                np.zeros((1, phi_positive.shape[1]), dtype=np.float32),
                np.cumsum(phi_padded, axis=0, dtype=np.float32),
            ])
            local_phi = (csum_phi[2 * window + 1:2 * window + 1 + N] - csum_phi[:N])
            local_phi /= max(2 * window + 1, 1)
            local_phi_t = torch.from_numpy(local_phi.astype(np.float32))
            phi_weights = self.W_phi_context[:, active_byte_indices].T  # (K, d_embedding)
            phi_part = (local_phi_t @ phi_weights) * self.phi_context_scale

        result = static_part + dynamic_part + phi_part
        return result.detach().cpu().numpy()

    def update(self, grad: np.ndarray, lr: float = 0.001):
        """Оновити параметри ембедингу (backward-compatible legacy interface)."""
        with torch.no_grad():
            grad_t = torch.from_numpy(grad.astype(np.float32))
            if grad.shape == (self.n_bytes, self.d_embedding):
                self.u.data -= lr * grad_t
            elif grad.shape == (self.d_embedding, 5):
                self.W_context.data -= lr * grad_t


# =============================================================================
# 2. Full Tensor Interaction V6 — Рівняння (3-7)
# =============================================================================

