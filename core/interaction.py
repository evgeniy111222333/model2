import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _softmax

class FullTensorInteractionV6:
    """
    Повний тензор байтової взаємодії.

    T_{ijkm} = f(b_i, b_j, |i-j|, h_i, h_j) σ(k-b_i) σ(m-b_j)   (3)

    f = α(r) · β(b_i, b_j) · γ(h_i, h_j)                          (4)

    α(r) = exp(-r / λ(h_i, h_j))                                    (5)
    β(b_i, b_j) = u_{b_i}^T W_β u_{b_j}                            (6)
    γ(h_i, h_j) = exp(h_i^T A h_j / √d) / Σ_k exp(h_i^T A h_k / √d)  (7)

    Оптимізація: не зберігаємо повний 4D тензор,
    обчислюємо ефективне поле взаємодії W(i) для Gray-Scott.
    """

    def __init__(
        self,
        d_embedding: int = 64,
        d_beta: int = 32,
        lambda_base: float = 8.0,
        k_neighbors: int = 16,
    ):
        self.d_embedding = d_embedding
        self.d_beta = d_beta
        self.lambda_base = lambda_base
        self.k_neighbors = k_neighbors

        # Параметри, що навчаються
        self.W_beta = np.random.randn(d_beta, d_beta).astype(np.float32) * 0.05
        self.A_attention = np.random.randn(d_embedding, d_embedding).astype(np.float32) * 0.05

        # Байтові вектори для β (може відрізнятися від DynamicByteEmbedding.u)
        self.u_beta = np.random.randn(256, d_beta).astype(np.float32) * 0.05

        # Параметр адаптивної довжини взаємодії
        self.lambda_net_W = np.random.randn(1, 2 * d_embedding).astype(np.float32) * 0.01
        self.lambda_net_b = np.array([np.log(lambda_base)], dtype=np.float32)

    def compute_lambda(
        self,
        h_i: np.ndarray,
        h_j: np.ndarray,
    ) -> float:
        """Адаптивна довжина взаємодії λ(h_i, h_j)."""
        combined = np.concatenate([h_i, h_j])
        if len(combined) > self.lambda_net_W.shape[1]:
            combined = combined[:self.lambda_net_W.shape[1]]
        elif len(combined) < self.lambda_net_W.shape[1]:
            combined = np.pad(combined, (0, self.lambda_net_W.shape[1] - len(combined)))
        log_lambda = float(self.lambda_net_W @ combined + self.lambda_net_b)
        return max(np.exp(np.clip(log_lambda, 0.5, 8.0)), 1.0)

    def compute_beta(
        self,
        b_i: int,
        b_j: int,
    ) -> float:
        """Функція сумісності байтових значень β(b_i, b_j) = u^T W u."""
        u_i = self.u_beta[b_i]
        u_j = self.u_beta[b_j]
        return float(u_i @ self.W_beta @ u_j)

    def compute_gamma_vectorized(
        self,
        h_query: np.ndarray,
        h_all: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        Векторизована функція контекстної спорідненості γ.

        γ(h_i, h_j) = softmax(h_i^T A h_j / √d) для j ∈ neighbors
        """
        if len(indices) == 0:
            return np.array([], dtype=np.float32)

        d = h_all.shape[1]
        Ah_query = self.A_attention.T @ h_query  # (d,)

        # Обчислення для всіх сусідів одночасно
        h_neighbors = h_all[indices]  # (K, d)
        scores = h_neighbors @ Ah_query / np.sqrt(d)  # (K,)
        scores = np.clip(scores, -20.0, 20.0)
        return _softmax(scores)

    def deterministic_embeddings_from_bytes(self, byte_vals: np.ndarray, N: int) -> np.ndarray:
        dims = np.arange(self.d_embedding, dtype=np.float32)[None, :] + 1.0
        b = byte_vals.astype(np.float32)[:, None] + 1.0
        emb = 0.05 * np.sin(b * dims * 0.017) + 0.05 * np.cos(b * dims * 0.031)
        return emb.astype(np.float32).reshape(N, self.d_embedding)

    def compute_interaction_field(
        self,
        substrate,
        embeddings: Optional[np.ndarray] = None,
        use_adaptive_lambda: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Обчислити ефективне поле взаємодії W(i) для кожної позиції.

        W(i) = Σ_j α(r) · β(b_i, b_j) · γ(h_i, h_j) · h_j

        V6 FIX: α(r) тепер використовує адаптивне λ(h_i, h_j) замість
        фіксованого lambda_base. Згідно Рівняння (5):
        α(r) = exp(−r / λ(h_i, h_j))

        Для продуктивності: обчислюємо λ на підгрупах сусідів.

        Оптимізація: лише k_neighbors найближчих за β·γ.
        """
        data = substrate.raw_data
        N = len(data)
        if N == 0:
            return np.zeros(0, dtype=np.float32)
        byte_vals = np.frombuffer(data, dtype=np.uint8)

        if embeddings is None:
            # Fallback: статичні ембединги
            embeddings = self.deterministic_embeddings_from_bytes(byte_vals, N)

        # Попередньо обчислюємо β матрицю (256x256) — один раз
        beta_matrix = self.u_beta @ self.W_beta @ self.u_beta.T  # (256, 256)

        W_field = np.zeros(N, dtype=np.float32)

        for i in range(N):
            r_max = min(int(self.lambda_base * 3), N)
            lo = max(0, i - r_max)
            hi = min(N, i + r_max + 1)
            neighbors = np.arange(lo, hi)
            neighbors = neighbors[neighbors != i]

            if len(neighbors) == 0:
                continue

            # Відстані
            r = np.abs(neighbors - i).astype(np.float32)

            # β(b_i, b_j) — векторизовано
            bi_val = byte_vals[i]
            beta_vals = beta_matrix[bi_val, byte_vals[neighbors]]

            # V6 FIX: α(r) з АДАПТИВНИМ λ(h_i, h_j)
            # Згідно Рівняння (5): α(r) = exp(−r / λ(h_i, h_j))
            if use_adaptive_lambda:
                # Обчислюємо λ для кожного сусіда на основі ембедингів
                # Для продуктивності: групуємо по λ інтервалах
                h_i = embeddings[i]
                # Векторизоване наближення: використовуємо середнє λ по групах
                n_groups = min(4, len(neighbors))
                group_size = max(1, len(neighbors) // n_groups)
                alpha_vals = np.zeros(len(neighbors), dtype=np.float32)
                for g_idx in range(n_groups):
                    g_start = g_idx * group_size
                    g_end = min((g_idx + 1) * group_size, len(neighbors))
                    g_neighbors = neighbors[g_start:g_end]
                    if len(g_neighbors) == 0:
                        continue
                    # Беремо представника групи для обчислення λ
                    rep_j = g_neighbors[len(g_neighbors) // 2]
                    lam = self.compute_lambda(h_i, embeddings[rep_j])
                    alpha_vals[g_start:g_end] = np.exp(-r[g_start:g_end] / lam)
            else:
                alpha_vals = np.exp(-r / max(self.lambda_base, 1e-6))

            # γ — обмежуємо до k_neighbors
            if len(neighbors) > self.k_neighbors:
                scores = alpha_vals * np.maximum(beta_vals, 0)
                top_k_idx = np.argsort(scores)[-self.k_neighbors:]
                neighbors = neighbors[top_k_idx]
                alpha_vals = alpha_vals[top_k_idx]
                beta_vals = beta_vals[top_k_idx]

            gamma_vals = self.compute_gamma_vectorized(
                embeddings[i], embeddings, neighbors
            )

            # Повна взаємодія: α · β · γ
            interaction = alpha_vals * np.maximum(beta_vals, 0) * gamma_vals

            # Ефективне поле: зважена сума
            W_field[i] = float(np.sum(interaction))

        # Нормалізація
        if not np.all(np.isfinite(W_field)):
            W_field = np.nan_to_num(W_field, nan=0.0, posinf=1.0, neginf=-1.0)
        max_w = np.max(np.abs(W_field))
        if max_w > 1e-10:
            W_field = W_field / max_w

        return W_field




class FFTSpaceValueInteractionV7:
    """
    Повний тензор взаємодії V7 з підтримкою 2D вибіркового спрямування (value-specific steering)
    та швидких Фур'є-згорток (FFT).

    Реалізує два режими:
    - 'dense_bilinear': Точне обчислення 2D-поля сумісності через просторову увагу:
      W_interaction = M * V, де V = Phi * B_active^T, B = u_beta * W_beta * u_beta^T.
    - 'fft_separable': Обчислення просторової взаємодії через FFT-згортку по просторовій осі.
    """

    def __init__(
        self,
        d_embedding: int = 64,
        d_beta: int = 32,
        lambda_base: float = 8.0,
        k_neighbors: int = 16,
        mode: str = 'dense_bilinear',
        r_rank: int = 4,
    ):
        self.d_embedding = d_embedding
        self.d_beta = d_beta
        self.lambda_base = lambda_base
        self.k_neighbors = k_neighbors
        self.mode = mode
        self.r_rank = r_rank
        self.adaptive_mix = 0.65

        # Параметри, що навчаються
        self.W_beta = np.random.randn(d_beta, d_beta).astype(np.float32) * 0.05
        self.A_attention = np.random.randn(d_embedding, d_embedding).astype(np.float32) * 0.05
        self.u_beta = np.random.randn(256, d_beta).astype(np.float32) * 0.05
        self.B_residual = np.zeros((256, 256), dtype=np.float32)

        # Параметри для адаптивної довжини взаємодії
        self.lambda_net_W = np.random.randn(1, 2 * d_embedding).astype(np.float32) * 0.01
        self.lambda_net_b = np.array([np.log(lambda_base)], dtype=np.float32)

    def compute_lambda(self, h_i: np.ndarray, h_j: np.ndarray) -> float:
        combined = np.concatenate([h_i, h_j])
        if len(combined) > self.lambda_net_W.shape[1]:
            combined = combined[:self.lambda_net_W.shape[1]]
        elif len(combined) < self.lambda_net_W.shape[1]:
            combined = np.pad(combined, (0, self.lambda_net_W.shape[1] - len(combined)))
        log_lambda = float(self.lambda_net_W @ combined + self.lambda_net_b)
        return max(np.exp(np.clip(log_lambda, 0.5, 8.0)), 1.0)

    def compute_gamma_vectorized(
        self,
        h_query: np.ndarray,
        h_all: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=np.float32)
        d = h_all.shape[1]
        Ah_query = self.A_attention.T @ h_query
        h_neighbors = h_all[indices]
        scores = h_neighbors @ Ah_query / np.sqrt(d)
        scores = np.clip(scores, -20.0, 20.0)
        return _softmax(scores)

    def deterministic_embeddings_from_bytes(self, byte_vals: np.ndarray, N: int) -> np.ndarray:
        """Deterministic fallback when no DynamicByteEmbedding is configured."""
        dims = np.arange(self.d_embedding, dtype=np.float32)[None, :] + 1.0
        b = byte_vals.astype(np.float32)[:, None] + 1.0
        emb = 0.05 * np.sin(b * dims * 0.017) + 0.05 * np.cos(b * dims * 0.031)
        return emb.astype(np.float32).reshape(N, self.d_embedding)

    def compute_beta_matrix(self) -> np.ndarray:
        """Full 256x256 value compatibility: low-rank beta plus residual table."""
        B = self.u_beta @ self.W_beta @ self.u_beta.T
        if hasattr(self, 'B_residual') and self.B_residual is not None:
            B = B + self.B_residual
        return B.astype(np.float32)

    def compute_lambda_vectorized(self, h_i: np.ndarray, h_neighbors: np.ndarray) -> np.ndarray:
        if h_neighbors.size == 0:
            return np.array([], dtype=np.float32)
        d = min(h_neighbors.shape[1], self.d_embedding)
        w = self.lambda_net_W.reshape(-1)
        log_lambda = (
            h_i[:d] @ w[:d]
            + h_neighbors[:, :d] @ w[self.d_embedding:self.d_embedding + d]
            + float(self.lambda_net_b[0])
        )
        lambdas = np.exp(np.clip(log_lambda, 0.5, 8.0))
        return np.maximum(lambdas, 1.0).astype(np.float32)

    def compute_adaptive_pair_field(
        self,
        V: np.ndarray,
        embeddings: np.ndarray,
        use_adaptive_lambda: bool = True,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """Fully vectorized sliding-window pair field calculation in PyTorch to avoid CPU/GPU OOM."""
        N = V.shape[0]
        if N == 0:
            return np.zeros_like(V)

        device = "cpu"
        # Convert inputs to PyTorch tensors
        V_t = torch.from_numpy(V.astype(np.float32)).to(device)
        emb_t = torch.from_numpy(embeddings.astype(np.float32)).to(device)
        
        d = emb_t.shape[1]
        K = V_t.shape[1]
        r_max = min(max(1, int(self.lambda_base * 3)), max(N - 1, 1))

        # Create window index tensor: (N, 2 * r_max + 1)
        offsets = torch.arange(-r_max, r_max + 1, dtype=torch.long, device=device)
        idx = torch.arange(N, dtype=torch.long, device=device).unsqueeze(1) + offsets.unsqueeze(0)
        
        # Mask valid neighbors (those inside sequence bounds)
        valid_mask = (idx >= 0) & (idx < N)
        idx_clamped = torch.clamp(idx, 0, N - 1)

        # Distance matrix (N, 2 * r_max + 1)
        r = torch.abs(offsets.float().unsqueeze(0)).repeat(N, 1)

        # Retrieve neighbor representations
        h_neighbors = emb_t[idx_clamped]  # (N, 2 * r_max + 1, d)
        V_neighbors = V_t[idx_clamped]    # (N, 2 * r_max + 1, K)
        h_i = emb_t.unsqueeze(1)          # (N, 1, d)

        # Compute adaptive lambda decay
        W_beta_interaction = self.lambda_net_W if isinstance(self.lambda_net_W, torch.Tensor) else torch.from_numpy(self.lambda_net_W).to(device)
        b_beta_interaction = self.lambda_net_b if isinstance(self.lambda_net_b, torch.Tensor) else torch.from_numpy(self.lambda_net_b).to(device)
        
        if use_adaptive_lambda:
            term_i = torch.matmul(h_i, W_beta_interaction[0, :d].unsqueeze(1)).squeeze(-1)  # (N, 1)
            term_j = torch.matmul(h_neighbors, W_beta_interaction[0, d:2*d].unsqueeze(1)).squeeze(-1)  # (N, 2 * r_max + 1)
            log_lambda = term_i + term_j + b_beta_interaction[0]
            lambdas = torch.exp(torch.clamp(log_lambda, 0.5, 8.0))
            alpha = torch.exp(-r / torch.clamp(lambdas, min=1e-6))
        else:
            alpha = torch.exp(-r / max(self.lambda_base, 1e-6))

        # Compute contextual affinity gamma
        A = self.A_attention if isinstance(self.A_attention, torch.Tensor) else torch.from_numpy(self.A_attention).to(device)
        Ah_i = torch.matmul(h_i, A)  # (N, 1, d)
        scores = torch.sum(Ah_i * h_neighbors, dim=-1) / np.sqrt(d)  # (N, 2 * r_max + 1)
        scores = torch.clamp(scores, -20.0, 20.0)
        
        # Softmax over window dimension with validity mask
        scores = torch.where(valid_mask, scores, torch.tensor(-1e9, dtype=torch.float32, device=device))
        gamma = torch.softmax(scores, dim=-1)

        # Weights: alpha * gamma
        weights = alpha * gamma
        # Mask out self-interaction
        weights[:, r_max] = 0.0
        weights = torch.where(valid_mask, weights, torch.tensor(0.0, dtype=torch.float32, device=device))

        row_sum = weights.sum(dim=-1, keepdim=True)
        normalized_weights = torch.where(row_sum > 1e-10, weights / row_sum, torch.zeros_like(weights))

        # Matrix multiply: sum over neighbors
        W_adaptive = torch.bmm(normalized_weights.unsqueeze(1), V_neighbors).squeeze(1)
        if not torch.all(torch.isfinite(W_adaptive)):
            W_adaptive = torch.nan_to_num(W_adaptive, nan=0.0, posinf=1.0, neginf=-1.0)

        return W_adaptive.detach().cpu().numpy()

    def compute_interaction_field(
        self,
        substrate,
        embeddings: np.ndarray,
        use_adaptive_lambda: bool = True,
        field = None,
    ) -> np.ndarray:
        data = substrate.raw_data
        N = len(data)
        if N == 0:
            n_active = len(field.active_byte_indices) if field is not None else 256
            return np.zeros((0, n_active), dtype=np.float32)
        byte_vals = np.frombuffer(data, dtype=np.uint8)

        if embeddings is None:
            embeddings = self.deterministic_embeddings_from_bytes(byte_vals, N)

        # 1. Спільна матриця когнітивної сумісності B (256, 256)
        B = self.compute_beta_matrix()  # (256, 256)

        # 2. Отримання стану поля Phi та active_indices
        if field is not None:
            Phi = field.Phi
            active_indices = field.active_byte_indices
        else:
            # Fallback для ініціалізації
            active_indices = np.arange(256)
            Phi = np.zeros((N, 256), dtype=np.float32)
            theta = 0.5
            sqrt_theta = np.sqrt(theta)
            for i in range(N):
                Phi[i, :] = -sqrt_theta
                Phi[i, byte_vals[i]] = sqrt_theta

        # Враховуємо підмножину B для активних індексів
        B_active = B[active_indices][:, active_indices]  # (n_active, n_active)
        # Only active (positive) activations propagate the interaction field
        V = np.maximum(Phi, 0.0) @ B_active.T  # (N, n_active)

        if self.mode == 'fft_separable':
            # Одновимірний зсув згасання уздовж позицій
            r_indices = np.arange(N)
            # Симетричний фільтр
            x = np.exp(-np.minimum(r_indices, N - r_indices) / max(self.lambda_base, 1e-6)).astype(np.float32)
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x_sum = x.sum()
            if x_sum > 1e-10:
                x /= x_sum
            X_fft = np.fft.fft(x)

            W_interaction = np.zeros_like(V)
            V_clean = V
            if not np.all(np.isfinite(V_clean)):
                V_clean = np.nan_to_num(V_clean, nan=0.0, posinf=1.0, neginf=-1.0)
            for k in range(V_clean.shape[1]):
                Vk_fft = np.fft.fft(V_clean[:, k])
                W_interaction[:, k] = np.real(np.fft.ifft(X_fft * Vk_fft))
            if self.adaptive_mix > 1e-8:
                W_adaptive = self.compute_adaptive_pair_field(
                    V,
                    embeddings,
                    use_adaptive_lambda=use_adaptive_lambda,
                    top_k=self.k_neighbors,
                )
                W_interaction = ((1.0 - self.adaptive_mix) * W_interaction
                                 + self.adaptive_mix * W_adaptive)
        else:
            # Dense/adaptive mode as a lazy operator, avoiding explicit N x N storage.
            W_interaction = self.compute_adaptive_pair_field(
                V,
                embeddings,
                use_adaptive_lambda=use_adaptive_lambda,
                top_k=None,
            )

        # Signed interaction -> concept field in [0, 1], with 0.5 as neutral.
        if not np.all(np.isfinite(W_interaction)):
            W_interaction = np.nan_to_num(W_interaction, nan=0.0, posinf=1.0, neginf=-1.0)
        max_w = np.max(np.abs(W_interaction))
        if max_w > 1e-10:
            W_interaction = 0.5 + 0.5 * (W_interaction / max_w)
        else:
            W_interaction = np.ones_like(W_interaction, dtype=np.float32) * 0.5

        return W_interaction




class TorchSpaceValueInteractionV8(torch.nn.Module):
    """
    Повний тензор взаємодії V8 на базі PyTorch.
    Векторизована глобальна взаємодія з адаптивними параметрами без циклів Python.
    """

    def __init__(
        self,
        d_embedding: int = 64,
        d_beta: int = 32,
        lambda_base: float = 8.0,
        k_neighbors: int = 16,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_embedding = d_embedding
        self.d_beta = d_beta
        self.lambda_base = lambda_base
        self.k_neighbors = k_neighbors
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Параметри, що навчаються
        self.W_beta = torch.nn.Parameter(
            torch.randn(d_beta, d_beta, dtype=torch.float32) * 0.05
        )
        self.A_attention = torch.nn.Parameter(
            torch.randn(d_embedding, d_embedding, dtype=torch.float32) * 0.05
        )
        self.u_beta = torch.nn.Parameter(
            torch.randn(256, d_beta, dtype=torch.float32) * 0.05
        )
        self.B_residual = torch.nn.Parameter(
            torch.zeros(256, 256, dtype=torch.float32)
        )

        self.lambda_net_W = torch.nn.Parameter(
            torch.randn(1, 2 * d_embedding, dtype=torch.float32) * 0.01
        )
        self.lambda_net_b = torch.nn.Parameter(
            torch.tensor([np.log(lambda_base)], dtype=torch.float32)
        )

        self.to(self.device)

    def forward(
        self,
        V: torch.Tensor,
        embeddings: torch.Tensor,
        use_adaptive_lambda: bool = True,
    ) -> torch.Tensor:
        """
        Векторизований прямий прохід з O(N * r_max) пам'яттю (sliding window).
        """
        N = V.size(0)
        d = embeddings.size(1)
        K = V.size(1)
        device = self.device
        
        r_max = min(max(1, int(self.lambda_base * 3)), max(N - 1, 1))

        # Create window index tensor: (N, 2 * r_max + 1)
        offsets = torch.arange(-r_max, r_max + 1, dtype=torch.long, device=device)
        idx = torch.arange(N, dtype=torch.long, device=device).unsqueeze(1) + offsets.unsqueeze(0)
        
        valid_mask = (idx >= 0) & (idx < N)
        idx_clamped = torch.clamp(idx, 0, N - 1)

        r = torch.abs(offsets.float().unsqueeze(0)).repeat(N, 1)

        h_neighbors = embeddings[idx_clamped]  # (N, 2 * r_max + 1, d)
        V_neighbors = V[idx_clamped]    # (N, 2 * r_max + 1, K)
        h_i = embeddings.unsqueeze(1)          # (N, 1, d)

        if use_adaptive_lambda:
            term_i = torch.matmul(h_i, self.lambda_net_W[0, :d].unsqueeze(1)).squeeze(-1)  # (N, 1)
            term_j = torch.matmul(h_neighbors, self.lambda_net_W[0, d:2*d].unsqueeze(1)).squeeze(-1)  # (N, 2 * r_max + 1)
            log_lambda = term_i + term_j + self.lambda_net_b[0]
            lambdas = torch.exp(torch.clamp(log_lambda, 0.5, 8.0))
            alpha = torch.exp(-r / torch.clamp(lambdas, min=1e-6))
        else:
            alpha = torch.exp(-r / max(self.lambda_base, 1e-6))

        Ah_i = torch.matmul(h_i, self.A_attention)  # (N, 1, d)
        scores = torch.sum(Ah_i * h_neighbors, dim=-1) / np.sqrt(max(d, 1))  # (N, 2 * r_max + 1)
        scores = torch.clamp(scores, -20.0, 20.0)
        
        scores = torch.where(valid_mask, scores, torch.tensor(-1e9, dtype=torch.float32, device=device))
        gamma = torch.softmax(scores, dim=-1)

        weights = alpha * gamma
        weights[:, r_max] = 0.0
        weights = torch.where(valid_mask, weights, torch.tensor(0.0, dtype=torch.float32, device=device))

        row_sum = weights.sum(dim=-1, keepdim=True)
        normalized_weights = torch.where(row_sum > 1e-10, weights / row_sum, torch.zeros_like(weights))

        W_interaction = torch.bmm(normalized_weights.unsqueeze(1), V_neighbors).squeeze(1)
        if not torch.all(torch.isfinite(W_interaction)):
            W_interaction = torch.nan_to_num(W_interaction, nan=0.0, posinf=1.0, neginf=-1.0)
        return W_interaction

    def compute_interaction_field(
        self,
        substrate,
        embeddings_np: np.ndarray,
        use_adaptive_lambda: bool = True,
        field = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Обгортка для сумісності з NumPy кодом BCSModelV6.
        """
        data = substrate.raw_data
        N = len(data)
        if N == 0:
            n_active = len(field.active_byte_indices) if field is not None else 256
            return np.zeros((0, n_active), dtype=np.float32)
        byte_vals = np.frombuffer(data, dtype=np.uint8)

        # 1. Ембединги
        if embeddings_np is None:
            dims = np.arange(self.d_embedding, dtype=np.float32)[None, :] + 1.0
            b = byte_vals.astype(np.float32)[:, None] + 1.0
            emb = 0.05 * np.sin(b * dims * 0.017) + 0.05 * np.cos(b * dims * 0.031)
            embeddings_np = emb.astype(np.float32).reshape(N, self.d_embedding)

        # 2. Propagated activations (V)
        # B = u_beta @ W_beta @ u_beta.T + B_residual
        B = torch.matmul(self.u_beta, torch.matmul(self.W_beta, self.u_beta.T)) + self.B_residual
        if not torch.all(torch.isfinite(B)):
            B = torch.nan_to_num(B, nan=0.0, posinf=1.0, neginf=-1.0)

        if field is not None:
            Phi_np = field.Phi
            active_indices = field.active_byte_indices
        else:
            active_indices = np.arange(256)
            Phi_np = np.zeros((N, 256), dtype=np.float32)
            theta = 0.5
            sqrt_theta = np.sqrt(theta)
            for i in range(N):
                Phi_np[i, :] = -sqrt_theta
                Phi_np[i, byte_vals[i]] = sqrt_theta

        Phi = torch.tensor(Phi_np, device=self.device, dtype=torch.float32)
        
        # Propagate only positive activations
        V = torch.matmul(
            torch.clamp(Phi, min=0.0), 
            B[active_indices][:, active_indices].T
        )

        # Convert embeddings_np to torch
        embeddings = torch.tensor(embeddings_np, device=self.device, dtype=torch.float32)

        # Compute
        W_interaction = self.forward(V, embeddings, use_adaptive_lambda)

        # Normalize (signed interaction -> [0, 1] with 0.5 as neutral)
        if not torch.all(torch.isfinite(W_interaction)):
            W_interaction = torch.nan_to_num(W_interaction, nan=0.0, posinf=1.0, neginf=-1.0)
        max_w = torch.max(torch.abs(W_interaction))
        if max_w > 1e-10:
            W_interaction = 0.5 + 0.5 * (W_interaction / max_w)
        else:
            W_interaction = torch.ones_like(W_interaction) * 0.5

        return W_interaction.detach().cpu().numpy()



# =============================================================================
# 2b. Prediction Error Loop — Фрістонівська мінімізація вільної енергії
# =============================================================================


