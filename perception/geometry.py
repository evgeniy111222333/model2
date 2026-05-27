import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from bcs.utils import _safe_normalize

class FisherInformationGeometry:
    """
    Інформаційна геометрія байтового субстрату.

    Обчислює метрику Фішера g_ij для простору байтових розподілів,
    природний градієнт для навчання, та геодезичні відстані.

    Ключові формули:
    - Метрика Фішера: g_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
    - Природний градієнт: ∇̃L = G⁻¹∇L
    - Геодезична відстань (Bhattacharyya): d(p,q) = -ln(Σ√p_i·√q_i)
    - Відстань Фішера-Рао: d(p,q) = arccos(Σ√p_i·√q_i)

    V2: Повний перехід на PyTorch Autograd для обчислення матриці Фішера
    через точний Якобіан (замість скінченних різниць).
    Забезпечує аналітичну точність та підтримку Гессіана (других похідних).
    """

    def __init__(self):
        self.G = None  # Кеш матриці Фішера
        self._G_eigenvalues = None
        self._G_condition_number = None

    # =========================================================================
    # Матриця Фішера через PyTorch Autograd
    # =========================================================================

    def compute_fisher_matrix_params(
        self,
        log_probs_fn: Callable,
        params: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Обчислення матриці Фішера через PyTorch Autograd (точний Якобіан).

        G_ij = Σ_k p(k|θ) · ∂log p(k|θ)/∂θ_i · ∂log p(k|θ)/∂θ_j
             = J^T · diag(p) · J

        де J = ∂log p / ∂θ — Якобіан, обчислений аналітично через autograd.

        Args:
            log_probs_fn: функція params → log ймовірностей.
                          Може бути torch-based (params: Tensor → Tensor)
                          або numpy-based (params: ndarray → ndarray).
                          Numpy-функції автоматично обгортаються в torch.
            params: вектор параметрів θ ∈ R^d (numpy або torch)
            eps: (ігнорується, збережено для backward compatibility)

        Returns:
            G: матриця Фішера (d, d) як numpy array
        """
        # Конвертуємо params у torch
        if isinstance(params, np.ndarray):
            params_np = params.copy()
            params_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        else:
            params_np = params.detach().cpu().numpy().copy()
            params_t = params.detach().clone().double().requires_grad_(True)

        # Визначаємо torch-compatible обгортку для log_probs_fn
        torch_fn = self._wrap_log_probs_fn(log_probs_fn, params_np, params_t)

        # Обчислюємо Якобіан J = ∂log_p / ∂θ, shape (K, d)
        J = torch.autograd.functional.jacobian(torch_fn, params_t)

        # Обчислюємо ймовірності p(k|θ)
        with torch.no_grad():
            log_p = torch_fn(params_t)
            probs = torch.exp(log_p)  # (K,)

        # G = J^T · diag(p) · J
        G_torch = J.t() @ torch.diag(probs) @ J
        G = G_torch.detach().numpy()

        self.G = G
        self._update_eigen_stats(G)
        return G

    def _wrap_log_probs_fn(
        self,
        log_probs_fn: Callable,
        params_np: np.ndarray,
        params_t: torch.Tensor,
    ) -> Callable:
        """
        Створити torch-compatible обгортку для log_probs_fn.

        Автоматично визначає тип функції:
        - Якщо функція приймає torch.Tensor і повертає torch.Tensor → використовуємо напряму
        - Якщо функція приймає np.ndarray → обгортаємо через torch.autograd.Function
        """
        # Спроба 1: прямий torch виклик
        try:
            test_result = log_probs_fn(params_t)
            if isinstance(test_result, torch.Tensor):
                # Функція вже torch-compatible
                # Перевіряємо що повертає 1D вектор
                if test_result.ndim == 1:
                    return log_probs_fn
                elif test_result.ndim == 2:
                    # (N, K) → flatten до (N*K,) для Якобіану
                    def flat_fn(theta):
                        return log_probs_fn(theta).reshape(-1)
                    return flat_fn
        except (TypeError, RuntimeError):
            pass

        # Спроба 2: numpy-based функція → обгортаємо в torch через скінченні різниці autograd
        # (для сумісності зі старими numpy-only log_probs_fn)
        d = len(params_np)
        log_p0_np = log_probs_fn(params_np)
        if hasattr(log_p0_np, 'numpy'):
            log_p0_np = log_p0_np.detach().numpy()
        if log_p0_np.ndim > 1:
            log_p0_np = log_p0_np.reshape(-1)
        K = len(log_p0_np)

        # Обчислюємо Якобіан чисельно один раз і побудуємо лінеаризовану torch-модель
        eps = 1e-5
        J_np = np.zeros((K, d), dtype=np.float64)
        for i in range(d):
            params_plus = params_np.copy()
            params_plus[i] += eps
            log_p_plus = log_probs_fn(params_plus)
            if hasattr(log_p_plus, 'numpy'):
                log_p_plus = log_p_plus.detach().numpy()
            if log_p_plus.ndim > 1:
                log_p_plus = log_p_plus.reshape(-1)
            J_np[:, i] = (log_p_plus - log_p0_np) / eps

        # Лінеаризована torch-модель: log_p(θ) ≈ log_p(θ₀) + J · (θ - θ₀)
        J_t = torch.tensor(J_np, dtype=torch.float64)
        log_p0_t = torch.tensor(log_p0_np, dtype=torch.float64)
        theta0_t = torch.tensor(params_np, dtype=torch.float64)

        def linearized_fn(theta):
            return log_p0_t + J_t @ (theta - theta0_t)

        return linearized_fn

    def _update_eigen_stats(self, G: np.ndarray):
        """Оновити кешовану статистику власних значень."""
        eigvals = np.linalg.eigvalsh(G)
        self._G_eigenvalues = eigvals
        positive_eigvals = eigvals[eigvals > 1e-10]
        if len(positive_eigvals) > 0:
            self._G_condition_number = float(positive_eigvals[-1] / positive_eigvals[0])
        else:
            self._G_condition_number = 0.0

    def compute_natural_gradient_autograd(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: Union[np.ndarray, torch.Tensor],
        log_probs_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        damping: float = 1e-3,
    ) -> np.ndarray:
        """
        Природний градієнт через PyTorch Autograd.

        ∇̃L = (G + λI)⁻¹ · ∇L

        Градієнт ∇L обчислюється аналітично через autograd.backward().
        Матриця Фішера G — або з кешу, або обчислюється через log_probs_fn.

        Args:
            loss_fn: функція params → scalar loss
            params: вектор параметрів θ
            log_probs_fn: (опціонально) для обчислення G, якщо не кешовано
            damping: Tikhonov-регуляризація λ

        Returns:
            natural_grad: природний градієнт як numpy array
        """
        # Конвертуємо params
        if isinstance(params, np.ndarray):
            params_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        else:
            params_t = params.detach().clone().double().requires_grad_(True)

        # Обчислюємо евклідів градієнт
        loss = loss_fn(params_t)
        loss.backward()
        eucl_grad = params_t.grad.detach().numpy()

        # Обчислюємо/оновлюємо матрицю Фішера якщо потрібно
        if log_probs_fn is not None:
            self.compute_fisher_matrix_params(log_probs_fn, params)

        if self.G is None:
            return eucl_grad.astype(np.float32)

        # Природний градієнт: (G + λI)⁻¹ · ∇L
        d = self.G.shape[0]
        G_reg = self.G + damping * np.eye(d, dtype=np.float64)
        try:
            natural_grad = np.linalg.solve(G_reg, eucl_grad.astype(np.float64))
            return natural_grad.astype(np.float32)
        except np.linalg.LinAlgError:
            return eucl_grad.astype(np.float32)

    # =========================================================================
    # Аналітична метрика для категоріальних розподілів
    # =========================================================================

    def compute_fisher_matrix_distributions(
        self,
        distributions: np.ndarray,
    ) -> np.ndarray:
        """
        Обчислення матриці Фішера для набору байтових розподілів.
        G = diag(p_bar) - (P^T @ P) / N
        """
        N, K = distributions.shape
        p = np.maximum(distributions, 1e-10)
        p = p / np.sum(p, axis=1, keepdims=True)

        # 1. Середній розподіл по позиціях
        p_bar = np.mean(p, axis=0)

        # 2. Векторизована коваріація через матричний добуток (outer products sum)
        P_outer = (p.T @ p) / N

        # 3. Аналітична матриця Фішера: diag(p_bar) - P_outer
        G = np.diag(p_bar) - P_outer

        self.G = G
        self._update_eigen_stats(G)
        return G

    # =========================================================================
    # Природний градієнт (legacy API, працює з кешованою G)
    # =========================================================================

    def natural_gradient(
        self,
        euclidean_grad: np.ndarray,
        damping: float = 1e-3,
    ) -> np.ndarray:
        """
        Природний градієнт: ∇̃ = (G + λI)⁻¹ ∇L

        Tikhonov-регуляризація для числової стабільності.

        Args:
            euclidean_grad: евклідів градієнт ∇L
            damping: параметр регуляризації λ

        Returns:
            natural_grad: природний градієнт
        """
        if self.G is None:
            return euclidean_grad

        G_reg = self.G + damping * np.eye(self.G.shape[0], dtype=np.float64)
        try:
            natural_grad = np.linalg.solve(G_reg, euclidean_grad.astype(np.float64))
            return natural_grad.astype(np.float32)
        except np.linalg.LinAlgError:
            return euclidean_grad

    # =========================================================================
    # Геодезичні відстані
    # =========================================================================

    def bhattacharyya_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Геодезична відстань Бхаттачар'я: d = -ln(Σ√p_i·√q_i)"""
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)
        bc = float(np.sum(np.sqrt(p * q)))
        return -np.log(max(bc, 1e-10))

    def fisher_rao_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Відстань Фішера-Рао: d = 2 * arccos(Σ√p_i·√q_i)"""
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)
        bc = float(np.sum(np.sqrt(p * q)))
        # Додано коефіцієнт 2.0 відповідно до метричної геометрії простору розподілів
        return 2.0 * float(np.arccos(np.clip(bc, 0.0, 1.0)))

    def compute_geodesic(
        self,
        p: np.ndarray,
        q: np.ndarray,
        n_steps: int = 20,
    ) -> List[np.ndarray]:
        """
        Обчислення геодезичної кривої між p та q в інформаційному многовиді.

        Використовує експоненційне відображення в просторі √p.
        """
        p = _safe_normalize(p)
        q = _safe_normalize(q)

        sqrt_p = np.sqrt(p)
        sqrt_q = np.sqrt(q)

        # Кут між p та q
        cos_angle = np.clip(np.sum(sqrt_p * sqrt_q), 0.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle < 1e-10:
            return [p.copy()]

        # Геодезична інтерполяція через сферичні координати
        geodesic = []
        for t in np.linspace(0, 1, n_steps + 1):
            # sin((1-t)θ)/sin(θ) · √p + sin(tθ)/sin(θ) · √q
            sin_angle = np.sin(angle)
            if sin_angle < 1e-10:
                interp_sqrt = sqrt_p.copy()
            else:
                interp_sqrt = (np.sin((1 - t) * angle) / sin_angle) * sqrt_p + \
                              (np.sin(t * angle) / sin_angle) * sqrt_q

            # Зворотне перетворення: p_t = (interp_sqrt)²
            p_t = interp_sqrt ** 2
            p_t = _safe_normalize(p_t)
            geodesic.append(p_t)

        return geodesic

    def fisher_rao_kmeans(
        self,
        distributions: np.ndarray,
        n_clusters: int = 5,
        max_iter: int = 30,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        K-means кластеризація з відстанню Фішера-Рао замість евклідової.

        Args:
            distributions: (N, K) розподіли
            n_clusters: кількість кластерів
            max_iter: максимальна кількість ітерацій

        Returns:
            labels: (N,) призначення кластерів
            centroids: список центроїдів
        """
        N, K = distributions.shape
        n_clusters = min(n_clusters, N)

        # Ініціалізація: вибір K рівномірно розподілених точок
        indices = np.linspace(0, N - 1, n_clusters, dtype=int)
        centroids = [distributions[i].copy() for i in indices]

        labels = np.zeros(N, dtype=int)

        for iteration in range(max_iter):
            # Призначення кластерів
            for i in range(N):
                min_dist = float('inf')
                for c_idx, centroid in enumerate(centroids):
                    d = self.fisher_rao_distance(distributions[i], centroid)
                    if d < min_dist:
                        min_dist = d
                        labels[i] = c_idx

            # Оновлення центроїдів: геодезичне середнє
            new_centroids = []
            for c_idx in range(n_clusters):
                mask = labels == c_idx
                if np.sum(mask) == 0:
                    new_centroids.append(centroids[c_idx])
                    continue

                cluster_dists = distributions[mask]
                # Геодезичне середнє: Fréchet mean через √p простір
                sqrt_dists = np.sqrt(np.maximum(cluster_dists, 1e-10))
                mean_sqrt = np.mean(sqrt_dists, axis=0)
                mean_sqrt = np.maximum(mean_sqrt, 1e-10)
                centroid_new = mean_sqrt ** 2
                centroid_new = _safe_normalize(centroid_new)
                new_centroids.append(centroid_new)

            # Перевірка конвергенції
            converged = True
            for c_idx in range(n_clusters):
                if self.fisher_rao_distance(centroids[c_idx], new_centroids[c_idx]) > 1e-4:
                    converged = False
                    break

            centroids = new_centroids
            if converged:
                break

        return labels, centroids

    def get_stats(self) -> Dict:
        """Статистика інформаційної геометрії."""
        return {
            'fisher_matrix_trace': float(np.trace(self.G)) if self.G is not None else 0.0,
            'fisher_matrix_rank': int(np.sum(np.linalg.eigvalsh(self.G) > 1e-6)) if self.G is not None else 0,
            'condition_number': self._G_condition_number or 0.0,
        }
