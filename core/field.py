import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.core.policy import AdaptiveNumericPolicy
from bcs.utils import _sigmoid

class PredictionErrorLoop:
    """
    Цикл помилки передбачення (Prediction Error Loop) — реалізація
    принципу вільної енергії Фрістона для БКС.

    V8 Byte-Grounded PEL:
    Замість самореферентного передбачення u з u, PEL тепер реалізує
    ПРАВИЛЬНИЙ принцип вільної енергії: передбачує ЗОВНІШНІ спостереження
    (байти субстрату) з ВНУТРІШНЬОГО стану поля (Φ контекст).

    Архітектура:
        Спостереження:  byte[i] ∈ {active_bytes}     ← ЗОВНІШНІЙ сигнал (фіксований!)
                            ↑
        Генеративна модель: P(byte[i] | Φ[i-k:i+k, :])  ← PEL навчає це
                            ↑
        Внутрішній стан:   Φ(i,k) поле               ← Allen-Cahn еволюціонує

    Friston's Free Energy Principle:
    F = CE(byte_observed, byte_predicted) + λ·||W||² → min

    Де:
    - CE = Cross-Entropy між спостереженими байтами та передбаченнями
    - complexity = ||W||² — складність моделі
    - Мінімізація F через градієнтний спуск одночасно:
      1) Оновлює параметри моделі (Linear) → точніші передбачення байтів
      2) Коригує стан поля Φ → краще відповідає спостереженням

    КЛЮЧОВА ПЕРЕВАГА: ціль (байти) НІКОЛИ не змінюється, тому PEL
    навчається на стаціонарній цілі, на відміну від попередньої версії
    де u дрейфувала в 2× через Allen-Cahn динаміку.
    """

    def __init__(
        self,
        n_active_bytes: int = 256,
        active_byte_indices: Optional[np.ndarray] = None,
        context_size: int = 8,
        learning_rate: float = 0.01,
        field_correction_rate: float = 0.001,
        complexity_weight: float = 0.01,
    ):
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.field_correction_rate = field_correction_rate
        self.complexity_weight = complexity_weight
        self.n_active_bytes = n_active_bytes

        # Маппінг: byte_value → class_index для categorical prediction
        # active_byte_indices[class_idx] = byte_value
        if active_byte_indices is not None:
            self.active_byte_indices = active_byte_indices.copy()
        else:
            self.active_byte_indices = np.arange(n_active_bytes, dtype=np.int64)

        # Зворотній маппінг: byte_value → class_index
        self._byte_to_class = np.full(256, -1, dtype=np.int64)
        for cls_idx, byte_val in enumerate(self.active_byte_indices):
            self._byte_to_class[byte_val] = cls_idx

        # --- Модель: Linear(Φ_context_flat → n_active_bytes logits) ---
        # Input: Φ[i-k:i+k, :] flattened = (2*context_size + 1) * n_active_bytes
        input_dim = (2 * context_size + 1) * n_active_bytes
        self._linear = nn.Linear(input_dim, n_active_bytes, bias=True)
        # Xavier init for better convergence
        nn.init.xavier_uniform_(self._linear.weight)
        nn.init.zeros_(self._linear.bias)

        self._optimizer = torch.optim.SGD(
            self._linear.parameters(), lr=learning_rate, momentum=0.9
        )
        self._ce_loss = nn.CrossEntropyLoss(reduction='none')  # per-position loss

        # Історія для моніторингу
        self.prediction_error_history = []
        self.free_energy_history = []

        # Backward compatibility: expose a weights property
        self._weights_param = self._linear.weight  # for external access

    @property
    def weights(self) -> np.ndarray:
        """Backward-compatible: return weight matrix as flat numpy array."""
        return self._linear.weight.detach().cpu().numpy().ravel()

    @weights.setter
    def weights(self, value: np.ndarray):
        """Backward-compatible setter (reshapes to weight matrix)."""
        with torch.no_grad():
            w = np.asarray(value, dtype=np.float32)
            if w.shape == self._linear.weight.shape:
                self._linear.weight.copy_(torch.from_numpy(w))

    def _build_phi_windows(self, Phi: np.ndarray) -> torch.Tensor:
        """
        Build sliding windows over Phi field for context extraction.

        Args:
            Phi: (N, n_active_bytes) field state

        Returns:
            windows_flat: (N, (2k+1)*n_active_bytes) — flattened Phi context
        """
        half = self.context_size
        N = Phi.shape[0]
        n_bytes = Phi.shape[1]

        # Pad along position axis
        Phi_padded = np.pad(Phi, ((half, half), (0, 0)), mode='edge')  # (N+2k, n_bytes)

        # Build sliding windows: (N, 2k+1, n_bytes)
        windows = np.lib.stride_tricks.sliding_window_view(
            Phi_padded, (2 * half + 1, n_bytes)
        ).squeeze(axis=1).copy()  # (N, 2k+1, n_bytes)

        # Zero out center position (don't cheat by looking at self)
        windows[:, half, :] = 0.0

        # Flatten: (N, (2k+1)*n_bytes)
        windows_flat = windows.reshape(N, -1)
        return torch.from_numpy(windows_flat.astype(np.float32))

    def _bytes_to_targets(self, byte_values: np.ndarray) -> torch.Tensor:
        """
        Convert raw byte values to class indices using active_byte_indices mapping.

        Args:
            byte_values: (N,) uint8 array of raw byte values

        Returns:
            targets: (N,) int64 tensor of class indices
        """
        class_indices = self._byte_to_class[byte_values]
        # Handle unmapped bytes: assign to class 0 (fallback)
        class_indices[class_indices < 0] = 0
        return torch.from_numpy(class_indices.astype(np.int64))

    def predict(self, Phi_or_u: np.ndarray, field_system=None) -> np.ndarray:
        """
        Генерація передбачень для кожної позиції.

        V8: Якщо передано Phi (2D), передбачує класи байтів.
        Backward-compatible: якщо передано u (1D), повертає u-predictions
        (для зворотної сумісності з PredictiveCoding).

        Returns:
            predictions: (N,) — predicted class indices або u-values
        """
        # Backward compatibility: 1D u_field → old behavior (simple mean predictor)
        if Phi_or_u.ndim == 1:
            # Fallback: return input as-is (trivial prediction)
            return Phi_or_u.copy()

        Phi = Phi_or_u
        N = Phi.shape[0]
        if N == 0:
            return np.zeros(0, dtype=np.float32)

        with torch.no_grad():
            windows = self._build_phi_windows(Phi)
            logits = self._linear(windows)  # (N, n_classes)
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
        return predicted_classes.astype(np.float32)

    def compute_prediction_error(
        self, Phi_or_u: np.ndarray, byte_targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Обчислити помилку передбачення та вільну енергію.

        V8: Cross-entropy між передбаченими та реальними байтами.

        Returns:
            errors: (N,) per-position CE loss
            predictions: (N,) predicted class indices
            free_energy: F = mean(CE) + λ·||W||²
        """
        # Backward compatibility: 1D input
        if Phi_or_u.ndim == 1:
            predictions = self.predict(Phi_or_u)
            errors = Phi_or_u - predictions
            accuracy = float(np.mean(errors ** 2))
            with torch.no_grad():
                complexity = sum(float(torch.sum(p ** 2).item()) for p in self._linear.parameters())
            return errors, predictions, accuracy + self.complexity_weight * complexity

        Phi = Phi_or_u
        N = Phi.shape[0]
        if N == 0 or byte_targets is None:
            return np.zeros(0), np.zeros(0), 0.0

        with torch.no_grad():
            windows = self._build_phi_windows(Phi)
            logits = self._linear(windows)  # (N, n_classes)
            targets_t = self._bytes_to_targets(byte_targets)
            per_pos_loss = self._ce_loss(logits, targets_t)  # (N,)

            errors = per_pos_loss.cpu().numpy().astype(np.float32)
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy().astype(np.float32)

            accuracy = float(per_pos_loss.mean().item())
            complexity = sum(float(torch.sum(p ** 2).item()) for p in self._linear.parameters())
            free_energy = accuracy + self.complexity_weight * complexity

        return errors, predicted_classes, free_energy

    def update_model(
        self, Phi_or_u: np.ndarray, byte_targets: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Оновити параметри предиктивної моделі через PyTorch autograd.

        V8: Навчає Linear(Φ_context → byte_class) через cross-entropy.

        Returns:
            free_energy: значення вільної енергії після оновлення
            errors: (N,) per-position CE loss
        """
        # Backward compatibility: 1D input → old u-based learning
        if Phi_or_u.ndim == 1:
            # Simplified: just track error without modifying model
            errors, _, fe = self.compute_prediction_error(Phi_or_u)
            self.prediction_error_history.append(float(np.mean(errors ** 2)))
            self.free_energy_history.append(fe)
            return fe, errors

        Phi = Phi_or_u
        N = Phi.shape[0]
        if N == 0 or byte_targets is None:
            return 0.0, np.array([], dtype=np.float32)

        # Build input features
        windows = self._build_phi_windows(Phi)  # (N, input_dim)
        targets_t = self._bytes_to_targets(byte_targets)  # (N,)

        # Forward pass
        self._optimizer.zero_grad()
        logits = self._linear(windows)  # (N, n_classes)
        per_pos_loss = self._ce_loss(logits, targets_t)  # (N,)

        # Total loss = mean CE + L2 regularization
        ce_mean = per_pos_loss.mean()
        l2_reg = sum(torch.sum(p ** 2) for p in self._linear.parameters())
        loss = ce_mean + self.complexity_weight * l2_reg
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self._linear.parameters(), max_norm=1.0)

        # SGD step
        self._optimizer.step()

        # Recompute after update
        with torch.no_grad():
            new_logits = self._linear(windows)
            new_per_pos_loss = self._ce_loss(new_logits, targets_t)
            new_errors = new_per_pos_loss.cpu().numpy().astype(np.float32)
            new_ce = float(new_per_pos_loss.mean().item())
            new_l2 = sum(float(torch.sum(p ** 2).item()) for p in self._linear.parameters())
            new_fe = new_ce + self.complexity_weight * new_l2

        self.prediction_error_history.append(float(np.mean(new_errors)))
        self.free_energy_history.append(new_fe)

        return new_fe, new_errors

    def correct_field(
        self, field_system, byte_targets: np.ndarray
    ) -> np.ndarray:
        """
        Коригувати стан поля Φ на основі помилки передбачення байтів.

        V8: Gradient-based field correction — обчислює dCE/dΦ через autograd
        і зсуває Φ в напрямку кращого передбачення спостережень.

        За Фрістоном: "зміни внутрішній стан (Φ) так, щоб він краще
        пояснював зовнішні спостереження (байти)."

        Returns:
            corrections: (N, n_active_bytes) застосовані корекції
        """
        if not hasattr(field_system, 'Phi') or field_system.Phi is None:
            return np.zeros(0, dtype=np.float32)

        N = field_system.N
        n_bytes = field_system.n_active_bytes

        if len(byte_targets) != N:
            return np.zeros((N, n_bytes), dtype=np.float32)

        half = self.context_size
        Phi_np = field_system.Phi.astype(np.float32)

        # Build Phi windows WITH gradient tracking on Phi
        Phi_padded = np.pad(Phi_np, ((half, half), (0, 0)), mode='edge')
        windows_np = np.lib.stride_tricks.sliding_window_view(
            Phi_padded, (2 * half + 1, n_bytes)
        ).squeeze(axis=1).copy()
        windows_np[:, half, :] = 0.0

        # Create differentiable Phi tensor
        Phi_t = torch.from_numpy(Phi_np).requires_grad_(True)

        # Reconstruct windows from Phi_t for autograd
        # We need the gradient to flow back to Phi_t, so we rebuild windows
        Phi_padded_t = torch.nn.functional.pad(
            Phi_t.unsqueeze(0), (0, 0, half, half), mode='replicate'
        ).squeeze(0)  # (N+2k, n_bytes)

        # Extract windows manually for autograd
        indices = torch.arange(N).unsqueeze(1) + torch.arange(2 * half + 1).unsqueeze(0)  # (N, 2k+1)
        windows_t = Phi_padded_t[indices]  # (N, 2k+1, n_bytes)
        # Zero center
        mask = torch.ones(2 * half + 1, dtype=torch.float32)
        mask[half] = 0.0
        windows_t = windows_t * mask[None, :, None]

        windows_flat = windows_t.reshape(N, -1)  # (N, input_dim)

        # Forward through frozen model
        with torch.no_grad():
            linear_weight = self._linear.weight.detach().clone()
            linear_bias = self._linear.bias.detach().clone()

        logits = torch.nn.functional.linear(windows_flat, linear_weight, linear_bias)
        targets_t = self._bytes_to_targets(byte_targets)
        loss = torch.nn.functional.cross_entropy(logits, targets_t)

        # Compute gradient w.r.t. Phi
        grad_Phi = torch.autograd.grad(loss, Phi_t, retain_graph=False)[0]
        if not torch.all(torch.isfinite(grad_Phi)):
            grad_Phi = torch.nan_to_num(grad_Phi, nan=0.0, posinf=10.0, neginf=-10.0)
        # Clamp gradient for stability
        grad_Phi = torch.clamp(grad_Phi, -10.0, 10.0)

        # Apply corrections: move Phi to reduce byte prediction error
        corrections = -self.field_correction_rate * grad_Phi.cpu().numpy()
        field_system.Phi = np.clip(field_system.Phi + corrections, -1.5, 2.0)

        # Recompute derived fields u, v
        Phi_positive = np.maximum(field_system.Phi, 0.0)
        field_system.u = np.sum(Phi_positive, axis=1).astype(np.float32)
        field_system.v = np.max(Phi_positive, axis=1).astype(np.float32)

        return corrections

    def step(
        self, field_system, byte_targets: np.ndarray
    ) -> Dict:
        """
        Повний крок Prediction Error Loop (V8 Byte-Grounded):

        1. Будуємо Φ-контекст для кожної позиції
        2. Передбачуємо байт через Linear(Φ_context)
        3. Обчислюємо Cross-Entropy з реальними байтами
        4. Оновлюємо модель (Linear weights)
        5. Коригуємо стан поля Φ через dCE/dΦ

        Args:
            field_system: FieldSystemV6 з поточним станом Φ
            byte_targets: (N,) uint8 — реальні байти субстрату

        Returns:
            dict з free_energy, mean_error, corrections_applied
        """
        # 1-4: Оновлення моделі на основі байтових спостережень
        free_energy, errors = self.update_model(field_system.Phi, byte_targets)

        # 5: Корекція поля через gradient dCE/dΦ
        corrections = self.correct_field(field_system, byte_targets)

        # Accuracy: % правильно передбачених байтів
        with torch.no_grad():
            windows = self._build_phi_windows(field_system.Phi)
            logits = self._linear(windows)
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            targets_cls = self._byte_to_class[byte_targets]
            accuracy = float(np.mean(predicted == targets_cls))

        return {
            'free_energy': free_energy,
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)) if len(errors) > 0 else 0.0,
            'corrections_applied': float(np.mean(np.abs(corrections))) if corrections.size > 0 else 0.0,
            'byte_accuracy': accuracy,
        }


# =============================================================================
# 3. Field System V6 — Рівняння (9-11) з double-well potential
# =============================================================================



class FieldSystemV6:
    """
    Польова система з per-byte-value дифузією та double-well потенціалом.

    ∂Φ(i,k,t)/∂t = D_k Σ_{j∈N(i)} [Φ(j,k,t)−Φ(i,k,t)] + R_k(Φ,θ_i) − μΦ(i,k,t)  (9)

    R_k(ϕ,θ) = −∂E_k/∂ϕ_k                                                            (10)

    E_k(ϕ_k, θ_k) = a_k(ϕ_k² − θ_k)²                                                 (11)

    V6 FIX: ОСНОВНА динаміка — Allen-Cahn double-well (Рівняння 9-11).
    Gray-Scott більше НЕ є базовою моделлю. Φ(i,k) — ПЕРВИННЕ поле
    для ВСІХ 256 байтових значень (n_active_bytes=256 за замовчуванням).

    Похідні поля u та v обчислюються АГРЕГАЦІЄЮ з Φ:
    - u(i) = Σ_k Φ(i,k) — агрегована активація
    - v(i) = max_k Φ(i,k) — максимальна активація (інгібітор)

    Це відповідає концепції: Φ є фундаментальним полем, u/v — похідні.
    """

    def __init__(
        self,
        substrate,
        D_u: float = 0.008,   # CONCEPT FIX: зменшено з 0.08, бо Laplacian тепер
                               # використовує Σ замість MEAN (в 2*ns разів сильніший)
        D_v: float = 0.04,    # CONCEPT FIX: пропорційно зменшено
        F_base: float = 0.035,
        k_base: float = 0.060,
        dt: float = 0.1,      # CONCEPT FIX: зменшено з 1.0 для стабільності
                               # (CFL: dt·D·(2·ns)² має бути < 1)
        neighborhood_size: int = 5,
        interaction_field: Optional[np.ndarray] = None,
        n_active_bytes: int = 256,  # V6 FIX: за замовчуванням ВСІ 256 байтів
        numeric_policy: Optional[AdaptiveNumericPolicy] = None,
    ):
        self.substrate = substrate
        self.N = substrate.length
        self.D_u = D_u
        self.D_v = D_v
        self.F_base = F_base
        self.k_base = k_base
        self.dt = dt
        self.neighborhood_size = neighborhood_size
        self.interaction_field = interaction_field
        self.n_active_bytes = min(n_active_bytes, 256)
        self.numeric_policy = numeric_policy or AdaptiveNumericPolicy()

        byte_vals = np.frombuffer(substrate.raw_data, dtype=np.uint8)

        # === Per-byte-value дифузія D_k (Рівняння 9) ===
        byte_freq = substrate.byte_distribution
        byte_freq_norm = byte_freq / max(byte_freq.sum(), 1e-10)
        self.numeric_field_policy = self.numeric_policy.field_policy(byte_freq_norm, self.N)
        # D_k: часті байти дифундують швидше (сепаратори розповсюджуються)
        # CONCEPT FIX: D_k тепер відповідає Рівнянню (9) з Σ-формою Laplacian
        self.D_k = np.zeros(256, dtype=np.float32)
        for k in range(256):
            if byte_freq_norm[k] > self.numeric_field_policy['freq_high_threshold']:
                self.D_k[k] = D_u * self.numeric_field_policy['diffusion_frequent_scale']
            elif byte_freq_norm[k] > self.numeric_field_policy['freq_mid_threshold']:
                self.D_k[k] = D_u * self.numeric_field_policy['diffusion_mid_scale']
            else:
                self.D_k[k] = D_u * self.numeric_field_policy['diffusion_rare_scale']

        # === Double-well potential параметри (Рівняння 11) ===
        self.a_k = np.ones(256, dtype=np.float32) * 1.0
        theta_base = float(self.numeric_field_policy['theta_base'])
        theta_gain = float(self.numeric_field_policy['theta_freq_gain'])
        self.theta_k = np.ones(256, dtype=np.float32) * theta_base

        # Адаптація θ_k на основі частоти
        for b in range(256):
            if byte_freq_norm[b] > 0:
                self.theta_k[b] = theta_base + theta_gain * byte_freq_norm[b]

        self.interaction_modulation_scale = float(
            self.numeric_field_policy['interaction_modulation_scale']
        )
        self.kinetic_energy_weight = float(self.numeric_field_policy['kinetic_energy_weight'])

        # Коефіцієнт затухання μ
        self.mu = 0.001

        # === ПЕРВИННЕ поле Φ(i,k) — Рівняння (9) ===
        # Для n_active_bytes найчастіших байтів (або всіх 256)
        top_bytes = np.argsort(byte_freq)[-self.n_active_bytes:]
        self.active_byte_indices = top_bytes
        self.Phi = np.zeros((self.N, self.n_active_bytes), dtype=np.float32)

        # Векторизована ініціалізація Φ згідно з концепцією:
        # Байт присутній → Φ = +√θ_k ("увімкнений" стан, Рівняння 11)
        # Байт відсутній → Φ = -√θ_k ("вимкнений" стан, Рівняння 11)
        # CONCEPT FIX: Рівняння 11 каже E_k = a_k(Φ²-θ_k)² має мінімуми при Φ=±√θ_k.
        # Відповідно до концепції: "+√θ_k" — стан "увімкнено", "−√θ_k" — стан "вимкнено".
        # Попередній код ініціалізував non-matching байти як 0.01 (майже нуль),
        # але 0 є НЕСТАБІЛЬНОЮ точкою double-well (дивергенція!):
        # будь-який малий поштовх від дифузії перетворює 0.01 → +√θ_k.
        # Це призводило до того, що ВСІ 256 байтів ставали "увімкненими" →
        # однорідна активація → немає кластерів.
        # Ініціалізація -√θ_k є СТАБІЛЬНИМ станом: дифузія має подолати
        # бар'єр від -√θ до √(θ/3) (значно більша відстань ніж від 0).
        byte_val_matrix = np.tile(byte_vals[:, None], (1, self.n_active_bytes))  # (N, n_active)
        active_matrix = self.active_byte_indices[None, :]  # (1, n_active)
        match_mask = (byte_val_matrix == active_matrix)  # (N, n_active)
        theta_vec = self.theta_k[self.active_byte_indices]  # (n_active,)
        sqrt_theta = np.sqrt(theta_vec)  # (n_active,)
        self.Phi = np.where(match_mask, sqrt_theta[None, :], -sqrt_theta[None, :]).astype(np.float32)

        # === ПОХІДНІ поля u та v (агрегація з Φ) ===
        # u(i) = Σ_k Φ(i,k) — агрегована активація
        # v(i) = max_k Φ(i,k) — максимальна активація
        self.u = np.sum(self.Phi, axis=1).astype(np.float32)
        self.v = np.max(self.Phi, axis=1).astype(np.float32)

        # CONCEPT FIX: НЕ нормалізуємо u та v!
        # Рівняння (9) каже u(i)=ΣΦ(i,k) — це ПОХІДНА агрегована величина,
        # а не нормалізований [0,1] сигнал. Нормалізація після КОЖНОГО кроку
        # руйнувала Allen-Cahn динаміку — прибивала різницю між позиціями
        # до std≈0.01, що не давало boundary detector знайти жодних границь.
        # V4 компоненти (PredictiveCoding, SelfOrganizer) працюють з
        # реальними значеннями u, ім не потрібен [0,1] діапазон.

        self.step_count = 0

        # V7 FIX: Контекстна ін'єкція — неперервний терм в ОДР (Рівняння 37)
        # ∂Φ/∂t += κ · ctx(t) · σ_gate(ctx, Φ) · dt
        # Замість імпульсного зсуву раз на 100 кроків — плавне
        # безперервне додавання, масштабоване на dt
        self.context_injection_vector = np.zeros(self.n_active_bytes, dtype=np.float32)
        self.context_injection_kappa = 0.0  # κ (встановлюється ContextResonance)

    def step(self):
        """
        Один крок еволюції поля з Allen-Cahn double-well реакцією.

        V6 FIX: ОСНОВНА динаміка — Allen-Cahn (Рівняння 9-11).
        Gray-Scott БІЛЬШЕ НЕ використовується як базова модель.
        u та v обчислюються як АГРЕГАТНІ величини з Φ після кожного кроку.

        ∂Φ(i,k,t)/∂t = D_k·∇²Φ + R_k(Φ,θ) − μ·Φ
        R_k = −4·a_k·Φ·(Φ² − θ_k)   (double-well force)

        PERFORMANCE FIX: Векторизований лапласіан для ВСІХ k ОДНОЧАСНО.
        Замість Python циклу з 256 ітерацій — один векторизований pass
        з використанням cumulative sum для обчислення сусідньої суми.
        """
        N = self.N
        if N == 0:
            return
        ns = self.neighborhood_size
        dt = self.dt
        Phi = self.Phi  # (N, n_active_bytes)

        # === Векторизований Laplacian для всіх k одночасно ===
        # Замість 256 окремих convolve — одна операція з cumulative sum
        # Для кожної позиції i: neighbor_sum[i,k] = Σ_{j=i-ns}^{i+ns} Phi[j,k] (без center)
        # Це еквівалентно: sum_2ns_plus1[i,k] - Phi[i,k]

        # Pad Phi along position axis (axis 0)
        Phi_padded = np.pad(Phi, ((ns, ns), (0, 0)), mode='edge')  # (N+2*ns, n_active)

        # Cumulative sum along axis 0
        cumsum = np.cumsum(Phi_padded, axis=0)  # (N+2*ns, n_active)
        # Insert zero at the beginning for proper windowed sum
        cumsum_ext = np.vstack([np.zeros((1, Phi.shape[1]), dtype=np.float32), cumsum])  # (N+2*ns+1, n_active)

        # window_sum[i] = cumsum_ext[i + 2*ns + 1] - cumsum_ext[i]
        # Сума елементів Phi_padded[i : i + 2*ns + 1] = сума вікна навколо позиції i
        # Після pad, позиція i у Phi відповідає позиції i+ns у Phi_padded
        # Вікно [i-ns, i+ns] у Phi = [i, i+2*ns] у Phi_padded
        window_sum = cumsum_ext[2*ns + 1 : 2*ns + 1 + N] - cumsum_ext[:N]  # (N, n_active)

        # Віднімаємо центральний елемент щоб отримати суму сусідів
        neighbor_sum = window_sum - Phi  # (N, n_active_bytes)

        # Векторизований Laplacian: Σ_{j∈N(i)} [Φ(j,k) − Φ(i,k)]
        # CONCEPT FIX: Рівняння (9) каже Σ (сума), а НЕ середнє!
        # Попередній код ділив на 2*ns (перетворяв SUM → MEAN), що
        # зменшувало дифузію в 2*ns разів (10x при ns=5) — це НЕ
        # відповідало концепції і призводило до відсутності самоорганізації.
        # Згідно Рівняння (9): D_k · Σ_{j∈N(i)} [Φ(j)−Φ(i)]
        # = D_k · (neighbor_sum − 2·ns·Φ(i))
        laplacian_Phi = neighbor_sum - 2.0 * ns * Phi  # (N, n_active) — БЕЗ ділення!

        # Параметри для кожного k як вектори
        D_k_vec = self.D_k[self.active_byte_indices]  # (n_active_bytes,)
        a_k_vec = self.a_k[self.active_byte_indices]    # (n_active_bytes,)
        theta_k_vec = self.theta_k[self.active_byte_indices]  # (n_active_bytes,)

        # Модуляція тензорною взаємодією W(i)
        # CONCEPT FIX: Рівняння (3-7) описують тензор взаємодії як КЛЮЧОВИЙ
        # механізм формування патернів. Раніше коефіцієнт 0.1 робив модуляцію
        # лише 1.3% від сили double-well (0.1*0.5*0.5 = 0.025 vs spring = 2.0),
        # тому тензорна взаємодія ПРАКТИЧНО не впливала на динаміку.
        # Згідно з концепцією, тензор взаємодії має БУТИ порівнянним за силою
        # з double-well реакцією, щоб створювати просторові патерни.
        # Новий підхід: W_mod діє як ADDITIVE драйвер (не мультиплікативний),
        # що штовхає Phi до стану, визначеного тензором взаємодії.
        # W(i) > 0.5 → позиція у "когерентному" регіоні → підсилюємо Phi
        # W(i) < 0.5 → позиція на границі → послаблюємо Phi
        if self.interaction_field is not None:
            if self.interaction_field.ndim == 1:
                # 1D backward-compatible mode
                W_mod = 2.0 * (self.interaction_field - 0.5)
                W_mod_expanded = W_mod[:, None]
            else:
                # 2D value-specific mode
                W_mod = 2.0 * (self.interaction_field - 0.5)
                W_mod_expanded = W_mod
        else:
            W_mod_expanded = np.zeros_like(Phi)

        # === Векторизована Allen-Cahn еволюція для ВСІХ k ===
        # R_k = -4·a_k·Phi·(Phi² − θ_k) для кожного k
        # D_k · laplacian_Phi[k] для кожного k
        # Всі операції є поелементними → одна векторизована операція

        # Double-well reaction: R_k = -4·a_k·Phi·(Phi² − θ_k)  (поелементно)
        R = -4.0 * a_k_vec[None, :] * Phi * (Phi ** 2 - theta_k_vec[None, :])  # (N, n_active)

        # Повне рівняння Allen-Cahn (Рівняння 9):
        # ∂Φ/∂t = D_k·∇²Φ + R_k − μ·Φ + κ·W_mod·σ(Φ)
        # CONCEPT FIX: Модуляція взаємодією тепер ADDITIVE (не мультиплікативна).
        # Коефіцієнт κ = 0.5 робить тензорну взаємодію порівнянною з double-well.
        interaction_modulation = (
            self.interaction_modulation_scale * W_mod_expanded * _sigmoid(Phi)
        )  # (N, n_active)

        dphi = (D_k_vec[None, :] * laplacian_Phi  # дифузія (поелементно)
                + R                                   # реакція
                - self.mu * Phi                       # затухання
                + interaction_modulation)              # модуляція взаємодією (ADDITIVE)

        # V7 FIX: Контекстна ін'єкція як неперервний терм в ОДР (Рівняння 37)
        # ∂Φ(i,k,t)/∂t += κ · ctx(k,t) · σ_gate(ctx, Φ(i,k,t))
        # де ctx(k,t) — контекстний вектор, адаптований до розмірності Φ
        # Замість імпульсного зсуву раз на 100 кроків — плавне додавання,
        # масштабоване на dt, інтегроване в кожен крок еволюції.
        if self.context_injection_kappa > 1e-10:
            ctx_vec = self.context_injection_vector  # (n_active_bytes,)
            ctx_norm = float(np.linalg.norm(ctx_vec))
            if ctx_norm > 1e-10:
                # σ_gate(ctx, Φ) = sigmoid(‖ctx‖ · mean(Φ))
                phi_mean = float(np.mean(Phi))
                gate = float(_sigmoid(np.array([ctx_norm * phi_mean]))[0])
                # Контекстний терм: κ · ctx · gate
                # ctx_vec[k] діє на Φ[:, k] для всіх позицій i
                ctx_term = self.context_injection_kappa * ctx_vec[None, :] * gate  # (1, n_active)
                dphi = dphi + ctx_term

        # CONCEPT FIX: Дозволяємо негативні значення Phi!
        # Рівняння 11: double-well має стабільні стани при Φ=±√θ_k.
        # "Вимкнений" стан — це Φ=-√θ_k, а НЕ Φ=0 (0 — нестабільна точка!).
        # Попередній clip [0, 2] блокував "вимкнений" стан → усі байти
        # дрейфували до "увімкненого" → однорідна активація → немає кластерів.
        # Новий кліп: [-1.5, 2.0] — дозволяє "вимкнений" стан при -√θ_k ≈ -0.5.
        if not np.all(np.isfinite(dphi)):
            dphi = np.nan_to_num(dphi, nan=0.0, posinf=10.0, neginf=-10.0)
        self.Phi = np.clip(Phi + dt * dphi, -1.5, 2.0)

        # === ПОХІДНІ поля u та v (агрегація з Φ) ===
        # CONCEPT FIX: u рахує лише "увімкнені" байти (Phi > 0),
        # бо "увімкнений" стан = +√θ_k (позитивний), а "вимкнений" = -√θ_k.
        # u(i) = Σ_k max(Φ(i,k), 0) — сума активації "увімкнених" байтів.
        # v(i) = max_k max(Φ(i,k), 0) — максимальна "увімкнена" активація.
        # Попередній код: u = Σ Phi — рахував і негативні значення,
        # що давало u ≈ 0 (кілька позитивних + багато негативних ≈ нуль).
        Phi_positive = np.maximum(self.Phi, 0.0)
        self.u = np.sum(Phi_positive, axis=1).astype(np.float32)
        self.v = np.max(Phi_positive, axis=1).astype(np.float32)

        self.step_count += 1

    def update_feed_rate(self, interaction_field: np.ndarray):
        """Оновити поле взаємодії (зворотний зв'язок)."""
        self.interaction_field = interaction_field

    def get_concept_activation(self) -> np.ndarray:
        """Map signed double-well Phi back to the concept-level [0, 1] activation."""
        theta_k_vec = self.theta_k[self.active_byte_indices]
        sqrt_theta = np.sqrt(np.maximum(theta_k_vec, 1e-10))[None, :]
        return np.clip(0.5 * (self.Phi / sqrt_theta + 1.0), 0.0, 1.0).astype(np.float32)

    def compute_double_well_gradient(self) -> np.ndarray:
        """Gradient of the mean double-well energy with respect to Phi using autograd."""
        a_k_vec = self.a_k[self.active_byte_indices]
        theta_k_vec = self.theta_k[self.active_byte_indices]
        
        Phi_t = torch.from_numpy(self.Phi.astype(np.float32)).requires_grad_(True)
        a_k_t = torch.from_numpy(a_k_vec.astype(np.float32))
        theta_k_t = torch.from_numpy(theta_k_vec.astype(np.float32))
        
        # Mean double-well energy: E = mean(a_k * (Phi^2 - theta_k)^2)
        E_dw = torch.mean(a_k_t[None, :] * (Phi_t ** 2 - theta_k_t[None, :]) ** 2)
        grad_Phi = torch.autograd.grad(E_dw, Phi_t)[0]
        if not torch.all(torch.isfinite(grad_Phi)):
            grad_Phi = torch.nan_to_num(grad_Phi, nan=0.0, posinf=10.0, neginf=-10.0)
        # V6 Stability Fix: Clamp gradient to prevent overflow
        grad_Phi = torch.clamp(grad_Phi, -10.0, 10.0)
        
        return grad_Phi.cpu().numpy().astype(np.float32)

    def compute_free_energy(self, temperature: float = 1.0) -> float:
        """
        Вільна енергія: F = E_total - T·S (Рівняння 17).

        V6 FIX: E_total обчислюється з Allen-Cahn double-well потенціалу,
        а НЕ з Gray-Scott моделі. Векторизовано для продуктивності.
        """
        # Векторизована Allen-Cahn double-well енергія
        a_k_vec = self.a_k[self.active_byte_indices]  # (K,)
        theta_k_vec = self.theta_k[self.active_byte_indices]  # (K,)
        # E = mean_k(a_k * mean_i(Phi_ik^2 - theta_k)^2)
        E_dw = float(np.mean(
            a_k_vec[None, :] * (self.Phi ** 2 - theta_k_vec[None, :]) ** 2
        ))

        # Кінетична енергія: градієнт Φ (векторизовано для всіх k)
        grad_Phi = np.gradient(self.Phi, axis=0)  # (N, K)
        D_k_vec = self.D_k[self.active_byte_indices]  # (K,)
        E_kinetic = float(np.mean(0.5 * D_k_vec[None, :] * grad_Phi ** 2))

        E_total = E_dw + self.kinetic_energy_weight * E_kinetic

        # CONCEPT FIX: Ентропія поля Φ згідно Рівняння (18):
        # S = -Σ_i Σ_k ϕ̃(i,k) · log(ϕ̃(i,k))
        # де ϕ̃(i,k) = |Φ(i,k)| / Σ_m |Φ(i,m)| — нормалізація по байтових значеннях
        # для КОЖНОЇ позиції окремо.
        # Попередній код обчислював ентропію u по позиціях — це НЕ відповідає
        # концепції. Ентропія за Рівнянням (18) вимірює "фокусування" поля на
        # конкретних байтових значеннях у кожній позиції (наскільки впевнене
        # поле, яке байтове значення присутнє), а не розподіл активації по позиціях.
        Phi_abs = np.abs(self.Phi)  # (N, n_active_bytes)
        Phi_row_sums = Phi_abs.sum(axis=1, keepdims=True)  # (N, 1)
        phi_tilde = Phi_abs / np.maximum(Phi_row_sums, 1e-10)  # (N, n_active_bytes)
        phi_tilde = np.maximum(phi_tilde, 1e-10)  # Для log без -inf
        row_entropy = -np.sum(phi_tilde * np.log(phi_tilde), axis=1)
        entropy_density = float(np.mean(row_entropy)) if row_entropy.size else 0.0
        entropy_sum = float(np.sum(row_entropy))
        free_energy = E_total + temperature * entropy_density

        # Calculate mathematically complete physical terms
        E_dw_sum = E_dw * len(self.active_byte_indices)
        E_kinetic_sum = E_kinetic * len(self.active_byte_indices)
        E_decay = float(np.sum(0.5 * self.mu * np.mean(self.Phi ** 2, axis=0)))
        if self.interaction_field is not None:
            W_mod_expanded = 2.0 * (self.interaction_field - 0.5) if self.interaction_field.ndim == 2 else (2.0 * (self.interaction_field - 0.5))[:, None]
        else:
            W_mod_expanded = np.zeros_like(self.Phi)
        softplus_Phi = np.log(1.0 + np.exp(np.clip(self.Phi, -20, 20)))
        E_interaction = -float(np.sum(self.interaction_modulation_scale * np.mean(W_mod_expanded * softplus_Phi, axis=0)))
        E_total_physical = E_dw_sum + self.kinetic_energy_weight * E_kinetic_sum + E_decay + E_interaction
        physical_free_energy = E_total_physical - temperature * entropy_density

        self._last_free_energy_components = {
            'free_energy': float(free_energy),
            'energy_total': float(E_total),
            'double_well_energy': float(E_dw),
            'kinetic_energy': float(E_kinetic),
            'entropy_density': float(entropy_density),
            'entropy_sum': float(entropy_sum),
            'legacy_free_energy': float(E_total - temperature * entropy_sum),
            'physical_free_energy': float(physical_free_energy),
            'physical_energy_total': float(E_total_physical),
            'decay_energy': float(E_decay),
            'interaction_energy': float(E_interaction),
        }

        return self._last_free_energy_components['free_energy']

    def compute_free_energy_components(self, temperature: float = 1.0) -> Dict[str, float]:
        """Return normalized free-energy components; lower free_energy is better."""
        self.compute_free_energy(temperature)
        return dict(self._last_free_energy_components)

    def get_field_statistics(self) -> Dict:
        """Статистика польової системи."""
        return {
            'u_mean': float(np.mean(self.u)),
            'u_std': float(np.std(self.u)),
            'v_mean': float(np.mean(self.v)),
            'v_std': float(np.std(self.v)),
            'pattern_strength': float(np.std(self.v)),
            'phi_mean': float(np.mean(self.Phi)),
            'phi_std': float(np.std(self.Phi)),
            'field_policy': dict(self.numeric_field_policy),
            'numeric_policy': self.numeric_policy.report(),
        }


# =============================================================================
# 4. Variational Inference — Рівняння (25-27)
# =============================================================================


