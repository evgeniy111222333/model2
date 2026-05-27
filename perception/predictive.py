import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union

class PredictiveCoding:
    """
    Механізм предиктивного кодування БКС.

    Розділ 6 концепції: система генерує передбачення û_i на основі
    контексту і використовує помилку передбачення ε_i = u_i - û_i
    для навчання та виявлення аномалій.

    V4: Векторизовано через sliding_window_view та матричні операції.
    Замість O(N·C) Python циклу — O(N) матрична операція.

    V7: PyTorch autograd для обчислення градієнтів.
    Замість ручного grad = -2.0 * (errors @ windows) / N,
    autograd обчислює точний ∂MSE/∂w через loss.backward().
    """

    def __init__(
        self,
        context_size: int = 8,
        learning_rate: float = 0.01,
    ):
        self.context_size = context_size
        self.learning_rate = learning_rate
        # Ваги предиктивного кодування як torch.nn.Parameter
        w_init = np.random.randn(2 * context_size + 1).astype(np.float32) * 0.1
        w_init[context_size] = 0.0  # Не використовуємо саму позицію
        self._weights_param = nn.Parameter(torch.from_numpy(w_init))
        self._optimizer = torch.optim.SGD([self._weights_param], lr=learning_rate)
        self.prediction_error_history = []

    @property
    def weights(self) -> np.ndarray:
        """Backward-compatible NumPy accessor."""
        return self._weights_param.detach().cpu().numpy()

    @weights.setter
    def weights(self, value: np.ndarray):
        """Backward-compatible NumPy setter."""
        with torch.no_grad():
            self._weights_param.copy_(torch.from_numpy(np.asarray(value, dtype=np.float32)))

    def predict(self, u_field: np.ndarray) -> np.ndarray:
        """
        Генерація передбачень û_i для кожної позиції.
        û_i = Σ_{j} w[j] · u_{i+j-offset}

        V4: Векторизовано через sliding_window_view + матричний добуток.
        """
        half = self.context_size
        N = len(u_field)
        if N == 0:
            return np.zeros(0, dtype=np.float32)

        # Доповнення граничними значеннями (як у V3 — еквівалент пропуску)
        u_padded = np.pad(u_field, half, mode='edge')

        # Матриця вікон: windows[i, k] = u_padded[i + k]
        windows = np.lib.stride_tricks.sliding_window_view(
            u_padded, 2 * half + 1
        )  # (N, 2*half+1)

        predictions = windows @ self.weights  # (N,)
        return predictions.astype(np.float32)

    def compute_prediction_error(self, u_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обчислення помилки передбачення.

        Returns:
            errors: ε_i = u_i - û_i (помилка)
            predictions: û_i (передбачення)
        """
        predictions = self.predict(u_field)
        errors = u_field - predictions
        return errors, predictions

    def learn(self, u_field: np.ndarray) -> float:
        """
        Один крок навчання предиктивного кодування.
        PyTorch autograd: loss = MSE(u, û) → loss.backward() → optimizer.step()

        V7: Замість ручного градієнта grad = -2*(errors @ windows)/N,
        використовуємо torch.autograd для точного ∂MSE/∂w.

        Returns:
            mean_squared_error: середньоквадратична помилка після оновлення
        """
        half = self.context_size
        N = len(u_field)
        if N == 0:
            return 0.0

        # Побудова матриці вікон для forward pass
        u_padded = np.pad(u_field, half, mode='edge')
        windows_np = np.lib.stride_tricks.sliding_window_view(
            u_padded, 2 * half + 1
        ).copy()  # (N, 2*half+1)

        # Torch forward pass
        windows_t = torch.from_numpy(windows_np.astype(np.float32))
        u_target = torch.from_numpy(u_field.astype(np.float32))

        self._optimizer.zero_grad()
        predictions = windows_t @ self._weights_param  # (N,)
        loss = torch.mean((u_target - predictions) ** 2)
        loss.backward()

        # Не оновлюємо центральну вагу
        if self._weights_param.grad is not None:
            self._weights_param.grad[half] = 0.0

        self._optimizer.step()

        # Нова помилка
        new_errors, _ = self.compute_prediction_error(u_field)
        mse = float(np.mean(new_errors ** 2))
        self.prediction_error_history.append(mse)

        return mse

    def detect_anomalies(self, u_field: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """
        Виявлення аномалій на основі помилки передбачення.
        Позиції з |ε_i| > threshold · std(ε) є аномальними.
        """
        errors, _ = self.compute_prediction_error(u_field)
        std_err = np.std(errors)
        if std_err < 1e-10:
            return np.array([], dtype=int)

        anomaly_mask = np.abs(errors) > threshold * std_err
        return np.where(anomaly_mask)[0]



class HierarchicalPredictiveCoding:
    """
    Ієрархічне предиктивне кодування — багаторівневі передбачення.

    Кожен конвертаційний рівень генерує передбачення для рівня нижче.
    Помилка передбачення поширюється знизу вгору (bottom-up error)
    та зверху вниз (top-down prediction).

    Рівень ℓ: ε_ℓ = x_ℓ - f_ℓ(x_{ℓ+1})   (помилка передбачення)
    Рівень ℓ+1: x_{ℓ+1} = g_{ℓ+1}(ε_ℓ)      (оновлення через помилку)

    Відповідність концепції:
    - Розділ 5.2: Зворотний зв'язок та предиктивна обробка
    - Рівняння (20-21): Ієрархічна агрегація з помилкою

    V7: PyTorch autograd для update_weights().
    BUGFIX: Попередня версія обчислювала grad = -2·outer(err, x_upper),
    що НЕ враховувало tanh активацію в predict_topdown().
    Правильний градієнт потребує dtanh = 1 - pred². Autograd це
    обчислює автоматично через loss.backward().
    """

    def __init__(
        self,
        n_levels: int = 4,
        d_representations: Optional[List[int]] = None,
    ):
        self.n_levels = n_levels

        if d_representations is None:
            d_representations = [256, 128, 64, 32]
        self.d_representations = d_representations[:n_levels]

        # PyTorch parameters for top-down and bottom-up weights
        self._W_topdown_params = nn.ParameterList()
        self._W_bottomup_params = nn.ParameterList()

        for i in range(n_levels - 1):
            d_lower = self.d_representations[i]
            d_upper = self.d_representations[i + 1]

            # Use NumPy randn to preserve seed reproducibility
            w_td = np.random.randn(d_lower, d_upper).astype(np.float32) * 0.05
            w_bu = np.random.randn(d_upper, d_lower).astype(np.float32) * 0.05

            # Top-down: проєкція з верхнього рівня на нижчий
            self._W_topdown_params.append(
                nn.Parameter(torch.from_numpy(w_td))
            )
            # Bottom-up: проєкція помилки з нижнього рівня на верхній
            self._W_bottomup_params.append(
                nn.Parameter(torch.from_numpy(w_bu))
            )

        self.error_history = []

    @property
    def W_topdown(self) -> list:
        """Backward-compatible NumPy accessor for W_topdown."""
        return [p.detach().cpu().numpy() for p in self._W_topdown_params]

    @W_topdown.setter
    def W_topdown(self, value: list):
        """Backward-compatible NumPy setter for W_topdown."""
        with torch.no_grad():
            for i, v in enumerate(value):
                self._W_topdown_params[i].copy_(torch.from_numpy(np.asarray(v, dtype=np.float32)))

    @property
    def W_bottomup(self) -> list:
        """Backward-compatible NumPy accessor for W_bottomup."""
        return [p.detach().cpu().numpy() for p in self._W_bottomup_params]

    @W_bottomup.setter
    def W_bottomup(self, value: list):
        """Backward-compatible NumPy setter for W_bottomup."""
        with torch.no_grad():
            for i, v in enumerate(value):
                self._W_bottomup_params[i].copy_(torch.from_numpy(np.asarray(v, dtype=np.float32)))

    def _adapt_dim(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        """Адаптувати розмірність вектора: обрізати або доповнити нулями."""
        if len(x) == target_dim:
            return x
        elif len(x) > target_dim:
            return x[:target_dim]
        else:
            return np.pad(x, (0, target_dim - len(x)))

    def predict_topdown(
        self,
        representations: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Генерувати передбачення зверху вниз для кожного рівня.

        pred_ℓ = tanh(W_topdown[ℓ] · x_{ℓ+1})
        """
        predictions = [None] * self.n_levels

        # Найвищий рівень не має передбачення зверху
        predictions[-1] = np.zeros(self.d_representations[-1], dtype=np.float32)

        for i in range(self.n_levels - 2, -1, -1):
            # Передбачення для рівня i з рівня i+1
            d_lower = self.d_representations[i]
            d_upper = self.d_representations[i + 1]
            x_upper = self._adapt_dim(representations[i + 1], d_upper)
            W_td = self._W_topdown_params[i].detach().cpu().numpy()
            pred = W_td @ x_upper
            pred = np.tanh(pred)  # Активація
            # pred має розмірність d_lower
            predictions[i] = pred

        return predictions

    def compute_errors(
        self,
        representations: List[np.ndarray],
        predictions: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Обчислити помилки передбачення на кожному рівні.
        ε_ℓ = x_ℓ - pred_ℓ
        """
        errors = []
        for i in range(self.n_levels):
            d_i = self.d_representations[i]
            x_i = self._adapt_dim(representations[i], d_i)
            if predictions[i] is not None:
                pred_i = self._adapt_dim(predictions[i], d_i)
                err = x_i - pred_i
            else:
                err = np.zeros(d_i, dtype=np.float32)
            errors.append(err)
        return errors

    def update_representations(
        self,
        representations: List[np.ndarray],
        errors: List[np.ndarray],
        learning_rate: float = 0.01,
    ) -> List[np.ndarray]:
        """
        Оновити представлення на основі помилок.

        Bottom-up: x_{ℓ+1} += η · W_bottomup[ℓ]^T · ε_ℓ
        Top-down: x_ℓ += η · ε_ℓ (пряма корекція)
        """
        new_reps = []
        for i in range(self.n_levels):
            d_i = self.d_representations[i]
            rep = self._adapt_dim(representations[i], d_i).copy()

            # Пряма корекція: x_ℓ += η · ε_ℓ
            if errors[i] is not None and len(errors[i]) > 0:
                err_i = self._adapt_dim(errors[i], d_i)
                rep = rep + learning_rate * err_i

            new_reps.append(rep)

        # Bottom-up вплив
        for i in range(self.n_levels - 1):
            d_upper = self.d_representations[i + 1]
            d_lower = self.d_representations[i]
            if errors[i] is not None and len(errors[i]) > 0:
                err_i = self._adapt_dim(errors[i], d_lower)
                W_bu = self._W_bottomup_params[i].detach().cpu().numpy()
                bottom_up_signal = W_bu @ err_i
                new_reps[i + 1] = new_reps[i + 1] + learning_rate * 0.5 * bottom_up_signal

        return new_reps

    def update_weights(
        self,
        representations: List[np.ndarray],
        errors: List[np.ndarray],
        learning_rate: float = 0.001,
    ):
        """
        Оновити ваги предиктивного кодування через PyTorch autograd.

        V7 BUGFIX: Попередня версія обчислювала:
            grad_td = -2.0 * np.outer(err_i, x_upper)
        Це НЕ враховувало tanh в predict_topdown: pred = tanh(W @ x).
        Правильний градієнт: ∂||ε||²/∂W = -2·ε·(1 - pred²)·x^T,
        де (1 - pred²) = dtanh — похідна tanh.

        Autograd обчислює це автоматично через loss.backward(),
        проходячи градієнт через tanh коректно.
        """
        # Build the full hierarchical loss in torch and backprop
        all_params = list(self._W_topdown_params) + list(self._W_bottomup_params)
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()

        total_loss = torch.tensor(0.0)

        for i in range(self.n_levels - 1):
            d_lower = self.d_representations[i]
            d_upper = self.d_representations[i + 1]

            x_i_np = self._adapt_dim(representations[i], d_lower)
            x_upper_np = self._adapt_dim(representations[i + 1], d_upper)

            x_i = torch.from_numpy(x_i_np.astype(np.float32))
            x_upper = torch.from_numpy(x_upper_np.astype(np.float32))

            # Forward pass: pred = tanh(W_topdown @ x_upper) — autograd traces through tanh
            pred_td = torch.tanh(self._W_topdown_params[i] @ x_upper)
            err_td = x_i - pred_td
            loss_td = torch.sum(err_td ** 2)

            # Bottom-up: pred_bu = W_bottomup @ err
            err_np = self._adapt_dim(
                errors[i] if errors[i] is not None else np.zeros(d_lower, dtype=np.float32),
                d_lower,
            )
            err_t = torch.from_numpy(err_np.astype(np.float32))
            pred_bu = self._W_bottomup_params[i] @ err_t
            x_upper_target = torch.from_numpy(x_upper_np.astype(np.float32))
            loss_bu = 0.5 * torch.sum((x_upper_target - pred_bu) ** 2)

            total_loss = total_loss + loss_td + loss_bu

        if total_loss.requires_grad:
            total_loss.backward()

            # Manual SGD step (to match original learning_rate semantics)
            with torch.no_grad():
                for p in all_params:
                    if p.grad is not None:
                        p.data -= learning_rate * p.grad

    def free_energy_hierarchical(
        self,
        errors: List[np.ndarray],
        complexity_weight: float = 0.01,
    ) -> float:
        """
        Ієрархічна вільна енергія: F = Σ_ℓ ||ε_ℓ||² + λ · Σ_ℓ ||W_ℓ||²
        """
        accuracy_term = 0.0
        complexity_term = 0.0

        for i, err in enumerate(errors):
            if err is not None:
                accuracy_term += float(np.sum(err ** 2))

        for i in range(self.n_levels - 1):
            W_td = self._W_topdown_params[i].detach().cpu().numpy()
            W_bu = self._W_bottomup_params[i].detach().cpu().numpy()
            complexity_term += float(np.sum(W_td ** 2))
            complexity_term += float(np.sum(W_bu ** 2))

        return accuracy_term + complexity_weight * complexity_term

    def learn_step(
        self,
        representations: List[np.ndarray],
        learning_rate: float = 0.01,
        weight_lr: float = 0.001,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Один крок ієрархічного предиктивного кодування.

        Returns:
            updated_representations: оновлені представлення
            free_energy: значення вільної енергії
        """
        # 1. Top-down передбачення
        predictions = self.predict_topdown(representations)

        # 2. Обчислення помилок
        errors = self.compute_errors(representations, predictions)

        # 3. Оновлення представлень
        new_reps = self.update_representations(representations, errors, learning_rate)

        # 4. Оновлення ваг (через autograd — fixes tanh derivative bug)
        self.update_weights(representations, errors, weight_lr)

        # 5. Вільна енергія
        fe = self.free_energy_hierarchical(errors)
        self.error_history.append(fe)

        return new_reps, fe
