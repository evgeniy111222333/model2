import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _kl_divergence, _safe_normalize, _sigmoid, _softmax

class VariationalInference(nn.Module):
    """
    Варіаційна інференція для ієрархічної генеративної моделі БКС.

    Генеративна модель:
    p(z^(l)|z^(l+1)) = N(z^(l); W^(l+1) z^(l+1), σ²I)     (25)
    p(s|z^(1)) = Π Cat(b_i; softmax(V z^(1)))                (26)

    Вільна енергія:
    F = D_KL(q(z) || p(z|s)) - E_q[log p(s|z)]              (27)

    Реалізація: Amortized variational inference з
    encoder/decoder для кожного рівня.

    V7: PyTorch autograd замість ручного ELBO-градієнту.
    Попередня версія мала ~120 рядків ручних градієнтних обчислень:
    - grad_enc_recon = outer(error * sigmoid_deriv, input)
    - grad_enc_kl = outer(kl_grad_mu * sigmoid_deriv, input)
    - grad_logvar = outer(recon_grad_logvar + kl_grad_logvar, input)
    Ці обчислення мали задокументовані баги (V6 FIX #3).
    Тепер loss.backward() обчислює всі градієнти автоматично.
    """

    def __init__(
        self,
        n_levels: int = 4,
        d_latent: Optional[List[int]] = None,
        d_observation: int = 256,
    ):
        super().__init__()
        self.n_levels = n_levels
        if d_latent is None:
            d_latent = [128, 64, 32, 16]
        self.d_latent = d_latent[:n_levels]
        self.d_observation = d_observation

        # Encoder: x → z (bottom-up)
        self.W_encoder = nn.ParameterList()
        # V6 FIX #3: W_logvar для параметризації log(σ²) — НАВЧАЄТЬСЯ
        self.W_logvar = nn.ParameterList()
        # Decoder: z → x (top-down, генерація)
        self.W_decoder = nn.ParameterList()
        # Generation: z^(1) → байти
        v_gen_init = np.random.randn(d_observation, d_latent[0]).astype(np.float32) * 0.05
        self.V_gen = nn.Parameter(torch.from_numpy(v_gen_init))

        for i in range(n_levels):
            d_in = d_observation if i == 0 else d_latent[i - 1]
            d_out = d_latent[i]

            w_enc_init = np.random.randn(d_out, d_in).astype(np.float32) * 0.05
            self.W_encoder.append(
                nn.Parameter(torch.from_numpy(w_enc_init))
            )
            # V6 FIX #3: W_logvar — незалежні параметри для log(σ²)
            # Ініціалізуємо близько до 0 (log(1.0) = 0 → σ² = 1.0)
            w_lv_init = np.random.randn(d_out, d_in).astype(np.float32) * 0.01
            self.W_logvar.append(
                nn.Parameter(torch.from_numpy(w_lv_init))
            )
            w_dec_init = np.random.randn(d_in, d_out).astype(np.float32) * 0.05
            self.W_decoder.append(
                nn.Parameter(torch.from_numpy(w_dec_init))
            )

        self.elbo_history = []
        # FIX ELBO: KL-аннеалінг — починаємо з малої ваги KL, плавно
        # збільшуємо до 1.0. Запобігає posterior collapse (q(z)→p(z))
        # на початку навчання, коли reconstruction gradient ще нестабільний.
        # β_KL зростає від 0.01 до 1.0 за n_warmup кроків.
        self.kl_weight = 0.01
        self.kl_weight_max = 1.0
        self.kl_anneal_step = 0
        self.kl_warmup_steps = 200  # За скільки кроків β_KL → 1.0

    def _project(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Адаптувати розмірність тензора: обрізати або доповнити нулями."""
        if x.shape[-1] == target_dim:
            return x
        elif x.shape[-1] > target_dim:
            return x[..., :target_dim]
        else:
            pad_size = target_dim - x.shape[-1]
            return torch.nn.functional.pad(x, (0, pad_size))

    def encode_torch(
        self,
        observations: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Bottom-up encoding with full torch graph for autograd.

        Returns:
            latents, z_means, z_log_vars — all torch tensors in the compute graph
        """
        latents = []
        z_means = []
        z_log_vars = []
        current = observations

        for i in range(self.n_levels):
            W = self.W_encoder[i]
            W_lv = self.W_logvar[i]
            current_proj = self._project(current, W.shape[1])

            # μ = sigmoid(W_enc · x)
            z_mean = torch.sigmoid(W @ current_proj)

            # log(σ²) = W_logvar · x, clamped for stability
            z_log_var = torch.clamp(W_lv @ current_proj, -10.0, 2.0)

            # Reparameterization: z = μ + σ * ε
            z_std = torch.exp(0.5 * z_log_var)
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps

            latents.append(z)
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
            current = z

        return latents, z_means, z_log_vars

    def decode_torch(
        self,
        latents: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Top-down decoding with full torch graph.

        CONCEPT FIX (Рівняння 26): p(s|z^(1)) = Π Cat(b_i; softmax(V z^(1)))
        Перший рівень: softmax(V_gen · z^(1))
        Вищі рівні: sigmoid(W_dec · z)
        """
        predictions = [None] * self.n_levels

        for i in range(self.n_levels - 1, -1, -1):
            W = self.W_decoder[i]
            z = latents[i]
            z_proj = self._project(z, W.shape[1])

            if i == 0:
                # Категоріальна генерація байтів
                z1_proj = self._project(latents[0], self.V_gen.shape[1])
                logits = self.V_gen @ z1_proj
                pred = torch.softmax(logits, dim=-1)
            else:
                pred = torch.sigmoid(W @ z_proj)

            predictions[i] = pred

        return predictions

    def compute_elbo_torch(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute ELBO as a differentiable torch scalar.

        Returns:
            elbo_tensor: differentiable ELBO for .backward()
            recon_error_val: float reconstruction error
            kl_val: float KL divergence
        """
        latents, z_means, z_log_vars = self.encode_torch(observations)
        predictions = self.decode_torch(latents)

        # Reconstruction: categorical cross-entropy
        obs_safe = torch.clamp(observations, min=1e-10)
        obs_norm = obs_safe / obs_safe.sum()

        if predictions[0] is not None:
            pred_safe = torch.clamp(
                self._project(predictions[0], self.d_observation), min=1e-10
            )
            recon_error = -torch.sum(obs_norm * torch.log(pred_safe))
        else:
            recon_error = torch.tensor(0.0)

        # KL divergence: hierarchical prior
        kl = torch.tensor(0.0)
        for i in range(self.n_levels):
            mu_q = z_means[i]
            log_var_q = z_log_vars[i]
            var_q = torch.exp(log_var_q)

            if i == self.n_levels - 1:
                mu_p = torch.zeros_like(mu_q)
            else:
                W_dec = self.W_decoder[i + 1]
                z_upper = latents[i + 1]
                z_upper_proj = self._project(z_upper, W_dec.shape[1])
                mu_p_raw = torch.sigmoid(W_dec @ z_upper_proj)
                mu_p = self._project(mu_p_raw, mu_q.shape[-1])

            kl = kl + 0.5 * torch.sum(
                (mu_q - mu_p) ** 2 + var_q - log_var_q - 1.0
            )

        # KL annealing
        beta_kl = min(
            self.kl_weight_max,
            self.kl_weight + (self.kl_weight_max - self.kl_weight)
            * min(self.kl_anneal_step / max(self.kl_warmup_steps, 1), 1.0),
        )

        elbo = -recon_error - beta_kl * kl

        return elbo, float(recon_error.item()), float(kl.item())

    # =========================================================================
    # Legacy NumPy-compatible API (backward compatibility)
    # =========================================================================

    def encode(
        self,
        observations: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Bottom-up: x → z^(1) → z^(2) → ... → z^(L)
        Legacy NumPy interface.
        """
        obs_t = torch.from_numpy(observations.astype(np.float32))
        with torch.no_grad():
            latents_t, means_t, logvars_t = self.encode_torch(obs_t)
        return (
            [z.numpy() for z in latents_t],
            [m.numpy() for m in means_t],
            [lv.numpy() for lv in logvars_t],
        )

    def decode(
        self,
        latents: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Top-down генерація. Legacy NumPy interface.
        """
        latents_t = [torch.from_numpy(z.astype(np.float32)) for z in latents]
        with torch.no_grad():
            predictions_t = self.decode_torch(latents_t)
        predictions = [p.numpy() if p is not None else None for p in predictions_t]
        reconstructions = [p for p in predictions if p is not None]
        return predictions, reconstructions

    def compute_elbo(
        self,
        observations: np.ndarray,
    ) -> float:
        """
        ELBO = E_q[log p(x|z)] - D_KL(q(z) || p(z))
        Legacy NumPy interface.
        """
        obs_full = np.zeros(self.d_observation, dtype=np.float32)
        obs_full[:len(observations)] = observations[:self.d_observation]
        obs_t = torch.from_numpy(obs_full)

        with torch.no_grad():
            elbo, _, _ = self.compute_elbo_torch(obs_t)

        val = float(elbo.item())
        self.elbo_history.append(val)
        return val

    def update(
        self,
        observations: np.ndarray,
        lr: float = 0.001,
    ) -> float:
        """
        Оновити параметри варіаційної моделі через autograd.

        V7: Замість ~120 рядків ручних градієнтних обчислень
        (grad_enc_recon, grad_enc_kl, grad_logvar), один виклик
        loss.backward() обчислює всі градієнти автоматично.
        """
        if not np.all(np.isfinite(observations)):
            observations = np.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)
            
        obs_full = np.zeros(self.d_observation, dtype=np.float32)
        obs_full[:len(observations)] = observations[:self.d_observation]
        obs_t = torch.from_numpy(obs_full)

        # Zero gradients
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass — full compute graph
        elbo, recon_err, kl_val = self.compute_elbo_torch(obs_t)

        # Maximize ELBO = minimize -ELBO
        neg_elbo = -elbo
        neg_elbo.backward()

        # Stability Fix: Clip gradients of all parameters to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Manual SGD step (to match original lr semantics)
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    if not torch.all(torch.isfinite(p.grad)):
                        p.grad.data.copy_(torch.nan_to_num(p.grad.data, nan=0.0, posinf=1.0, neginf=-1.0))
                    p.data -= lr * p.grad

        # KL annealing step
        self.kl_anneal_step += 1

        elbo_val = float(elbo.item())
        self.elbo_history.append(elbo_val)
        return elbo_val


# =============================================================================
# 5. IB Optimizer — Рівняння (28)
# =============================================================================

