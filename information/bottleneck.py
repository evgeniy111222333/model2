import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _softmax

class InformationBottleneck:
    """
    Розділ 6.2 концепції: Information Bottleneck як дизайн-принцип.

    min I(X; Z) - β·I(Z; Y)

    Реалізація: BLAST з ітераційним оновленням p(z|x) та p(y|z).
    """

    def __init__(self, n_clusters: int = 10, beta: float = 1.0, max_iter: int = 50, random_state: Optional[int] = 42):
        self.n_clusters = n_clusters
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, local_distributions: np.ndarray) -> np.ndarray:
        """
        Information Bottleneck кластеризація.

        Args:
            local_distributions: (N, 256) — локальні байтові розподіли

        Returns:
            labels: (N,) — призначення кластерів
        """
        N = local_distributions.shape[0]
        K = min(self.n_clusters, N)

        # Deterministic random state to stabilize clustering & phase transition
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        labels = rng.randint(0, K, size=N)

        for iteration in range(self.max_iter):
            # Крок 1: Обчислення p(y|z)
            p_y_given_z = np.zeros((K, 256), dtype=np.float64)

            for z in range(K):
                mask = labels == z
                if np.sum(mask) > 0:
                    p_y_given_z[z] = np.mean(local_distributions[mask], axis=0)
                    p_y_given_z[z] = np.maximum(p_y_given_z[z], 1e-10)
                    p_y_given_z[z] /= p_y_given_z[z].sum()

            # Крок 2: Оновлення p(z|x) через IB-об'єктив
            p_z_given_x = np.zeros((N, K), dtype=np.float64)

            # Векторизоване обчислення KL-дивергенцій
            p_x = np.maximum(local_distributions, 1e-10)
            p_x = p_x / p_x.sum(axis=1, keepdims=True)

            for z in range(K):
                # KL(p_x || p_y_given_z[z]) для всіх x одночасно
                kl = np.sum(p_x * np.log(p_x / (p_y_given_z[z] + 1e-10)), axis=1)
                p_z = np.sum(labels == z) / N
                p_z_given_x[:, z] = -self.beta * kl + np.log(max(p_z, 1e-10))

            # Softmax
            p_z_given_x -= p_z_given_x.max(axis=1, keepdims=True)
            exp_scores = np.exp(p_z_given_x)
            p_z_given_x = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Hard assignment
            new_labels = np.argmax(p_z_given_x, axis=1)

            n_changed = np.sum(new_labels != labels)
            labels = new_labels

            if n_changed < N * 0.01:
                break

        return labels

    def compute_ib_objective(
        self, local_distributions: np.ndarray, labels: np.ndarray,
    ) -> Dict:
        """Обчислення I(X;Z) та I(Z;Y) для аналізу IB."""
        N = len(labels)
        unique_labels = sorted(set(labels))

        p_z_map = {}
        for z in unique_labels:
            p_z_map[z] = np.sum(labels == z) / N
        p_z_arr = np.array([p_z_map[z] for z in unique_labels])
        p_z_arr = p_z_arr[p_z_arr > 0]
        I_XZ = max(0.0, float(-np.sum(p_z_arr * np.log(p_z_arr))))

        I_ZY = 0.0
        global_dist = np.mean(local_distributions, axis=0)
        global_dist = np.maximum(global_dist, 1e-10)
        global_dist /= global_dist.sum()

        for z in unique_labels:
            mask = labels == z
            if np.sum(mask) > 0:
                cluster_dist = np.mean(local_distributions[mask], axis=0)
                cluster_dist = np.maximum(cluster_dist, 1e-10)
                cluster_dist /= cluster_dist.sum()
                kl = np.sum(cluster_dist * np.log(cluster_dist / global_dist))
                I_ZY += p_z_map[z] * kl

        return {
            'I_XZ': float(I_XZ),
            'I_ZY': float(I_ZY),
            'IB_objective': float(I_XZ - self.beta * I_ZY),
            'compression': float(I_XZ),
            'relevance': float(I_ZY),
        }



class VIBModel(nn.Module):
    """
    Variational Information Bottleneck (VIB) Neural Network.
    
    Encoder: parameterizes p(T|X) as a Gaussian distribution.
    Decoder: parameterizes q(Y|T) as a Categorical distribution over 256 classes.
    """
    def __init__(self, d_in: int, d_latent: int = 16, n_classes: int = 256):
        super().__init__()
        self.encoder_mean = nn.Linear(d_in, d_latent)
        self.encoder_logvar = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, n_classes)
        
    def encode(self, x: torch.Tensor) -> dists.Normal:
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        return dists.Normal(mean, std)
        
    def forward(self, x: torch.Tensor) -> Tuple[dists.Normal, torch.Tensor]:
        p_z_given_x = self.encode(x)
        z = p_z_given_x.rsample()  # Reparameterization trick
        logits = self.decoder(z)
        return p_z_given_x, logits


class IBOptimizer:
    """
    Information Bottleneck на кожному рівні конвертації.

    min I(S;T) - β I(T;Y)  (28)

    де S — вхід, T — стиснене представлення, Y — ціль.

    На кожному рівні конвертації обчислюємо:
    - I(S;T): скільки інформації зберігається
    - I(T;Y): скільки інформації про ціль зберігається
    - Оптимальний β через ТОЧКУ ФАЗОВОГО ПЕРЕХОДУ на IB кривій

    V6 FIX #1 (CRITICAL): Попередня реалізація обчислювала I(S;T) та I(T;Y)
    як СТАТИЧНІ KL-дивергенції, які НЕ залежали від β. Це робило IB-криву
    однією точкою, а β*=0.54 — просто другим елементом linspace.

    Правильний підхід: Blahut-Arimoto IB алгоритм, де представлення T
    дійсно змінюється з β. При β→0 T не залежить від S (максимальне
    стиснення), при β→∞ T зберігає всю інформацію про S. Це створює
    РЕАЛЬНУ IB-криву з фазовим переходом.

    Алгоритм Blahut-Arimoto для IB:
    1. p(s,y) — спільний розподіл вхід-ціль з даних
    2. Ітерації: p(t|s) ∝ p(t) exp(-β D_KL[p(y|s) || p(y|t)])
    3. I(S;T) та I(T;Y) обчислюються з оптимізованого p(t|s)
    4. Фазовий перехід знаходиться на РЕАЛЬНІЙ IB-кривій
    """

    def __init__(self, beta_range: Tuple[float, float] = (0.1, 50.0), n_beta: int = 25,
                 n_T: int = 12, ba_iters: int = 80, use_vib: bool = True,
                 vib_epochs: int = 50, vib_lr: float = 0.05):
        self.beta_range = beta_range
        self.n_beta = n_beta
        self.n_T = n_T          # Кількість станів представлення T
        self.ba_iters = ba_iters  # Ітерацій Blahut-Arimoto
        self.use_vib = use_vib
        self.vib_epochs = vib_epochs
        self.vib_lr = vib_lr
        self.level_ib_results = {}

    def _run_vib_training(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        beta: float,
        d_in: int,
    ) -> Tuple[float, float]:
        """
        Train a VIB model for a specific beta and return (I_XT, I_TY).
        """
        device = X.device
        model = VIBModel(d_in=d_in, d_latent=self.n_T, n_classes=256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.vib_lr)
        
        # Training loop
        for epoch in range(self.vib_epochs):
            optimizer.zero_grad()
            p_z_given_x, logits = model(X)
            
            # Prior r(T) = N(0, I)
            prior = dists.Normal(torch.zeros_like(p_z_given_x.mean), torch.ones_like(p_z_given_x.stddev))
            
            # KL divergence: D_KL(p(T|X) || r(T)) summed over dimensions
            kl_divs = dists.kl_divergence(p_z_given_x, prior).sum(dim=-1)  # (N,)
            kl_loss = kl_divs.mean()
            
            # Reconstruction loss: cross entropy between target distribution Y and predictions
            log_probs = F.log_softmax(logits, dim=-1)
            recon_loss = -torch.sum(Y * log_probs, dim=-1).mean()
            
            loss = beta * recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            
        # Evaluation of MI bounds
        model.eval()
        with torch.no_grad():
            p_z_given_x, logits = model(X)
            prior = dists.Normal(torch.zeros_like(p_z_given_x.mean), torch.ones_like(p_z_given_x.stddev))
            I_XT = float(dists.kl_divergence(p_z_given_x, prior).sum(dim=-1).mean().item())
            
            # Average reconstruction loss over multiple samples
            recon_sum = 0.0
            n_samples = 10
            for _ in range(n_samples):
                p_z_given_x_s = model.encode(X)
                z_s = p_z_given_x_s.sample()
                logits_s = model.decoder(z_s)
                log_probs_s = F.log_softmax(logits_s, dim=-1)
                recon_sum += -torch.sum(Y * log_probs_s, dim=-1).mean().item()
            recon_loss = recon_sum / n_samples
            
            # Baseline entropy of Y
            Y_bar = Y.mean(dim=0)
            Y_bar = torch.clamp(Y_bar, min=1e-15)
            Y_bar /= Y_bar.sum()
            H_Y = float(-torch.sum(Y_bar * torch.log(Y_bar)).item())
            
            I_TY = max(H_Y - recon_loss, 0.0)
            
        return max(I_XT, 0.0), I_TY

    def _blahut_arimoto_ib(
        self,
        p_sy: np.ndarray,
        beta: float,
    ) -> Tuple[float, float]:
        """
        Blahut-Arimoto IB алгоритм для одного β.

        Вхід: p_sy — матриця сумісного розподілу (|S| × |Y|)
        Вихід: (I_ST, I_TY) — взаємні інформації для оптимізованого p(t|s)

        Алгоритм:
        1. p(s) = Σ_y p(s,y),  p(y|s) = p(s,y)/p(s)
        2. Ініціалізуємо p(t|s) випадково
        3. Ітеруємо:
           a) p(t) = Σ_s p(s) p(t|s)
           b) p(y|t) = Σ_s [p(t|s)p(s)/p(t)] p(y|s)
           c) p(t|s) ∝ p(t) exp(-β D_KL[p(y|s) || p(y|t)])
        4. I(S;T) = Σ_{s,t} p(s)p(t|s) log[p(t|s)/p(t)]
        5. I(T;Y) = Σ_{t,y} p(t)p(y|t) log[p(y|t)/p(y)]

        FIX: Було problem що для малих p_sy матриць KL між p(y|s) та
        p(y|t) міг бути нескінченним (нульові елементи в p(y|t)).
        Тепер: використовуємо згладжені розподіли та clamp KL
        для стабільності.
        """
        n_S, n_Y = p_sy.shape
        n_T = min(self.n_T, n_S)  # T не може бути більшим за S

        # Якщо занадто мало даних — повертаємо нуль
        if n_S < 2 or n_Y < 2:
            return 0.0, 0.0

        # Маргінальні розподіли
        p_s = p_sy.sum(axis=1)  # (|S|,)
        p_y = p_sy.sum(axis=0)  # (|Y|,)

        # Перевірка: чи є достатньо маси в розподілах
        if p_s.sum() < 1e-10 or p_y.sum() < 1e-10:
            return 0.0, 0.0

        # Умовний розподіл p(y|s) — ЗГЛАДЖЕНИЙ для стабільності KL
        # Додаємо невелику константу до p_sy перед нормалізацією
        p_sy_smooth = p_sy + 1e-8 / n_Y
        p_s_smooth = p_sy_smooth.sum(axis=1)
        p_y_given_s = p_sy_smooth / np.maximum(p_s_smooth[:, None], 1e-15)  # (|S|, |Y|)

        # Ініціалізація p(t|s) — детерміністична з шумом
        rng = np.random.RandomState(int(beta * 1000) % 2**31)
        p_t_given_s = rng.dirichlet(np.ones(n_T), size=n_S)  # (|S|, |T|)
        p_t_given_s = np.maximum(p_t_given_s, 1e-10)
        p_t_given_s /= p_t_given_s.sum(axis=1, keepdims=True)

        for iteration in range(self.ba_iters):
            # a) p(t) = Σ_s p(s) p(t|s)
            p_t = (p_s[:, None] * p_t_given_s).sum(axis=0)  # (|T|,)
            p_t = np.maximum(p_t, 1e-15)
            p_t /= p_t.sum()

            # b) p(y|t) = Σ_s [p(t|s)p(s)/p(t)] p(y|s)
            # p(s|t) = p(t|s)p(s)/p(t)
            p_s_given_t = (p_t_given_s * p_s[:, None]) / p_t[None, :]  # (|S|, |T|)
            # Згладжування p(s|t) для стабільності
            p_s_given_t = np.maximum(p_s_given_t, 1e-15)
            p_s_given_t /= p_s_given_t.sum(axis=0, keepdims=True)

            p_y_given_t = p_s_given_t.T @ p_y_given_s  # (|T|, |Y|)
            p_y_given_t = np.maximum(p_y_given_t, 1e-15)
            p_y_given_t /= p_y_given_t.sum(axis=1, keepdims=True)

            # c) p(t|s) ∝ p(t) exp(-β D_KL[p(y|s) || p(y|t)])
            # D_KL[p(y|s) || p(y|t)] — CLAMP для стабільності
            # Замість повного KL з потенційними нескінченностями:
            # Використовуємо обмежений KL (clamp log ratios)
            log_ratio = np.log(np.maximum(p_y_given_s[:, None, :], 1e-15) /
                               np.maximum(p_y_given_t[None, :, :], 1e-15))
            # Обмежуємо log ratios для стабільності
            log_ratio = np.clip(log_ratio, -20.0, 20.0)
            kl_divs = np.sum(p_y_given_s[:, None, :] * log_ratio, axis=2)  # (|S|, |T|)
            # Обмежуємо KL для стабільності експоненти
            kl_divs = np.clip(kl_divs, 0.0, 50.0)

            log_p_t_given_s = np.log(p_t[None, :]) - beta * kl_divs  # (|S|, |T|)
            # Log-sum-exp для стабільності
            log_p_t_given_s -= log_p_t_given_s.max(axis=1, keepdims=True)
            p_t_given_s = np.exp(log_p_t_given_s)
            p_t_given_s = np.maximum(p_t_given_s, 1e-15)
            p_t_given_s /= p_t_given_s.sum(axis=1, keepdims=True)

        # Обчислення I(S;T)
        # I(S;T) = Σ_{s,t} p(s)p(t|s) log[p(t|s)/p(t)]
        p_st = p_s[:, None] * p_t_given_s  # (|S|, |T|)
        p_t_final = p_st.sum(axis=0)  # (|T|,)
        log_ratio_st = np.log(np.maximum(p_t_given_s, 1e-15) /
                              np.maximum(p_t_final[None, :], 1e-15))
        log_ratio_st = np.clip(log_ratio_st, -20.0, 20.0)
        I_ST = float(np.sum(p_st * log_ratio_st))

        # Обчислення I(T;Y)
        # I(T;Y) = Σ_{t,y} p(t,y) log[p(y|t)/p(y)]
        p_y_given_t_final = p_s_given_t.T @ p_y_given_s  # (|T|, |Y|)
        p_y_given_t_final = np.maximum(p_y_given_t_final, 1e-15)
        p_y_given_t_final /= p_y_given_t_final.sum(axis=1, keepdims=True)
        p_ty = p_t_final[:, None] * p_y_given_t_final  # (|T|, |Y|)
        log_ratio_ty = np.log(np.maximum(p_y_given_t_final, 1e-15) /
                              np.maximum(p_y[None, :], 1e-15))
        log_ratio_ty = np.clip(log_ratio_ty, -20.0, 20.0)
        I_TY = float(np.sum(p_ty * log_ratio_ty))

        return max(I_ST, 0.0), max(I_TY, 0.0)

    def _build_joint_distribution(
        self,
        clusters: List[Dict],
        substrate,
        Y_context_size: int = 8,
    ) -> np.ndarray:
        """
        Побудувати спільний розподіл p(s,y) з даних кластерів.

        s — байтове значення всередині кластера
        y — байтове значення в контексті після кластера

        Повертає: p_sy — матрицю (|S_used| × |Y_used|) де |S_used| та |Y_used|
        — кількість унікальних байтів що зустрічаються (зазвичай < 256).
        Також повертає маппінги byte→index.
        """
        data = substrate.raw_data
        N = len(data)

        # Збираємо унікальні байти з кластерів та контекстів
        s_bytes = set()
        y_bytes = set()

        for cluster in clusters:
            start, end = cluster['start'], cluster['end']
            for pos in range(start, end):
                s_bytes.add(data[pos])
            for pos in range(end, min(end + Y_context_size, N)):
                y_bytes.add(data[pos])

        s_bytes = sorted(s_bytes)
        y_bytes = sorted(y_bytes)
        s_map = {b: i for i, b in enumerate(s_bytes)}
        y_map = {b: i for i, b in enumerate(y_bytes)}

        n_S = len(s_bytes)
        n_Y = len(y_bytes)

        # Рахуємо спільні входження
        counts = np.zeros((n_S, n_Y), dtype=np.float64)
        for cluster in clusters:
            start, end = cluster['start'], cluster['end']
            for pos in range(start, end):
                s_idx = s_map[data[pos]]
                # Контекст: наступні Y_context_size байтів
                for offset in range(1, Y_context_size + 1):
                    y_pos = pos + offset
                    if y_pos < N:
                        y_idx = y_map.get(data[y_pos])
                        if y_idx is not None:
                            counts[s_idx, y_idx] += 1.0

        # Нормалізація до спільного розподілу
        total = counts.sum()
        if total > 0:
            p_sy = counts / total
        else:
            p_sy = np.ones((n_S, n_Y), dtype=np.float64) / (n_S * n_Y)

        return p_sy

    def _find_phase_transition(self, results_per_beta: List[Dict]) -> Dict:
        """
        Знайти точку фазового переходу на IB кривій.

        IB крива: I(S;T) як функція I(T;Y), параметризована β.
        Фазовий перехід — точка де нахил кривої різко змінюється,
        тобто друга похідна d²I(S;T)/dI(T;Y)² максимальна.

        Це відповідає β*, де компроміс між стисненням та інфоємністю
        оптимальний — більше збільшення I(T;Y) не дає пропорційного
        зростання I(S;T).
        """
        if len(results_per_beta) < 3:
            return results_per_beta[len(results_per_beta) // 2] if results_per_beta else \
                   {'beta': 1.0, 'I_ST': 0.0, 'I_TY': 0.0, 'ib_loss': 0.0}

        # Сортуємо за I_TY для побудови IB кривої
        sorted_results = sorted(results_per_beta, key=lambda x: x['I_TY'])

        # Обчислюємо нахил dI(S;T)/dI(T;Y) для кожної пари сусідніх точок
        slopes = []
        for idx in range(1, len(sorted_results)):
            d_IST = sorted_results[idx]['I_ST'] - sorted_results[idx - 1]['I_ST']
            d_ITY = sorted_results[idx]['I_TY'] - sorted_results[idx - 1]['I_TY']
            if abs(d_ITY) > 1e-12:
                slope = d_IST / d_ITY
            else:
                slope = 0.0
            slopes.append(abs(slope))

        # Фазовий перехід — де друга похідна (зміна нахилу) максимальна
        if len(slopes) < 2:
            return sorted_results[len(sorted_results) // 2]

        second_derivs = []
        for idx in range(1, len(slopes)):
            second_derivs.append(abs(slopes[idx] - slopes[idx - 1]))

        # Знаходимо точку максимальної кривизни
        max_curvature_idx = int(np.argmax(second_derivs))
        # Індекс у sorted_results: зміщення на +1 бо slopes розмір на 1 менший
        phase_idx = min(max_curvature_idx + 1, len(sorted_results) - 1)

        return sorted_results[phase_idx]

    def _compute_mi_fallback(
        self,
        clusters: List[Dict],
        substrate,
    ) -> Tuple[float, float]:
        """
        Fallback обчислення mutual information коли BA дає нуль.

        Використовує прямий підрахунок:
        I(S;T) = H(T) - H(T|S) — інформація що зберігається
        I(T;Y) = H(Y) - H(Y|T) — інформація що є релевантною

        Де S = кластерний індекс, T = байтове значення, Y = контекст.
        """
        data = substrate.raw_data
        N = len(data)
        n_clusters = len(clusters)
        if n_clusters < 2 or N < 10:
            return 0.0, 0.0

        # P(S) — розподіл кластерів
        cluster_sizes = np.array([c['size'] for c in clusters], dtype=np.float64)
        p_s = cluster_sizes / cluster_sizes.sum()

        # H(S) — ентропія кластерів
        H_S = -np.sum(p_s * np.log(np.maximum(p_s, 1e-15)))

        # I(S;T) — MI між кластером та байтовим значенням
        # = Σ_s Σ_b p(s,b) log[p(s,b) / (p(s)·p(b))]
        byte_dist = substrate.byte_distribution.astype(np.float64)
        byte_dist = np.maximum(byte_dist, 1e-15)
        byte_dist /= byte_dist.sum()

        I_ST = 0.0
        for i, c in enumerate(clusters):
            c_dist = np.maximum(c['distribution'].astype(np.float64), 1e-15)
            c_dist /= c_dist.sum()
            # p(s,b) = p(s) · p(b|s) = p_s[i] * c_dist
            p_joint = p_s[i] * c_dist
            # MI contribution: p(s,b) · log[p(b|s) / p(b)]
            I_ST += float(np.sum(p_joint * np.log(c_dist / byte_dist)))

        # I(T;Y) — MI між байтовим значенням та контекстом
        # Використовуємо перехідну матрицю як міру контекстної залежності
        trans = substrate.compute_byte_transitions().astype(np.float64)
        trans = np.maximum(trans, 1e-15)
        # MI = Σ_k Σ_l p(k,l) log[p(l|k) / p(l)]
        p_kl = byte_dist[:, None] * trans  # (256, 256)
        I_TY = float(np.sum(p_kl * np.log(trans / byte_dist[None, :])))

        return max(I_ST, 0.0), max(I_TY, 0.0)

    def compute_ib_for_level(
        self,
        clusters: List[Dict],
        substrate,
        level: int,
        Y_context_size: int = 8,
        items: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Обчислити IB для одного рівня конвертації.
        
        Якщо self.use_vib = True, використовує Variational Information Bottleneck (VIB) на PyTorch.
        Інакше — класичний Blahut-Arimoto.
        """
        if not self.use_vib:
            # Legacy Blahut-Arimoto flow
            data = substrate.raw_data
            N = len(data)

            if len(clusters) == 0:
                return {'I_ST': 0.0, 'I_TY': 0.0, 'beta_opt': 1.0, 'ib_loss': 0.0}

            # Побудувати спільний розподіл p(s,y) з даних кластерів
            p_sy = self._build_joint_distribution(clusters, substrate, Y_context_size)

            if p_sy.shape[0] < 2 or p_sy.shape[1] < 2:
                return {'I_ST': 0.0, 'I_TY': 0.0, 'beta_opt': 1.0, 'ib_loss': 0.0}

            results_per_beta = []

            for beta in np.linspace(self.beta_range[0], self.beta_range[1], self.n_beta):
                # Blahut-Arimoto: представлення T змінюється з β
                I_ST, I_TY = self._blahut_arimoto_ib(p_sy, beta)

                ib_loss = I_ST - beta * I_TY
                results_per_beta.append({
                    'beta': float(beta),
                    'I_ST': float(I_ST),
                    'I_TY': float(I_TY),
                    'ib_loss': float(ib_loss),
                })
        else:
            # PyTorch-based VIB flow
            if len(clusters) < 2:
                return {'I_ST': 0.0, 'I_TY': 0.0, 'beta_opt': 1.0, 'ib_loss': 0.0}
                
            # Extract inputs X
            if items is not None and len(items) > 0 and 'representation' in items[0]:
                reps = []
                for item in items:
                    r = item['representation']
                    if isinstance(r, torch.Tensor):
                        reps.append(r.detach())
                    else:
                        reps.append(torch.tensor(r, dtype=torch.float32))
                # Pad to same length if sizes differ (defensive fallback)
                max_len = max(len(r) for r in reps)
                padded_reps = []
                for r in reps:
                    if len(r) < max_len:
                        padded_reps.append(F.pad(r, (0, max_len - len(r))))
                    else:
                        padded_reps.append(r[:max_len])
                X = torch.stack(padded_reps)
            else:
                reps = []
                for c in clusters:
                    dist = c['distribution']
                    if isinstance(dist, torch.Tensor):
                        reps.append(dist.detach())
                    else:
                        reps.append(torch.tensor(dist, dtype=torch.float32))
                # Pad to same length if sizes differ
                max_len = max(len(r) for r in reps)
                padded_reps = []
                for r in reps:
                    if len(r) < max_len:
                        padded_reps.append(F.pad(r, (0, max_len - len(r))))
                    else:
                        padded_reps.append(r[:max_len])
                X = torch.stack(padded_reps)
                
            # Extract targets Y (aggregated right context distributions for each cluster)
            data = substrate.raw_data
            N = len(data)
            Y_list = []
            for c in clusters:
                start, end = c['start'], c['end']
                y_counts = np.zeros(256, dtype=np.float32)
                for pos in range(start, end):
                    for offset in range(1, Y_context_size + 1):
                        y_pos = pos + offset
                        if y_pos < N:
                            y_counts[data[y_pos]] += 1.0
                total = y_counts.sum()
                if total > 0:
                    y_dist = y_counts / total
                else:
                    y_dist = np.ones(256, dtype=np.float32) / 256.0
                Y_list.append(torch.tensor(y_dist, dtype=torch.float32))
            Y = torch.stack(Y_list).to(device=X.device)
            
            d_in = X.shape[1]
            if d_in == 0:
                return {'I_ST': 0.0, 'I_TY': 0.0, 'beta_opt': 1.0, 'ib_loss': 0.0}
                
            results_per_beta = []
            for beta in np.linspace(self.beta_range[0], self.beta_range[1], self.n_beta):
                I_ST, I_TY = self._run_vib_training(X, Y, beta, d_in)
                ib_loss = I_ST - beta * I_TY
                results_per_beta.append({
                    'beta': float(beta),
                    'I_ST': float(I_ST),
                    'I_TY': float(I_TY),
                    'ib_loss': float(ib_loss),
                })

        # Знайти оптимальний β через ФАЗОВИЙ ПЕРЕХІД на РЕАЛЬНІЙ IB кривій
        if results_per_beta:
            # Перевірка: якщо ВСІ I_ST = 0 → BA/VIB не знайшов структури
            # Використовуємо прямий підрахунок MI як fallback
            all_zero = all(r['I_ST'] < 1e-10 for r in results_per_beta)
            if all_zero:
                # Fallback: прямий підрахунок mutual information між
                # кластерними розподілами та їх контекстами
                I_ST_fallback, I_TY_fallback = self._compute_mi_fallback(clusters, substrate)
                if I_ST_fallback > 1e-10 or I_TY_fallback > 1e-10:
                    if I_TY_fallback > 1e-10:
                        beta_star = I_ST_fallback / I_TY_fallback
                    else:
                        beta_star = float(np.sqrt(self.beta_range[0] * self.beta_range[1]))
                    # Обмежуємо β* до [beta_range[0], beta_range[1]]
                    beta_star = float(np.clip(beta_star, self.beta_range[0], self.beta_range[1]))
                    best = {
                        'beta': beta_star,
                        'I_ST': I_ST_fallback,
                        'I_TY': I_TY_fallback,
                        'ib_loss': I_ST_fallback - beta_star * I_TY_fallback,
                    }
                else:
                    best = results_per_beta[len(results_per_beta) // 2]
            else:
                best = self._find_phase_transition(results_per_beta)
        else:
            best = {'beta': 1.0, 'I_ST': 0.0, 'I_TY': 0.0, 'ib_loss': 0.0}

        result = {
            'level': level,
            'I_ST': best['I_ST'],
            'I_TY': best['I_TY'],
            'beta_opt': best['beta'],
            'ib_loss': best['ib_loss'],
            'all_betas': results_per_beta,
        }

        self.level_ib_results[level] = result
        return result


# =============================================================================
# 6. Bayesian Modality Detector — Рівняння (24)
# =============================================================================


