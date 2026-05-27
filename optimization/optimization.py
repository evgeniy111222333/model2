import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.core.policy import AdaptiveNumericPolicy

class CMAESOptimizer:
    """
    Еволюційна стратегія CMA-ES для мета-оптимізації параметрів БКС.

    Розділ 14.2: "еволюційна стратегія (CMA-ES), яка шукає оптимальні
    параметри D_k, θ_k, μ, a_k. Фітнес-функція — швидкість зменшення
    вільної енергії на валідаційному наборі байтових потоків."

    Реалізація: спрощений (1+λ)-CMA-ES для простору параметрів Θ.
    Повний CMA-ES (з адаптацією коваріаційної матриці) є надто
    витратним для inline виклику, але (1+λ)-ES з адаптивним кроком
    зберігає ключову ідею: ПОПУЛЯЦІЙНА оптимізація замість градієнтної.

    V2: Оптимізовано механізм збереження/відновлення стану:
    - Одноразове збереження перед циклом кандидатів (замість λ копіювань)
    - Pre-allocated буфери з np.copyto для in-place відновлення
    - Виділено _save_state / _restore_state / _evaluate_candidate

    Крок:
    1. Обираємо λ нащадків з центроїда m ∈ ℝ^d, σ ∈ ℝ⁺
    2. Оцінюємо фітнес F(x) = -ΔF_free/Δt (швидкість зменшення F)
    3. Оновлюємо m ← wmean(x_best), σ ← σ · exp(...)
    4. Застосовуємо найкращі параметри до field_system
    """

    def __init__(
        self,
        population_size: int = 8,
        sigma0: float = 0.1,
        dim: int = 8,
        adaptation_rate: float = 0.1,
    ):
        self.lam = population_size
        self.sigma = sigma0
        self.dim = dim
        self.adaptation_rate = adaptation_rate

        # Центроїд розподілу параметрів
        # Параметри: [log(D_u), log(D_v), log(F_base), log(k_base),
        #             mean(log(theta_k)), log(mu), mean(log(a_k)), log(dt)]
        self.mean = np.zeros(dim, dtype=np.float64)
        self.best_params = None
        self.best_fitness = -np.inf
        self.generation = 0

        # Історія для моніторингу
        self.fitness_history = []

        # Pre-allocated буфери стану (ініціалізуються при першому save)
        self._state_buffers = None

    def _params_from_field(self, field_system) -> np.ndarray:
        """Екстракт параметрів з field_system у вектор θ."""
        params = np.zeros(self.dim, dtype=np.float64)
        params[0] = np.log(max(field_system.D_u, 1e-10))
        params[1] = np.log(max(field_system.D_v, 1e-10))
        params[2] = np.log(max(field_system.F_base, 1e-10))
        params[3] = np.log(max(field_system.k_base, 1e-10))
        # Середні θ_k та a_k
        theta_mean = float(np.mean(field_system.theta_k[field_system.active_byte_indices]))
        a_mean = float(np.mean(field_system.a_k[field_system.active_byte_indices]))
        params[4] = np.log(max(theta_mean, 1e-10))
        params[5] = np.log(max(field_system.mu, 1e-10))
        params[6] = np.log(max(a_mean, 1e-10))
        params[7] = np.log(max(field_system.dt, 1e-10))
        self.mean = params.copy()
        return params

    def _apply_params_to_field(self, params: np.ndarray, field_system):
        """Застосувати вектор параметрів θ до field_system."""
        field_system.D_u = np.exp(params[0])
        field_system.D_v = np.exp(params[1])
        field_system.F_base = np.exp(params[2])
        field_system.k_base = np.exp(params[3])
        # Масштабуємо ВСІ θ_k та a_k пропорційно
        new_theta_scale = np.exp(params[4]) / max(
            float(np.mean(field_system.theta_k[field_system.active_byte_indices])), 1e-10)
        new_a_scale = np.exp(params[6]) / max(
            float(np.mean(field_system.a_k[field_system.active_byte_indices])), 1e-10)
        # Обмежуємо масштабування [0.5, 2.0] для стабільності
        new_theta_scale = np.clip(new_theta_scale, 0.5, 2.0)
        new_a_scale = np.clip(new_a_scale, 0.5, 2.0)
        field_system.theta_k[field_system.active_byte_indices] *= new_theta_scale
        field_system.a_k[field_system.active_byte_indices] *= new_a_scale
        # Clip для безпеки
        field_system.theta_k = np.clip(field_system.theta_k, 0.01, 1.0)
        field_system.a_k = np.clip(field_system.a_k, 0.1, 5.0)
        field_system.mu = np.exp(params[5])
        field_system.mu = np.clip(field_system.mu, 1e-5, 0.1)
        field_system.dt = np.exp(params[7])
        field_system.dt = np.clip(field_system.dt, 0.01, 1.0)
        # Перерахувати D_k відповідно до нового D_u
        byte_freq_norm = field_system.substrate.byte_distribution / max(
            field_system.substrate.byte_distribution.sum(), 1e-10)
        policy = getattr(field_system, 'numeric_field_policy', None)
        if policy is None:
            numeric_policy = getattr(field_system, 'numeric_policy', AdaptiveNumericPolicy())
            policy = numeric_policy.field_policy(byte_freq_norm, field_system.N)
        for k_idx, k in enumerate(field_system.active_byte_indices):
            if byte_freq_norm[k] > policy['freq_high_threshold']:
                field_system.D_k[k] = field_system.D_u * policy['diffusion_frequent_scale']
            elif byte_freq_norm[k] > policy['freq_mid_threshold']:
                field_system.D_k[k] = field_system.D_u * policy['diffusion_mid_scale']
            else:
                field_system.D_k[k] = field_system.D_u * policy['diffusion_rare_scale']

    def _save_state(self, field_system) -> dict:
        """
        Зберегти повний стан field_system в pre-allocated буфери.

        Використовує np.copyto для in-place копіювання у існуючі буфери,
        уникаючи повторної алокації пам'яті на кожному кандидаті.
        """
        if self._state_buffers is None or self._state_buffers.get('_shape_key') != field_system.Phi.shape:
            # Перша ініціалізація або зміна розміру — створюємо буфери
            self._state_buffers = {
                '_shape_key': field_system.Phi.shape,
                'Phi': field_system.Phi.copy(),
                'u': field_system.u.copy(),
                'v': field_system.v.copy(),
                'theta_k': field_system.theta_k.copy(),
                'a_k': field_system.a_k.copy(),
                'D_k': field_system.D_k.copy(),
                'mu': field_system.mu,
                'dt': field_system.dt,
                'D_u': field_system.D_u,
                'D_v': field_system.D_v,
            }
        else:
            # In-place копіювання без алокації нових масивів
            np.copyto(self._state_buffers['Phi'], field_system.Phi)
            np.copyto(self._state_buffers['u'], field_system.u)
            np.copyto(self._state_buffers['v'], field_system.v)
            np.copyto(self._state_buffers['theta_k'], field_system.theta_k)
            np.copyto(self._state_buffers['a_k'], field_system.a_k)
            np.copyto(self._state_buffers['D_k'], field_system.D_k)
            self._state_buffers['mu'] = field_system.mu
            self._state_buffers['dt'] = field_system.dt
            self._state_buffers['D_u'] = field_system.D_u
            self._state_buffers['D_v'] = field_system.D_v

        return self._state_buffers

    def _restore_state(self, field_system, state: dict):
        """
        Відновити стан field_system з буферів.

        Використовує np.copyto для in-place відновлення масивів.
        """
        np.copyto(field_system.Phi, state['Phi'])
        np.copyto(field_system.u, state['u'])
        np.copyto(field_system.v, state['v'])
        np.copyto(field_system.theta_k, state['theta_k'])
        np.copyto(field_system.a_k, state['a_k'])
        np.copyto(field_system.D_k, state['D_k'])
        field_system.mu = state['mu']
        field_system.dt = state['dt']
        field_system.D_u = state['D_u']
        field_system.D_v = state['D_v']

    def _evaluate_candidate(
        self,
        candidate: np.ndarray,
        field_system,
        state: dict,
        n_validation_steps: int,
    ) -> float:
        """
        Оцінити фітнес одного кандидата.

        1. Застосувати параметри кандидата
        2. Виконати коротку симуляцію
        3. Обчислити фітнес = швидкість зменшення вільної енергії
        4. Відновити стан field_system з буферів

        Args:
            candidate: вектор параметрів кандидата
            field_system: система поля
            state: збережений стан (pre-allocated буфери)
            n_validation_steps: кількість кроків симуляції

        Returns:
            fitness: швидкість зменшення вільної енергії
        """
        self._apply_params_to_field(candidate, field_system)

        # Коротка симуляція
        fe_before = field_system.compute_free_energy(1.0)
        for _ in range(n_validation_steps):
            field_system.step()
        fe_after = field_system.compute_free_energy(1.0)

        fitness = -(fe_after - fe_before) / n_validation_steps

        # Відновлюємо стан з буферів (in-place)
        self._restore_state(field_system, state)

        return fitness

    def step(
        self,
        field_system,
        free_energy: float,
        prev_free_energy: float,
        n_validation_steps: int = 10,
    ) -> Dict:
        """
        Один крок CMA-ES мета-оптимізації.

        Генерує λ нащадків, оцінює фітнес (швидкість зменшення F_free),
        вибирає найкращого, застосовує параметри.

        V2: Оптимізовано — одноразове збереження стану перед циклом
        кандидатів, np.copyto для in-place відновлення.

        Fitness = -(F_after - F_before) / n_steps = швидкість зменшення F_free
        Більше fitness = краще (F швидко падає).

        Returns:
            dict з fitness, best_params, sigma, improvement
        """
        current_params = self._params_from_field(field_system)
        dF_current = -(free_energy - prev_free_energy)

        # Одноразове збереження стану перед циклом кандидатів
        saved_state = self._save_state(field_system)

        # Генеруємо та оцінюємо нащадків
        candidates = []
        fitnesses = []
        for _ in range(self.lam):
            # Мутація: x = m + σ · N(0, I)
            noise = np.random.randn(self.dim).astype(np.float64)
            candidate = current_params + self.sigma * noise
            candidates.append(candidate)

            # Оцінка фітнесу: evaluate → restore (in-place)
            fitness = self._evaluate_candidate(
                candidate, field_system, saved_state, n_validation_steps
            )
            fitnesses.append(fitness)

        # Вибираємо найкращого
        best_idx = int(np.argmax(fitnesses))
        best_fitness = fitnesses[best_idx]
        best_candidate = candidates[best_idx]

        # Оновлюємо σ: якщо найкращий нащадок кращий за поточний → збільшуємо σ
        # інакше → зменшуємо (1/5 rule)
        if best_fitness > dF_current:
            self.sigma *= 1.2  # Успіх → досліджуємо далі
        else:
            self.sigma *= 0.8  # Невдача → звужуємо пошук
        self.sigma = np.clip(self.sigma, 0.001, 1.0)

        # Застосовуємо найкращі параметри якщо покращення
        improvement = best_fitness - dF_current
        if improvement > 0:
            # Плавне застосування (не стрибок) — адитивне змішування
            alpha = self.adaptation_rate
            blended = (1.0 - alpha) * current_params + alpha * best_candidate
            self._apply_params_to_field(blended, field_system)
            self.best_params = blended.copy()
            self.best_fitness = best_fitness

        self.generation += 1
        result = {
            'generation': self.generation,
            'best_fitness': float(best_fitness),
            'current_dF': float(dF_current),
            'improvement': float(improvement),
            'sigma': float(self.sigma),
            'applied': improvement > 0,
        }
        self.fitness_history.append(result)
        return result


# =============================================================================
# 7. Multi-Timescale Optimizer — Рівняння (29-30)
# =============================================================================



class MultiTimescaleOptimizer:
    """
    Багатомасштабна емерджентна оптимізація.

    Рівень 1 (субстрат): τ_fast ≈ мілісекунди
        dθ_k/dt = -η_θ ∂F/∂θ_k, dD_k/dt = -η_D ∂F/∂D_k     (29)

    Рівень 2 (тензори): τ_medium ≈ секунди
        W_β ← W_β - η ∇_{W_β} F, u_k ← u_k - η ∇_{u_k} F    (30)

    Рівень 3 (конвертація): τ_slow ≈ хвилини
        Варіаційний висновок для конвертаційних шарів

    Рівень 4 (мета): τ_meta ≈ години
        Оптимізація архітектури, кількість шарів, d_ℓ, T_{c,ℓ}
    """

    def __init__(
        self,
        lr_substrate: float = 0.01,
        lr_tensor: float = 0.001,
        lr_conversion: float = 0.0001,
        lr_meta: float = 0.00001,
        substrate_every: int = 10,
        tensor_every: int = 100,
        conversion_every: int = 500,
        meta_every: int = 2000,
    ):
        self.lr = {
            'substrate': lr_substrate,
            'tensor': lr_tensor,
            'conversion': lr_conversion,
            'meta': lr_meta,
        }
        self.every = {
            'substrate': substrate_every,
            'tensor': tensor_every,
            'conversion': conversion_every,
            'meta': meta_every,
        }
        self.history = {
            'substrate': [],
            'tensor': [],
            'conversion': [],
            'meta': [],
        }

    def should_update(self, level: str, step: int) -> bool:
        """Чи потрібно оновлювати на даному кроці?"""
        return step > 0 and step % self.every[level] == 0

    def substrate_step(
        self,
        field_system,
        free_energy: float,
        prev_free_energy: float,
    ) -> Dict:
        """
        Оновлення параметрів субстрату: θ_k, D_k.

        dθ_k/dt = -η_θ ∂F/∂θ_k
        dD_k/dt = -η_D ∂F/∂D_k

        Наближення: скінченні різниці вільної енергії.
        """
        lr = self.lr['substrate']
        dF = free_energy - prev_free_energy

        # Адаптація θ_k: збільшуємо для байтів з високою активацією
        if hasattr(field_system, 'theta_k'):
            for idx in range(field_system.n_active_bytes):
                k = field_system.active_byte_indices[idx]
                phi_k_mean = float(np.mean(field_system.Phi[:, idx]))
                # Якщо phi_k близько до √θ_k → стабільно, зменшуємо θ
                # Якщо phi_k мале → збільшуємо θ для кращої активації
                field_system.theta_k[k] += lr * (0.5 - phi_k_mean) * np.sign(dF)
                field_system.theta_k[k] = np.clip(field_system.theta_k[k], 0.01, 1.0)

        result = {'dF': float(dF), 'lr': lr}
        self.history['substrate'].append(result)
        return result

    def tensor_step(
        self,
        tensors,
        embeddings: np.ndarray,
        free_energy: float,
        prev_free_energy: float,
    ) -> Dict:
        """
        Оновлення тензорів взаємодії: W_β, A, u_k.

        W_β ← W_β - η ∇_{W_β} F
        """
        lr = self.lr['tensor']
        dF = free_energy - prev_free_energy

        # Неградієнтна адаптація: якщо F зменшується, підсилюємо взаємодію
        if dF < 0 and hasattr(tensors, 'W_beta'):
            scale = 1.0 + lr * abs(dF) / max(abs(free_energy), 1e-10)
            tensors.W_beta *= min(scale, 1.01)

        result = {'dF': float(dF), 'lr': lr}
        self.history['tensor'].append(result)
        return result


# =============================================================================
# 8b. Time Scale System — Шкала часу БКС
# =============================================================================



class TimeScaleSystem:
    """
    Шкала часу БКС — узгоджені часові масштаби для всіх механізмів.

    Кожен механізм має свій часовий масштаб:
    - τ_field   = 10-50 τ_0      (польова релаксація)
    - τ_cluster = 50-200 τ_0     (формування кластера)
    - τ_WM      = 10³-10⁴ τ_0    (робоча пам'ять)
    - τ_ctx     = 10⁴-10⁵ τ_0    (контекстний стан)
    - τ_cryst   = 10⁵-10⁶ τ_0    (кристалізація)
    - τ_split   = 10⁶-10⁸ τ_0    (розщеплення рівнів)
    - τ_forget  = 10⁷-10⁹ τ_0    (забування)

    Правило: τ_{k+1} ≥ 10·τ_k (аналог критерію Куранта)

    τ_0 адаптивний: τ_0(t) = τ_base / max(1, |ΔB(t)|)
    де |ΔB(t)| — кількість нових байтів за останній фізичний інтервал.
    """

    def __init__(self, tau_base: float = 1.0):
        self.tau_base = tau_base
        self.tau_0 = tau_base  # Адаптивний, змінюється з часом

        # Часові сталі у тактах τ_0 (кожна >= 10·попередня)
        self.tau_field   = 30 * tau_base              # 10-50
        self.tau_cluster = 500 * tau_base              # >= 10·τ_field
        self.tau_WM      = 10000 * tau_base            # >= 10·τ_cluster
        self.tau_ctx     = 200000 * tau_base           # >= 10·τ_WM
        self.tau_cryst   = 5000000 * tau_base          # >= 10·τ_ctx
        self.tau_split   = 100000000 * tau_base        # >= 10·τ_cryst
        self.tau_forget  = 5000000000 * tau_base       # >= 10·τ_split

        # Похідні сталі
        self.tau_decay   = self.tau_cryst          # Для забування кристалів
        self.tau_context = self.tau_ctx             # Для контекстного резонансу
        self.tau_calib   = 10 * self.tau_cluster    # Для калібрації нового рівня

        # Кількість нових байтів за останній інтервал
        self._delta_bytes = 1
        self._last_byte_count = 0

    def update_tau_0(self, new_byte_count: int):
        """
        Адаптивний τ_0 на основі щільності входу.

        τ_0(t) = τ_base / max(1, |ΔB(t)|)

        Для тексту: 1 байт → τ_0 = τ_base
        Для зображення: 10⁶ байтів → τ_0 = 10⁻⁶·τ_base
        """
        self._delta_bytes = max(1, abs(new_byte_count - self._last_byte_count))
        self._last_byte_count = new_byte_count
        self.tau_0 = self.tau_base / self._delta_bytes

    def get_tau(self, mechanism: str) -> float:
        """Отримати часову сталу для механізму."""
        mapping = {
            'field': self.tau_field,
            'cluster': self.tau_cluster,
            'WM': self.tau_WM,
            'ctx': self.tau_ctx,
            'cryst': self.tau_cryst,
            'split': self.tau_split,
            'forget': self.tau_forget,
            'decay': self.tau_decay,
            'context': self.tau_context,
            'calib': self.tau_calib,
        }
        return mapping.get(mechanism, self.tau_base)

    def validate_separation(self) -> Dict:
        """Перевірити правило τ_{k+1} ≥ 10·τ_k."""
        scales = [
            ('field', self.tau_field),
            ('cluster', self.tau_cluster),
            ('WM', self.tau_WM),
            ('ctx', self.tau_ctx),
            ('cryst', self.tau_cryst),
            ('split', self.tau_split),
            ('forget', self.tau_forget),
        ]
        violations = []
        for i in range(len(scales) - 1):
            name_next, val_next = scales[i + 1]
            name_curr, val_curr = scales[i]
            if val_next < 10 * val_curr:
                violations.append(f"{name_curr}→{name_next}: "
                                  f"τ_{name_next}={val_next:.0f} < 10·τ_{name_curr}={10*val_curr:.0f}")
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'scales': {name: val for name, val in scales},
        }


# =============================================================================
# 9. Crystallized Memory — Довгострокова пам'ять
# =============================================================================


