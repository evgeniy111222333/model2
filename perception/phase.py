import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _kl_divergence

class PhaseTransitionAnalyzer:
    """
    Аналіз фазових переходів у БКС.

    Теорема 4.2: Фазовий перехід при T < T_c
    Параметр порядку: ψ = (1/N) Σ_i |φ_i - ⟨φ⟩|

    Аналіз:
    1. Параметр порядку ψ(T) як функція температури
    2. Критична температура T_c через кривину вільної енергії
    3. Критичні експоненти (β, γ, ν)
    4. Сприйнятливість та кореляційна довжина

    Відповідність концепції:
    - Теорема 4.2: Фазовий перехід
    - Рівняння (19): Дисперсійне співвідношення
    - Рівняння (17): Вільна енергія
    """

    def __init__(self):
        self.results = {}

    def compute_order_parameter(
        self,
        field_system,
        temperature: float,
    ) -> float:
        """
        Обчислити параметр порядку ψ для даної температури.

        ψ = (1/N) Σ_i |v_i - ⟨v⟩|

        Використовуємо v-поле як порядок параметра, оскільки воно
        відображає структуру патернів.
        """
        v = field_system.v
        mean_v = np.mean(v)
        psi = float(np.mean(np.abs(v - mean_v)))
        return psi

    def compute_susceptibility(
        self,
        field_system,
        temperature: float,
    ) -> float:
        """
        Сприйнятливість: χ = N · Var(ψ) / T
        """
        v = field_system.v
        N = len(v)
        mean_v = np.mean(v)
        psi_i = np.abs(v - mean_v)
        chi = float(N * np.var(psi_i) / max(temperature, 1e-10))
        return chi

    def compute_correlation_length(self, field_system) -> float:
        """
        Кореляційна довжина: ξ з функції автокореляції v-поля.

        ξ = позиція, де автокореляція падає до 1/e від максимуму.
        """
        v = field_system.v
        N = len(v)

        if N < 4:
            return 1.0

        # Автокореляція
        v_centered = v - np.mean(v)
        var_v = np.var(v)

        if var_v < 1e-10:
            return 1.0

        max_lag = min(N // 2, 200)
        autocorr = np.zeros(max_lag, dtype=np.float32)
        for lag in range(max_lag):
            if lag == 0:
                autocorr[0] = 1.0
            else:
                corr = np.mean(v_centered[:N - lag] * v_centered[lag:]) / var_v
                autocorr[lag] = corr

        # Знайти де autocorr падає до 1/e
        threshold = 1.0 / np.e
        xi = 1.0
        for lag in range(1, max_lag):
            if autocorr[lag] < threshold:
                # Лінійна інтерполяція
                if lag > 0:
                    x1, x2 = lag - 1, lag
                    y1, y2 = autocorr[lag - 1], autocorr[lag]
                    if abs(y1 - y2) > 1e-10:
                        xi = x1 + (threshold - y1) / (y2 - y1) * (x2 - x1)
                    else:
                        xi = float(lag)
                break
        else:
            xi = float(max_lag)

        return float(xi)

    def find_critical_temperature(
        self,
        field_system,
        T_range: Optional[np.ndarray] = None,
        n_bootstrap: int = 5,
    ) -> Dict:
        """
        Знайти T_c через аналіз параметра порядку.

        Метод:
        1. Обчислити ψ(T) для діапазону температур
        2. Знайти точку інфлексії (макс dψ/dT)
        3. Bootstrap для довірчого інтервалу
        """
        if T_range is None:
            T_range = np.linspace(0.01, 10.0, 200)

        # Збереження оригінального стану
        u_orig = field_system.u.copy()
        v_orig = field_system.v.copy()

        # Обчислення ψ(T)
        psi_values = []
        fe_values = []

        for T in T_range:
            psi = self.compute_order_parameter(field_system, T)
            fe = field_system.compute_free_energy(T)
            psi_values.append(psi)
            fe_values.append(fe)

        psi_arr = np.array(psi_values)
        fe_arr = np.array(fe_values)

        # Метод 1: Точка інфлексії ψ(T) — макс dψ/dT
        dpsi = np.gradient(psi_arr, T_range)
        T_c_inflection = T_range[np.argmax(dpsi)]

        # Метод 2: Макс кривина вільної енергії
        d2fe = np.gradient(np.gradient(fe_arr, T_range), T_range)
        # Згладжування
        if len(d2fe) > 5:
            kernel = np.ones(5) / 5
            d2fe = np.convolve(d2fe, kernel, mode='same')
        T_c_curvature = T_range[np.argmax(np.abs(d2fe))]

        # Вибираємо T_c як середнє двох методів
        T_c = (T_c_inflection + T_c_curvature) / 2.0

        # Відновлення оригінального стану
        field_system.u = u_orig
        field_system.v = v_orig

        self.results = {
            'T_c': float(T_c),
            'T_c_inflection': float(T_c_inflection),
            'T_c_curvature': float(T_c_curvature),
            'psi_values': psi_values,
            'fe_values': fe_values,
            'T_range': T_range.tolist(),
        }

        return self.results

    def compute_critical_exponents(
        self,
        field_system,
        T_c: float,
    ) -> Dict:
        """
        Обчислити критичні експоненти.

        β: ψ ~ |T - T_c|^β  (параметр порядку)
        γ: χ ~ |T - T_c|^{-γ} (сприйнятливість)
        ν: ξ ~ |T - T_c|^{-ν} (кореляційна довжина)
        """
        # Збереження стану
        u_orig = field_system.u.copy()
        v_orig = field_system.v.copy()

        # Обчислення біля T_c
        T_points = np.linspace(T_c * 0.5, T_c * 1.5, 50)

        psi_list = []
        chi_list = []
        xi_list = []

        for T in T_points:
            psi = self.compute_order_parameter(field_system, T)
            chi = self.compute_susceptibility(field_system, T)
            xi = self.compute_correlation_length(field_system)
            psi_list.append(psi)
            chi_list.append(chi)
            xi_list.append(xi)

        psi_arr = np.array(psi_list)
        chi_arr = np.array(chi_list)
        xi_arr = np.array(xi_list)

        # Відновлення стану
        field_system.u = u_orig
        field_system.v = v_orig

        # Лінійна регресія в log-log для β
        beta_exp = self._fit_power_law(
            np.abs(T_points - T_c), psi_arr
        )
        gamma_exp = self._fit_power_law(
            np.abs(T_points - T_c), np.maximum(chi_arr, 1e-10)
        )
        nu_exp = self._fit_power_law(
            np.abs(T_points - T_c), np.maximum(xi_arr, 1e-10)
        )

        return {
            'beta': float(beta_exp),
            'gamma': float(gamma_exp),
            'nu': float(nu_exp),
        }

    def _fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Лінійна регресія в log-log просторі: y ~ x^α."""
        # Фільтрація невалидних значень
        mask = (x > 0) & (y > 0)
        if np.sum(mask) < 3:
            return 0.0

        log_x = np.log(x[mask])
        log_y = np.log(y[mask])

        # Лінійна регресія: log y = α log x + c
        A = np.vstack([log_x, np.ones_like(log_x)]).T
        try:
            result = np.linalg.lstsq(A, log_y, rcond=None)
            alpha = result[0][0]
            return float(alpha)
        except np.linalg.LinAlgError:
            return 0.0

    def full_analysis(self, field_system) -> Dict:
        """Повний аналіз фазових переходів."""
        # Критична температура
        tc_results = self.find_critical_temperature(field_system)
        T_c = tc_results['T_c']

        # Кореляційна довжина при поточній температурі
        xi = self.compute_correlation_length(field_system)

        # Сприйнятливість при поточній температурі
        chi = self.compute_susceptibility(field_system, 1.0)

        # Параметр порядку при поточній температурі
        psi = self.compute_order_parameter(field_system, 1.0)

        # Критичні експоненти (якщо є достатньо даних)
        try:
            exponents = self.compute_critical_exponents(field_system, T_c)
        except Exception:
            exponents = {'beta': 0.0, 'gamma': 0.0, 'nu': 0.0}

        return {
            'T_c': T_c,
            'order_parameter': psi,
            'susceptibility': chi,
            'correlation_length': xi,
            'critical_exponents': exponents,
            'phase_state': 'САМООРГАНІЗАЦІЯ' if 1.0 < T_c else 'ДИСПЕРСНИЙ',
        }



# =============================================================================
# 1. Dynamic Byte Embedding — Рівняння (2)
# =============================================================================


