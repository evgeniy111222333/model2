import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _softmax, _js_divergence, _kl_divergence

class CrystallizedMemory:
    """
    Кристалізована пам'ять (довгострокова) — аналог синаптичної пластичності
    та довгострокового потенціювання (LTP).

    Когнітивний кластер, який зустрічається повторно, укріплюється.
    Якщо φ(C) > Θ_consolidate → C стає "кристалом".
    Кристал зберігається як стабільна конфігурація поля.
    Кристал модифікує параметри W_β, A, u_k перманентно.

    Банк кристалів K = {K_1, K_2, ...} де K_j = (p_j, h_j, n_j, τ_j):
    - p_j — розподіл байтових значень
    - h_j — векторне представлення вищого рівня
    - n_j — кількість зустрічей (підсилення)
    - τ_j — час останньої зустрічі (для забування)

    Правило консолідації:
    n_j(t+1) = n_j(t) + 1                     якщо C_j впізнаний
    n_j(t+1) = n_j(t) · e^{-Δt/τ_decay}       інакше (забування)

    Кристал з n_j < n_min розчиняється — аналог забування.
    """

    def __init__(
        self,
        theta_consolidate: float = 0.3,
        n_min: int = 3,
        tau_decay: float = 500000.0,
        max_crystals: int = 10000,
        # CONCEPT FIX (Розділ 14.3): Параметри стадії імпринтингу
        imprint_mode: bool = True,
        imprint_theta_consolidate: float = 0.5,
        imprint_tau_decay: float = 50000.0,
        imprint_repeat_threshold: int = 3,
    ):
        self.theta_consolidate = theta_consolidate
        self.n_min = n_min
        self.tau_decay = tau_decay
        self.max_crystals = max_crystals

        # CONCEPT FIX (Розділ 14.3): Стадія імпринтингу
        # "На початку навчання система працює в спеціальному режимі:
        # високий поріг кристалізації Θ_cryst (кристалізуються лише
        # найсильніші кластери), швидке забування τ_decay (шумові
        # кристали швидко розчиняються), та підвищена увага до повторень
        # (кластер, що з'явився ≥3 разів, кристалізується негайно,
        # оминаючи стандартний поріг)."
        self.imprint_mode = imprint_mode
        self.imprint_theta_consolidate = imprint_theta_consolidate
        self.imprint_tau_decay = imprint_tau_decay
        self.imprint_repeat_threshold = imprint_repeat_threshold
        self._repeat_counter = {}  # peak_byte → count (для відстеження повторень)

        # CONCEPT FIX (Розділ 14.5): Двоетапна фільтрація шуму
        # Етап 1: Поріг активації Θ_active — кластер C розглядається як
        # кандидат на кристалізацію лише якщо його пікова активація
        # max Φ(i,k,t) > Θ_active.
        # Етап 2: Перевірка стійкості — кластер має витримати ≥3 циклів
        # польової релаксації без суттєвої зміни форми (KL між
        # послідовними станами < ε_stable). Шумові кластери, як правило,
        # нестійкі — вони "розпливаються" при повторній обробці.
        self.theta_active = 0.1          # Поріг активації (Етап 1)
        self.stability_cycles = 3        # Кількість циклів релаксації (Етап 2)
        self.epsilon_stable = 0.1        # KL-поріг стійкості (Етап 2)
        self._pending_candidates = []    # Кандидати, що проходять перевірку стійкості

        # Банк кристалів: список словників
        self.crystals = []
        self.global_time = 0.0

    def try_consolidate(
        self,
        cluster_dist: np.ndarray,
        cluster_repr: np.ndarray,
        activation: float,
        peak_byte: int,
        field_system=None,
        cluster_start: int = 0,
        cluster_end: int = 0,
    ) -> Optional[int]:
        """
        Спробувати консолідацію кластера в кристал.

        CONCEPT FIX (Розділ 14.3): Стадія імпринтингу.
        Якщо imprint_mode=True:
        1. Високий Θ_cryst (ліше найсильніші кластери кристалізуються)
        2. Швидке забування τ_decay (шумові кристали швидко розчиняються)
        3. Кластер, що з'явився ≥ imprint_repeat_threshold разів,
           кристалізується НЕГАЙНО, оминаючи стандартний поріг.
           Це аналог критичного періоду розвитку — мозок дитини має
           підвищену пластичність, але і підвищену "вибірковість".

        CONCEPT FIX (Розділ 14.5): Двоетапна фільтрація шуму.
        Перед кристалізацією обов'язково:
        1. Перевіряємо поріг активації: max Φ > Θ_active
        2. Перевіряємо стійкість: KL між послідовними станами < ε_stable
           протягом ≥3 циклів релаксації.

        Returns:
            crystal_index якщо консолідовано, None інакше
        """
        # CONCEPT FIX (Розділ 14.5): Етап 1 — Поріг активації
        # Кластер C розглядається як кандидат на кристалізацію лише якщо
        # його пікова активація max Φ(i,k,t) > Θ_active.
        # Це відсіває шумові флуктуації, що мають малу амплітуду.
        if not self.check_activation_threshold(cluster_dist, activation):
            return None

        # CONCEPT FIX (Розділ 14.5): Етап 2 — Перевірка стійкості
        # Кластер має витримати ≥3 циклів польової релаксації без
        # суттєвої зміни форми (KL < ε_stable). Шумові кластери
        # нестійкі — вони "розпливаються" при повторній обробці.
        # Увага: якщо field_system не передано — пропускаємо (зворотна сумісність)
        if field_system is not None and cluster_end > cluster_start:
            if not self.check_stability(cluster_dist, field_system,
                                        cluster_start, cluster_end):
                return None

        # CONCEPT FIX (Розділ 14.3): Відстеження повторень для імпринтингу
        # Використовуємо унікальний ключ (peak_byte, hash) замість лише peak_byte,
        # щоб різні кластери з однаковим peak_byte не зливались.
        _repeat_key = (peak_byte, hash(cluster_dist.tobytes()))
        if self.imprint_mode:
            self._repeat_counter[_repeat_key] = self._repeat_counter.get(_repeat_key, 0) + 1
            repeat_count = self._repeat_counter[_repeat_key]

            # Негайна кристалізація при ≥ imprint_repeat_threshold повторень
            # (пройшли обидва етапи фільтрації шуму вище)
            if repeat_count >= self.imprint_repeat_threshold:
                # Перевіряємо чи вже є кристал з цим peak_byte
                for idx, crystal in enumerate(self.crystals):
                    if crystal['peak_byte'] == peak_byte and crystal.get('_repeat_key') == _repeat_key:
                        # Вже існує — підсилюємо
                        crystal['n'] += 1
                        crystal['tau'] = self.global_time
                        self._update_crystal_params(idx)
                        return idx
                # Створюємо новий кристал (негайно!)
                if len(self.crystals) >= self.max_crystals:
                    self._remove_weakest()
                crystal = {
                    'p': cluster_dist.copy(),
                    'h': cluster_repr.copy(),
                    'n': repeat_count,
                    'tau': self.global_time,
                    'peak_byte': peak_byte,
                    '_repeat_key': _repeat_key,
                    'omega': self._alpha_n(repeat_count),
                    'sigma': self._sigma_n(repeat_count),
                }
                self.crystals.append(crystal)
                return len(self.crystals) - 1

        # Стандартний поріг кристалізації (вищий в режимі імпринтингу)
        # (кластер вже пройшов двоетапну фільтрацію шуму вище)
        effective_threshold = (self.imprint_theta_consolidate
                              if self.imprint_mode
                              else self.theta_consolidate)
        if activation < effective_threshold:
            return None

        # Перевіряємо, чи є вже схожий кристал
        best_idx = -1
        best_similarity = 0.0
        for idx, crystal in enumerate(self.crystals):
            h_c = crystal['h']
            # Приведення розмірності
            min_len = min(len(h_c), len(cluster_repr))
            h_c_crop = h_c[:min_len]
            repr_crop = cluster_repr[:min_len]
            norm_c = np.linalg.norm(h_c_crop)
            norm_r = np.linalg.norm(repr_crop)
            if norm_c > 1e-10 and norm_r > 1e-10:
                sim = float(np.dot(h_c_crop, repr_crop) / (norm_c * norm_r))
            else:
                sim = 1.0 - _js_divergence(crystal['p'][:256], cluster_dist[:256])
            if sim > best_similarity:
                best_similarity = sim
                best_idx = idx

        if best_similarity > 0.85 and best_idx >= 0:
            # Впізнано існуючий кристал — підсилюємо
            self.crystals[best_idx]['n'] += 1
            self.crystals[best_idx]['tau'] = self.global_time
            n = self.crystals[best_idx]['n']
            self.crystals[best_idx]['p'] = (
                (n - 1) / n * self.crystals[best_idx]['p'] +
                1.0 / n * cluster_dist
            )
            self._update_crystal_params(best_idx)
            return best_idx
        else:
            # Новий кристал
            if len(self.crystals) >= self.max_crystals:
                self._remove_weakest()
            crystal = {
                'p': cluster_dist.copy(),
                'h': cluster_repr.copy(),
                'n': 1,
                'tau': self.global_time,
                'peak_byte': peak_byte,
                '_repeat_key': _repeat_key,
                'omega': self._alpha_n(1),
                'sigma': self._sigma_n(1),
            }
            self.crystals.append(crystal)
            return len(self.crystals) - 1

    def apply_forgetting(self, delta_t: float, cluster_recognition=None):
        """
        Застосувати забування: n_j(t+1) = n_j(t) · e^{-Δt/τ_decay}
        Кристали з n_j < n_min розчиняються.

        CONCEPT FIX (Розділ 14.3): В режимі імпринтингу використовуємо
        швидке забування (imprint_tau_decay) — шумові кристали
        розчиняються швидше, звільняючи місце для справжніх патернів.

        CONCEPT FIX (Розділ 14.7): Якщо кристали були видалені, LSH індекс
        має бути перебудований (старі індекси вже не валідні).
        """
        effective_tau = (self.imprint_tau_decay
                         if self.imprint_mode
                         else self.tau_decay)
        n_before = len(self.crystals)
        surviving = []
        for crystal in self.crystals:
            crystal['n'] *= np.exp(-delta_t / effective_tau)
            if crystal['n'] >= self.n_min:
                self._update_crystal_params(len(surviving))
                surviving.append(crystal)
        self.crystals = surviving
        # CONCEPT FIX (Розділ 14.7): Якщо кристали були видалені —
        # перебудовуємо LSH індекс з нуля (старі індекси не валідні).
        if cluster_recognition is not None and len(self.crystals) < n_before:
            cluster_recognition._lsh_tables = [
                {} for _ in range(cluster_recognition.lsh_n_tables)
            ]
            for ci in range(len(self.crystals)):
                cluster_recognition.update_lsh_index(ci, self)

    def _alpha_n(self, n: int) -> float:
        """Функція підсилення α(n_j) — глибина ями атрактора.

        CONCEPT FIX (Розділ 14.4): ω_j ∝ log(n_j + 1).
        Концепція каже: "глибина атракторної ями ω пропорційна log(n_j + 1)".
        Попередній код: min(1.0 + 0.5·log1p(n), 5.0) — вже відповідав концепції.
        Зберігаємо з обрізкою зверху для стабільності.
        """
        return min(1.0 + 0.5 * np.log1p(n), 5.0)

    def _sigma_n(self, n: int) -> float:
        """Ширина ями атрактора — обернено пропорційна n_j.

        CONCEPT FIX (Розділ 14.4): σ_j ∝ 1/√n_j.
        Концепція каже: "ширина σ обернено пропорційна n_j" — це степеневий
        закон 1/√n, а НЕ логарифмічний 1/(1+c·log n).

        Різниця суттєва:
        - 1/√n: при n=10 → σ=0.316, при n=100 → σ=0.1, при n=1000 → σ=0.032
        - 1/(1+0.1·log n): при n=10 → σ=0.81, при n=100 → σ=0.71, при n=1000 → σ=0.59

        Степеневий закон забезпечує ШВИДКЕ звуження — чим частіше бачили кристал,
        тим точніше впізнавання (вузька яма = точний атрактор). Логарифмічний закон
        звужує занадто повільно — навіть при n=1000 кристал має широку яму σ≈0.59,
        що означає неточне впізнавання (забагато хибних спрацювань).

        Мінімум σ = 0.05 для запобігання занадто вузьким ямам (неможливо впізнати
        при малому шумі). n=1 → σ=1.0 (широко, бо вперше бачимо — невизначеність).
        """
        if n <= 0:
            return 1.0
        return max(1.0 / np.sqrt(n), 0.05)

    def _update_crystal_params(self, idx: int):
        """Оновити параметри кристала на основі n_j."""
        if idx < 0 or idx >= len(self.crystals):
            return
        crystal = self.crystals[idx]
        crystal['omega'] = self._alpha_n(int(crystal['n']))
        crystal['sigma'] = self._sigma_n(int(crystal['n']))

    def _remove_weakest(self):
        """Видалити найслабший кристал (з найменшим n_j)."""
        if not self.crystals:
            return
        min_idx = min(range(len(self.crystals)),
                      key=lambda i: self.crystals[i]['n'])
        self.crystals.pop(min_idx)

    def check_activation_threshold(
        self,
        cluster_dist: np.ndarray,
        peak_activation: float,
    ) -> bool:
        """
        CONCEPT FIX (Розділ 14.5): Етап 1 — Поріг активації.

        Кластер C розглядається як кандидат на кристалізацію лише якщо
        його пікова активація max Φ(i,k,t) > Θ_active.

        Args:
            cluster_dist: розподіл байтових значень кластера
            peak_activation: максимальна активація Φ в кластері

        Returns:
            True якщо кластер проходить поріг активації
        """
        return peak_activation > self.theta_active

    def check_stability(
        self,
        cluster_dist: np.ndarray,
        field_system,
        cluster_start: int,
        cluster_end: int,
    ) -> bool:
        """
        CONCEPT FIX (Розділ 14.5): Етап 2 — Перевірка стійкості.

        Кластер має витримати ≥3 циклів польової релаксації без
        суттєвої зміни форми (KL-дивергенція між послідовними
        станами < ε_stable).

        Реалізація: робимо stability_cycles кроків поля, обчислюємо
        KL між розподілом до та після. Якщо KL < ε_stable для ВСІХ
        циклах → кластер стійкий.

        Args:
            cluster_dist: поточний розподіл кластера
            field_system: польова система для релаксації
            cluster_start: початкова позиція кластера
            cluster_end: кінцева позиція кластера

        Returns:
            True якщо кластер стійкий
        """
        if field_system is None:
            return True  # Без поля не можемо перевірити — пропускаємо

        current_dist = cluster_dist.copy()
        for cycle in range(self.stability_cycles):
            # Зберігаємо стан поля
            phi_orig = field_system.Phi.copy()
            u_orig = field_system.u.copy()
            v_orig = field_system.v.copy()

            # Один крок релаксації
            field_system.step()

            # Обчислюємо новий розподіл кластера
            positions = np.arange(cluster_start, min(cluster_end, field_system.N))
            if len(positions) == 0:
                field_system.Phi = phi_orig
                field_system.u = u_orig
                field_system.v = v_orig
                return False

            local_dist = field_system.substrate.compute_local_distributions(
                window=max(field_system.N // 20, 4)
            )
            new_dist = np.mean(local_dist[positions], axis=0)
            new_dist = new_dist / max(new_dist.sum(), 1e-10)

            # KL-дивергенція між поточним та новим розподілом
            kl = _kl_divergence(
                np.maximum(current_dist, 1e-10),
                np.maximum(new_dist, 1e-10)
            )

            # Відновлюємо стан поля (не хочемо змінювати його під час перевірки)
            field_system.Phi = phi_orig
            field_system.u = u_orig
            field_system.v = v_orig

            if kl > self.epsilon_stable:
                return False  # Нестійкий — розпливається

            current_dist = new_dist

        return True  # Стійкий — пройшов усі цикли

    def modify_parameters(self, tensors, embeddings):
        """
        Кристали модифікують параметри W_β, A, u_k перманентно.
        Кожен кристал зсуває параметри у напрямку, що підсилює
        його атракторний басейн.
        """
        if not self.crystals or tensors is None:
            return
        for crystal in self.crystals:
            n = int(crystal['n'])
            if n < self.n_min:
                continue
            peak = crystal['peak_byte']
            strength = 0.001 * self._alpha_n(n)
            if hasattr(tensors, 'u_beta') and 0 <= peak < 256:
                tensors.u_beta[peak] *= (1.0 + strength * 0.01)

    def get_attractor_potential(
        self,
        h_current: np.ndarray,
        k: int,
    ) -> float:
        """
        Атракторний потенціал V_attract для позиції та байтового значення k.

        V_attract(i,k,t) = -Σ_{j∈K} ω_j · exp(-||h_i-h_j||²/(2σ_j²))
                                    · δ(k - peak(K_j))
        """
        potential = 0.0
        for crystal in self.crystals:
            if crystal['peak_byte'] != k:
                continue  # δ(k - peak(K_j)) = 0
            h_j = crystal['h']
            min_len = min(len(h_current), len(h_j))
            h_c = h_current[:min_len]
            h_j_c = h_j[:min_len]
            dist_sq = float(np.sum((h_c - h_j_c) ** 2))
            sigma_sq = crystal['sigma'] ** 2
            potential -= crystal['omega'] * np.exp(-dist_sq / (2.0 * sigma_sq))
        return potential

    def step(self, delta_t: float = 1.0, cluster_recognition=None):
        """Оновити глобальний час та періодично застосовувати забування.

        CONCEPT FIX (Розділ 14.7): Передаємо cluster_recognition для
        перебудови LSH індексу при видаленні кристалів.
        """
        self.global_time += delta_t
        if int(self.global_time) % 1000 == 0 and int(self.global_time) > 0:
            self.apply_forgetting(delta_t, cluster_recognition=cluster_recognition)


# =============================================================================
# 10. Working Memory — Короткострокова пам'ять
# =============================================================================



class WorkingMemory:
    """
    Робоча пам'ять (короткострокова) — кільцевий буфер кластерів.

    W(t) = [C_{t-M+1}, C_{t-M+2}, ..., C_t]

    Не простий FIFO. Кластери мають вагу релевантності, яка залежить від:
    1. Часу (свіжіший = важливіший): recency = e^{-time_decay · age}
    2. Зв'язку з поточним вхідним потоком (γ-спорідненість)
    3. Емоційного забарвлення (новизна / аномальність)
    """

    def __init__(
        self,
        capacity: int = 50,
        time_decay: float = 0.01,
        gamma_exponent: float = 1.0,
        novelty_exponent: float = 1.0,
        recency_exponent: float = 1.0,
    ):
        self.capacity = capacity
        self.time_decay = time_decay
        # Експоненти для мультиплікативної формули (Рівняння 33):
        # relevance = recency^α_r · γ^α_γ · novelty^α_n
        # α > 1 підсилює чутливість, α < 1 послаблює
        self.gamma_exponent = gamma_exponent
        self.novelty_exponent = novelty_exponent
        self.recency_exponent = recency_exponent

        self.buffer = []  # {cluster, timestamp, novelty, gamma_affinity, relevance}
        self.current_time = 0.0

    def add(
        self,
        cluster: Dict,
        gamma_affinity: float = 0.5,
        novelty: float = 0.5,
    ):
        """
        Додати кластер до робочої пам'яті.
        Якщо буфер повний — видаляємо кластер з найменшою вагою.
        """
        self.current_time += 1.0
        entry = {
            'cluster': cluster,
            'timestamp': self.current_time,
            'novelty': novelty,
            'gamma_affinity': gamma_affinity,
            'relevance': 0.0,
        }
        if len(self.buffer) >= self.capacity:
            self._update_relevances()
            min_idx = min(range(len(self.buffer)),
                          key=lambda i: self.buffer[i]['relevance'])
            self.buffer[min_idx] = entry
        else:
            self.buffer.append(entry)

    def _update_relevances(self):
        """
        Обчислити ваги релевантності для всіх кластерів.

        Рівняння 33: relevance = recency · γ_affinity · novelty
        Мультиплікативний добуток (не адитивна сума!):
        якщо будь-який множник ≈ 0 → весь добуток ≈ 0,
        тому застарілі ТА/АБО нерелевантні записи відсікаються ПОВНІСТЮ.
        """
        if not self.buffer:
            return
        for entry in self.buffer:
            age = self.current_time - entry['timestamp']
            recency = np.exp(-self.time_decay * age)
            gamma_aff = entry['gamma_affinity']
            novelty = entry['novelty']
            # Мультиплікативний добуток (Рівняння 33):
            # кожен множник ∈ [0, 1], тому добуток ∈ [0, 1]
            # Якщо хоч один ≈ 0 → весь добуток ≈ 0
            # Експоненти підсилюють/послаблюють чутливість кожного фактора
            entry['relevance'] = float(
                (recency ** self.recency_exponent) *
                (gamma_aff ** self.gamma_exponent) *
                (novelty ** self.novelty_exponent)
            )

    def get_context_vector(self, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Контекстний вектор робочої пам'яті.
        Зважена сума представлень кластерів, де ваги = релевантність.
        """
        self._update_relevances()
        if not self.buffer:
            return np.zeros(256, dtype=np.float32)
        repr_dim = 256
        context = np.zeros(repr_dim, dtype=np.float32)
        total_weight = 0.0
        for entry in self.buffer:
            w = entry['relevance']
            dist = entry['cluster'].get('distribution', np.zeros(repr_dim))
            if len(dist) < repr_dim:
                dist = np.pad(dist, (0, repr_dim - len(dist)))
            context += w * dist[:repr_dim]
            total_weight += w
        if total_weight > 0:
            context /= total_weight
        return context

    def get_most_relevant(self, n: int = 5) -> List[Dict]:
        """Отримати n найбільш релевантних кластерів."""
        self._update_relevances()
        sorted_entries = sorted(self.buffer,
                                key=lambda e: e['relevance'], reverse=True)
        return sorted_entries[:n]

    def clear(self):
        """Очистити робочу пам'ять."""
        self.buffer = []
        self.current_time = 0.0


# =============================================================================
# 11. Cluster Recognition — Впізнавання через атрактори поля
# =============================================================================



class SequenceAssociativeMemory:
    """Context-indexed byte sequence memory for human/text-like streams."""

    def __init__(
        self,
        max_radius: int = 8,
        min_count: int = 2,
        max_entries: int = 200000,
        unknown_bytes: Tuple[int, ...] = (0, ord('?')),
    ):
        self.max_radius = max(1, int(max_radius))
        self.min_count = max(1, int(min_count))
        self.max_entries = max_entries
        self.unknown_bytes = set(int(b) for b in unknown_bytes)
        self.counts: Dict[Tuple[str, bytes, bytes], Dict[int, int]] = {}
        self.total_observations = 0

    def _add(self, key: Tuple[str, bytes, bytes], target: int):
        if len(self.counts) >= self.max_entries and key not in self.counts:
            return
        bucket = self.counts.setdefault(key, {})
        bucket[target] = bucket.get(target, 0) + 1

    def observe(self, data: bytes):
        if isinstance(data, str):
            data = data.encode('utf-8')
        arr = bytes(data)
        n = len(arr)
        for i, target in enumerate(arr):
            target = int(target)
            if target in self.unknown_bytes:
                continue
            for radius in range(1, self.max_radius + 1):
                left = arr[max(0, i - radius):i]
                right = arr[i + 1:min(n, i + 1 + radius)]
                if left and right:
                    self._add(('lr', left, right), target)
                if left:
                    self._add(('l', left, b''), target)
                if right:
                    self._add(('r', b'', right), target)
            self.total_observations += 1

    def predict_distribution(self, data: bytes, index: int) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, str):
            data = data.encode('utf-8')
        arr = bytes(data)
        n = len(arr)
        scores = np.zeros(256, dtype=np.float64)
        support = 0.0
        matched_keys = 0

        for radius in range(self.max_radius, 0, -1):
            left = arr[max(0, index - radius):index]
            right = arr[index + 1:min(n, index + 1 + radius)]
            keys = []
            if left and right:
                keys.append((('lr', left, right), 3.0))
            if left:
                keys.append((('l', left, b''), 1.0))
            if right:
                keys.append((('r', b'', right), 1.0))

            radius_weight = (radius / self.max_radius) ** 2
            for key, mode_weight in keys:
                bucket = self.counts.get(key)
                if not bucket:
                    continue
                total = sum(bucket.values())
                if total < self.min_count:
                    continue
                weight = radius_weight * mode_weight
                for byte_val, count in bucket.items():
                    scores[byte_val] += weight * count
                support += weight * total
                matched_keys += 1
            if matched_keys > 0 and np.max(scores) > 0:
                break

        if np.sum(scores) <= 0:
            return np.zeros(256, dtype=np.float32), {
                'confidence': 0.0,
                'support': 0.0,
                'matched_keys': 0,
                'top_byte': None,
            }

        probs = scores / np.sum(scores)
        top_byte = int(np.argmax(probs))
        return probs.astype(np.float32), {
            'confidence': float(probs[top_byte]),
            'support': float(support),
            'matched_keys': int(matched_keys),
            'top_byte': top_byte,
        }

    def apply_to_field(
        self,
        field_system,
        raw_data: bytes,
        strength: float = 1.25,
        min_confidence: float = 0.55,
        placeholder_only: bool = True,
    ) -> Dict:
        if field_system is None or not self.counts:
            return {'applied': 0, 'mean_confidence': 0.0}
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')

        active_to_col = {
            int(byte_val): col
            for col, byte_val in enumerate(field_system.active_byte_indices)
        }
        applied = 0
        confidences = []
        for i, observed in enumerate(raw_data[:field_system.N]):
            observed = int(observed)
            if placeholder_only and observed not in self.unknown_bytes:
                continue
            probs, meta = self.predict_distribution(raw_data, i)
            confidence = float(meta['confidence'])
            if confidence < min_confidence:
                continue

            top_indices = np.argsort(probs)[-5:][::-1]
            for byte_val in top_indices:
                prob = float(probs[byte_val])
                if prob <= 0.0 or int(byte_val) not in active_to_col:
                    continue
                col = active_to_col[int(byte_val)]
                field_system.Phi[i, col] += strength * prob
            if observed in active_to_col:
                field_system.Phi[i, active_to_col[observed]] -= 0.5 * strength * confidence
            applied += 1
            confidences.append(confidence)

        if applied:
            field_system.Phi = np.clip(field_system.Phi, -1.5, 2.0)
            phi_positive = np.maximum(field_system.Phi, 0.0)
            field_system.u = np.sum(phi_positive, axis=1).astype(np.float32)
            field_system.v = np.max(phi_positive, axis=1).astype(np.float32)

        return {
            'applied': int(applied),
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
        }



