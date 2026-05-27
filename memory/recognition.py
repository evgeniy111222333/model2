import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _softmax, _sigmoid, _safe_normalize

class ClusterRecognition:
    """
    Впізнавання кластерів через атрактори поля.

    Кожен кристалізований кластер K_j створює в полі атракторний басейн —
    область простору станів, яка "притягує" поле, якщо вхід схожий.

    V_attract(i,k,t) = -Σ_{j∈K} ω_j · exp(-||h_i(t)-h_j||²/(2σ_j²))
                              · δ(k - peak(K_j))

    Три результати впізнавання:
    1. Поле впало в існуючий басейн K_j → Впізнано → n_j += 1
    2. Поле впало на межі двох басейнів → Амбівалентно → новий гібридний кристал
    3. Поле не впало в жоден басейн → Нове → новий кристал K_new

    CONCEPT FIX (Розділ 14.7): LSH для O(1) пошуку кристалів.
    Замість O(|K|) повного перебору, використовуємо локально-чутливе
    хешування (LSH) для отримання короткого списку кандидатів
    K_cand ⊂ K з |K_cand| ≈ 100 замість |K|.
    "LSH гарантовано знаходить усі кристали з ||h_i-h_j|| < r з
    ймовірністю ≥ 1-δ_LSH."
    """

    def __init__(
        self,
        recognition_threshold: float = 0.6,
        ambivalence_threshold: float = 0.3,
        # CONCEPT FIX (Розділ 14.7): LSH параметри
        use_lsh: bool = True,
        lsh_n_tables: int = 8,
        lsh_n_bits: int = 12,
        lsh_max_candidates: int = 100,
    ):
        self.recognition_threshold = recognition_threshold
        self.ambivalence_threshold = ambivalence_threshold
        self.recognition_history = []

        # CONCEPT FIX (Розділ 14.7): LSH індекс
        # n_tables хеш-таблиць з n_bits біт кожна.
        # Більше таблиць → вища ймовірність знайти сусідів (recall).
        # Більше біт → точніші хеші, але менше кандидатів (precision).
        self.use_lsh = use_lsh
        self.lsh_n_tables = lsh_n_tables
        self.lsh_n_bits = lsh_n_bits
        self.lsh_max_candidates = lsh_max_candidates
        # Хеш-таблиці: list of dicts {hash_key → [crystal_indices]}
        self._lsh_tables = [{} for _ in range(lsh_n_tables)]
        # Випадкові проекції для LSH: (n_tables, n_bits, d_repr)
        # Ініціалізуються ледачо при першому використанні
        self._lsh_projections = None
        self._lsh_d_repr = 64  # Розмірність за замовчуванням

    def _init_lsh_projections(self, d_repr: int):
        """Ініціалізувати випадкові проекції для LSH."""
        self._lsh_d_repr = d_repr
        self._lsh_projections = np.random.randn(
            self.lsh_n_tables, self.lsh_n_bits, d_repr
        ).astype(np.float32) * 0.1

    def _compute_lsh_hash(self, h: np.ndarray, table_idx: int) -> str:
        """
        Обчислити LSH-хеш вектора h для таблиці table_idx.

        LSH: hash = sign(projection · h) → бітовий рядок.
        Близькі вектори (малий кут) мають високу ймовірність
        однакового хешу (властивість LSH).
        """
        if self._lsh_projections is None:
            self._init_lsh_projections(min(len(h), 256))

        h_proj = h[:self._lsh_d_repr] if len(h) >= self._lsh_d_repr else \
                 np.pad(h, (0, self._lsh_d_repr - len(h)))
        proj = self._lsh_projections[table_idx]  # (n_bits, d_repr)
        bits = (proj @ h_proj > 0).astype(int)
        return ''.join(map(str, bits))

    def update_lsh_index(self, crystal_idx: int, crystal_memory):
        """
        Додати кристал до LSH індексу.

        Викликати після кожної консолідації нового кристала.
        """
        if not self.use_lsh or crystal_idx >= len(crystal_memory.crystals):
            return
        crystal = crystal_memory.crystals[crystal_idx]
        h = crystal['h']
        for t in range(self.lsh_n_tables):
            key = self._compute_lsh_hash(h, t)
            if key not in self._lsh_tables[t]:
                self._lsh_tables[t][key] = []
            self._lsh_tables[t][key].append(crystal_idx)

    def _get_lsh_candidates(self, h_current: np.ndarray, crystal_memory) -> List[int]:
        """
        Отримати кандидатів через LSH замість O(|K|) перебору.

        Повертає список індексів кристалів-кандидатів,
        обмежений lsh_max_candidates.
        """
        if not self.use_lsh or not crystal_memory.crystals:
            return list(range(len(crystal_memory.crystals)))

        candidate_set = set()
        for t in range(self.lsh_n_tables):
            key = self._compute_lsh_hash(h_current, t)
            if key in self._lsh_tables[t]:
                candidate_set.update(self._lsh_tables[t][key])

        # Обмежуємо кількість кандидатів
        candidates = list(candidate_set)[:self.lsh_max_candidates]

        # Fallback: якщо LSH не знайшов жодного (рідкісний випадок),
        # повертаємо всі кристали
        if not candidates and crystal_memory.crystals:
            candidates = list(range(min(len(crystal_memory.crystals), self.lsh_max_candidates)))

        return candidates

    def recognize(
        self,
        h_current: np.ndarray,
        crystal_memory,  # CrystallizedMemory
        cluster_dist: np.ndarray,
        activation: float,
        peak_byte: int,
        field_system=None,
        cluster_start: int = 0,
        cluster_end: int = 0,
    ) -> Dict:
        """
        Впізнати кластер через атракторну динаміку.

        CONCEPT FIX (Розділ 14.5): Передаємо field_system та позиції кластера
        в try_consolidate для двоетапної фільтрації шуму.

        Returns:
            dict з 'result': 'recognized' | 'ambivalent' | 'novel'
        """
        if not crystal_memory.crystals:
            new_idx = crystal_memory.try_consolidate(
                cluster_dist, h_current, activation, peak_byte,
                field_system=field_system,
                cluster_start=cluster_start,
                cluster_end=cluster_end,
            )
            result = {
                'result': 'novel',
                'crystal_idx': None,
                'ambivalent_pair': None,
                'new_crystal_idx': new_idx,
            }
            self.recognition_history.append(result)
            return result

        # Атракторні оцінки для кандидатів (LSH замість O(|K|) перебору)
        # CONCEPT FIX (Розділ 14.7): використовуємо LSH для O(|K_cand|) замість O(|K|)
        candidate_indices = self._get_lsh_candidates(h_current, crystal_memory)
        score_rows = []
        for idx in candidate_indices:
            if idx >= len(crystal_memory.crystals):
                continue
            crystal = crystal_memory.crystals[idx]
            h_j = crystal['h']
            min_len = min(len(h_current), len(h_j))
            h_c = h_current[:min_len]
            h_j_c = h_j[:min_len]
            dist_sq = float(np.sum((h_c - h_j_c) ** 2))
            sigma_sq = max(float(crystal['sigma'] ** 2), 1e-12)
            score = float(crystal['omega'] * np.exp(-dist_sq / (2.0 * sigma_sq)))
            score_rows.append((idx, score, dist_sq))

        score_rows.sort(key=lambda row: row[1], reverse=True)
        best_idx = int(score_rows[0][0]) if score_rows else -1
        best_score = float(score_rows[0][1]) if score_rows else 0.0
        best_dist_sq = float(score_rows[0][2]) if score_rows else float('inf')
        second_score = float(score_rows[1][1]) if len(score_rows) > 1 else 0.0

        if best_score > self.recognition_threshold:
            # 1. Впізнано
            idx = best_idx
            crystal_memory.crystals[idx]['n'] += 1
            crystal_memory.crystals[idx]['tau'] = crystal_memory.global_time
            crystal_memory._update_crystal_params(idx)
            result = {
                'result': 'recognized',
                'crystal_idx': idx,
                'ambivalent_pair': None,
                'new_crystal_idx': None,
                'score': best_score,
                'second_score': second_score,
                'dist_sq': best_dist_sq,
            }
        elif (best_score > self.ambivalence_threshold and
              second_score > self.ambivalence_threshold and
              best_score - second_score < self.ambivalence_threshold):
            # 2. Амбівалентно — на межі двох басейнів
            idx1, idx2 = int(score_rows[0][0]), int(score_rows[1][0])
            h_hybrid = 0.5 * (crystal_memory.crystals[idx1]['h'] +
                               crystal_memory.crystals[idx2]['h'])
            p_hybrid = 0.5 * (crystal_memory.crystals[idx1]['p'] +
                               crystal_memory.crystals[idx2]['p'])
            p_hybrid = _safe_normalize(p_hybrid)
            new_idx = crystal_memory.try_consolidate(
                p_hybrid, h_hybrid, activation, peak_byte,
                field_system=field_system,
                cluster_start=cluster_start,
                cluster_end=cluster_end,
            )
            result = {
                'result': 'ambivalent',
                'crystal_idx': None,
                'ambivalent_pair': (idx1, idx2),
                'new_crystal_idx': new_idx,
                'score': best_score,
                'second_score': second_score,
                'dist_sq': best_dist_sq,
            }
        else:
            # 3. Нове
            new_idx = crystal_memory.try_consolidate(
                cluster_dist, h_current, activation, peak_byte,
                field_system=field_system,
                cluster_start=cluster_start,
                cluster_end=cluster_end,
            )
            result = {
                'result': 'novel',
                'crystal_idx': None,
                'ambivalent_pair': None,
                'new_crystal_idx': new_idx,
                'score': best_score,
                'second_score': second_score,
                'dist_sq': best_dist_sq,
            }

        self.recognition_history.append(result)
        return result

    def inject_attractor_field(
        self,
        field_system,
        crystal_memory,
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Ін'єкція атракторного потенціалу V_attract у польову динаміку.
        Поле "скочується" до найближчого атрактора при релаксації.
        """
        if not crystal_memory.crystals:
            return
        N = field_system.N
        n_bytes = field_system.n_active_bytes
        for idx in range(n_bytes):
            k = field_system.active_byte_indices[idx]
            phi_k = field_system.Phi[:, idx]
            for i in range(N):
                if embeddings is not None and i < len(embeddings):
                    h_i = embeddings[i]
                else:
                    h_i = phi_k[i:i+1] if i < len(phi_k) else np.zeros(1)
                V = crystal_memory.get_attractor_potential(h_i, k)
                field_system.Phi[i, idx] += 0.01 * V * phi_k[i]
        # CONCEPT FIX: Дозволяємо негативні Phi (див. FieldSystemV6.step)
        field_system.Phi = np.clip(field_system.Phi, -1.5, 2.0)
        Phi_positive = np.maximum(field_system.Phi, 0.0)
        field_system.u = np.sum(Phi_positive, axis=1).astype(np.float32)
        field_system.v = np.max(Phi_positive, axis=1).astype(np.float32)
        # CONCEPT FIX: без нормалізації u/v (див. FieldSystemV6.step)


# =============================================================================
# 12. Context Resonance — Послідовний контекст
# =============================================================================



class ContextResonance:
    """
    Контекстний резонанс — механізм послідовного контексту.

    Кожен новий кластер C_t резонує з кристалізованим банком:
    r_j(t) = γ(h_t, h_j) · α(n_j) · e^{-Δt_j / τ_context}

    Контекстний вектор:
    ctx(t) = Σ_{j∈K} r_j(t) · h_j

    Ін'єкція в польове рівняння:
    ∂Φ(i,k,t)/∂t += κ · ctx(t) · σ_gate(ctx, Φ)

    Накопичувальний контекст:
    s(t+1) = β_s · s(t) + (1 - β_s) · ctx(t),  β_s ≈ 0.99
    """

    def __init__(
        self,
        kappa: float = 0.1,
        tau_context: float = 50000.0,
        d_representation: int = 64,
        beta_s: float = 0.99,
    ):
        self.kappa = kappa
        self.tau_context = tau_context
        self.d_representation = d_representation
        self.beta_s = beta_s

        # Поточний контекстний вектор
        self.ctx = np.zeros(d_representation, dtype=np.float32)
        # Накопичувальний контекст
        self.s = np.zeros(d_representation, dtype=np.float32)

        self.resonance_history = []

    def compute_resonance(
        self,
        h_current: np.ndarray,
        crystal_memory,
    ) -> np.ndarray:
        """
        Контекстний вектор через резонанс з банком кристалів.

        r_j(t) = γ(h_t, h_j) · α(n_j) · e^{-Δt_j / τ_context}
        ctx(t) = Σ_j r_j(t) · h_j
        """
        if not crystal_memory.crystals:
            self.ctx = np.zeros(self.d_representation, dtype=np.float32)
            return self.ctx

        ctx = np.zeros(self.d_representation, dtype=np.float32)
        total_resonance = 0.0

        for crystal in crystal_memory.crystals:
            h_j = crystal['h']
            n_j = int(crystal['n'])
            tau_j = crystal['tau']

            # γ(h_t, h_j) — косинусна спорідненість
            min_len = min(len(h_current), len(h_j))
            h_c = h_current[:min_len]
            h_j_c = h_j[:min_len]
            norm_c = np.linalg.norm(h_c)
            norm_j = np.linalg.norm(h_j_c)
            if norm_c > 1e-10 and norm_j > 1e-10:
                gamma = float(np.dot(h_c, h_j_c) / (norm_c * norm_j))
                gamma = max(0.0, gamma)
            else:
                gamma = 0.0

            alpha = crystal.get('omega', 1.0)
            delta_t = crystal_memory.global_time - tau_j
            time_decay = np.exp(-delta_t / self.tau_context)

            r_j = gamma * alpha * time_decay

            h_j_full = self._adapt(h_j, self.d_representation)
            ctx += r_j * h_j_full
            total_resonance += r_j

        if total_resonance > 0:
            ctx /= total_resonance

        self.ctx = ctx.astype(np.float32)
        # Накопичувальний контекст: s(t+1) = β_s · s(t) + (1-β_s) · ctx(t)
        self.s = (self.beta_s * self.s +
                  (1.0 - self.beta_s) * self.ctx).astype(np.float32)

        self.resonance_history.append({
            'total_resonance': float(total_resonance),
            'ctx_norm': float(np.linalg.norm(self.ctx)),
            's_norm': float(np.linalg.norm(self.s)),
        })

        return self.ctx

    def inject_into_field(
        self,
        field_system,
        ctx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        V7 FIX: Ін'єкція контексту — неперервний терм в ОДР (Рівняння 37).

        Замість імпульсного зсуву Phi раз на 100 кроків, ми встановлюємо
        context_injection_vector та context_injection_kappa в FieldSystemV6,
        після чого кожен виклик field_system.step() автоматично додає:
            ∂Φ/∂t += κ · ctx · σ_gate(ctx, Φ)
        як неперервний терм диференціального рівняння, масштабований на dt.

        Це гарантує ПЛАВНУ еволюцію поля під впливом накопиченого контексту,
        замість різких стрибків кожні 100 кроків.
        """
        if ctx is None:
            ctx = self.ctx
        ctx_norm = float(np.linalg.norm(ctx))
        if ctx_norm < 1e-10:
            # Немає контексту — обнуляємо ін'єкцію
            field_system.context_injection_vector = np.zeros(
                field_system.n_active_bytes, dtype=np.float32)
            field_system.context_injection_kappa = 0.0
            return np.zeros(field_system.n_active_bytes, dtype=np.float32)

        # Напрямок контексту у просторі байтових значень
        ctx_direction = np.abs(ctx[:field_system.n_active_bytes]) \
            if len(ctx) >= field_system.n_active_bytes else \
            np.pad(np.abs(ctx), (0, field_system.n_active_bytes - len(ctx)))
        ctx_norm_dir = float(np.linalg.norm(ctx_direction))
        if ctx_norm_dir > 1e-10:
            ctx_direction = ctx_direction / ctx_norm_dir
        else:
            ctx_direction = np.zeros(field_system.n_active_bytes, dtype=np.float32)

        # Встановлюємо контекстний вектор та κ для неперервної ін'єкції
        # в FieldSystemV6.step() — Рівняння 37:
        # ∂Φ(i,k,t)/∂t += κ · ctx(k,t) · σ_gate(ctx, Φ(i,k,t))
        field_system.context_injection_vector = ctx_direction.astype(np.float32)
        field_system.context_injection_kappa = self.kappa

        # Повертаємо вектор ін'єкції для логування
        injection = self.kappa * ctx_direction * ctx_norm
        return injection

    def _adapt(self, v: np.ndarray, target_len: int) -> np.ndarray:
        """Адаптувати розмірність вектора."""
        if len(v) == target_len:
            return v
        elif len(v) > target_len:
            return v[:target_len]
        else:
            return np.pad(v, (0, target_len - len(v)))


# =============================================================================
# 13. Knowledge Transfer — Перенесення знань між модальностями
# =============================================================================


