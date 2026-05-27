import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _kl_divergence, _softmax

class BayesianModalityDetector:
    """
    Баєсівська ідентифікація модальності: M(p) = argmax_m P(m|p)

    P(m|p) ∝ P(p|m) · P(m)

    Модальності: text_ascii, text_utf8, image, audio, binary, structured

    Характерні ознаки:
    - text_ascii: байти 0x20-0x7E переважають, 0x0A/0x0D є
    - text_utf8: байти 0xC0-0xEF присутні (багатобайтові UTF-8)
    - image: приблизно рівномірний розподіл (стиснуті)
    - audio: концентрація навколо 0x80 (signed 8-bit center)
    - binary: високі частоти для 0x00 та 0xFF
    - structured: регулярні патерни (JSON, XML)
    """

    def __init__(self):
        self.modalities = ['text_ascii', 'text_utf8', 'image', 'audio', 'binary', 'structured']

        # Апріорні ймовірності
        self.prior = {
            'text_ascii': 0.25,
            'text_utf8': 0.20,
            'image': 0.15,
            'audio': 0.10,
            'binary': 0.15,
            'structured': 0.15,
        }

        # Характерні розподіли для кожної модальності (спрощені)
        self.modality_signatures = self._build_signatures()

    def _build_signatures(self) -> Dict[str, np.ndarray]:
        """Побудувати сигнатури модальностей."""
        sigs = {}

        # text_ascii: пік на 0x20-0x7E
        text_ascii = np.ones(256) * 0.001
        text_ascii[0x20:0x7F] = 1.0
        text_ascii[0x0A] = 0.5  # newline
        text_ascii[0x0D] = 0.3  # CR
        sigs['text_ascii'] = text_ascii / text_ascii.sum()

        # text_utf8: як ASCII + сильні байти 0xC0-0xEF (leading) + 0x80-0xBF (continuation)
        text_utf8 = text_ascii.copy()
        text_utf8[0x80:0xC0] = 1.5  # Continuation bytes
        text_utf8[0xC0:0xF0] = 3.0  # Leading bytes (сильний сигнал)
        sigs['text_utf8'] = text_utf8 / text_utf8.sum()

        # image: майже рівномірний
        sigs['image'] = np.ones(256) / 256

        # audio: концентрація навколо 0x80
        audio = np.exp(-0.5 * ((np.arange(256) - 128) / 40) ** 2)
        sigs['audio'] = audio / audio.sum()

        # binary: сильні піки на 0x00 та 0xFF, інші дуже низькі
        binary = np.ones(256) * 0.0005
        binary[0x00] = 10.0
        binary[0xFF] = 5.0
        sigs['binary'] = binary / binary.sum()

        # structured: numeric/tabular data plus JSON/XML-like delimiters.
        # Keep letters possible for JSON keys, but make digit/separator-heavy
        # streams distinguishable from ordinary ASCII prose.
        structured = np.ones(256) * 0.001
        structured[ord('a'):ord('z') + 1] = 0.20
        structured[ord('A'):ord('Z') + 1] = 0.20
        structured[ord('0'):ord('9') + 1] = 2.50
        for ch in ' \t\r\n':
            structured[ord(ch)] = 1.20
        for ch in ',;|':
            structured[ord(ch)] = 3.00
        for ch in '.:-_=+/':
            structured[ord(ch)] = 2.00
        for ch in '<>{}[]()"\\':
            structured[ord(ch)] = 3.00
        sigs['structured'] = structured / structured.sum()

        return sigs

    def detect(
        self,
        byte_distribution: np.ndarray,
        N: int = 1,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Визначити модальність даних.

        Args:
            byte_distribution: (256,) розподіл байтів
            N: довжина вхідної послідовності байтів для правильного масштабування правдоподібності

        Returns:
            modality: назва модальності
            posteriors: P(m|p) для кожної модальності
        """
        p = np.maximum(byte_distribution.astype(np.float64), 1e-10)
        p = p / p.sum()

        log_posteriors = {}
        for mod in self.modalities:
            sig = self.modality_signatures[mod]
            # log P(p|m) = N * Σ p(k) log sig_m(k)
            log_likelihood = float(np.sum(p * np.log(np.maximum(sig, 1e-10)))) * N
            log_prior = np.log(self.prior[mod])
            log_posteriors[mod] = log_likelihood + log_prior

        # Нормалізація
        max_log = max(log_posteriors.values())
        posteriors = {}
        total = 0.0
        for mod, log_p in log_posteriors.items():
            posteriors[mod] = np.exp(log_p - max_log)
            total += posteriors[mod]

        for mod in posteriors:
            posteriors[mod] /= total

        best_mod = max(posteriors, key=posteriors.get)

        return best_mod, posteriors


# =============================================================================
# 6b. CMA-ES Meta-Optimizer — Розділ 14.2 концепції
# =============================================================================



class KnowledgeTransfer:
    """
    Перенесення знань між модальностями.

    Чотири механізми:
    1. Структурний ізоморфізм: cos(f_m1^(ℓ), f_m2^(ℓ)) > θ_transfer
    2. Універсальний байтовий скаффолд: однакова архітектура → сумісні простори
    3. Ланцюгове перенесення: w_1,3_indirect = w_1,2 · w_2,3 (транзитивність)
    4. Мовне заземлення: мова як мета-модальність для міжмодального вирівнювання
    """

    def __init__(
        self,
        theta_transfer: float = 0.7,
        link_decay: float = 0.001,
    ):
        self.theta_transfer = theta_transfer
        self.link_decay = link_decay

        # Граф зв'язків: {(mod1,idx1,mod2,idx2): weight}
        self.cross_modal_links = {}
        # Модальність кожного кристала
        self.crystal_modalities = {}
        # Мовний банк: мета-модальність
        self.language_bank = {}

    def register_modality(self, crystal_idx: int, modality: str):
        """Зареєструвати модальність кристала."""
        self.crystal_modalities[crystal_idx] = modality

    def check_structural_isomorphism(
        self,
        h_1: np.ndarray,
        h_2: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Перевірити структурний ізоморфізм: cos(f_m1^(ℓ), f_m2^(ℓ)) > θ_transfer.
        """
        min_len = min(len(h_1), len(h_2))
        h_1, h_2 = h_1[:min_len], h_2[:min_len]
        norm_1 = np.linalg.norm(h_1)
        norm_2 = np.linalg.norm(h_2)
        if norm_1 < 1e-10 or norm_2 < 1e-10:
            return False, 0.0
        cos_sim = float(np.dot(h_1, h_2) / (norm_1 * norm_2))
        return cos_sim > self.theta_transfer, cos_sim

    def create_cross_modal_link(
        self,
        mod1: str, idx1: int,
        mod2: str, idx2: int,
        weight: float = 1.0,
    ):
        """
        Створити зв'язок між кристалами різних модальностей.
        w_transfer(H_text, H_img) = Σ_t γ(h_text(t), h_img(t))
        """
        key = (mod1, idx1, mod2, idx2)
        self.cross_modal_links[key] = weight
        rev_key = (mod2, idx2, mod1, idx1)
        self.cross_modal_links[rev_key] = weight

    def chain_transfer(
        self,
        mod1: str, idx1: int,
        mod3: str, idx3: int,
    ) -> float:
        """
        Ланцюгове перенесення: w_1,3_indirect = w_1,2 · w_2,3.
        Транзитивність: якщо m1↔m2 та m2↔m3 → m1↔m3 через міст.
        """
        best_indirect = 0.0
        for key, weight in self.cross_modal_links.items():
            m1, i1, m2, i2 = key
            if m1 != mod1 or i1 != idx1:
                continue
            for key2, weight2 in self.cross_modal_links.items():
                m2b, i2b, m3, i3 = key2
                if m2b == m2 and i2b == i2 and m3 == mod3 and i3 == idx3:
                    indirect = weight * weight2
                    best_indirect = max(best_indirect, indirect)
        return best_indirect

    def language_grounding(
        self,
        text_crystal_idx: int,
        target_crystal_idx: int,
        target_modality: str,
    ):
        """
        Мовне заземлення: зв'язування текстового концепту з концептом
        іншої модальності. Мова — мета-модальність.
        """
        self.create_cross_modal_link(
            'language', text_crystal_idx,
            target_modality, target_crystal_idx,
            weight=1.0
        )
        self.language_bank[text_crystal_idx] = target_crystal_idx

    def find_related(
        self,
        crystal_idx: int,
        modality: str,
        target_modality: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        """Знайти пов'язані кристали в інших модальностях."""
        related = []
        for key, weight in self.cross_modal_links.items():
            m1, i1, m2, i2 = key
            if i1 == crystal_idx and m1 == modality:
                if target_modality is None or m2 == target_modality:
                    related.append((m2, i2, weight))
        return related

    def decay_links(self, delta_t: float = 1.0):
        """Затухання зв'язків з часом."""
        to_remove = []
        for key in self.cross_modal_links:
            self.cross_modal_links[key] *= (1.0 - self.link_decay * delta_t)
            if self.cross_modal_links[key] < 0.01:
                to_remove.append(key)
        for key in to_remove:
            del self.cross_modal_links[key]

    def universal_byte_scaffold_check(
        self,
        modality_1: str,
        modality_2: str,
    ) -> Dict:
        """
        Універсальний байтовий скаффолд: однакові W_β, A, D_k, μ
        створюють спільний алфавіт абстракцій між модальностями.

        Принципи, вбудовані в архітектуру:
        - Суміжність → групування
        - Повторення → патерн
        - Межа → сегментація
        - Контрастність → виділення
        """
        return {
            'shared_principles': ['contiguity', 'repetition', 'boundary', 'contrast'],
            'compatible': True,
            'modality_1': modality_1,
            'modality_2': modality_2,
            'explanation': (
                f"Обидві модальності обробляються однією архітектурою "
                f"(однакові W_β, A, D_k, μ) → простори ознак сумісні"
            ),
        }


# =============================================================================
# 14. Level Splitting — Автокаталітичне розщеплення рівнів
# =============================================================================


