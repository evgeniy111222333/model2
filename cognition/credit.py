import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _softmax, _sigmoid

class CreditEpisodeResult(dict):
    """Dictionary-compatible result for one global credit-assignment episode."""

    METRIC_KEYS = (
        'answer_correctness',
        'target_evidence_rank',
        'target_probability',
        'graph_cf_drop',
        'decoder_nll',
        'conditioned_nll',
        'field_alignment',
        'random_probe_confidence',
        'calibration_error',
    )




class AdaptiveCalibration:
    """Data-calibrated policy values for semantic readout and credit assignment.

    Numeric epsilons and dimensional caps stay as invariants. Behavioral gates,
    margins, loss scales and acceptance bands are estimated from observed
    positive/negative episodes whenever the system has enough evidence.
    """

    def __init__(self, max_history: int = 512, ema_alpha: float = 0.08):
        self.max_history = int(max(32, max_history))
        self.ema_alpha = float(np.clip(ema_alpha, 0.001, 0.50))
        self.stats = {}

    def observe(self, name: str, value):
        try:
            v = float(value)
        except Exception:
            return
        if not np.isfinite(v):
            return
        key = str(name)
        stat = self.stats.setdefault(key, {'values': [], 'ema': None, 'n': 0})
        stat['values'].append(v)
        if len(stat['values']) > self.max_history:
            del stat['values'][:len(stat['values']) - self.max_history]
        stat['n'] = int(stat.get('n', 0)) + 1
        if stat['ema'] is None:
            stat['ema'] = v
        else:
            stat['ema'] = (1.0 - self.ema_alpha) * float(stat['ema']) + self.ema_alpha * v

    def observe_many(self, items: Dict[str, float]):
        for key, value in (items or {}).items():
            self.observe(key, value)

    def _values(self, name: str) -> List[float]:
        stat = self.stats.get(str(name), {})
        return list(stat.get('values', []))

    def count(self, name: str) -> int:
        return int(self.stats.get(str(name), {}).get('n', 0))

    def quantile(self, name: str, q: float, fallback: Optional[float] = None) -> Optional[float]:
        vals = self._values(name)
        if not vals:
            return fallback
        return float(np.percentile(np.asarray(vals, dtype=np.float32), float(np.clip(q, 0.0, 100.0))))

    def mean(self, name: str, fallback: Optional[float] = None) -> Optional[float]:
        vals = self._values(name)
        if not vals:
            return fallback
        return float(np.mean(np.asarray(vals, dtype=np.float32)))

    def std(self, name: str, fallback: float = 0.0) -> float:
        vals = self._values(name)
        if len(vals) < 2:
            return float(fallback)
        return float(np.std(np.asarray(vals, dtype=np.float32)))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(np.clip(float(value), float(low), float(high)))

    def discriminative_threshold(
        self,
        signal_name: str,
        prior: float,
        low: float,
        high: float,
        positive_q: float = 25.0,
        negative_q: float = 90.0,
    ) -> float:
        pos = self.quantile(f'positive_{signal_name}', positive_q, None)
        neg = self.quantile(f'negative_{signal_name}', negative_q, None)
        if pos is not None and neg is not None:
            value = 0.5 * (float(pos) + float(neg))
        elif pos is not None:
            value = 0.82 * float(pos)
        elif neg is not None:
            value = float(neg) + 0.05 * max(abs(float(neg)), 1.0)
        else:
            value = float(prior)
        return self._clamp(value, low, high)

    def robust_scale(self, name: str, prior: float, low: float, high: float, q: float = 90.0) -> float:
        vals = self._values(name)
        if len(vals) >= 4:
            value = max(float(np.percentile(np.asarray(vals, dtype=np.float32), q)), float(prior) * 0.35)
        else:
            value = float(prior)
        return self._clamp(value, low, high)

    def answer_policy(self, min_confidence: float, n_candidates: int = 1, n_graph_nodes: int = 1) -> Dict[str, float]:
        threshold = self.discriminative_threshold(
            'confidence',
            prior=float(min_confidence),
            low=0.35,
            high=0.92,
            positive_q=20.0,
            negative_q=92.0,
        )
        evidence_center = self.discriminative_threshold(
            'evidence_support',
            prior=0.43,
            low=-0.10,
            high=0.90,
            positive_q=20.0,
            negative_q=92.0,
        )
        byte_center = self.discriminative_threshold(
            'byte_support',
            prior=0.06,
            low=0.0,
            high=0.55,
            positive_q=20.0,
            negative_q=92.0,
        )
        score_center = self.discriminative_threshold(
            'answer_score',
            prior=0.64,
            low=0.05,
            high=0.95,
            positive_q=20.0,
            negative_q=92.0,
        )
        margin_center = self.discriminative_threshold(
            'answer_margin',
            prior=0.03,
            low=-0.20,
            high=0.55,
            positive_q=20.0,
            negative_q=92.0,
        )
        graph_prior = max(1.0 / max(int(n_graph_nodes), 1), 0.01)
        graph_rerank = self.discriminative_threshold(
            'graph_support',
            prior=max(0.34, graph_prior),
            low=graph_prior,
            high=0.92,
            positive_q=20.0,
            negative_q=92.0,
        )
        base_floor = self.discriminative_threshold(
            'graph_base_support',
            prior=0.10,
            low=-0.05,
            high=0.70,
            positive_q=20.0,
            negative_q=92.0,
        )
        conf_weights = np.array([
            self.robust_scale('answer_support_conf_weight', 0.72, 0.10, 0.90),
            self.robust_scale('answer_reason_conf_weight', 0.28, 0.10, 0.90),
        ], dtype=np.float32)
        conf_weights = conf_weights / max(float(conf_weights.sum()), 1e-8)
        return {
            'answer_confidence_threshold': threshold,
            'evidence_support_center': evidence_center,
            'byte_support_center': byte_center,
            'answer_score_center': score_center,
            'answer_margin_center': margin_center,
            'graph_rerank_probability': graph_rerank,
            'graph_base_support_floor': base_floor,
            'graph_rerank_weight': self.robust_scale('graph_rerank_weight', 0.18, 0.02, 0.60),
            'evidence_gate_floor': self.robust_scale('evidence_gate_floor', 0.25, 0.05, 0.55),
            'evidence_gate_gain': self.robust_scale('evidence_gate_gain', 0.75, 0.30, 0.95),
            'support_conf_weight': float(conf_weights[0]),
            'reason_conf_weight': float(conf_weights[1]),
            'margin_conf_base': self.robust_scale('margin_conf_base', 0.95, 0.70, 1.00),
            'margin_conf_gain': self.robust_scale('margin_conf_gain', 0.05, 0.00, 0.30),
            'support_gate_slope': self.robust_scale('support_gate_slope', 12.0, 4.0, 24.0),
            'byte_gate_slope': self.robust_scale('byte_gate_slope', 18.0, 4.0, 30.0),
            'score_gate_slope': self.robust_scale('score_gate_slope', 8.0, 3.0, 18.0),
            'margin_gate_slope': self.robust_scale('margin_gate_slope', 16.0, 4.0, 32.0),
            'n_candidates': int(max(1, n_candidates)),
            'n_graph_nodes': int(max(1, n_graph_nodes)),
        }

    def generation_policy(self, min_confidence: float, n_graph_nodes: int = 1) -> Dict[str, float]:
        answer_policy = self.answer_policy(min_confidence, n_graph_nodes=n_graph_nodes)
        evidence_floor = self.discriminative_threshold(
            'generation_evidence_support',
            prior=0.35,
            low=-0.10,
            high=0.85,
            positive_q=20.0,
            negative_q=92.0,
        )
        evidence_override = self.discriminative_threshold(
            'generation_evidence_override',
            prior=0.52,
            low=0.05,
            high=0.95,
            positive_q=20.0,
            negative_q=92.0,
        )
        graph_sparse_scale = self.robust_scale('graph_sparse_floor_scale', 0.70, 0.15, 2.50)
        graph_floor_abs = self.discriminative_threshold(
            'generation_graph_support',
            prior=0.018,
            low=0.0,
            high=0.60,
            positive_q=20.0,
            negative_q=92.0,
        )
        decoder_floor = self.discriminative_threshold(
            'decoder_confidence',
            prior=0.012,
            low=0.0,
            high=0.40,
            positive_q=15.0,
            negative_q=95.0,
        )
        weights = np.array([
            self.robust_scale('generation_retrieval_weight', 0.25, 0.05, 0.70),
            self.robust_scale('generation_score_weight', 0.45, 0.05, 0.80),
            self.robust_scale('generation_decoder_weight', 0.30, 0.05, 0.70),
        ], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        return {
            'evidence_support_floor': evidence_floor,
            'evidence_override_floor': evidence_override,
            'graph_probability_floor': graph_floor_abs,
            'graph_sparse_floor_scale': graph_sparse_scale,
            'decoder_confidence_floor': decoder_floor,
            'generation_confidence_floor': max(0.30, answer_policy['answer_confidence_threshold'] * 0.5),
            'retrieval_weight': float(weights[0]),
            'score_weight': float(weights[1]),
            'decoder_weight': float(weights[2]),
        }

    def episode_policy(self) -> Dict[str, float]:
        random_floor = self.discriminative_threshold(
            'random_probe_confidence',
            prior=0.35,
            low=0.05,
            high=0.75,
            positive_q=10.0,
            negative_q=92.0,
        )
        return {
            'decoder_nll_scale': self.robust_scale('conditioned_nll', 5.5, 0.50, 12.0),
            'counterfactual_drop_target': self.robust_scale('positive_graph_cf_drop', 0.08, 0.01, 0.35, q=25.0),
            'random_probe_floor': random_floor,
            'random_probe_band': max(0.03, min(0.20, 2.0 * self.std('negative_confidence', 0.04))),
            'layer_loss_tolerance': self.robust_scale('accepted_loss_tolerance', 0.075, 0.01, 0.18),
            'episode_loss_tolerance': self.robust_scale('episode_loss_tolerance', 0.10, 0.02, 0.22),
            'correctness_overlap_full': self.discriminative_threshold(
                'byte_overlap_correctness',
                prior=0.62,
                low=0.25,
                high=0.90,
                positive_q=20.0,
                negative_q=92.0,
            ),
            'projection_margin': self.discriminative_threshold(
                'projection_margin',
                prior=0.22,
                low=0.03,
                high=0.65,
                positive_q=20.0,
                negative_q=92.0,
            ),
            'semantic_margin': self.discriminative_threshold(
                'semantic_margin',
                prior=0.24,
                low=0.03,
                high=0.70,
                positive_q=20.0,
                negative_q=92.0,
            ),
            'evidence_margin': self.discriminative_threshold(
                'evidence_margin',
                prior=0.18,
                low=0.02,
                high=0.60,
                positive_q=20.0,
                negative_q=92.0,
            ),
            'epoch_beta_min': 0.45,
            'epoch_beta_max': 1.75,
            'lr_beta_min': 0.45,
            'lr_beta_max': 1.45,
            'semantic_lr_scale': self.robust_scale('semantic_lr_scale', 0.65, 0.20, 1.20),
            'decoder_lr_scale': self.robust_scale('decoder_lr_scale', 0.80, 0.20, 1.30),
            'graph_lr_scale': self.robust_scale('graph_lr_scale', 0.78, 0.20, 1.30),
        }

    def observe_answer_outcome(
        self,
        confidence: float,
        correct: bool,
        evidence_support: float,
        byte_support: float,
        answer_score: float,
        margin: float,
        graph_support: float,
        graph_base_support: float,
    ):
        prefix = 'positive' if bool(correct) else 'negative'
        self.observe(f'{prefix}_confidence', confidence)
        self.observe(f'{prefix}_evidence_support', evidence_support)
        self.observe(f'{prefix}_byte_support', byte_support)
        self.observe(f'{prefix}_answer_score', answer_score)
        self.observe(f'{prefix}_answer_margin', margin)
        self.observe(f'{prefix}_graph_support', graph_support)
        self.observe(f'{prefix}_graph_base_support', graph_base_support)

    def observe_episode_metrics(self, pre: Dict, post: Dict):
        for metrics in (pre or {}, post or {}):
            self.observe('conditioned_nll', metrics.get('conditioned_nll', 0.0))
            self.observe('decoder_nll', metrics.get('decoder_nll', 0.0))
            self.observe('random_probe_confidence', metrics.get('random_probe_confidence', 0.0))
            self.observe('graph_cf_drop', metrics.get('graph_cf_drop', 0.0))
            self.observe('answer_confidence', metrics.get('response', {}).get('confidence', 0.0))
            correct = float(metrics.get('answer_correctness', 0.0)) >= 0.75
            prefix = 'positive' if correct else 'negative'
            self.observe(f'{prefix}_confidence', metrics.get('response', {}).get('confidence', 0.0))
            self.observe(f'{prefix}_graph_cf_drop', metrics.get('graph_cf_drop', 0.0))
        if post:
            self.observe('episode_loss', post.get('episode_loss', 0.0))
            self.observe('byte_overlap_correctness', post.get('answer_correctness', 0.0))

    def observe_generation(self, evidence_support: float, graph_support: float, decoder_confidence: float, accepted: bool):
        prefix = 'positive' if bool(accepted) else 'negative'
        self.observe(f'{prefix}_generation_evidence_support', evidence_support)
        self.observe(f'{prefix}_generation_graph_support', graph_support)
        self.observe(f'{prefix}_decoder_confidence', decoder_confidence)

    def observe_negative_probe(self, confidence: float):
        self.observe('negative_confidence', confidence)
        self.observe('negative_random_probe_confidence', confidence)

    def report(self) -> Dict:
        stats = {}
        for key, stat in sorted(self.stats.items()):
            vals = np.asarray(stat.get('values', []), dtype=np.float32)
            if vals.size == 0:
                continue
            stats[key] = {
                'n': int(stat.get('n', 0)),
                'mean': float(np.mean(vals)),
                'ema': float(stat['ema']) if stat.get('ema') is not None else None,
                'q10': float(np.percentile(vals, 10)),
                'q50': float(np.percentile(vals, 50)),
                'q90': float(np.percentile(vals, 90)),
            }
        return {
            'type': 'adaptive_empirical_quantile_calibration',
            'stats': stats,
            'episode_policy': self.episode_policy(),
            'answer_policy': self.answer_policy(min_confidence=0.58),
            'generation_policy': self.generation_policy(min_confidence=0.58),
        }




class AdaptiveEvidenceScorer:
    """Learns how retrieval evidence channels should be fused.

    This replaces fixed semantic/byte/state coefficients with a trainable,
    role-aware scorer. It keeps byte evidence as one causal signal, but the
    final ranking is learned from target-vs-hard-negative episodes.
    """

    FEATURE_NAMES = (
        'semantic_consensus',
        'projection_similarity',
        'legacy_similarity',
        'semantic_state_similarity',
        'associative_support',
        'memory_focus',
        'byte_overlap',
    )

    ROLES = ('qa', 'segment')

    def __init__(self, seed: int = 6061):
        self.rng = np.random.default_rng(seed)
        n = len(self.FEATURE_NAMES)
        self.logits = {role: np.zeros(n, dtype=np.float32) for role in self.ROLES}
        initial_p = max(2.0, 2.0 * float(n))
        initial_logit = float(np.log(np.expm1(max(initial_p - 1.0, 1e-4))))
        self.sharpness_logit = {role: np.float32(initial_logit) for role in self.ROLES}
        self.training_history = []

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    @staticmethod
    def _role(kind: str) -> str:
        return 'qa' if str(kind) == 'qa_memory' else 'segment'

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        centered = arr - float(np.max(arr))
        exp_vals = np.exp(np.clip(centered, -60.0, 60.0))
        return (exp_vals / max(float(exp_vals.sum()), 1e-12)).astype(np.float32)

    @staticmethod
    def _softplus(x: float) -> float:
        val = float(x)
        if val > 20.0:
            return val
        if val < -20.0:
            return float(np.exp(val))
        return float(np.log1p(np.exp(val)))

    def weights(self, role: str) -> np.ndarray:
        role_key = self._role(role)
        return self._softmax(self.logits[role_key])

    def sharpness(self, role: str) -> float:
        role_key = self._role(role)
        return float(1.0 + self._softplus(float(self.sharpness_logit[role_key])))

    def feature_vector(self, channels: Dict) -> np.ndarray:
        raw_semantic = self._clip01(max(0.0, float(channels.get('semantic_similarity', 0.0))))
        projection = self._clip01(max(0.0, float(channels.get('projection_similarity', 0.0))))
        legacy = self._clip01(max(0.0, float(channels.get('legacy_similarity', 0.0))))
        state_sim = self._clip01(max(0.0, float(channels.get('semantic_state_similarity', 0.0))))
        memory_gate = self._clip01(channels.get('memory_gate', 0.0))
        memory_support = self._clip01(max(0.0, float(channels.get('memory_support', 0.0))))
        memory_focus = self._clip01(max(
            float(channels.get('memory_focus', 0.0)),
            memory_support * memory_gate,
            float(channels.get('associative_support', 0.0)),
        ))
        byte = self._clip01(channels.get('byte_overlap', 0.0))
        independent_support = max(projection, legacy, state_sim, memory_focus, byte)
        semantic_consensus = self._clip01(np.sqrt(max(raw_semantic * independent_support, 0.0)))
        values = np.array([
            semantic_consensus,
            projection,
            legacy,
            state_sim,
            self._clip01(channels.get('associative_support', 0.0)),
            memory_focus,
            byte,
        ], dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def score_vector(self, role: str, x: np.ndarray) -> Dict:
        role_key = self._role(role)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(x) != len(self.FEATURE_NAMES):
            resized = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
            n = min(len(resized), len(x))
            resized[:n] = x[:n]
            x = resized
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        w = self.weights(role_key)
        p = self.sharpness(role_key)
        powered = np.power(np.maximum(x, 0.0), p).astype(np.float32)
        mean_power = float(np.dot(w, powered))
        score = float(np.power(max(mean_power, 0.0), 1.0 / max(p, 1e-6)))
        return {
            'score': score,
            'evidence_role': role_key,
            'evidence_features': {name: float(val) for name, val in zip(self.FEATURE_NAMES, x)},
            'evidence_weights': {name: float(val) for name, val in zip(self.FEATURE_NAMES, w)},
            'evidence_sharpness': float(p),
        }

    def score(self, role: str, channels: Dict) -> Dict:
        return self.score_vector(role, self.feature_vector(channels))

    def _candidate_score(self, cand: Dict) -> float:
        return float(cand.get('score', cand.get('base_score', 0.0)))

    def _loss(self, candidates: List[Dict], target_ids: List[str], margin: float = 0.18) -> Dict:
        target_set = {str(t) for t in target_ids}
        targets = [c for c in candidates if str(c.get('candidate_id')) in target_set]
        negatives = [c for c in candidates if str(c.get('candidate_id')) not in target_set]
        if not targets:
            return {
                'loss': 0.0,
                'target_score': 0.0,
                'hardest_negative_score': 0.0,
                'target_rank': 0,
                'n_targets': 0,
                'n_negatives': int(len(negatives)),
            }
        target_scores = [self._candidate_score(c) for c in targets]
        pos = float(max(target_scores))
        hard = float(max([self._candidate_score(c) for c in negatives], default=-1.0))
        all_scores = sorted(
            [(self._candidate_score(c), str(c.get('candidate_id')) in target_set) for c in candidates],
            key=lambda row: row[0],
            reverse=True,
        )
        rank = 1
        for idx, (_, is_target) in enumerate(all_scores, start=1):
            if is_target:
                rank = idx
                break
        return {
            'loss': float(max(0.0, float(margin) - pos + hard)) if negatives else float(max(0.0, 1.0 - pos)),
            'target_score': pos,
            'hardest_negative_score': hard,
            'target_rank': int(rank),
            'n_targets': int(len(targets)),
            'n_negatives': int(len(negatives)),
        }

    def _rescore_candidates(self, candidates: List[Dict]) -> List[Dict]:
        out = []
        for cand in candidates:
            row = dict(cand)
            role = row.get('evidence_role', self._role(row.get('kind', 'segment')))
            if row.get('evidence_channels') is not None:
                scored = self.score(role, row.get('evidence_channels', {}))
            else:
                scored = self.score_vector(
                    role,
                    np.array([row.get('evidence_features', {}).get(name, 0.0) for name in self.FEATURE_NAMES], dtype=np.float32),
                )
            row.update(scored)
            row['base_score'] = float(scored['score'])
            out.append(row)
        return out

    def _regularize(self):
        for role in self.ROLES:
            self.logits[role] = np.nan_to_num(
                self.logits[role],
                nan=0.0,
                posinf=6.0,
                neginf=-6.0,
            ).astype(np.float32)
            self.logits[role] = np.clip(self.logits[role], -6.0, 6.0).astype(np.float32)
            upper = max(6.0, 3.0 * float(len(self.FEATURE_NAMES)))
            self.sharpness_logit[role] = np.float32(np.clip(float(self.sharpness_logit[role]), -3.0, upper))

    def train_candidates(
        self,
        candidates: List[Dict],
        target_ids: List[str],
        epochs: int = 8,
        lr: float = 0.035,
        margin: float = 0.18,
    ) -> Dict:
        target_set = {str(t) for t in target_ids}
        if not candidates or not target_set:
            return {'status': 'skipped', 'reason': 'empty_candidates_or_targets'}

        work = self._rescore_candidates(candidates)
        before = self._loss(work, target_ids, margin=margin)
        for _ in range(max(1, int(epochs))):
            work = self._rescore_candidates(work)
            targets = [c for c in work if str(c.get('candidate_id')) in target_set]
            negatives = [c for c in work if str(c.get('candidate_id')) not in target_set]
            if not targets:
                break
            pos_row = max(targets, key=self._candidate_score)
            neg_row = max(negatives, key=self._candidate_score) if negatives else None
            if neg_row is None:
                violation = max(0.0, 1.0 - self._candidate_score(pos_row))
            else:
                violation = max(0.0, float(margin) - self._candidate_score(pos_row) + self._candidate_score(neg_row))
            if violation <= 0.0:
                continue

            pos_role = self._role(pos_row.get('evidence_role', pos_row.get('kind', 'segment')))
            pos_x = np.array([pos_row.get('evidence_features', {}).get(name, 0.0) for name in self.FEATURE_NAMES], dtype=np.float32)
            pos_w = self.weights(pos_role)
            pos_score = float(np.dot(pos_w, pos_x))
            self.logits[pos_role] += np.float32(lr) * (pos_w * (pos_x - pos_score)).astype(np.float32)
            self.sharpness_logit[pos_role] += np.float32(lr * 0.08 * (float(np.max(pos_x)) - float(np.mean(pos_x))))

            if neg_row is not None:
                neg_role = self._role(neg_row.get('evidence_role', neg_row.get('kind', 'segment')))
                neg_x = np.array([neg_row.get('evidence_features', {}).get(name, 0.0) for name in self.FEATURE_NAMES], dtype=np.float32)
                neg_w = self.weights(neg_role)
                neg_score = float(np.dot(neg_w, neg_x))
                self.logits[neg_role] -= np.float32(lr) * (neg_w * (neg_x - neg_score)).astype(np.float32)
                self.sharpness_logit[neg_role] -= np.float32(lr * 0.04 * max(0.0, float(np.max(neg_x)) - float(np.mean(neg_x))))
            self._regularize()

        after_work = self._rescore_candidates(work)
        after = self._loss(after_work, target_ids, margin=margin)
        record = {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'margin': float(margin),
            'loss_before': before['loss'],
            'loss_after': after['loss'],
            'target_score_before': before['target_score'],
            'target_score_after': after['target_score'],
            'hardest_negative_before': before['hardest_negative_score'],
            'hardest_negative_after': after['hardest_negative_score'],
            'target_rank_before': before['target_rank'],
            'target_rank_after': after['target_rank'],
            'n_targets': before['n_targets'],
            'n_negatives': before['n_negatives'],
            'weights': self.report()['weights'],
        }
        self.training_history.append(record)
        return record

    def snapshot(self) -> Dict:
        return {
            'logits': {role: vals.copy() for role, vals in self.logits.items()},
            'sharpness_logit': {role: np.float32(val) for role, val in self.sharpness_logit.items()},
        }

    def restore(self, snap: Dict):
        for role in self.ROLES:
            if role in snap.get('logits', {}):
                self.logits[role] = snap['logits'][role].copy()
            if role in snap.get('sharpness_logit', {}):
                self.sharpness_logit[role] = np.float32(snap['sharpness_logit'][role])
        self._regularize()

    def report(self) -> Dict:
        return {
            'type': 'adaptive_evidence_scorer',
            'feature_names': list(self.FEATURE_NAMES),
            'weights': {
                role: {name: float(val) for name, val in zip(self.FEATURE_NAMES, self.weights(role))}
                for role in self.ROLES
            },
            'sharpness': {role: float(self.sharpness(role)) for role in self.ROLES},
            'training_steps': int(len(self.training_history)),
        }




class EpisodeCreditEngine:
    """Episode-level credit routing across semantic, readout, graph, field and decoder layers."""

    LAYER_ORDER = ('semantic', 'state', 'evidence', 'projection', 'field', 'decoder', 'graph', 'policy')

    def __init__(self):
        self.history = []

    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            value = float(x)
        except Exception:
            return float(default)
        if not np.isfinite(value):
            return float(default)
        return value

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _snapshot(self, readout, semantic=None, field_system=None) -> Dict:
        snap = {
            'readout': {
                'W_query': readout.W_query.copy(),
                'W_segment': readout.W_segment.copy(),
                'field_byte_coupling': readout.field_byte_coupling.copy(),
            },
            'state_constructor': (
                readout.state_constructor.snapshot()
                if hasattr(readout, 'state_constructor') and readout.state_constructor is not None
                else None
            ),
            'evidence_scorer': (
                readout.evidence_scorer.snapshot()
                if hasattr(readout, 'evidence_scorer') and readout.evidence_scorer is not None
                else None
            ),
            'decoder': {
                'W_condition': readout.byte_decoder.W_condition.copy(),
                'condition_bias': readout.byte_decoder.condition_bias.copy(),
            },
            'reasoner': {
                'score_weights': readout.reasoner.score_weights.copy(),
                'state_mix_logits': readout.reasoner.state_mix_logits.copy(),
                'state_mix': readout.reasoner.state_mix.copy(),
            },
            'graph': {
                'W_query': readout.memory_graph.W_query.copy(),
                'W_key': readout.memory_graph.W_key.copy(),
                'W_value': readout.memory_graph.W_value.copy(),
                'edge_weights': readout.memory_graph.edge_weights.copy(),
                'node_weights': readout.memory_graph.node_weights.copy(),
                'state_mix_logits': readout.memory_graph.state_mix_logits.copy(),
                'flow_strength': np.float32(readout.memory_graph.flow_strength),
            },
        }
        if semantic is not None:
            snap['semantic'] = {
                'W_enc': semantic.W_enc.copy(),
                'W_dec': semantic.W_dec.copy(),
                'W_pred': semantic.W_pred.copy(),
            }
        if field_system is not None and hasattr(field_system, 'Phi'):
            snap['field'] = {
                'Phi': field_system.Phi.copy() if hasattr(field_system, 'Phi') else None,
                'u': field_system.u.copy() if hasattr(field_system, 'u') else None,
                'v': field_system.v.copy() if hasattr(field_system, 'v') else None,
                'context_injection_vector': (
                    field_system.context_injection_vector.copy()
                    if hasattr(field_system, 'context_injection_vector')
                    and field_system.context_injection_vector is not None
                    else None
                ),
                'context_injection_kappa': float(getattr(field_system, 'context_injection_kappa', 0.0)),
            }
        return snap

    def _restore_layer(self, readout, snap: Dict, layer: str, semantic=None, field_system=None):
        if layer in ('all', 'projection') and 'readout' in snap:
            readout.W_query = snap['readout']['W_query'].copy()
            readout.W_segment = snap['readout']['W_segment'].copy()
            readout._refresh_projection_cache()
        if (
            layer in ('all', 'state')
            and snap.get('state_constructor') is not None
            and hasattr(readout, 'state_constructor')
            and readout.state_constructor is not None
        ):
            readout.state_constructor.restore(snap['state_constructor'])
            readout._refresh_projection_cache()
        if (
            layer in ('all', 'evidence')
            and snap.get('evidence_scorer') is not None
            and hasattr(readout, 'evidence_scorer')
            and readout.evidence_scorer is not None
        ):
            readout.evidence_scorer.restore(snap['evidence_scorer'])
        if layer in ('all', 'field') and 'readout' in snap:
            readout.field_byte_coupling = snap['readout']['field_byte_coupling'].copy()
        if layer in ('all', 'decoder') and 'decoder' in snap:
            readout.byte_decoder.W_condition = snap['decoder']['W_condition'].copy()
            readout.byte_decoder.condition_bias = snap['decoder']['condition_bias'].copy()
        if layer in ('all', 'policy') and 'reasoner' in snap:
            readout.reasoner.score_weights = snap['reasoner']['score_weights'].copy()
            readout.reasoner.state_mix_logits = snap['reasoner']['state_mix_logits'].copy()
            readout.reasoner.state_mix = snap['reasoner']['state_mix'].copy()
        if layer in ('all', 'graph') and 'graph' in snap:
            graph = readout.memory_graph
            graph.W_query = snap['graph']['W_query'].copy()
            graph.W_key = snap['graph']['W_key'].copy()
            graph.W_value = snap['graph']['W_value'].copy()
            graph.edge_weights = snap['graph']['edge_weights'].copy()
            graph.node_weights = snap['graph']['node_weights'].copy()
            graph.state_mix_logits = snap['graph']['state_mix_logits'].copy()
            graph.flow_strength = np.float32(snap['graph']['flow_strength'])
        if layer in ('all', 'semantic') and semantic is not None and 'semantic' in snap:
            semantic.W_enc = snap['semantic']['W_enc'].copy()
            semantic.W_dec = snap['semantic']['W_dec'].copy()
            semantic.W_pred = snap['semantic']['W_pred'].copy()
            readout._refresh_feature_cache(semantic)
        if layer in ('all', 'field') and field_system is not None and 'field' in snap:
            fs = snap['field']
            if fs.get('Phi') is not None and hasattr(field_system, 'Phi'):
                field_system.Phi = fs['Phi'].copy()
            if fs.get('u') is not None and hasattr(field_system, 'u'):
                field_system.u = fs['u'].copy()
            if fs.get('v') is not None and hasattr(field_system, 'v'):
                field_system.v = fs['v'].copy()
            if hasattr(field_system, 'context_injection_vector'):
                civ = fs.get('context_injection_vector')
                field_system.context_injection_vector = civ.copy() if civ is not None else civ
            if hasattr(field_system, 'context_injection_kappa'):
                field_system.context_injection_kappa = float(fs.get('context_injection_kappa', 0.0))

    def _target_rank(self, candidates: List[Dict], target_ids: List[str]) -> int:
        target_set = {str(t) for t in target_ids}
        if not candidates:
            return 0
        ranked = sorted(
            candidates,
            key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
            reverse=True,
        )
        for idx, row in enumerate(ranked, start=1):
            if str(row.get('candidate_id')) in target_set:
                return int(idx)
        return int(len(ranked) + 1)

    def _answer_correctness(self, readout, target_raw: bytes, response_raw: bytes) -> float:
        target_raw = readout._to_bytes(target_raw)
        response_raw = readout._to_bytes(response_raw)
        if not target_raw or not response_raw:
            return 0.0
        if target_raw in response_raw or (
            len(response_raw) >= min(12, len(target_raw))
            and response_raw in target_raw
        ):
            return 1.0
        policy = getattr(readout, 'calibration', AdaptiveCalibration()).episode_policy()
        overlap_full = max(float(policy.get('correctness_overlap_full', 0.62)), 1e-6)
        return self._clip01(readout._bytegram_overlap(target_raw, response_raw) / overlap_full)

    def _metrics(
        self,
        readout,
        intent_raw: bytes,
        target_raw: bytes,
        semantic=None,
        target_ids: Optional[List[str]] = None,
        field_system=None,
        negative_probes: Optional[List[bytes]] = None,
        betas: Optional[Dict[str, float]] = None,
    ) -> Dict:
        target_ids = [str(t) for t in (target_ids or [])]
        q_feature = readout.feature_from_bytes(intent_raw, semantic)
        q_hidden = readout._project_query(q_feature)
        candidates = readout._candidate_records(q_feature, q_hidden, intent_raw)
        pool = readout._training_candidate_pool(
            candidates,
            target_ids,
            max_candidates=min(readout.memory_graph.max_nodes, 256),
        )
        graph = readout.memory_graph.rollout(q_hidden, pool, target_ids=target_ids, top_k=7)
        first_target = target_ids[0] if target_ids else None
        graph_cf = readout.memory_graph.rollout(
            q_hidden,
            pool,
            target_ids=target_ids,
            ablate_ids=[first_target] if first_target else None,
            top_k=7,
        )
        response = readout.respond(
            intent_raw,
            semantic=semantic,
            mode='auto',
            max_bytes=min(240, max(64, len(target_raw) + 48)),
            top_k=7,
        )
        response_raw = response.get('response_bytes', b'')
        correctness = self._answer_correctness(readout, target_raw, response_raw)
        confidence = self._safe_float(response.get('confidence', 0.0))
        decoder_nll = readout.byte_decoder.nll(target_raw)
        conditioned_nll = readout.byte_decoder.conditioned_nll(target_raw, q_hidden, seed=intent_raw)
        field_alignment = readout._field_alignment(field_system, target_raw)
        calibration = getattr(readout, 'calibration', AdaptiveCalibration())
        episode_policy = calibration.episode_policy()
        probes = negative_probes if negative_probes is not None else [b'zxq unrelated counterfactual bytes']
        probe_reports = []
        random_conf = 0.0
        random_answered = False
        for probe in probes:
            probe_resp = readout.respond(
                readout._to_bytes(probe),
                semantic=semantic,
                mode='auto',
                max_bytes=96,
                top_k=7,
            )
            pc = self._safe_float(probe_resp.get('confidence', 0.0))
            calibration.observe_negative_probe(pc)
            probe_resp = readout.respond(
                readout._to_bytes(probe),
                semantic=semantic,
                mode='auto',
                max_bytes=96,
                top_k=7,
            )
            pc = self._safe_float(probe_resp.get('confidence', pc))
            random_conf = max(random_conf, pc)
            random_answered = random_answered or bool(probe_resp.get('responded', False))
            probe_reports.append({
                'probe': readout._to_bytes(probe)[:96].decode('utf-8', errors='replace'),
                'responded': bool(probe_resp.get('responded', False)),
                'confidence': pc,
                'response_kind': probe_resp.get('response_kind'),
            })
        rank = self._target_rank(candidates, target_ids)
        n_candidates = max(len(candidates), 1)
        target_p = self._safe_float(graph.get('target_probability', 0.0))
        cf_p = self._safe_float(graph_cf.get('target_probability', 0.0))
        cf_drop = max(0.0, target_p - cf_p)
        calibration_error = abs(1.0 - confidence) if correctness >= 0.75 else confidence
        graph_norm = max(float(np.log(max(int(graph.get('n_nodes', n_candidates)), 2))), 1.0)
        decoder_scale = max(float(episode_policy.get('decoder_nll_scale', 5.5)), 1e-6)
        cf_target = max(float(episode_policy.get('counterfactual_drop_target', 0.08)), 1e-6)
        random_floor = float(episode_policy.get('random_probe_floor', 0.35))
        components = {
            'L_answer': 1.0 - correctness,
            'L_evidence': min(1.0, max(rank - 1, 0) / max(n_candidates - 1, 1)),
            'L_graph': min(1.0, -float(np.log(max(target_p, 1e-9))) / graph_norm),
            'L_decoder': min(1.0, self._safe_float(conditioned_nll.get('mean_nll', 0.0)) / decoder_scale),
            'L_field': 0.0 if field_system is None else max(0.0, 1.0 - field_alignment),
            'L_calibration': min(1.0, calibration_error),
            'L_counterfactual': min(1.0, max(0.0, cf_target - cf_drop) / cf_target),
            'L_random_suppression': min(1.0, max(0.0, random_conf - random_floor) / max(1.0 - random_floor, 1e-6)),
        }
        weights = {
            'L_answer': 1.20,
            'L_evidence': 0.80,
            'L_graph': 1.00,
            'L_decoder': 0.65,
            'L_field': 0.45,
            'L_calibration': 0.60,
            'L_counterfactual': 0.80,
            'L_random_suppression': 1.10,
        }
        if betas:
            weights['L_graph'] *= self._safe_float(betas.get('graph', 1.0), 1.0)
            weights['L_decoder'] *= self._safe_float(betas.get('decoder', 1.0), 1.0)
            weights['L_field'] *= self._safe_float(betas.get('field', 1.0), 1.0)
            weights['L_answer'] *= self._safe_float(betas.get('policy', 1.0), 1.0)
            weights['L_evidence'] *= max(
                self._safe_float(betas.get('projection', 1.0), 1.0),
                self._safe_float(betas.get('evidence', 1.0), 1.0),
            )
        total_weight = max(float(sum(weights.values())), 1e-8)
        episode_loss = float(sum(weights[k] * components[k] for k in components) / total_weight)
        return {
            'answer_correctness': float(correctness),
            'target_evidence_rank': int(rank),
            'target_probability': float(target_p),
            'target_prior_probability': float(graph.get('target_prior_probability', 0.0)),
            'target_path_probability': float(graph.get('target_path_probability', target_p)),
            'target_lift_over_prior': float(graph.get('target_lift_over_prior', 0.0)),
            'graph_cf_drop': float(cf_drop),
            'decoder_nll': float(decoder_nll.get('mean_nll', 0.0)),
            'conditioned_nll': float(conditioned_nll.get('mean_nll', 0.0)),
            'field_alignment': float(field_alignment),
            'random_probe_confidence': float(random_conf),
            'random_probe_answered': bool(random_answered),
            'calibration_error': float(calibration_error),
            'episode_loss': episode_loss,
            'loss_components': components,
            'response': {
                'responded': bool(response.get('responded', False)),
                'response_kind': response.get('response_kind'),
                'confidence': float(confidence),
                'response_text': response.get('response_text', '')[:240],
                'counterfactual_sensitivity': float(response.get('counterfactual_sensitivity', 0.0)),
            },
            'graph': {
                'n_nodes': int(graph.get('n_nodes', 0)),
                'n_steps': int(graph.get('n_steps', 0)),
                'target_probability': float(target_p),
                'counterfactual_probability': float(cf_p),
                'counterfactual_drop': float(cf_drop),
                'target_lift_over_prior': float(graph.get('target_lift_over_prior', 0.0)),
                'trace': graph.get('trace', []),
                'ranked_path': graph.get('ranked_path', []),
            },
            'negative_probes': probe_reports,
            'candidate_count': int(len(candidates)),
            'pool_count': int(len(pool)),
            'calibration_policy': episode_policy,
        }

    def _causal_trace(self, readout, intent_raw: bytes, target_raw: bytes, semantic, target_ids: List[str], field_system=None) -> Dict:
        q_feature = readout.feature_from_bytes(intent_raw, semantic)
        target_feature = readout.feature_from_bytes(target_raw, semantic)
        q_hidden = readout._project_query(q_feature)
        candidates = readout._candidate_records(q_feature, q_hidden, intent_raw)
        pool = readout._training_candidate_pool(
            candidates,
            target_ids,
            max_candidates=min(readout.memory_graph.max_nodes, 256),
        )
        graph = readout.memory_graph.rollout(q_hidden, pool, target_ids=target_ids, top_k=7)
        target_p = self._safe_float(graph.get('target_probability', 0.0))
        first_target = target_ids[0] if target_ids else None
        without_target = readout.memory_graph.rollout(
            q_hidden,
            pool,
            target_ids=target_ids,
            ablate_ids=[first_target] if first_target else None,
            top_k=7,
        )
        flow_orig = np.float32(readout.memory_graph.flow_strength)
        readout.memory_graph.flow_strength = np.float32(0.0)
        no_flow = readout.memory_graph.rollout(q_hidden, pool, target_ids=target_ids, top_k=7)
        readout.memory_graph.flow_strength = flow_orig
        base_nll = readout.byte_decoder.nll(target_raw)
        cond_nll = readout.byte_decoder.conditioned_nll(target_raw, q_hidden, seed=intent_raw)
        hard_rows = sorted(
            [c for c in candidates if str(c.get('candidate_id')) not in set(target_ids)],
            key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
            reverse=True,
        )
        calibration = getattr(readout, 'calibration', AdaptiveCalibration())
        episode_policy = calibration.episode_policy()
        negative_features = [row['feature'] for row in hard_rows[:24]]
        proj = readout._projection_loss(
            q_feature,
            target_feature,
            negative_features,
            margin=float(episode_policy.get('projection_margin', 0.22)),
        )
        target_ablation_drop = max(0.0, target_p - self._safe_float(without_target.get('target_probability', 0.0)))
        graph_edge_drop = max(0.0, target_p - self._safe_float(no_flow.get('target_probability', 0.0)))
        decoder_gain = max(0.0, self._safe_float(base_nll.get('mean_nll', 0.0)) - self._safe_float(cond_nll.get('mean_nll', 0.0)))
        field_credit = readout._field_alignment(field_system, target_raw) if field_system is not None else 0.0
        projection_margin = max(0.0, self._safe_float(proj.get('positive_similarity', 0.0)) - self._safe_float(proj.get('hardest_negative_similarity', -1.0)))
        state_report = (
            readout.state_constructor.candidate_score(q_feature, target_feature)
            if hasattr(readout, 'state_constructor') and readout.state_constructor is not None
            else {'state_similarity': 0.0, 'associative_support': 0.0, 'memory_support': 0.0}
        )
        target_set = {str(t) for t in target_ids}
        target_scores = [
            float(c.get('base_score', c.get('score', 0.0)))
            for c in candidates
            if str(c.get('candidate_id')) in target_set
        ]
        negative_scores = [
            float(c.get('base_score', c.get('score', 0.0)))
            for c in candidates
            if str(c.get('candidate_id')) not in target_set
        ]
        evidence_target_score = float(max(target_scores, default=0.0))
        evidence_hard_negative = float(max(negative_scores, default=-1.0))
        evidence_margin = float(evidence_target_score - evidence_hard_negative)
        evidence_report = (
            readout.evidence_scorer.report()
            if hasattr(readout, 'evidence_scorer') and readout.evidence_scorer is not None
            else {'type': 'none'}
        )
        response = readout.respond(intent_raw, semantic=semantic, mode='auto', max_bytes=min(180, max(64, len(target_raw) + 32)), top_k=7)
        layer_credit = {
            'semantic': float(max(0.0, projection_margin) * 0.60 + target_ablation_drop * 0.40),
            'state': float(max(0.0, state_report.get('state_similarity', 0.0)) + state_report.get('associative_support', 0.0)),
            'evidence': float(max(0.0, evidence_target_score) + max(0.0, evidence_margin)),
            'projection': float(projection_margin),
            'field': float(field_credit),
            'decoder': float(min(1.0, decoder_gain / 2.5)),
            'graph': float(target_ablation_drop + graph_edge_drop),
            'policy': float(max(0.0, self._safe_float(response.get('counterfactual_sensitivity', 0.0)))),
        }
        betas = {
            layer: float(1.0 + min(1.25, max(0.0, credit))) if credit > 1e-6 else 0.55
            for layer, credit in layer_credit.items()
        }
        return {
            'selected_evidence': graph.get('ranked_path', []),
            'graph_trace': graph.get('trace', []),
            'target_ablation_drop': float(target_ablation_drop),
            'graph_edge_flow_drop': float(graph_edge_drop),
            'decoder_conditioned_gain': float(decoder_gain),
            'field_credit_score': float(field_credit),
            'projection_margin': float(projection_margin),
            'state_constructor': state_report,
            'evidence_scorer': evidence_report,
            'evidence_target_score': float(evidence_target_score),
            'evidence_hard_negative': float(evidence_hard_negative),
            'evidence_margin': float(evidence_margin),
            'layer_credit': layer_credit,
            'adaptive_betas': betas,
        }

    def _accept_layer(self, layer: str, before: Dict, after: Dict, layer_stats: Dict) -> Tuple[bool, str]:
        loss_before = self._safe_float(before.get('episode_loss', 0.0))
        loss_after = self._safe_float(after.get('episode_loss', 0.0))
        policy = layer_stats.get('_episode_policy', {}) if isinstance(layer_stats, dict) else {}
        random_floor = float(policy.get('random_probe_floor', 0.55))
        random_band = float(policy.get('random_probe_band', 0.08))
        random_limit = max(random_floor, self._safe_float(before.get('random_probe_confidence', 0.0)) + random_band)
        if self._safe_float(after.get('random_probe_confidence', 0.0)) > random_limit:
            return False, 'random_probe_regression'
        if bool(after.get('random_probe_answered', False)) and not bool(before.get('random_probe_answered', False)):
            return False, 'random_probe_became_answered'
        if after.get('answer_correctness', 0.0) + 1e-6 < before.get('answer_correctness', 0.0) and loss_after > loss_before:
            return False, 'answer_correctness_regression'
        if layer == 'field' and after.get('field_alignment', 0.0) + 1e-6 < before.get('field_alignment', 0.0):
            return False, 'field_alignment_regression'
        tolerance = float(policy.get('layer_loss_tolerance', 0.075))
        if loss_after <= loss_before + tolerance:
            return True, 'episode_loss_ok'
        if layer_stats:
            lb = layer_stats.get('loss_before')
            la = layer_stats.get('loss_after')
            if lb is not None and la is not None and self._safe_float(la, 1e9) <= self._safe_float(lb, 1e9):
                return True, 'local_loss_ok'
        return False, 'episode_loss_regression'

    def learn(
        self,
        readout,
        intent,
        target,
        semantic=None,
        context=None,
        field_system=None,
        negative_probes: Optional[List[bytes]] = None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
        max_negatives: int = 24,
    ) -> Dict:
        intent_raw = readout._to_bytes(intent)
        target_raw = readout._to_bytes(target)
        if context is not None:
            readout.observe_episode(context, semantic)

        # --- Generate Temporal Hard Negatives ---
        temporal_negatives_raw = []
        stride = getattr(readout, 'stride', 80)
        
        # 1. Extract temporal neighbors from context if possible
        context_bytes = None
        if context is not None:
            context_bytes = readout._to_bytes(context)
            
        if context_bytes is not None and len(context_bytes) > len(target_raw):
            target_idx = context_bytes.find(target_raw)
            if target_idx != -1:
                offsets = [-stride, -stride // 2, stride // 2, stride]
                for d in offsets:
                    start_idx = target_idx + d
                    if 0 <= start_idx <= len(context_bytes) - len(target_raw):
                        neg_bytes = context_bytes[start_idx : start_idx + len(target_raw)]
                        if neg_bytes != target_raw and len(neg_bytes) == len(target_raw):
                            if neg_bytes not in temporal_negatives_raw:
                                temporal_negatives_raw.append(neg_bytes)

        # 2. Fallback / Augment: Circular phase shifts of target_raw (rolls)
        n_target = len(target_raw)
        if n_target >= 4:
            roll_offsets = sorted(list(set([
                max(1, n_target // 10),
                max(2, n_target // 4),
                max(3, n_target // 2),
                max(4, (3 * n_target) // 4)
            ])))
            for r in roll_offsets:
                neg_bytes = target_raw[r:] + target_raw[:r]
                if neg_bytes != target_raw and len(neg_bytes) == n_target:
                    if neg_bytes not in temporal_negatives_raw:
                        temporal_negatives_raw.append(neg_bytes)
        # ----------------------------------------

        digest = readout._digest(intent_raw + b'\x03' + target_raw, semantic)
        segment_idx = readout._add_segment(
            target_raw,
            semantic,
            digest,
            0,
            len(target_raw),
            None,
            kind='credit_target',
        )
        target_path = [f'segment:{segment_idx}'] if segment_idx >= 0 else []
        q_feature = readout.feature_from_bytes(intent_raw, semantic)
        target_feature = readout.feature_from_bytes(target_raw, semantic)
        q_hidden = readout._project_query(q_feature)
        candidates = readout._candidate_records(q_feature, q_hidden, intent_raw)
        hard_rows = sorted(
            [c for c in candidates if str(c.get('candidate_id')) not in set(target_path)],
            key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
            reverse=True,
        )
        
        # Initial negative features
        temporal_negs_init = [
            readout.feature_from_bytes(raw_neg, semantic)
            for raw_neg in temporal_negatives_raw
        ]
        retrieved_negs_init = [row['feature'] for row in hard_rows]
        negative_features = (temporal_negs_init + retrieved_negs_init)[:max(1, int(max_negatives))]

        causal_before = self._causal_trace(readout, intent_raw, target_raw, semantic, target_path, field_system=field_system)
        betas = causal_before['adaptive_betas']
        calibration = getattr(readout, 'calibration', AdaptiveCalibration())
        episode_policy = calibration.episode_policy()
        pre_metrics = self._metrics(
            readout,
            intent_raw,
            target_raw,
            semantic=semantic,
            target_ids=target_path,
            field_system=field_system,
            negative_probes=negative_probes,
            betas=betas,
        )
        current_metrics = pre_metrics
        full_snapshot = self._snapshot(readout, semantic=semantic, field_system=field_system)
        accepted_updates = {}
        rejected_updates = {}
        layer_stats = {
            'semantic': None,
            'state': None,
            'evidence': None,
            'projection': None,
            'field': None,
            'decoder': None,
            'graph': None,
            'policy': None,
        }

        def refresh_state():
            qf = readout.feature_from_bytes(intent_raw, semantic)
            tf = readout.feature_from_bytes(target_raw, semantic)
            qh = readout._project_query(qf)
            cands = readout._candidate_records(qf, qh, intent_raw)
            rows = sorted(
                [c for c in cands if str(c.get('candidate_id')) not in set(target_path)],
                key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
                reverse=True,
            )
            retrieved_negs = [row['feature'] for row in rows]
            temporal_negs = [
                readout.feature_from_bytes(raw_neg, semantic)
                for raw_neg in temporal_negatives_raw
            ]
            negs = (temporal_negs + retrieved_negs)[:max(1, int(max_negatives))]
            return qf, tf, qh, negs

        for layer in self.LAYER_ORDER:
            before_layer = current_metrics
            snap = self._snapshot(readout, semantic=semantic, field_system=field_system)
            q_feature, target_feature, q_hidden, negative_features = refresh_state()
            beta = self._safe_float(betas.get(layer, 1.0), 1.0)
            layer_epochs = max(1, int(round(
                max(1, epochs)
                * min(
                    float(episode_policy.get('epoch_beta_max', 1.75)),
                    max(float(episode_policy.get('epoch_beta_min', 0.45)), beta),
                )
            )))
            layer_lr = float(lr) * min(
                float(episode_policy.get('lr_beta_max', 1.45)),
                max(float(episode_policy.get('lr_beta_min', 0.45)), beta),
            )
            stats = None
            if layer == 'semantic':
                if semantic is not None:
                    stats = semantic.train_contrastive_features(
                        q_feature,
                        target_feature,
                        negative_features,
                        epochs=max(4, layer_epochs // 2),
                        lr=layer_lr * float(episode_policy.get('semantic_lr_scale', 0.65)),
                        margin=float(episode_policy.get('semantic_margin', 0.24)),
                    )
                    readout._refresh_feature_cache(semantic)
            elif layer == 'state':
                if hasattr(readout, 'state_constructor') and readout.state_constructor is not None:
                    negative_query_features = [
                        readout.feature_from_bytes(readout._to_bytes(probe), semantic)
                        for probe in (negative_probes or [])
                    ]
                    stats = readout.state_constructor.train_episode(
                        q_feature,
                        target_feature,
                        negative_features,
                        target_id=target_path[0] if target_path else None,
                        association_id=readout._digest(intent_raw + b'\x07' + target_raw, semantic),
                        epochs=max(6, layer_epochs),
                        lr=layer_lr,
                        margin=float(episode_policy.get('semantic_margin', 0.24)),
                        negative_query_features=negative_query_features,
                    )
                    readout._refresh_projection_cache()
            elif layer == 'evidence':
                if hasattr(readout, 'evidence_scorer') and readout.evidence_scorer is not None:
                    stats = readout._train_evidence_scorer_target(
                        q_feature,
                        q_hidden,
                        intent_raw,
                        target_path,
                        epochs=max(6, layer_epochs),
                        lr=layer_lr,
                        margin=float(episode_policy.get('evidence_margin', episode_policy.get('projection_margin', 0.18))),
                    )
            elif layer == 'projection':
                stats = readout._train_projection_contrast(
                    q_feature,
                    target_feature,
                    negative_features,
                    epochs=max(6, layer_epochs),
                    lr=layer_lr,
                    margin=float(episode_policy.get('projection_margin', 0.22)),
                )
            elif layer == 'field':
                stats = readout.train_field_coupling(
                    intent_raw,
                    target_raw,
                    field_system=field_system,
                    epochs=max(3, layer_epochs // 3),
                    lr=layer_lr,
                    relax_steps=4 if field_system is not None else 0,
                )
            elif layer == 'decoder':
                for _ in range(max(1, int(repetitions))):
                    readout.byte_decoder.observe(intent_raw + b' ' + target_raw)
                latent_strength = 0.50 if causal_before['layer_credit'].get('graph', 0.0) > 0.0 else 0.42
                stats = readout.byte_decoder.observe_conditioned(
                    q_hidden,
                    target_raw,
                    seed=intent_raw,
                    epochs=max(6, layer_epochs),
                    lr=layer_lr * float(episode_policy.get('decoder_lr_scale', 0.80)),
                    latent_strength=latent_strength,
                )
            elif layer == 'graph':
                stats = readout._train_memory_graph_target(
                    q_feature,
                    q_hidden,
                    intent_raw,
                    target_path,
                    epochs=max(8, layer_epochs),
                    lr=layer_lr * float(episode_policy.get('graph_lr_scale', 0.78)),
                )
            elif layer == 'policy':
                stats = readout._train_reasoner_target(
                    q_feature,
                    q_hidden,
                    intent_raw,
                    target_path,
                    epochs=max(8, layer_epochs),
                    lr=layer_lr,
                )
            layer_stats[layer] = stats
            after_layer = self._metrics(
                readout,
                intent_raw,
                target_raw,
                semantic=semantic,
                target_ids=target_path,
                field_system=field_system,
                negative_probes=negative_probes,
                betas=betas,
            )
            gate_stats = dict(stats or {})
            gate_stats['_episode_policy'] = episode_policy
            accepted, reason = self._accept_layer(layer, before_layer, after_layer, gate_stats)
            if accepted:
                accepted_updates[layer] = {
                    'reason': reason,
                    'episode_loss_before': float(before_layer.get('episode_loss', 0.0)),
                    'episode_loss_after': float(after_layer.get('episode_loss', 0.0)),
                }
                current_metrics = after_layer
            else:
                self._restore_layer(readout, snap, layer, semantic=semantic, field_system=field_system)
                rejected_updates[layer] = {
                    'reason': reason,
                    'episode_loss_before': float(before_layer.get('episode_loss', 0.0)),
                    'episode_loss_after': float(after_layer.get('episode_loss', 0.0)),
                }
                layer_stats[layer] = {
                    'status': 'rejected',
                    'rejection_reason': reason,
                    'original': stats,
                }

        post_metrics = self._metrics(
            readout,
            intent_raw,
            target_raw,
            semantic=semantic,
            target_ids=target_path,
            field_system=field_system,
            negative_probes=negative_probes,
            betas=betas,
        )
        if post_metrics['episode_loss'] > pre_metrics['episode_loss'] + float(episode_policy.get('episode_loss_tolerance', 0.10)):
            self._restore_layer(readout, full_snapshot, 'all', semantic=semantic, field_system=field_system)
            post_metrics = self._metrics(
                readout,
                intent_raw,
                target_raw,
                semantic=semantic,
                target_ids=target_path,
                field_system=field_system,
                negative_probes=negative_probes,
                betas=betas,
            )
            rejected_updates['episode'] = {
                'reason': 'full_episode_loss_regression',
                'episode_loss_before': float(pre_metrics['episode_loss']),
                'episode_loss_after': float(post_metrics['episode_loss']),
            }

        causal_after = self._causal_trace(readout, intent_raw, target_raw, semantic, target_path, field_system=field_system)
        graph_stats = layer_stats.get('graph') or {}
        negative_report = (post_metrics.get('negative_probes') or [{}])[0]
        calibration.observe_episode_metrics(pre_metrics, post_metrics)
        for probe in post_metrics.get('negative_probes', []):
            calibration.observe_negative_probe(probe.get('confidence', 0.0))
        for layer_name, update in accepted_updates.items():
            delta = float(update.get('episode_loss_before', 0.0)) - float(update.get('episode_loss_after', 0.0))
            if delta >= 0.0:
                calibration.observe('accepted_loss_tolerance', max(0.01, delta))
        result = CreditEpisodeResult({
            'status': 'trained',
            'target_ids': target_path,
            'segments_total': len(readout.segments),
            'episode_loss_before': float(pre_metrics['episode_loss']),
            'episode_loss_after': float(post_metrics['episode_loss']),
            'layer_credit': causal_after.get('layer_credit', causal_before.get('layer_credit', {})),
            'adaptive_betas': betas,
            'accepted_updates': accepted_updates,
            'rejected_updates': rejected_updates,
            'counterfactual_report': {
                'before': {
                    'graph_cf_drop': float(pre_metrics.get('graph_cf_drop', 0.0)),
                    'target_ablation_drop': float(causal_before.get('target_ablation_drop', 0.0)),
                    'graph_edge_flow_drop': float(causal_before.get('graph_edge_flow_drop', 0.0)),
                },
                'after': {
                    'graph_cf_drop': float(post_metrics.get('graph_cf_drop', 0.0)),
                    'target_ablation_drop': float(causal_after.get('target_ablation_drop', 0.0)),
                    'graph_edge_flow_drop': float(causal_after.get('graph_edge_flow_drop', 0.0)),
                },
            },
            'calibration_report': {
                'before_error': float(pre_metrics.get('calibration_error', 0.0)),
                'after_error': float(post_metrics.get('calibration_error', 0.0)),
                'random_probe_confidence_before': float(pre_metrics.get('random_probe_confidence', 0.0)),
                'random_probe_confidence_after': float(post_metrics.get('random_probe_confidence', 0.0)),
                'policy': episode_policy,
            },
            'calibration_state': calibration.report(),
            'causal_trace_before': causal_before,
            'causal_trace_after': causal_after,
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
            'semantic_training': layer_stats.get('semantic'),
            'state_constructor_training': layer_stats.get('state'),
            'evidence_scorer_training': layer_stats.get('evidence'),
            'projection_training': layer_stats.get('projection'),
            'field_training': layer_stats.get('field'),
            'policy_training': layer_stats.get('policy'),
            'graph_training': graph_stats,
            'conditioned_decoder_training': layer_stats.get('decoder'),
            'decoder_nll_before': {
                'mean_nll': float(pre_metrics.get('decoder_nll', 0.0)),
                'bytes': len(target_raw),
            },
            'decoder_nll_after': {
                'mean_nll': float(post_metrics.get('decoder_nll', 0.0)),
                'bytes': len(target_raw),
            },
            'conditioned_nll_before': {
                'mean_nll': float(pre_metrics.get('conditioned_nll', 0.0)),
                'bytes': len(target_raw),
            },
            'conditioned_nll_after': {
                'mean_nll': float(post_metrics.get('conditioned_nll', 0.0)),
                'bytes': len(target_raw),
            },
            'before_response': pre_metrics.get('response', {}),
            'after_response': post_metrics.get('response', {}),
            'negative_probe': {
                'responded': bool(negative_report.get('responded', False)),
                'confidence': float(negative_report.get('confidence', 0.0)),
                'response_kind': negative_report.get('response_kind'),
            },
            'target_lift_before': float(pre_metrics.get('target_lift_over_prior', 0.0)),
            'target_lift_after': float(post_metrics.get('target_lift_over_prior', 0.0)),
        })
        self.history.append(result)
        return result



