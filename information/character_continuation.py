"""
Character Geometric Continuation Module

Геометрична інтерполяція в character space.
Це СЕРЕДНИК системи — між trajectory і output.

Принцип: "Give it raw bytes — it learns everything else"
Відстань, attention, continuation — все self-learned.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from bcs.information.character_manifold import CharacterManifold, CharacterPoint


class CharacterGeometricContinuation:
    """
    Geometric continuation в character space.
    
    Замість byte-level interpolation (що призводить до low confidence):
    - Працює з character embeddings
    - Використовує region-aware distances
    - Self-learned transition probabilities
    - BIGRAM/TRIGRAM support (V13)
    
    Алгоритм:
    1. Get character/bigram trajectory (recent context as points)
    2. Compute novelty for each historical element
    3. Geometric attention: exp(-dist²/T) weighted by novelty
    4. Aggregate features
    5. Predict next character/bigram
    """
    
    def __init__(
        self,
        manifold: CharacterManifold,
        temperature: float = 1.0,
        novelty_threshold: float = 0.3,
        max_context: int = 50,
        use_ngrams: bool = True,  # NEW: enable bigram/trigram
        max_ngram_order: int = 3,  # NEW: up to trigrams
    ):
        self.manifold = manifold
        self.temperature = temperature
        self.novelty_threshold = novelty_threshold
        self.max_context = max_context
        self.use_ngrams = use_ngrams  # NEW
        self.max_ngram_order = max_ngram_order  # NEW
        
        # Learned transition probabilities (overridden by manifold's)
        self._transition_cache = {}
        
        # History for adaptive temperature
        self.history_confidences = []
        
        # NEW: N-gram statistics
        self._bigram_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._trigram_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self._bigram_context: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._trigram_context: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._total_bigrams = 0
        self._total_trigrams = 0
        self.history_confidences = []
    
    def continue_from_trajectory(
        self,
        trajectory: List[Tuple[int, int]],  # [(codepoint, position), ...]
        region_hint: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Геометрична інтерполяція з траєкторії символів.
        
        Args:
            trajectory: [(codepoint, position), ...] — recent characters
            region_hint: optional region to bias toward
            
        Returns:
            Dict[int, float]: {codepoint: probability} for next char
        """
        if not trajectory:
            return self._uniform_distribution()
        
        # Limit context
        recent = trajectory[-self.max_context:]
        
        # 1. Get character points from manifold
        char_points = []
        for cp, pos in recent:
            if cp in self.manifold.characters:
                char_points.append(self.manifold.characters[cp])
        
        if not char_points:
            # Fallback to codepoint-based
            return self._codepoint_continuation(trajectory, region_hint)
        
        # 2. Compute novelty for each character
        novelties = self._compute_novelties(char_points)
        
        # 3. Geometric attention
        query_point = char_points[-1].embedding if char_points else None
        
        attention_weights = self._compute_attention(
            query_point, 
            [p.embedding for p in char_points],
            novelties
        )
        
        # 4. Aggregate character features
        aggregated_embedding = np.zeros(self.manifold.embedding_dim)
        for i, point in enumerate(char_points):
            aggregated_embedding += attention_weights[i] * point.embedding
        
        # 5. Predict next character (V13: combined n-gram + embedding)
        if self.use_ngrams and (self._total_bigrams > 0 or self._total_trigrams > 0):
            # Use combined prediction: n-gram + embedding
            next_char_probs = self._combined_prediction(
                trajectory=recent,
                context_embedding=aggregated_embedding,
                last_region=char_points[-1].region if char_points else None,
            )
        else:
            # Fallback to embedding-only
            next_char_probs = self._predict_from_embedding(
                aggregated_embedding,
                last_region=char_points[-1].region if char_points else None,
                region_hint=region_hint
            )
        
        return next_char_probs
    
    def _compute_novelties(self, char_points: List[CharacterPoint]) -> List[float]:
        """
        Compute novelty scores for each character point.
        
        Novelty = distance to nearest neighbor in trajectory.
        High novelty = unusual character = high attention weight.
        """
        if len(char_points) < 2:
            return [1.0]
        
        novelties = []
        
        for i, point in enumerate(char_points):
            # Find nearest neighbor (excluding self)
            min_dist = float('inf')
            for j, other in enumerate(char_points):
                if i == j:
                    continue
                dist = self.manifold.character_distance(point.codepoint, other.codepoint)
                if dist < min_dist:
                    min_dist = dist
            
            # Normalize novelty to [0, 1]
            novelty = min(min_dist * 10, 1.0)
            novelties.append(novelty)
        
        return novelties
    
    def _compute_attention(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        novelties: List[float],
    ) -> np.ndarray:
        """
        Compute geometric attention weights.
        
        attention_i = exp(-dist(query, key_i)² / T) * (1 + novelty_i)
        """
        n = len(keys)
        weights = np.zeros(n)
        
        for i, key in enumerate(keys):
            # Euclidean distance in embedding space
            dist = np.linalg.norm(query - key) if query is not None else 1.0
            
            # Geometric attention
            attn = np.exp(-(dist ** 2) / self.temperature)
            
            # Novelty boost
            if novelties[i] > self.novelty_threshold:
                attn *= (1 + novelties[i])
            
            weights[i] = attn
        
        # Normalize
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(n) / n
        
        return weights
    
    def _predict_from_embedding(
        self,
        context_embedding: np.ndarray,
        last_region: Optional[str] = None,
        region_hint: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Predict next characters from aggregated embedding.
        """
        # Get candidates from regions
        candidates = set()
        
        # Preferred region
        preferred_region = region_hint or last_region or 'cyrillic_lower'
        
        if preferred_region in self.manifold.regions:
            candidates.update(self.manifold.regions[preferred_region].codepoints)
        
        # Also include adjacent regions for natural continuation
        if last_region and last_region in self.manifold.regions:
            transitions = self.manifold.regions[last_region].transitions
            for key in transitions:
                if '->' in key:
                    next_region = key.split('->')[1]
                    if next_region in self.manifold.regions:
                        candidates.update(self.manifold.regions[next_region].codepoints)
        
        # Fallback to all characters
        if not candidates:
            candidates = set(self.manifold.characters.keys())
        
        # Compute probabilities
        probs = {}
        
        for cp in candidates:
            if cp not in self.manifold.characters:
                continue
            
            point = self.manifold.characters[cp]
            
            # Distance from context to this character
            dist = np.linalg.norm(context_embedding - point.embedding)
            
            # Transition probability
            trans_prob = 0.1  # Base
            if last_region:
                trans_prob = self.manifold.get_transition_prob(
                    context_embedding[0] if len(context_embedding) > 0 else 0,
                    cp
                ) * 0.5 + 0.1
            
            # Embedding similarity
            emb_sim = np.exp(-dist * 2.0)
            
            # Combined probability
            prob = 0.4 * emb_sim + 0.3 * trans_prob + 0.3 * (1.0 / (1.0 + dist))
            
            probs[cp] = prob
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {cp: p / total for cp, p in probs.items()}
        else:
            return self._uniform_distribution()
        
        # Update confidence history
        max_prob = max(probs.values()) if probs else 0
        self.history_confidences.append(max_prob)
        if len(self.history_confidences) > 100:
            self.history_confidences = self.history_confidences[-100:]
        
        return probs
    
    def _codepoint_continuation(
        self,
        trajectory: List[Tuple[int, int]],
        region_hint: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Fallback: continuation using only codepoint information.
        Used when manifold doesn't have embeddings for characters.
        """
        if not trajectory:
            return self._uniform_distribution()
        
        last_cp = trajectory[-1][0]
        last_region = self.manifold.get_region_for_cp(last_cp)
        
        # Get candidates from region
        candidates = []
        preferred_region = region_hint or last_region
        
        if preferred_region in self.manifold.regions:
            candidates = list(self.manifold.regions[preferred_region].codepoints)
        else:
            # Use transitions
            for (f, t), count in self.manifold.transitions.items():
                if f == last_cp and t not in candidates:
                    candidates.append(t)
        
        if not candidates:
            candidates = list(range(0x0410, 0x044F))  # Default Cyrillic range
        
        # Simple probability based on codepoint distance
        probs = {}
        for cp in candidates:
            dist = abs(cp - last_cp) / 0x10FFFF
            probs[cp] = np.exp(-dist * 10)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {cp: p / total for cp, p in probs.items()}
        
        return probs
    
    def _uniform_distribution(self) -> Dict[int, float]:
        """Fallback uniform distribution."""
        if self.manifold.characters:
            n = len(self.manifold.characters)
            return {cp: 1.0 / n for cp in self.manifold.characters}
        return {cp: 1.0 / 256 for cp in range(256)}
    
    def adjust_temperature(self, confidence: float) -> float:
        """
        Adaptive temperature based on recent confidence.
        
        If confidence is low (model uncertain) → increase temperature
        If confidence is high → decrease temperature
        """
        if len(self.history_confidences) < 5:
            return self.temperature
        
        avg_conf = np.mean(self.history_confidences[-10:])
        
        if avg_conf < 0.3:
            # Low confidence — be more exploratory
            return min(self.temperature * 1.5, 5.0)
        elif avg_conf > 0.7:
            # High confidence — be more decisive
            return max(self.temperature * 0.8, 0.3)
        
        return self.temperature
    
    def sample_from_probs(
        self,
        probs: Dict[int, float],
        method: str = 'nucleus',
        top_p: float = 0.9,
        top_k: int = 20,
    ) -> Tuple[int, float]:
        """
        Sample next character from probability distribution.
        
        Args:
            probs: {codepoint: probability}
            method: 'nucleus', 'top_k', or 'temperature'
            top_p: nucleus threshold
            top_k: top-k cutoff
            
        Returns:
            (codepoint, log_prob)
        """
        if not probs:
            return 0, float('-inf')
        
        cps = list(probs.keys())
        p = np.array([probs[cp] for cp in cps])
        
        # Sort by probability
        sorted_idx = np.argsort(-p)
        sorted_p = p[sorted_idx]
        sorted_cps = [cps[i] for i in sorted_idx]
        
        # Apply filtering
        if method == 'nucleus':
            # Nucleus sampling
            cumsum = np.cumsum(sorted_p)
            cutoff = cumsum <= top_p
            if not cutoff.any():
                cutoff = [True]  # At least one
            kept_idx = np.where(cutoff)[0]
            kept_p = sorted_p[kept_idx]
            kept_cps = [sorted_cps[i] for i in kept_idx]
            kept_p /= kept_p.sum()
            
            chosen = np.random.choice(len(kept_cps), p=kept_p)
            return kept_cps[chosen], np.log(kept_p[chosen])
        
        elif method == 'top_k':
            # Top-k sampling
            kept_cps = sorted_cps[:top_k]
            kept_p = sorted_p[:top_k]
            kept_p /= kept_p.sum()
            
            chosen = np.random.choice(len(kept_cps), p=kept_p)
            return kept_cps[chosen], np.log(kept_p[chosen])
        
        else:  # temperature
            # Temperature scaling
            temp = self.temperature
            scaled_p = sorted_p ** (1.0 / temp)
            scaled_p /= scaled_p.sum()
            
            chosen = np.random.choice(len(sorted_cps), p=scaled_p)
            return sorted_cps[chosen], np.log(scaled_p[chosen])

    # ============ N-GRAM LEARNING (V13) ============

    def learn_ngrams(self, sequences: List[Tuple[bytes, int]]):
        """Learn bigram and trigram statistics from sequences."""
        if not self.use_ngrams:
            return

        codepoints = []
        for seq_bytes, pos in sequences:
            try:
                cp = ord(seq_bytes.decode('utf-8'))
                codepoints.append(cp)
            except:
                continue

        for i in range(len(codepoints) - 1):
            bigram = (codepoints[i], codepoints[i + 1])
            self._bigram_counts[bigram] += 1
            self._total_bigrams += 1

        for i in range(len(codepoints) - 2):
            trigram = (codepoints[i], codepoints[i + 1], codepoints[i + 2])
            self._trigram_counts[trigram] += 1
            self._total_trigrams += 1

        for (cp1, cp2), count in self._bigram_counts.items():
            self._bigram_context[cp1][cp2] = count / self._total_bigrams
        for (cp1, cp2, cp3), count in self._trigram_counts.items():
            self._trigram_context[(cp1, cp2)][cp3] = count / self._total_trigrams

    def _ngram_prediction(self, trajectory: List[Tuple[int, int]]) -> Dict[int, float]:
        """Predict using n-gram statistics."""
        probs = {}
        if not self.use_ngrams or len(trajectory) < 2:
            return {}

        cps = [cp for cp, pos in trajectory]

        if len(cps) >= 2:
            prev2, prev1 = cps[-2], cps[-1]
            trigram_key = (prev2, prev1)
            if trigram_key in self._trigram_context:
                for cp, prob in self._trigram_context[trigram_key].items():
                    probs[cp] = probs.get(cp, 0) + prob * 0.6
                return probs

        prev1 = cps[-1]
        if prev1 in self._bigram_context:
            for cp, prob in self._bigram_context[prev1].items():
                probs[cp] = probs.get(cp, 0) + prob * 0.4
        return probs

    def _combined_prediction(
        self,
        trajectory: List[Tuple[int, int]],
        context_embedding: np.ndarray,
        last_region: Optional[str] = None,
    ) -> Dict[int, float]:
        """Combine n-gram and embedding predictions."""
        emb_probs = self._predict_from_embedding(context_embedding, last_region)
        ngram_probs = self._ngram_prediction(trajectory)
        if not ngram_probs:
            return emb_probs

        ngram_total = sum(ngram_probs.values())
        if ngram_total > 0:
            ngram_probs = {cp: p / ngram_total for cp, p in ngram_probs.items()}
        emb_total = sum(emb_probs.values())
        if emb_total > 0:
            emb_probs = {cp: p / emb_total for cp, p in emb_probs.items()}

        ngram_conf = min(1.0, self._total_trigrams / 10000)
        alpha = 0.5 * ngram_conf

        combined = {}
        all_cps = set(list(ngram_probs.keys()) + list(emb_probs.keys()))
        for cp in all_cps:
            ngram_p = ngram_probs.get(cp, 0)
            emb_p = emb_probs.get(cp, 0)
            combined[cp] = alpha * ngram_p + (1 - alpha) * emb_p

        total = sum(combined.values())
        if total > 0:
            combined = {cp: p / total for cp, p in combined.items()}
        return combined


class CharacterContinuationReader:
    """
    Readout layer для конвертації character probabilities → output bytes.
    
    Інтегрується з TrajectoryReadout в information/trajectory_first.py
    """
    
    def __init__(self, continuation: CharacterGeometricContinuation):
        self.continuation = continuation
    
    def read_characters(
        self,
        trajectory: List[Tuple[int, int]],
        n_chars: int = 1,
        method: str = 'nucleus',
        region_hint: Optional[str] = None,
    ) -> Tuple[bytes, List[float]]:
        """
        Зчитати наступні n символів.
        
        Args:
            trajectory: [(codepoint, position), ...]
            n_chars: кількість символів для генерації
            method: sampling method
            region_hint: optional region bias
            
        Returns:
            (generated_bytes, log_probs)
        """
        generated = []
        log_probs = []
        current_trajectory = list(trajectory)
        
        for _ in range(n_chars):
            probs = self.continuation.continue_from_trajectory(
                current_trajectory,
                region_hint=region_hint
            )
            
            cp, log_p = self.continuation.sample_from_probs(probs, method=method)
            
            try:
                char_bytes = chr(cp).encode('utf-8')
                generated.append(char_bytes)
                log_probs.append(log_p)
            except (ValueError, UnicodeEncodeError):
                continue
            
            current_trajectory.append((cp, -1))  # -1 for generated
        
        return b''.join(generated), log_probs
    
    def read_string(
        self,
        trajectory: List[Tuple[int, int]],
        max_len: int = 100,
        stop_at_space: bool = False,
    ) -> str:
        """
        Зчитати string до max_len або stop condition.
        """
        result = []
        current_trajectory = list(trajectory)
        
        for _ in range(max_len):
            probs = self.continuation.continue_from_trajectory(
                current_trajectory
            )
            
            cp, _ = self.continuation.sample_from_probs(probs)
            
            try:
                char = chr(cp)
                result.append(char)
                
                if stop_at_space and char == ' ':
                    break
                
                current_trajectory.append((cp, -1))
            except ValueError:
                break

        return ''.join(result)

    # ============ N-GRAM LEARNING (V13) ============

    def learn_ngrams(self, sequences: List[Tuple[bytes, int]]):
        """Learn bigram and trigram statistics from sequences."""
        if not self.use_ngrams:
            return

        codepoints = []
        for seq_bytes, pos in sequences:
            try:
                cp = ord(seq_bytes.decode('utf-8'))
                codepoints.append(cp)
            except:
                continue

        for i in range(len(codepoints) - 1):
            bigram = (codepoints[i], codepoints[i + 1])
            self._bigram_counts[bigram] += 1
            self._total_bigrams += 1

        for i in range(len(codepoints) - 2):
            trigram = (codepoints[i], codepoints[i + 1], codepoints[i + 2])
            self._trigram_counts[trigram] += 1
            self._total_trigrams += 1

        for (cp1, cp2), count in self._bigram_counts.items():
            self._bigram_context[cp1][cp2] = count / self._total_bigrams
        for (cp1, cp2, cp3), count in self._trigram_counts.items():
            self._trigram_context[(cp1, cp2)][cp3] = count / self._total_trigrams

    def _ngram_prediction(self, trajectory: List[Tuple[int, int]]) -> Dict[int, float]:
        """Predict using n-gram statistics."""
        probs = {}
        if not self.use_ngrams or len(trajectory) < 2:
            return {}

        cps = [cp for cp, pos in trajectory]

        if len(cps) >= 2:
            prev2, prev1 = cps[-2], cps[-1]
            trigram_key = (prev2, prev1)
            if trigram_key in self._trigram_context:
                for cp, prob in self._trigram_context[trigram_key].items():
                    probs[cp] = probs.get(cp, 0) + prob * 0.6
                return probs

        prev1 = cps[-1]
        if prev1 in self._bigram_context:
            for cp, prob in self._bigram_context[prev1].items():
                probs[cp] = probs.get(cp, 0) + prob * 0.4
        return probs

    def _combined_prediction(
        self,
        trajectory: List[Tuple[int, int]],
        context_embedding: np.ndarray,
        last_region: Optional[str] = None,
    ) -> Dict[int, float]:
        """Combine n-gram and embedding predictions."""
        emb_probs = self._predict_from_embedding(context_embedding, last_region)
        ngram_probs = self._ngram_prediction(trajectory)
        if not ngram_probs:
            return emb_probs

        ngram_total = sum(ngram_probs.values())
        if ngram_total > 0:
            ngram_probs = {cp: p / ngram_total for cp, p in ngram_probs.items()}
        emb_total = sum(emb_probs.values())
        if emb_total > 0:
            emb_probs = {cp: p / emb_total for cp, p in emb_probs.items()}

        ngram_conf = min(1.0, self._total_trigrams / 10000)
        alpha = 0.5 * ngram_conf

        combined = {}
        all_cps = set(list(ngram_probs.keys()) + list(emb_probs.keys()))
        for cp in all_cps:
            ngram_p = ngram_probs.get(cp, 0)
            emb_p = emb_probs.get(cp, 0)
            combined[cp] = alpha * ngram_p + (1 - alpha) * emb_p

        total = sum(combined.values())
        if total > 0:
            combined = {cp: p / total for cp, p in combined.items()}
        return combined


def create_continuation(manifold: CharacterManifold) -> CharacterGeometricContinuation:
    """Factory for creating character continuation."""
    return CharacterGeometricContinuation(manifold=manifold)


# Test
if __name__ == '__main__':
    from bcs.perception.utf8_segmenter import UTF8Segmenter
    from bcs.information.character_manifold import create_character_manifold
    
    test_text = "Привіт світе! Як справи?"
    data = test_text.encode('utf-8')
    
    # Parse
    segmenter = UTF8Segmenter()
    sequences = segmenter.segment(data)
    seq_data = [(seq.bytes_data, seq.start) for seq in sequences]
    
    # Create manifold
    manifold = create_character_manifold(seq_data, data)
    print(f"Manifold: {manifold.get_stats()}")
    
    # Create continuation
    continuation = CharacterGeometricContinuation(manifold)
    
    # Test continuation
    trajectory = [(seq.codepoint, seq.start) for seq in sequences[:5]]
    print(f"\nTrajectory: {[chr(cp) for cp, _ in trajectory]}")
    
    probs = continuation.continue_from_trajectory(trajectory)
    print(f"\nTop 5 predictions:")
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:5]
    for cp, p in sorted_probs:
        try:
            print(f"  '{chr(cp)}' (U+{cp:04X}): {p:.3f}")
        except:
            pass
    
    # Test generation
    generated, _ = continuation.continue_from_trajectory(trajectory)
    top_cp, top_p = continuation.sample_from_probs(probs)
    print(f"\nSampled: '{chr(top_cp)}' (U+{top_cp:04X}) with prob {top_p:.3f}")