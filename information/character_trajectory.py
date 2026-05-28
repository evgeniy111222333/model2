"""
Character-Aware Trajectory Extension

Інтегрує character-level processing в існуючий HierarchicalTrajectory.
Це АДАПТЕР — не змінює існуючий код, а додає нові можливості.

Принцип: "Give it raw bytes — it learns everything else"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from bcs.perception.utf8_segmenter import UTF8Segmenter, AdaptiveUTF8Segmenter
from bcs.information.character_manifold import CharacterManifold, create_character_manifold
from bcs.information.character_continuation import CharacterGeometricContinuation


class CharacterTrajectory:
    """
    Character-aware extension для HierarchicalTrajectory.
    
    Працює поверх існуючої траєкторії, додаючи:
    - UTF-8 sequence parsing
    - Character manifold integration
    - Geometric continuation в character space
    
    Використовує composition, не modification існуючого коду.
    """
    
    def __init__(
        self,
        base_trajectory=None,  # HierarchicalTrajectory reference
        embedding_dim: int = 64,
        use_adaptive: bool = True,
    ):
        """
        Args:
            base_trajectory: HierarchicalTrajectory для інтеграції
            embedding_dim: dimensionality для character embeddings
            use_adaptive: використовувати AdaptiveUTF8Segmenter
        """
        self.base_trajectory = base_trajectory
        
        # UTF-8 segmentation
        self.segmenter = AdaptiveUTF8Segmenter() if use_adaptive else UTF8Segmenter()
        
        # Character manifold (self-learned)
        self.manifold: Optional[CharacterManifold] = None
        
        # Character continuation
        self.continuation: Optional[CharacterGeometricContinuation] = None
        
        # Character-level trajectory
        self.character_points: List[Tuple[int, int]] = []  # [(codepoint, position), ...]
        
        # Character embeddings cached in base trajectory points
        self._character_embeddings: Dict[int, np.ndarray] = {}
        
        self.embedding_dim = embedding_dim
        self.is_trained = False
    
    def process_input(self, data: bytes) -> List[Tuple[bytes, int, str]]:
        """
        Обробити вхідні дані — розбір на UTF-8 sequences.
        
        Returns:
            List[(bytes, position, region)]
        """
        sequences = self.segmenter.segment_with_regions(data)
        
        # Update segmenter stats
        self.segmenter.learn_regions(data)
        
        return sequences
    
    def build_from_data(
        self,
        data: bytes,
        base_trajectory_points: List[np.ndarray] = None,
    ):
        """
        Побудувати character-level компоненти з даних.
        
        Args:
            data: raw bytes
            base_trajectory_points: points from HierarchicalTrajectory for alignment
        """
        # Parse UTF-8
        sequences = self.segmenter.segment(data)
        
        if len(sequences) < 5:
            return
        
        # Build character manifold
        seq_data = [(seq.bytes_data, seq.start) for seq in sequences]
        self.manifold = create_character_manifold(seq_data, data)
        
        # Build continuation
        self.continuation = CharacterGeometricContinuation(
            manifold=self.manifold,
            temperature=1.0,
        )
        
        # Build character trajectory
        self.character_points = [
            (seq.codepoint, seq.start) for seq in sequences
        ]
        
        self.is_trained = True
    
    def push_character(self, codepoint: int, position: int, embedding: np.ndarray = None):
        """
        Додати символ до character trajectory.
        
        Args:
            codepoint: unicode codepoint
            position: position in data
            embedding: optional embedding (if manifold trained)
        """
        self.character_points.append((codepoint, position))
        
        # Cache embedding
        if embedding is not None:
            self._character_embeddings[codepoint] = embedding
        
        # Forward to base trajectory if available
        if self.base_trajectory is not None and embedding is not None:
            self._push_to_base(embedding, position)
    
    def _push_to_base(self, embedding: np.ndarray, position: int):
        """Push character embedding as distribution to base trajectory."""
        if self.base_trajectory is None:
            return
        
        # Convert embedding to distribution-like array
        # This is a simplified representation
        pass  # Could create a distribution from embedding
    
    def get_character_trajectory(self) -> List[Tuple[int, int]]:
        """Get character trajectory as (codepoint, position) pairs."""
        return list(self.character_points)
    
    def predict_next_characters(
        self,
        n: int = 1,
        method: str = 'nucleus',
    ) -> List[Tuple[int, float]]:
        """
        Передбачити наступні символи з geometric continuation.
        
        Returns:
            List[(codepoint, probability)]
        """
        if not self.continuation or not self.character_points:
            return []
        
        trajectory = list(self.character_points)
        predictions = []
        
        for _ in range(n):
            probs = self.continuation.continue_from_trajectory(trajectory)
            cp, prob = self.continuation.sample_from_probs(probs, method=method)
            predictions.append((cp, prob))
            trajectory.append((cp, -1))  # -1 for generated
        
        return predictions
    
    def continue_string(
        self,
        prefix: str,
        max_len: int = 100,
    ) -> str:
        """
        Продовжити string з geometric continuation.
        
        Args:
            prefix: prefix string to continue
            max_len: max characters to generate
            
        Returns:
            generated continuation
        """
        prefix_bytes = prefix.encode('utf-8')
        sequences = self.segmenter.segment(prefix_bytes)
        
        # Update trajectory
        self.character_points = [
            (seq.codepoint, seq.start) for seq in sequences
        ]
        
        result = [prefix]
        
        for _ in range(max_len):
            if not self.character_points:
                break
            
            probs = self.continuation.continue_from_trajectory(
                self.character_points
            )
            
            cp, _ = self.continuation.sample_from_probs(probs)
            
            try:
                char = chr(cp)
                result.append(char)
                self.character_points.append((cp, -1))
                
                if char == ' ' and len(result) > 5:  # Stop at space after some text
                    break
            except ValueError:
                break
        
        return ''.join(result)
    
    def get_character_distribution(self, context_len: int = 20) -> Dict[int, float]:
        """
        Отримати розподіл ймовірностей символів для поточного контексту.
        
        Returns:
            {codepoint: probability}
        """
        if not self.continuation or not self.character_points:
            return {}
        
        recent = self.character_points[-context_len:] if len(self.character_points) > context_len else self.character_points
        return self.continuation.continue_from_trajectory(recent)
    
    def get_stats(self) -> Dict:
        """Get character trajectory stats."""
        stats = {
            'n_characters': len(self.character_points),
            'n_regions': len(self.segmenter.regions) if hasattr(self.segmenter, 'regions') else 0,
            'is_trained': self.is_trained,
        }
        
        if self.manifold:
            stats['manifold'] = self.manifold.get_stats()
        
        return stats


class MultiModalCharacterTrajectory:
    """
    Character-level trajectory для мульти-modal вводу.
    
    Автоматично визначає script і адаптує geometric continuation.
    """
    
    # Script detection patterns (self-learned)
    SCRIPT_PATTERNS = {
        'cyrillic': {'first_bytes': [0xD0, 0xD1, 0xD2], 'range': (0x0400, 0x04FF)},
        'latin': {'first_bytes': [0xC3, 0xC4], 'range': (0x0080, 0x024F)},
        'cjk': {'first_bytes': [0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF], 'range': (0x4E00, 0x9FFF)},
        'arabic': {'first_bytes': [0xD8, 0xD9, 0xDA, 0xDB, 0xDC], 'range': (0x0600, 0x06FF)},
        'devanagari': {'first_bytes': [0xE0], 'range': (0x0900, 0x097F)},
    }
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.trajectories: Dict[str, CharacterTrajectory] = {}
        self.current_script = 'cyrillic'  # Default
        self.script_switches = 0
    
    def detect_script(self, data: bytes) -> str:
        """
        Detect script from byte patterns.
        Self-learned через byte prefix clustering.
        """
        if len(data) < 3:
            return 'unknown'
        
        # Count first bytes
        first_bytes = defaultdict(int)
        for i in range(min(100, len(data))):
            if data[i] >= 0x80:
                first_bytes[data[i]] += 1
        
        # Match against learned patterns
        for script, pattern in self.SCRIPT_PATTERNS.items():
            match_count = sum(
                count for byte, count in first_bytes.items()
                if byte in pattern['first_bytes']
            )
            if match_count > 5:
                return script
        
        return 'latin'  # Default to latin for ASCII
    
    def switch_script(self, script: str, data: bytes):
        """Switch to new script, building new trajectory if needed."""
        if script not in self.trajectories:
            self.trajectories[script] = CharacterTrajectory(
                embedding_dim=self.embedding_dim
            )
            self.trajectories[script].build_from_data(data)
        
        self.current_script = script
        self.script_switches += 1
    
    def process(self, data: bytes) -> CharacterTrajectory:
        """Process data, auto-detecting script."""
        script = self.detect_script(data)
        
        if script != self.current_script:
            self.switch_script(script, data)
        
        return self.trajectories[self.current_script]
    
    def get_current_trajectory(self) -> CharacterTrajectory:
        """Get current script trajectory."""
        return self.trajectories.get(self.current_script)


def create_character_trajectory(
    data: bytes,
    base_trajectory=None,
) -> CharacterTrajectory:
    """
    Factory: create and train character trajectory from data.
    """
    ct = CharacterTrajectory(base_trajectory=base_trajectory)
    ct.build_from_data(data)
    return ct


# Test
if __name__ == '__main__':
    from bcs.information.trajectory_first import HierarchicalTrajectory
    
    # Test data
    test_text = "Привіт світе! Hello world. Привіт всім на світі!"
    data = test_text.encode('utf-8')
    
    # Create base trajectory
    base = HierarchicalTrajectory(base_size=50, max_levels=4)
    
    # Create character extension
    char_traj = CharacterTrajectory(base_trajectory=base)
    char_traj.build_from_data(data)
    
    print(f"Character trajectory stats: {char_traj.get_stats()}")
    
    # Test continuation
    print(f"\nOriginal: 'Привіт с'")
    result = char_traj.continue_string("Привіт с", max_len=10)
    print(f"Continued: '{result}'")
    
    # Test prediction
    preds = char_traj.predict_next_characters(n=3)
    print(f"\nTop 3 predictions:")
    for cp, prob in preds:
        try:
            print(f"  '{chr(cp)}': {prob:.3f}")
        except:
            pass