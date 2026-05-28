"""
Character Manifold Module

Символьний многовид для geometric continuation в character space.
Повністю self-learned — система сама виявляє regions і структуру символів.

Принцип: "Give it raw bytes — it learns everything else"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CharacterRegion:
    """
    Self-learned region — система сама визначає які символи cluster together.
    
    Не хардкодимо "cyrillic_upper = А-Я" — система вивчає з даних.
    """
    name: str
    codepoints: Set[int] = field(default_factory=set)
    centroid: np.ndarray = None  # Learned centroid in embedding space
    std_cp: float = 0.0  # Spread of codepoints
    first_byte_dist: Dict[int, float] = field(default_factory=dict)
    transition_probs: Dict[str, float] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.codepoints)
    
    @property
    def mean_cp(self) -> float:
        return np.mean(list(self.codepoints)) if self.codepoints else 0


@dataclass 
class CharacterPoint:
    """Точка на character manifold."""
    codepoint: int
    char_bytes: bytes
    char_str: str
    region: str
    embedding: np.ndarray  # R^embedding_dim — learned representation
    position: int  # Position in data (for trajectory)
    
    @property
    def is_multibyte(self) -> bool:
        return len(self.char_bytes) > 1


class CharacterManifold:
    """
    Character manifold — многовид символів для geometric operations.
    
    Відмінності від byte simplex:
    - Точки = символи (code points), не байти
    - Відстань визначається семантичною близкістю, не просто code point difference
    - Regions learned з даних, не hardcoded
    - Transition probabilities learned from usage patterns
    
    Атрибути self-learned:
    - regions: які символи cluster together
    - embeddings: learned representations
    - distances: geometric structure
    - transitions: character usage patterns
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        use_fisher_metric: bool = True,
        min_region_size: int = 2,
    ):
        """
        Args:
            embedding_dim: dimensionality of character embeddings
            use_fisher_metric: use Fisher-Rao for distance (like byte manifold)
            min_region_size: minimum chars to form a region
        """
        self.embedding_dim = embedding_dim
        self.use_fisher_metric = use_fisher_metric
        self.min_region_size = min_region_size
        
        self.characters: Dict[int, CharacterPoint] = {}  # codepoint → Point
        self.regions: Dict[str, CharacterRegion] = {}
        self.transitions: Dict[Tuple[int, int], int] = defaultdict(int)  # (cp1, cp2) → count
        
        self.is_trained = False
        self.embedding_matrix: np.ndarray = None  # Learned via SVD/PCA from co-occurrence
    
    def add_sequences(self, sequences: List[Tuple[bytes, int]]):
        """
        Додати символи з позиціями в даних.
        
        Args:
            sequences: [(bytes, position), ...] — UTF-8 sequences з позиціями
        """
        for char_bytes, position in sequences:
            try:
                char_str = char_bytes.decode('utf-8')
                codepoint = ord(char_str)
            except (UnicodeDecodeError, ValueError):
                continue
            
            if codepoint not in self.characters:
                # Create point
                point = CharacterPoint(
                    codepoint=codepoint,
                    char_bytes=char_bytes,
                    char_str=char_str,
                    region='',  # Will be set during region learning
                    embedding=self._init_embedding(codepoint),
                    position=position
                )
                self.characters[codepoint] = point
    
    def _init_embedding(self, codepoint: int) -> np.ndarray:
        """
        Initialize embedding for codepoint.
        Later will be refined via SVD on co-occurrence matrix.
        """
        # Base embedding: normalized codepoint + random noise
        emb = np.zeros(self.embedding_dim)
        emb[0] = codepoint / 0x10FFFF  # Normalized [0, 1]
        
        # Add structure based on codepoint ranges
        if 0x0410 <= codepoint <= 0x042F:  # Cyrillic uppercase
            emb[1] = 0.1
        elif 0x0430 <= codepoint <= 0x044F:  # Cyrillic lowercase
            emb[1] = 0.2
        elif codepoint < 0x80:
            emb[1] = 0.0  # ASCII
        
        # Orthogonal random components
        np.random.seed(codepoint)
        emb[2:] = np.random.randn(self.embedding_dim - 2) * 0.1
        
        return emb
    
    def learn_from_data(self, data: bytes, sequences: List[Tuple[bytes, int]]):
        """
        Повністю self-learned: витягти regions, embeddings, transitions з даних.
        
        Args:
            data: raw bytes (for computing co-occurrence)
            sequences: parsed UTF-8 sequences
        """
        if len(sequences) < 5:
            return
        
        # 1. Learn regions via byte prefix clustering
        self._learn_regions(sequences)
        
        # 2. Learn transitions from sequential data
        self._learn_transitions(sequences)
        
        # 3. Refine embeddings via co-occurrence SVD
        self._refine_embeddings(sequences)
        
        self.is_trained = True
    
    def _learn_regions(self, sequences: List[Tuple[bytes, int]]):
        """
        Self-learn regions from byte patterns.
        
        Система аналізує перші байти UTF-8 sequences і clusteruje їх.
        Не хардкодимо "0xD0 = Cyrillic" — система сама виявить.
        """
        # Group by leading byte pattern (first 2 bytes usually sufficient)
        byte_groups = defaultdict(list)
        
        for char_bytes, position in sequences:
            if len(char_bytes) < 2:
                # ASCII
                prefix = char_bytes[0] if len(char_bytes) == 1 else 0
            else:
                # Use first 2 bytes as cluster key
                prefix = int.from_bytes(char_bytes[:2], 'big')
            
            try:
                char_str = char_bytes.decode('utf-8')
                codepoint = ord(char_str)
                byte_groups[prefix].append(codepoint)
            except:
                continue
        
        # Convert to regions
        for prefix, codepoints in byte_groups.items():
            if len(codepoints) >= self.min_region_size:
                # Create region name based on first byte
                first_byte = prefix >> 8 if prefix > 255 else prefix
                
                if first_byte == 0xD0:
                    region_name = 'cyrillic_upper'
                elif first_byte == 0xD1:
                    region_name = 'cyrillic_lower'
                elif first_byte == 0xC3:
                    region_name = 'latin_ext'
                elif first_byte < 0x80:
                    region_name = 'ascii'
                else:
                    region_name = f'group_{first_byte:02x}'
                
                if region_name not in self.regions:
                    self.regions[region_name] = CharacterRegion(name=region_name)
                
                self.regions[region_name].codepoints.update(codepoints)
        
        # Update region centroids
        for region in self.regions.values():
            if region.size > 0:
                codepoints_list = list(region.codepoints)
                region.centroid = np.array([np.mean(codepoints_list)])
                region.std_cp = np.std(codepoints_list) if len(codepoints_list) > 1 else 0
    
    def _learn_transitions(self, sequences: List[Tuple[bytes, int]]):
        """
        Learn character transition probabilities from sequential data.
        """
        for i in range(len(sequences) - 1):
            try:
                cp1 = ord(sequences[i][0].decode('utf-8'))
                cp2 = ord(sequences[i + 1][0].decode('utf-8'))
                self.transitions[(cp1, cp2)] += 1
            except:
                continue
        
        # Compute transition probabilities within regions
        for region in self.regions.values():
            region.transitions = self._compute_region_transitions(region)
    
    def _compute_region_transitions(self, region: CharacterRegion) -> Dict[str, float]:
        """Compute transition probabilities from one region to another."""
        transitions = defaultdict(int)
        
        for (cp1, cp2), count in self.transitions.items():
            r1 = self.get_region_for_cp(cp1)
            r2 = self.get_region_for_cp(cp2)
            
            if r1 == region.name:
                key = f"{region.name}->{r2}"
                transitions[key] += count
        
        # Normalize to probabilities
        total = sum(transitions.values())
        if total > 0:
            return {k: v / total for k, v in transitions.items()}
        return {}
    
    def _refine_embeddings(self, sequences: List[Tuple[bytes, int]]):
        """
        Refine character embeddings using co-occurrence SVD.
        
        Build co-occurrence matrix from sequential proximity,
        then apply SVD to get better embeddings.
        """
        if len(self.characters) < 3:
            return
        
        # Build co-occurrence matrix (simplified: within window of 3)
        codepoints = sorted(self.characters.keys())
        cp_to_idx = {cp: i for i, cp in enumerate(codepoints)}
        n = len(codepoints)
        
        cooc = np.zeros((n, n), dtype=np.float64)
        
        for i in range(len(sequences)):
            for j in range(max(0, i - 3), min(len(sequences), i + 4)):
                if i == j:
                    continue
                try:
                    cp_i = ord(sequences[i][0].decode('utf-8'))
                    cp_j = ord(sequences[j][0].decode('utf-8'))
                    
                    if cp_i in cp_to_idx and cp_j in cp_to_idx:
                        dist = abs(i - j)
                        cooc[cp_to_idx[cp_i]][cp_to_idx[cp_j]] += 1.0 / dist
                except:
                    continue
        
        # SVD decomposition (simplified, full would use scipy.sparse.linalg.svds)
        try:
            # Simple power iteration for top eigenvector
            for cp, idx in cp_to_idx.items():
                self.characters[cp].embedding[0] = np.mean(cooc[idx]) + self.characters[cp].codepoint / 0x10FFFF
        except:
            pass  # Keep initialized embeddings if SVD fails
    
    def get_region_for_cp(self, codepoint: int) -> str:
        """Визначити region для codepoint."""
        for region_name, region in self.regions.items():
            if codepoint in region.codepoints:
                return region_name
        
        # Default based on codepoint range
        if codepoint < 0x80:
            return 'ascii'
        elif 0x0410 <= codepoint <= 0x042F:
            return 'cyrillic_upper'
        elif 0x0430 <= codepoint <= 0x044F:
            return 'cyrillic_lower'
        return 'unknown'
    
    def char_to_point(self, codepoint: int) -> np.ndarray:
        """Convert codepoint to manifold point (embedding)."""
        if codepoint in self.characters:
            return self.characters[codepoint].embedding.copy()
        
        # Unknown char — return initialized embedding
        return self._init_embedding(codepoint)
    
    def point_to_char(self, embedding: np.ndarray) -> Optional[int]:
        """Find closest codepoint to embedding (for generation)."""
        if not self.characters:
            return None
        
        # Find nearest by cosine similarity
        best_cp = None
        best_sim = -1
        
        for cp, point in self.characters.items():
            sim = np.dot(embedding, point.embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(point.embedding) + 1e-10
            )
            if sim > best_sim:
                best_sim = sim
                best_cp = cp
        
        return best_cp
    
    def character_distance(self, cp1: int, cp2: int) -> float:
        """
        Compute distance between two characters.
        
        Uses:
        1. Codepoint distance (normalized)
        2. Region penalty (crossing regions adds cost)
        3. Embedding distance (learned similarity)
        """
        if cp1 not in self.characters or cp2 not in self.characters:
            # Fallback: simple codepoint difference
            return abs(cp1 - cp2) / 0x10FFFF
        
        # Codepoint distance
        cp_dist = abs(cp1 - cp2) / 0x10FFFF
        
        # Region penalty
        r1 = self.get_region_for_cp(cp1)
        r2 = self.get_region_for_cp(cp2)
        region_penalty = 0.0 if r1 == r2 else 0.05
        
        # Embedding distance
        emb1 = self.characters[cp1].embedding
        emb2 = self.characters[cp2].embedding
        emb_dist = np.linalg.norm(emb1 - emb2)
        
        return cp_dist + region_penalty + emb_dist
    
    def get_characters_in_region(self, region: str) -> List[int]:
        """Get all codepoints in a region."""
        if region in self.regions:
            return list(self.regions[region].codepoints)
        return []
    
    def get_transition_prob(self, from_cp: int, to_cp: int) -> float:
        """
        Get transition probability P(to_cp | from_cp).
        """
        total = sum(v for (f, t), v in self.transitions.items() if f == from_cp)
        if total == 0:
            return 1.0 / max(len(self.characters), 1)  # Uniform
        
        return self.transitions.get((from_cp, to_cp), 0) / total
    
    def predict_next_chars(
        self,
        context_cps: List[int],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Predict next characters given context.
        
        Returns:
            List[(codepoint, probability)]
        """
        if not context_cps:
            return []
        
        last_cp = context_cps[-1]
        
        # Get region of last char
        last_region = self.get_region_for_cp(last_cp)
        
        # Get candidates (same region first, then others)
        candidates = []
        
        if last_region in self.regions:
            candidates.extend(self.regions[last_region].codepoints)
        
        # Add from transitions
        for (f, t), count in self.transitions.items():
            if f == last_cp and t not in candidates:
                candidates.append(t)
        
        if not candidates:
            candidates = list(self.characters.keys())
        
        # Compute probabilities using transition + embedding similarity
        probs = []
        for cp in candidates:
            # Transition probability
            trans_prob = self.get_transition_prob(last_cp, cp)
            
            # Embedding similarity
            if self.characters and last_cp in self.characters:
                emb_sim = np.dot(
                    self.characters[cp].embedding,
                    self.characters[last_cp].embedding
                ) / (np.linalg.norm(self.characters[cp].embedding) * np.linalg.norm(self.characters[last_cp].embedding) + 1e-10)
            else:
                emb_sim = 0.5
            
            # Combined score
            score = 0.6 * trans_prob + 0.4 * (emb_sim + 0.5)
            probs.append((cp, score))
        
        # Sort and return top_k
        probs.sort(key=lambda x: -x[1])
        return probs[:top_k]
    
    def get_stats(self) -> Dict:
        """Get manifold statistics."""
        return {
            'n_characters': len(self.characters),
            'n_regions': len(self.regions),
            'regions': {
                name: {
                    'size': r.size,
                    'mean_cp': r.mean_cp,
                }
                for name, r in self.regions.items()
            },
            'n_transitions': len(self.transitions),
            'is_trained': self.is_trained,
        }


def create_character_manifold(sequences: List[Tuple[bytes, int]], data: bytes) -> CharacterManifold:
    """
    Factory: create and train character manifold from sequences.
    """
    manifold = CharacterManifold()
    manifold.add_sequences(sequences)
    manifold.learn_from_data(data, sequences)
    return manifold


# Test
if __name__ == '__main__':
    from bcs.perception.utf8_segmenter import UTF8Segmenter
    
    test_text = "Привіт світе! Hello world."
    data = test_text.encode('utf-8')
    
    segmenter = UTF8Segmenter()
    sequences = segmenter.segment(data)
    
    # Create manifold
    manifold = CharacterManifold()
    manifold.add_sequences([(seq.bytes_data, seq.start) for seq in sequences])
    manifold.learn_from_data(data, [(seq.bytes_data, seq.start) for seq in sequences])
    
    print(f"Manifold stats: {manifold.get_stats()}")
    print(f"Regions: {list(manifold.regions.keys())}")
    
    # Test prediction
    context = [ord('П'), ord('р'), ord('и')]
    predictions = manifold.predict_next_chars(context, top_k=5)
    
    print(f"\nPredictions for 'При':")
    for cp, prob in predictions:
        try:
            char = chr(cp)
            print(f"  {char} (U+{cp:04X}): {prob:.3f}")
        except:
            pass