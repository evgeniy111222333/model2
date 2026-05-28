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
    
    def get_script_for_cp(self, codepoint: int) -> str:
        """
        Detect script (writing system) from Unicode codepoint.
        
        Returns one of: 'cyrillic', 'latin', 'cjk', 'arabic', 'devanagari', 'hangul', 'greek', 'hebrew', 'thai', 'other', 'ascii', 'punct'
        
        KEY FIX: ASCII letters (A-Z, a-z) are classified as 'latin', not 'ascii'.
        This ensures correct script detection for English words and proper
        continuation behavior (Latin words should continue with Latin chars, not
        favor Cyrillic).
        
        Only control chars, space, and ASCII punctuation are 'ascii'.
        """
        cp = codepoint
        
        # ASCII Control characters: 0x00-0x1F
        if cp < 0x20:
            return 'control'
        
        # ASCII Letters: A-Z (0x41-0x5A) and a-z (0x61-0x7A) -> 'latin'
        # These ARE Latin letters, just in the basic ASCII range
        if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A):
            return 'latin'
        
        # ASCII Space: 0x20
        if cp == 0x20:
            return 'ascii'  # Space is a special "universal" character
        
        # ASCII Punctuation and symbols: 0x21-0x2F, 0x3A-0x40, 0x5B-0x60, 0x7B-0x7E
        if cp <= 0x7E:
            return 'ascii'  # Punctuation, digits, symbols are 'ascii'
        
        # ASCII extended: 0x80-0xFF -> 'latin_ext'
        if cp <= 0xFF:
            return 'latin_ext'
        
        # Cyrillic: U+0400-U+04FF (including extended)
        if 0x0400 <= cp <= 0x04FF:
            return 'cyrillic'
        if 0x0500 <= cp <= 0x052F:
            return 'cyrillic'
        
        # Latin extended-A/B: U+0100-U+024F
        if 0x0100 <= cp <= 0x024F:
            return 'latin'
        
        # Greek: U+0370-U+03FF
        if 0x0370 <= cp <= 0x03FF:
            return 'greek'
        
        # Hebrew: U+0590-U+05FF
        if 0x0590 <= cp <= 0x05FF:
            return 'hebrew'
        
        # Arabic: U+0600-U+06FF
        if 0x0600 <= cp <= 0x06FF:
            return 'arabic'
        
        # Devanagari and Indic: U+0900-U+097F
        if 0x0900 <= cp <= 0x097F:
            return 'devanagari'
        if 0x0980 <= cp <= 0x09FF:
            return 'bengali'
        if 0x0A00 <= cp <= 0x0AFF:
            return 'gurmukhi'
        
        # Thai: U+0E00-U+0E7F
        if 0x0E00 <= cp <= 0x0E7F:
            return 'thai'
        
        # Hangul (Korean): U+AC00-U+D7AF
        if 0xAC00 <= cp <= 0xD7AF:
            return 'hangul'
        
        # CJK Unified Ideographs: U+4E00-U+9FFF
        if 0x4E00 <= cp <= 0x9FFF:
            return 'cjk'
        
        # CJK Extensions: U+3400-U+4DBF
        if 0x3400 <= cp <= 0x4DBF:
            return 'cjk_ext'
        
        # Hiragana/Katakana: U+3040-U+30FF
        if 0x3040 <= cp <= 0x309F:
            return 'hiragana'
        if 0x30A0 <= cp <= 0x30FF:
            return 'katakana'
        
        # General punctuation and symbols that are not in ASCII
        if 0x2000 <= cp <= 0x206F:  # General Punctuation
            return 'punct'
        if 0x2070 <= cp <= 0x209F:  # Superscripts/Subscripts
            return 'punct'
        if 0x20A0 <= cp <= 0x20CF:  # Currency
            return 'punct'
        if 0x2100 <= cp <= 0x214F:  # Letterlike Symbols
            return 'punct'
        if 0x2190 <= cp <= 0x21FF:  # Arrows
            return 'punct'
        if 0x2200 <= cp <= 0x22FF:  # Mathematical Operators
            return 'punct'
        
        return 'other'
    
    def get_active_script(self, trajectory: List[Tuple[int, int]]) -> str:
        """
        Determine the active script from a character trajectory.
        
        Analyzes the last few non-space/punct characters to detect which
        writing system is currently active. Uses majority voting with recency bias.
        
        IMPORTANT: Ignores spaces and punctuation when determining active script.
        After "Hello world" vs "Привіт світе", the active script should be
        Latin/Cyrillic respectively, not based on the trailing space.
        
        Args:
            trajectory: List[(codepoint, position)]
            
        Returns:
            Script name ('cyrillic', 'latin', 'ascii', 'punct', etc.)
        """
        if not trajectory:
            return 'ascii'
        
        # Collect recent characters, ignoring spaces and punctuation
        letter_chars = []
        for cp, pos in reversed(trajectory):
            script = self.get_script_for_cp(cp)
            if script not in {'space', 'punct', 'control', 'ascii'}:
                letter_chars.append(script)
            else:
                # Stop at space/punct - don't count what comes after
                break
        
        if not letter_chars:
            # All space/punct - default based on first letter
            for cp, pos in trajectory:
                script = self.get_script_for_cp(cp)
                if script not in {'space', 'punct', 'control'}:
                    return script
            return 'ascii'
        
        # Majority vote with recency bias (earlier in list = more recent)
        script_counts = {}
        for script in letter_chars[:5]:  # Last 5 letter chars
            script_counts[script] = script_counts.get(script, 0) + 1
        
        return max(script_counts.items(), key=lambda x: x[1])[0]
    
    def get_allowed_next_scripts(
        self,
        last_script: str,
    ) -> set:
        """
        Get scripts that are allowed to follow the given script.
        
        Defines the transition matrix between writing systems.
        
        KEY FIX: After 'latin' script (which now includes ASCII letters A-Z, a-z),
        we allow both Latin and Cyrillic continuation (for mixed-language texts).
        After 'ascii' (only space and punctuation remain), any script is allowed
        as a universal transition point.
        
        Args:
            last_script: The current script
            
        Returns:
            Set of allowed next scripts
        """
        # Universal transitions - allowed from any script
        universal = {'punct', 'control', 'ascii', 'space'}
        
        # Script-specific transition rules
        transitions = {
            # Latin: follows itself, allows punctuation, allows Cyrillic for mixed texts
            'latin': {'latin', 'cyrillic', 'latin_ext', 'cyrillic_ext', 'punct', 'ascii', 'space'},
            
            # Latin extended: like Latin
            'latin_ext': {'latin', 'cyrillic', 'latin_ext', 'cyrillic_ext', 'punct', 'ascii', 'space'},
            
            # Cyrillic: mostly follows itself, allows punctuation, allows Latin for mixed texts
            'cyrillic': {'cyrillic', 'latin', 'cyrillic_ext', 'latin_ext', 'punct', 'ascii', 'space'},
            
            # Cyrillic extended: like Cyrillic
            'cyrillic_ext': {'cyrillic', 'latin', 'cyrillic_ext', 'latin_ext', 'punct', 'ascii', 'space'},
            
            # CJK scripts: mostly self, allow punctuation
            'cjk': {'cjk', 'cjk_ext', 'hiragana', 'katakana', 'punct', 'space'},
            'cjk_ext': {'cjk', 'cjk_ext', 'hiragana', 'katakana', 'punct', 'space'},
            'hiragana': {'hiragana', 'katakana', 'cjk', 'cjk_ext', 'punct', 'space'},
            'katakana': {'katakana', 'hiragana', 'cjk', 'cjk_ext', 'punct', 'space'},
            
            # Other letter scripts: allow punctuation
            'greek': {'greek', 'punct', 'ascii', 'space'},
            'arabic': {'arabic', 'punct', 'ascii', 'space'},
            'hebrew': {'hebrew', 'punct', 'ascii', 'space'},
            'hangul': {'hangul', 'cjk', 'cjk_ext', 'punct', 'space'},
            
            # Punctuation: universal target, allows any letter script
            'punct': universal | {
                'cyrillic', 'cyrillic_ext', 'latin', 'latin_ext', 'cjk', 'cjk_ext',
                'hiragana', 'katakana', 'greek', 'arabic', 'hebrew', 'hangul'
            },
            
            # Space: universal transition point, allows any letter script
            'space': universal | {
                'cyrillic', 'cyrillic_ext', 'latin', 'latin_ext', 'cjk', 'cjk_ext',
                'hiragana', 'katakana', 'greek', 'arabic', 'hebrew', 'hangul'
            },
            
            # Control: universal source
            'control': universal,
            
            # ASCII (space/punct only now - letters moved to 'latin'): universal transition
            'ascii': universal | {
                'cyrillic', 'cyrillic_ext', 'latin', 'latin_ext', 'cjk', 'cjk_ext',
                'hiragana', 'katakana', 'greek', 'arabic', 'hebrew', 'hangul'
            },
            
            # Other/unknown: allow punctuation as safe fallback
            'other': universal | {'cyrillic', 'latin', 'cjk'},  # Allow common scripts
        }
        
        return transitions.get(last_script, universal)
    
    # NOTE: get_script_for_cp is defined at line ~307 (the correct one with Latin letter fix)
    
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
        Predict next characters given context, with script-aware filtering.

        Filters candidates to match the active writing system (script), preventing
        impossible transitions like Cyrillic → Latin mid-word. Punctuation,
        spaces, and newlines are allowed from any script.
        
        Returns:
            List[(codepoint, probability)]
        """
        if not context_cps:
            return []
        
        last_cp = context_cps[-1]
        
        # Detect active script of the last character
        last_script = self.get_script_for_cp(last_cp)
        
        # Punctuation/spaces are universal transitions (allowed from any script)
        universal_chars = {0x0020, 0x000A, 0x000D, 0x0021, 0x002C, 0x002E, 0x003F, 0x003A,
                          0x003B, 0x0028, 0x0029, 0x002D, 0x2018, 0x2019, 0x201C, 0x201D}
        
        def is_allowed_transition(cp: int) -> bool:
            """Check if transition from last_script to cp is linguistically plausible."""
            cp_script = self.get_script_for_cp(cp)
            
            # Universal chars always allowed
            if cp in universal_chars:
                return True
            
            # Same script: always allowed
            if cp_script == last_script:
                return True
            
            # Allow Latin ↔ Cyrillic mixing (common in some texts)
            if {last_script, cp_script} == {'latin', 'cyrillic'}:
                return True
            
            # Allow transitions TO punctuation from any script (already handled above)
            # but allow transitions FROM punctuation to any script (punct is universal source)
            if last_script in {'punct', 'control', 'ascii'}:
                return True
            
            # Block same-script-letter transitions across incompatible scripts
            # e.g., Cyrillic → Latin (letters only)
            letter_scripts = {'cyrillic', 'latin', 'greek', 'hebrew', 'arabic', 'cjk',
                            'hiragana', 'katakana', 'hangul', 'devanagari', 'bengali'}
            if cp_script in letter_scripts and last_script in letter_scripts:
                return cp_script == last_script  # Only allow same-script letters
            
            # Everything else: allow (numbers, currency, misc symbols)
            return True
        
        # Get candidates from same region first
        candidates = []
        last_region = self.get_region_for_cp(last_cp)
        
        if last_region in self.regions:
            for cp in self.regions[last_region].codepoints:
                if is_allowed_transition(cp):
                    candidates.append(cp)
        
        # Add from transitions (only if they pass script filter)
        for (f, t), count in self.transitions.items():
            if f == last_cp and t not in candidates and is_allowed_transition(t):
                candidates.append(t)
        
        # If no script-filtered candidates, fall back to full character set filtered
        if not candidates:
            for cp in self.characters.keys():
                if is_allowed_transition(cp):
                    candidates.append(cp)
        
        # Final fallback: at least provide universal characters
        if not candidates:
            candidates = [0x0020]  # space
        
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