"""
UTF-8 Sequence Segmentation Module

Розбір сирих байтів на валідні UTF-8 послідовності.
Це НЕ tokenizer — просто правила інтерпретації encoding.

Принцип: "Give it raw bytes — it learns everything else"
Система не знає що таке символ — вона просто знає правила кодування.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UTF8Sequence:
    """Валідна UTF-8 послідовність."""
    bytes_data: bytes
    start: int
    end: int
    codepoint: int
    char_str: str
    n_bytes: int
    
    @property
    def is_ascii(self) -> bool:
        return self.n_bytes == 1
    
    @property
    def is_multibyte(self) -> bool:
        return self.n_bytes > 1
    
    def __repr__(self):
        return f"UTF8Seq('{self.char_str}' @{self.start}-{self.end}, {self.n_bytes}bytes)"


class UTF8Segmenter:
    """
    Розбір байтів на UTF-8 послідовності.
    
    Використовує правила UTF-8 encoding — це HARDCODED бо encoding spec
    не є знанням про мову, а базовим правилом інтерпретації даних.
    
    Потім система САМА вирішує які послідовності = символи через
    emergent token discovery.
    """
    
    # UTF-8 encoding rules
    # Leading byte → expected sequence length
    LEADING_BYTES = {
        (0x00, 0x7F): 1,   # ASCII (0xxxxxxx)
        (0xC0, 0xDF): 2,   # 2-byte (110xxxxx 10xxxxxx)
        (0xE0, 0xEF): 3,   # 3-byte (1110xxxx 10xxxxxx 10xxxxxx)
        (0xF0, 0xF4): 4,   # 4-byte (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
    }
    
    # Valid continuation byte range
    CONT_MIN = 0x80
    CONT_MAX = 0xBF
    
    def __init__(self, skip_invalid: bool = True):
        """
        Args:
            skip_invalid: пропускати невалідні послідовності замість помилки
        """
        self.skip_invalid = skip_invalid
        self.stats = {
            'total_sequences': 0,
            'ascii_sequences': 0,
            'multibyte_sequences': 0,
            'invalid_skipped': 0,
        }
    
    def segment(self, data: bytes) -> List[UTF8Sequence]:
        """
        Розбір байтів на UTF-8 послідовності.
        
        Args:
            data: сирі байти
            
        Returns:
            List[UTF8Sequence]: валідні UTF-8 послідовності
        """
        sequences = []
        i = 0
        n = len(data)
        
        while i < n:
            seq = self._parse_sequence(data, i)
            
            if seq is not None:
                sequences.append(seq)
                i = seq.end
            else:
                if self.skip_invalid:
                    self.stats['invalid_skipped'] += 1
                    i += 1  # Skip invalid byte
                else:
                    # Return empty to indicate error
                    return []
        
        # Update stats
        self.stats['total_sequences'] = len(sequences)
        self.stats['ascii_sequences'] = sum(1 for s in sequences if s.is_ascii)
        self.stats['multibyte_sequences'] = sum(1 for s in sequences if s.is_multibyte)
        
        return sequences
    
    def _parse_sequence(self, data: bytes, start: int) -> Optional[UTF8Sequence]:
        """Спроба розпарсити UTF-8 послідовність з позиції start."""
        if start >= len(data):
            return None
        
        first_byte = data[start]
        
        # Determine expected length from leading byte
        expected_len = self._get_expected_length(first_byte)
        
        if expected_len == 0:
            # Invalid leading byte (e.g., 0x80-0xBF are continuation bytes)
            return None
        
        # Check if we have enough bytes
        if start + expected_len > len(data):
            # Incomplete sequence (at end of data)
            # Try to parse what's available, but mark as incomplete
            available = len(data) - start
            if available < expected_len:
                return None
        
        # Validate continuation bytes
        end = start + expected_len
        for j in range(1, expected_len):
            if data[start + j] < self.CONT_MIN or data[start + j] > self.CONT_MAX:
                # Invalid continuation byte
                return None
        
        # Extract bytes
        seq_bytes = data[start:end]
        
        # Decode to get codepoint
        try:
            char_str = seq_bytes.decode('utf-8')
            codepoint = ord(char_str)
        except UnicodeDecodeError:
            return None
        
        return UTF8Sequence(
            bytes_data=seq_bytes,
            start=start,
            end=end,
            codepoint=codepoint,
            char_str=char_str,
            n_bytes=expected_len
        )
    
    def _get_expected_length(self, byte: int) -> int:
        """Визначити очікувану довжину послідовності з lead byte."""
        for (min_b, max_b), length in self.LEADING_BYTES.items():
            if min_b <= byte <= max_b:
                return length
        return 0  # Invalid
    
    def segment_to_characters(self, data: bytes) -> List[bytes]:
        """
        Розбір на список байтових послідовностей (без об'єктів).
        
        Returns:
            List[bytes]: кожен елемент = один UTF-8 символ як bytes
        """
        sequences = self.segment(data)
        return [seq.bytes_data for seq in sequences]
    
    def get_byte_distribution(self, data: bytes) -> np.ndarray:
        """
        Обчислити розподіл байтів з урахуванням UTF-8 sequences.
        
        Useful для аналізу модальності.
        """
        dist = np.zeros(256, dtype=np.float64)
        
        for byte in data:
            dist[byte] += 1
        
        return dist / dist.sum()
    
    def get_codepoint_distribution(self, data: bytes) -> Dict[int, float]:
        """
        Обчислити розподіл code points (замість байтів).
        
        Returns:
            Dict[int, float]: codepoint → probability
        """
        sequences = self.segment(data)
        
        counts = {}
        total = len(sequences)
        
        for seq in sequences:
            cp = seq.codepoint
            counts[cp] = counts.get(cp, 0) + 1
        
        return {cp: count / total for cp, count in counts.items()}


class AdaptiveUTF8Segmenter(UTF8Segmenter):
    """
    UTF-8 segmenter з адаптивним навчанням region discovery.
    
    Автоматично виявляє script regions (Cyrillic, Latin, etc.)
    через кластеризацію байтових паттернів.
    """
    
    def __init__(self, skip_invalid: bool = True, min_cluster_size: int = 2):
        super().__init__(skip_invalid)
        self.min_cluster_size = min_cluster_size
        self.regions = {}  # Self-learned regions
        self.region_stats = {}
    
    def learn_regions(self, data: bytes):
        """
        Виявити Unicode regions з даних.
        
        Система САМА визначає які codepoints cluster together
        на основі byte prefix patterns.
        """
        sequences = self.segment(data)
        
        if len(sequences) < 10:
            return  # Not enough data for learning
        
        # Analyze byte prefix patterns
        clusters = {}
        
        for seq in sequences:
            if seq.is_ascii:
                region = 'ascii'
            else:
                # Cluster by first 2 bytes (almost always sufficient for script detection)
                prefix = seq.bytes_data[:2] if len(seq.bytes_data) >= 2 else seq.bytes_data[:1]
                prefix_int = int.from_bytes(prefix, 'big') if len(prefix) > 1 else prefix[0]
                
                # Approximate region by codepoint range
                cp = seq.codepoint
                if 0x0410 <= cp <= 0x042F:
                    region = 'cyrillic_upper'
                elif 0x0430 <= cp <= 0x044F:
                    region = 'cyrillic_lower'
                elif 0x0400 <= cp <= 0x04FF:
                    region = 'cyrillic_ext'
                elif 0x0041 <= cp <= 0x007A:
                    region = 'latin'
                elif 0x0030 <= cp <= 0x0039:
                    region = 'digit'
                elif 0x0020 <= cp <= 0x007E:
                    region = 'punct'
                else:
                    region = 'other'
            
            if region not in clusters:
                clusters[region] = []
            clusters[region].append(seq.codepoint)
        
        self.regions = clusters
        
        # Compute region stats
        for region, codepoints in clusters.items():
            if len(codepoints) >= self.min_cluster_size:
                self.region_stats[region] = {
                    'count': len(codepoints),
                    'mean_cp': np.mean(codepoints),
                    'std_cp': np.std(codepoints) if len(codepoints) > 1 else 0,
                    'min_cp': min(codepoints),
                    'max_cp': max(codepoints),
                    'first_bytes': [seq.bytes_data[0] for seq in sequences 
                                   if self._get_region_for_cp(seq.codepoint) == region][:100],
                }
    
    def _get_region_for_cp(self, cp: int) -> str:
        """Визначити region по codepoint."""
        if 0x0410 <= cp <= 0x042F:
            return 'cyrillic_upper'
        elif 0x0430 <= cp <= 0x044F:
            return 'cyrillic_lower'
        elif 0x0400 <= cp <= 0x04FF:
            return 'cyrillic_ext'
        elif 0x0041 <= cp <= 0x007A:
            return 'latin'
        elif 0x0030 <= cp <= 0x0039:
            return 'digit'
        elif 0x0020 <= cp <= 0x007E:
            return 'punct'
        return 'other'
    
    def get_region_for_sequence(self, seq: UTF8Sequence) -> str:
        """Отримати region для послідовності."""
        return self._get_region_for_cp(seq.codepoint)
    
    def segment_with_regions(self, data: bytes) -> List[Tuple[UTF8Sequence, str]]:
        """
        Розбір з region labeling.
        
        Returns:
            List[(sequence, region)]: sequences з region tags
        """
        sequences = self.segment(data)
        
        if not self.regions:
            self.learn_regions(data)
        
        return [(seq, self.get_region_for_sequence(seq)) for seq in sequences]


def create_segmenter(adaptive: bool = False, **kwargs) -> UTF8Segmenter:
    """
    Factory для створення segmenter.
    
    Args:
        adaptive: використовувати AdaptiveUTF8Segmenter з region learning
        **kwargs: аргументи для конструктора
    """
    if adaptive:
        return AdaptiveUTF8Segmenter(**kwargs)
    return UTF8Segmenter(**kwargs)


# Quick test
if __name__ == '__main__':
    test_data = "Привіт! Hello 123".encode('utf-8')
    
    segmenter = UTF8Segmenter()
    sequences = segmenter.segment(test_data)
    
    print(f"Test: '{test_data.decode('utf-8')}'")
    print(f"Sequences: {len(sequences)}")
    for seq in sequences:
        print(f"  {seq}")
    
    print(f"\nStats: {segmenter.stats}")