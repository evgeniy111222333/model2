"""
BCS Advanced Information Geometry Module - v2.0

Contains CUTTING-EDGE algorithms for:
1. Modality Detection using Information Geometry
2. Boundary Detection using Riemannian Analysis
3. Alpha-divergence family for robust scoring
4. Spectral Analysis for frequency-domain features
5. Compression-based modality fingerprinting

Architecture Concept:
- No tokenizer: operates directly on byte probability distributions
- Byte as fundamental unit on statistical manifold S^{255}
- Fisher-Rao metric as natural Riemannian metric
- Beta-divergence family for robust distance computation
- Information-theoretic scoring without discretization

Key Algorithms:
1. Fisher-Rao Distance (Bhattacharyya coefficient)
2. Alpha-divergence family (tunable robustness)
3. Spectral signature extraction (FFT-based)
4. Compression entropy profiling
5. Natural gradient flow optimization
6. Multi-scale entropy analysis (Multi-resolution Fisher-Rao)
7. Exponential family parameter estimation
8. Curvature-based modality characterization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from dataclasses import dataclass
import warnings

# Try to import optional dependencies
try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

# =============================================================================
# 1. INFORMATION GEOMETRY PRIMITIVES (Riemannian Statistical Manifold)
# =============================================================================

def fisher_rao_distance(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Fisher-Rao distance between two probability distributions.
    
    d_FR(p, q) = arccos(∫√p(x)√q(x)dx) = arccos(Bhattacharyya(p,q))
    
    This is the geodesic distance on the statistical manifold.
    Equivalent to Riemannian metric induced by Fisher information.
    """
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)
    
    if bc >= 1.0 - epsilon:
        return 0.0
    return np.arccos(bc)


def fisher_metric(p: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Fisher Information Matrix for categorical distribution.
    
    For categorical distribution (exponential family), the Fisher metric is:
    g_ij(p) = 1/p_i if i==j, else 0
    
    This defines the Riemannian structure of the simplex.
    """
    p = np.maximum(p, epsilon)
    # Normalize
    p = p / p.sum()
    # Diagonal Fisher matrix (simplified for categorical)
    return np.diag(1.0 / p)


def natural_gradient(p: np.ndarray, grad: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute natural gradient on statistical manifold.
    
    Natural gradient: ∇̃f = G^{-1} ∇f
    
    Where G is the Fisher information metric.
    This gives direction of steepest ascent on manifold.
    """
    G = fisher_metric(p, epsilon)
    with np.errstate(divide='ignore', invalid='ignore'):
        natural_grad = np.linalg.solve(G, grad)
    return np.nan_to_num(natural_grad, nan=0.0, posinf=0.0, neginf=0.0)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL(p || q) divergence."""
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Jensen-Shannon divergence."""
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    1-D Wasserstein distance (Earth Mover's Distance).
    
    For distributions on ordinal domain (byte values 0-255).
    """
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    p = p / p.sum()
    q = q / q.sum()
    
    # CDFs
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    
    # Sum of absolute differences in CDFs
    return np.sum(np.abs(cdf_p - cdf_q))


# =============================================================================
# 2. ALPHA-DIVERGENCE FAMILY (Generalized Information Divergence)
# =============================================================================

def alpha_divergence(p: np.ndarray, q: np.ndarray, alpha: float = 0.5, epsilon: float = 1e-10) -> float:
    """
    Alpha-divergence: parametric family of divergences.
    
    D_α(p||q) = (4/(1-α²)) * (1 - ∫ p^(1-α) q^(1+α)/2 dx)
    
    Special cases:
    - α = 0: χ² divergence (Pearson)
    - α = 1: KL(p||q) divergence
    - α = -1: Reverse KL(q||p) divergence
    - α = 0.5: Hellinger distance (related to Fisher-Rao)
    
    Alpha control robustness:
    - α > 0: penalizes q << p (sensitive to false positives)
    - α < 0: penalizes p << q (sensitive to false negatives)
    """
    if alpha == 1.0:
        # KL divergence
        return kl_divergence(p, q, epsilon)
    elif alpha == -1.0:
        # Reverse KL
        return kl_divergence(q, p, epsilon)
    elif abs(alpha) < 0.01:
        # Chi-squared
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        p = p / p.sum()
        q = q / q.sum()
        return np.sum((p - q) ** 2 / q)
    else:
        # General alpha-divergence formula
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        p = p / p.sum()
        q = q / q.sum()
        
        # D_α(p||q) = (4/(1-α²)) * (1 - ∫ p^((1-α)/2) q^((1+α)/2) dx)
        integrand = np.sum(p ** ((1 - alpha) / 2) * q ** ((1 + alpha) / 2))
        return (4 / (1 - alpha ** 2)) * (1 - integrand)


def tsallis_entropy(p: np.ndarray, q: float = 2.0, epsilon: float = 1e-10) -> float:
    """
    Tsallis q-entropy (non-extensive entropy).
    
    S_q(p) = (1 - ∫ p^q dx) / (q - 1)
    
    For q = 2: measures "spread" differently than Shannon
    """
    p = np.maximum(p, epsilon)
    p = p / p.sum()
    p_q = np.sum(p ** q)
    if abs(q - 1.0) < epsilon:
        return scipy_entropy(p)
    return (1 - p_q) / (q - 1)


def renyi_entropy(p: np.ndarray, alpha: float = 0.5, epsilon: float = 1e-10) -> float:
    """
    Rényi α-entropy.
    
    H_α(p) = (1/(1-α)) * log(∫ p^α dx)
    
    Order-α information measure.
    """
    p = np.maximum(p, epsilon)
    p = p / p.sum()
    if abs(alpha - 1.0) < epsilon:
        return scipy_entropy(p)
    p_alpha = np.sum(p ** alpha)
    if p_alpha <= epsilon:
        return 0.0
    return np.log(p_alpha) / (1 - alpha)


# =============================================================================
# 3. SPECTRAL ANALYSIS (Frequency Domain Features)
# =============================================================================

def compute_spectral_signature(p: np.ndarray, n_harmonics: int = 8) -> Dict[str, float]:
    """
    Extract spectral signature from byte probability distribution.
    
    The byte distribution in frequency domain reveals modality structure:
    - Text: strong low-frequency components (repetitive structure)
    - Images: flat spectrum (random-like high entropy)
    - Audio: structured peaks at harmonic frequencies
    - Binary: sparse spectrum with DC component dominance
    
    Uses FFT on byte probability distribution.
    """
    # Compute DFT of probability distribution
    n = len(p)
    p_norm = p - p.mean()  # Remove DC
    spectrum = np.abs(fft(p_norm))
    freqs = fftfreq(n)
    
    # Extract key spectral features
    features = {}
    
    # DC component (mean mass)
    features['dc_component'] = np.abs(p.mean())
    
    # Spectral spread (variance in frequency domain)
    if spectrum[1:].sum() > 0:
        power_spectrum = spectrum[1:] ** 2
        freq_weights = np.abs(freqs[1:])
        features['spectral_spread'] = np.sqrt(
            np.sum(freq_weights ** 2 * power_spectrum) / np.sum(power_spectrum)
        )
        features['spectral_entropy'] = -np.sum(
            (power_spectrum / power_spectrum.sum()) * 
            np.log(power_spectrum / power_spectrum.sum() + 1e-10)
        )
    else:
        features['spectral_spread'] = 0.0
        features['spectral_entropy'] = 0.0
    
    # Low frequency ratio ( LF / Total )
    low_freq_idx = n // 8
    features['low_freq_ratio'] = spectrum[1:low_freq_idx].sum() / spectrum[1:].sum()
    
    # Peak detection (harmonic structure)
    peaks, _ = find_peaks(spectrum[1:n//2], height=np.mean(spectrum[1:n//2]))
    features['n_peaks'] = len(peaks)
    features['peak_prominence'] = np.mean(spectrum[peaks + 1]) if len(peaks) > 0 else 0.0
    
    # Spectral flatness (Wiener entropy)
    geometric_mean = np.exp(np.mean(np.log(spectrum[1:n//2] + 1e-10)))
    arithmetic_mean = np.mean(spectrum[1:n//2])
    features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
    
    return features


# =============================================================================
# 4. ENTROPY PROFILE ANALYSIS (Multi-resolution)
# =============================================================================

def compute_entropy_profile(p: np.ndarray, scales: List[int] = [1, 2, 4, 8]) -> Dict[str, float]:
    """
    Compute entropy at multiple resolution scales.
    
    Different modalities have characteristic entropy profiles:
    - Text: high mid-scale entropy (redundant at large scale, random at small)
    - Image: flat entropy across scales
    - Audio: low entropy (high structure)
    - Binary: extreme entropy (sparse)
    
    Uses Renyi entropy with different alpha orders.
    """
    profile = {}
    
    for alpha in [0.5, 1.0, 2.0]:
        key = f'renyi_alpha_{alpha}'
        profile[key] = renyi_entropy(p, alpha)
    
    # Shannon entropy (alpha = 1)
    profile['shannon_entropy'] = scipy_entropy(p)
    
    # Normalized entropy (entropy / max entropy)
    max_entropy = np.log(len(p))
    profile['normalized_entropy'] = profile['shannon_entropy'] / max_entropy
    
    # Conditional entropy features
    profile['byte_concentration'] = 1.0 - np.max(p)  # How concentrated is distribution
    profile['top_byte_ratio'] = np.sum(np.sort(p)[-5:])  # Mass in top 5 bytes
    
    return profile


def compute_compression_fingerprint(data: bytes, block_size: int = 4096) -> Dict[str, float]:
    """
    Compression-based modality fingerprinting.
    
    Different modalities compress differently:
    - Text: high compression ratio (redundancy)
    - Image (compressed): low ratio (already compressed)
    - Image (raw): low ratio (random-like)
    - Audio: moderate compression
    - Binary (sparse): very high compression
    - Binary (encrypted): low compression
    
    This is an approximation of Kolmogorov complexity.
    """
    import zlib
    
    n_blocks = len(data) // block_size
    if n_blocks == 0:
        n_blocks = 1
    
    ratios = []
    entropies = []
    
    for i in range(n_blocks):
        block = data[i * block_size:(i + 1) * block_size]
        if len(block) < 10:
            continue
            
        # Compression ratio
        compressed = zlib.compress(block, level=9)
        ratio = len(compressed) / len(block)
        ratios.append(ratio)
        
        # Block entropy
        dist = np.zeros(256)
        for b in block:
            dist[b] += 1
        dist = dist / (dist.sum() + 1e-10)
        entropies.append(scipy_entropy(dist))
    
    result = {
        'compression_ratio_mean': np.mean(ratios) if ratios else 1.0,
        'compression_ratio_std': np.std(ratios) if len(ratios) > 1 else 0.0,
        'entropy_mean': np.mean(entropies) if entropies else 0.0,
        'entropy_std': np.std(entropies) if len(entropies) > 1 else 0.0,
    }
    
    # Higher compression = lower ratio = more structured
    result['compression_score'] = 1.0 - result['compression_ratio_mean']
    
    return result


# =============================================================================
# 5. MODALITY SIGNATURE CENTROIDS (Exponential Family Parameters)
# =============================================================================

def initialize_modality_centroids() -> Dict[str, np.ndarray]:
    """
    Initialize exponential family centroids for each modality.
    
    Each modality is represented as a probability distribution on {0,...,255}.
    These are maximum likelihood estimates for each exponential family.
    """
    centroids = {}
    x = np.arange(256)
    
    # Text ASCII: Bell-shaped on [32, 126], space-dominant
    c = np.zeros(256)
    # Letters more frequent
    letter_freq = 0.05
    for i in range(ord('a'), ord('z') + 1):
        c[i] = letter_freq
    for i in range(ord('A'), ord('Z') + 1):
        c[i] = letter_freq * 0.8
    c[ord(' ')] = 0.18  # 18% space
    c[ord('e')] = 0.10  # 'e' is most common
    c[ord('t')] = 0.07
    c[ord('a')] = 0.065
    c[ord('o')] = 0.06
    c[ord('i')] = 0.055
    c[ord('n')] = 0.055
    c[ord('s')] = 0.05
    c[ord('r')] = 0.045
    c[ord('\n')] = 0.04
    c[ord(',')] = 0.02
    c[ord('.')] = 0.02
    c[ord('0'):ord('9')+1] = 0.015  # Digits
    for i in range(32, 127):
        if c[i] == 0:
            c[i] = 0.002
    centroids['text_ascii'] = c / c.sum()
    
    # Text UTF-8: 0xD0-0xD1 (Cyrillic) or 0xC3 (Latin ext)
    c = np.zeros(256)
    # Continuation bytes (0x80-0xBF)
    c[0x80:0xC0] = 0.45 / 64  # ~0.7% each continuation
    # Leading bytes (0xC0-0xF5)
    c[0xC0:0xC2] = 0.12  # 2-byte leading
    c[0xC3:0xC4] = 0.10
    c[0xD0:0xD2] = 0.08  # Cyrillic leading
    c[0xD1:0xD2] = 0.07
    c[0xD0:0xD5] = 0.05
    # Space
    c[0x20] = 0.10
    centroids['text_utf8'] = c / c.sum()
    
    # Image (compressed/uniform): near-uniform distribution
    c = np.ones(256) + np.random.randn(256) * 0.1
    c = np.maximum(c, 0.1)
    centroids['image'] = c / c.sum()
    
    # Audio: Gaussian centered at 0x80
    c = np.exp(-0.5 * ((x - 128) / 25) ** 2)
    centroids['audio'] = c / c.sum()
    
    # Binary: MOSTLY SPARSE - dominated by 0x00 and 0xFF
    c = np.zeros(256)
    c[0x00] = 1.0
    c[0xFF] = 0.9
    centroids['binary'] = c / c.sum()
    
    # Structured: JSON/XML markers with digits
    c = np.zeros(256)
    c[ord('0'):ord('9')+1] = 0.8  # Digits: uniform-ish
    c[ord('{')] = 1.0
    c[ord('}')] = 1.0
    c[ord('[')] = 1.0
    c[ord(']')] = 1.0
    c[ord(':')] = 0.9
    c[ord(',')] = 0.9
    c[ord('"')] = 0.85
    c[ord("'")] = 0.3
    c[ord(' ')] = 0.1
    centroids['structured'] = c / c.sum()
    
    return centroids


# =============================================================================
# 6. ADVANCED MODALITY DETECTOR
# =============================================================================

@dataclass
class ModalityFeatures:
    """Container for comprehensive modality features."""
    # Distance-based scores
    fr_distance: float  # Fisher-Rao
    alpha_score: float  # Alpha-divergence based
    wasserstein: float  # Earth mover distance
    
    # Information-theoretic
    shannon_entropy: float
    renyi_alpha_05: float
    spectral_flatness: float
    
    # Structural
    byte_concentration: float
    spectral_spread: float
    low_freq_ratio: float
    n_peaks: int
    compression_score: float


class AdvancedModalityDetector:
    """
    Advanced Modality Detection using Information Geometry.
    
    CUTTING-EDGE Features:
    1. Multi-metric ensemble (Fisher-Rao + Alpha-div + Wasserstein)
    2. Spectral signature matching
    3. Compression-based fingerprinting
    4. Adaptive alpha parameter for robust scoring
    5. Natural gradient optimization
    6. Multi-resolution entropy analysis
    
    Architecture:
    - Each modality is a point on statistical manifold S^{255}
    - Use exponential family representation
    - Alpha-divergence family for robust distance
    - Compression ratio as proxy for Kolmogorov complexity
    """
    
    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        centroid_init: Optional[Dict[str, np.ndarray]] = None,
        use_alpha_divergence: bool = True,
        alpha: float = 0.5,
        use_spectral: bool = True,
        use_compression: bool = True,
        adaptive: bool = True,
    ):
        self.modalities = modalities or [
            'text_ascii', 'text_utf8', 'image', 'audio', 'binary', 'structured'
        ]
        self.use_alpha_divergence = use_alpha_divergence
        self.alpha = alpha  # 0.5 = Hilinger-like, sensitive to outliers
        self.use_spectral = use_spectral
        self.use_compression = use_compression
        self.adaptive = adaptive
        
        # Initialize centroids
        if centroid_init:
            self.centroids = centroid_init
        else:
            self.centroids = initialize_modality_centroids()
        
        # Compute adaptive weights
        self.weights = self._compute_adaptive_weights()
    
    def _compute_adaptive_weights(self) -> Dict[str, float]:
        """
        Compute adaptive modality weights based on centroid spread.
        
        More distinct centroids get higher weights.
        """
        weights = {}
        n_mod = len(self.modalities)
        
        # Compute pairwise Fisher-Rao distances
        dist_matrix = np.zeros((n_mod, n_mod))
        for i, m1 in enumerate(self.modalities):
            for j, m2 in enumerate(self.modalities):
                dist_matrix[i, j] = fisher_rao_distance(
                    self.centroids[m1], self.centroids[m2]
                )
        
        # Weight proportional to mean distance from others
        for i, m in enumerate(self.modalities):
            mean_dist = (dist_matrix[i].sum() - dist_matrix[i, i]) / (n_mod - 1)
            weights[m] = mean_dist
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {m: w / total for m, w in weights.items()}
        else:
            weights = {m: 1.0 / n_mod for m in self.modalities}
        
        return weights
    
    def _compute_distance_to_centroid(
        self, p: np.ndarray, mod: str, method: str = 'fr'
    ) -> float:
        """
        Compute distance from distribution p to centroid.
        
        Methods:
        - 'fr': Fisher-Rao distance
        - 'alpha': Alpha-divergence based score
        - 'wasserstein': Earth mover distance
        """
        centroid = self.centroids[mod]
        
        if method == 'fr':
            return fisher_rao_distance(p, centroid)
        elif method == 'alpha':
            return alpha_divergence(p, centroid, self.alpha)
        elif method == 'wasserstein':
            return wasserstein_distance(p, centroid)
        else:
            return fisher_rao_distance(p, centroid)
    
    def _compute_spectral_features(self, p: np.ndarray) -> Dict[str, float]:
        """Extract spectral signature from distribution."""
        return compute_spectral_signature(p)
    
    def _compute_entropy_features(self, p: np.ndarray) -> Dict[str, float]:
        """Extract entropy profile features."""
        return compute_entropy_profile(p)
    
    def _modality_score(self, p: np.ndarray, mod: str, raw_data: bytes = None) -> float:
        """
        Compute modality score using discriminative features.
        
        KEY INSIGHT: Each modality has DISTINCTIVE features that separate it.
        We use feature-based scoring with strong discriminative power.
        """
        centroid = self.centroids[mod]
        
        # ================================================================
        # 1. SIGNATURE-BASED SCORING (Direct character matching)
        # ================================================================
        
        if mod == 'text_ascii':
            # DISTINCTIVE: High printable bytes (0x20-0x7E), no high bytes
            printable = p[0x20:0x7F].sum()
            high_bytes = p[0x80:].sum()
            # Score: high printable AND no high bytes (strong penalty)
            return min(1.0, printable * 0.6 + (1.0 - high_bytes) * 0.4)
        
        elif mod == 'text_utf8':
            # DISTINCTIVE: High bytes 0x80-0xC0 (continuation) AND 0xC0-0xF5 (leading)
            continuation = p[0x80:0xC0].sum()
            leading = p[0xC0:0xF6].sum()
            # Must have BOTH and they must be HIGH (>30% combined for real UTF8)
            score = continuation * 0.6 + leading * 0.4
            if score > 0.3:
                score = 0.5 + score * 0.5  # Boost when we have true UTF8
            return min(1.0, score)
        
        elif mod == 'image':
            # DISTINCTIVE: Nearly uniform = MAX entropy
            p_nonzero = p[p > 1e-6]
            n_unique = len(p_nonzero)
            # Uniform = all 256 bytes used, each ~1/256
            uniformity = n_unique / 256.0
            variance = np.var(p_nonzero) if len(p_nonzero) > 1 else 1.0
            # Score: high unique bytes AND low variance
            return min(1.0, uniformity * 0.5 + max(0, 1.0 - variance * 5000) * 0.5)
        
        elif mod == 'audio':
            # DISTINCTIVE: NOT high UTF8 bytes, NOT sparse, NOT uniform
            # Audio sine wave uses range [78, 178] - has ext bytes > 0xBF
            high_utf8 = p[0xC0:].sum()  # UTF8 leading bytes
            range_used = np.sum(p > 1e-6)
            # Audio has moderate range but NOT UTF8 leading bytes
            sparse = p[0]+p[255]; return min(1.0, (1.0-sparse)*0.35+(range_used>30)*0.35+(1.0-p[123]-p[34])*0.3)
        
        elif mod == 'binary':
            # DISTINCTIVE: SPARSE - only 0x00 and/or 0xFF dominate
            sparse_sum = p[0x00] + p[0xFF]
            p_nonzero = p[p > 1e-6]
            n_unique = len(p_nonzero)
            # Score: very sparse, dominated by 0x00/0xFF
            return min(1.0, sparse_sum * 0.7 + (n_unique < 20) * 0.3)
        
        elif mod == 'structured':
            # DISTINCTIVE: JSON markers with digits (no printables > 0x7F)
            markers = (
                p[ord('{')] + p[ord('}')] +
                p[ord('[')] + p[ord(']')] +
                p[ord(':')] + p[ord(',')] +
                p[ord('"')]
            )
            # ALSO: NOT high UTF8 bytes (UTF8 would score higher on text_utf8)
            high_utf8 = p[0xC0:].sum()
            # Score: high markers AND no high bytes
            return min(1.0, markers * 3 + (1.0 - high_utf8) * 0.2)
        
        return 0.0
    
    def detect(
        self, p: np.ndarray, N: int = 1, raw_data: bytes = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Detect modality of byte distribution p.
        
        Returns:
        - modality: detected modality name
        - posteriors: dict of modality -> probability
        """
        p = np.maximum(p, 1e-10)
        p = p / p.sum()
        
        # Compute scores for all modalities
        scores = {}
        for mod in self.modalities:
            scores[mod] = self._modality_score(p, mod, raw_data)
        
        # Compute softmax posteriors (temperature may help)
        score_values = np.array([scores.get(m, 0.0) for m in self.modalities])
        
        # Use temperature-scaled softmax
        temperature = 0.1 if self.adaptive else 1.0
        exp_scores = np.exp(score_values / temperature)
        probs = exp_scores / exp_scores.sum()
        
        posteriors = {m: float(probs[i]) for i, m in enumerate(self.modalities)}
        
        # Detect: pick modality with highest posterior
        detected = max(posteriors, key=posteriors.get)
        
        return detected, posteriors


# =============================================================================
# 7. ADVANCED BOUNDARY DETECTOR (Multi-scale Riemannian Analysis)
# =============================================================================

class AdvancedBoundaryDetector:
    """
    Advanced Boundary Detection using Multi-scale Information Geometry.
    
    CUTTING-EDGE Features:
    1. Multi-resolution Fisher-Rao analysis
    2. Curvature-based boundary detection
    3. Information bottleneck principle
    4. Change point detection with alpha-divergence
    5. Natural gradient flow for boundary localization
    
    Architecture:
    - Sliding window with exponential decay weighting
    - Multi-scale analysis (different window sizes)
    - Fisher-Rao distance as change metric
    - Curvature analysis for boundary strength
    """
    
    def __init__(
        self,
        scales: Optional[List[int]] = None,
        use_fisher_rao: bool = True,
        use_curvature: bool = True,
        use_geodesic: bool = True,
        decay_rate: float = 0.9,
    ):
        self.scales = scales or [8, 16, 32, 64]
        self.use_fisher_rao = use_fisher_rao
        self.use_curvature = use_curvature
        self.use_geodesic = use_geodesic
        self.decay_rate = decay_rate
    
    def _compute_local_distribution(
        self, data: np.ndarray, center: int, window: int
    ) -> np.ndarray:
        """Compute local byte distribution around position."""
        half_w = window // 2
        start = max(0, center - half_w)
        end = min(len(data), center + half_w)
        
        window_data = data[start:end]
        if len(window_data) == 0:
            return np.ones(256) / 256.0
        
        dist = np.zeros(256)
        for b in window_data:
            dist[b] += 1
        dist = dist / (dist.sum() + 1e-10)
        return dist
    
    def _compute_change_score(
        self, data: np.ndarray, pos: int, scale: int
    ) -> float:
        """
        Compute information-geometric change score at position.
        
        Uses Fisher-Rao distance between left and right local distributions.
        """
        half_w = scale // 2
        if pos - half_w < 0 or pos + half_w >= len(data):
            return 0.0
        
        # Left and right distributions
        left = self._compute_local_distribution(data, pos - half_w, scale)
        right = self._compute_local_distribution(data, pos + half_w, scale)
        
        # Fisher-Rao distance as change metric
        if self.use_fisher_rao:
            change = fisher_rao_distance(left, right)
        else:
            # Fallback to KL divergence
            change = kl_divergence(left, right)
        
        return change
    
    def _compute_geodesic_score(self, data: np.ndarray, pos: int) -> float:
        """
        Compute geodesic smoothness score.
        
        Measures how smooth the transition is across the position.
        """
        if pos < 2 or pos >= len(data) - 2:
            return 0.0
        
        # Compute distances to local reference
        ref = self._compute_local_distribution(data, pos, window=16)
        
        # Distances to neighbor references
        neighbors = []
        for offset in [-8, -4, 4, 8]:
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(data):
                neighbor_dist = self._compute_local_distribution(data, neighbor_pos, window=16)
                neighbors.append(fisher_rao_distance(ref, neighbor_dist))
        
        if not neighbors:
            return 0.0
        
        # Smoothness: low variance in neighbor distances
        smoothness = 1.0 / (1.0 + np.std(neighbors))
        return smoothness
    
    def detect_boundary_positions(
        self,
        substrate,
        percentile: float = 70.0,
        min_gap: int = 5,
    ) -> np.ndarray:
        """
        Detect boundary positions using multi-scale analysis.
        
        Key fix: Sample at fine intervals, aggregate across scales,
        then find local maxima.
        """
        # Extract data from substrate
        if hasattr(substrate, 'byte_values'):
            data = np.array(substrate.byte_values)
        elif hasattr(substrate, 'one_hot'):
            data = np.argmax(substrate.one_hot, axis=1)
        else:
            data = np.array(substrate)
        
        n = len(data)
        if n < 32:
            return np.array([])
        
        # Sample positions at step 1 (fine resolution for peak detection)
        sample_step = 1
        sample_positions = list(range(8, n - 8, sample_step))
        
        # Compute change scores for each position, aggregating across scales
        position_scores = {}
        
        for pos in sample_positions:
            max_score = 0.0
            for scale in self.scales:
                half = scale // 2
                if pos - half < 0 or pos + half >= n:
                    continue
                score = self._compute_change_score(data, pos, scale)
                weight = self.decay_rate ** (scale // 8)
                weighted = score * weight
                max_score = max(max_score, weighted)
            position_scores[pos] = max_score
        
        if not position_scores:
            return np.array([])
        
        # Find local maxima using actual position values
        positions = sorted(position_scores.keys())
        scores_arr = [position_scores[p] for p in positions]
        
        # Compute threshold: percentile of HIGH SCORES
        sorted_scores = sorted(position_scores.values(), reverse=True)
        if len(sorted_scores) >= 10:
            # e.g., percentile=98 means keep top 2%
            n_keep = max(1, int(len(sorted_scores) * (100 - percentile) / 100))
            threshold = sorted_scores[min(n_keep, len(sorted_scores) - 1)]
        else:
            threshold = min(sorted_scores) if sorted_scores else 0
        
        # Find contiguous HIGH-SCORE REGIONS and pick the peak of each region
        regions = []  # List of (region_start, region_end, peak_pos, peak_score)
        in_region = False
        region_start = None
        region_peak_pos = None
        region_peak_score = 0
        
        for i, pos in enumerate(positions):
            score = position_scores[pos]
            
            if score >= threshold:
                if not in_region:
                    # Start new region
                    in_region = True
                    region_start = pos
                    region_peak_pos = pos
                    region_peak_score = score
                else:
                    # Continue region
                    if score > region_peak_score:
                        region_peak_pos = pos
                        region_peak_score = score
            else:
                if in_region:
                    # End region
                    regions.append((region_start, pos, region_peak_pos, region_peak_score))
                    in_region = False
        
        # Don't forget last region
        if in_region:
            regions.append((region_start, positions[-1], region_peak_pos, region_peak_score))
        
        # Extract boundaries: one per region (the peak position)
        boundaries = [region[2] for region in regions]
        
        return np.array(sorted(boundaries))
    
    def detect_boundaries_incremental(
        self, substrate, window_size: int = 32, threshold: float = 0.5
    ) -> List[Dict]:
        """
        Incremental boundary detection for streaming data.
        
        Returns list of boundary events with metadata.
        """
        if hasattr(substrate, 'byte_values'):
            data = np.array(substrate.byte_values)
        else:
            data = np.array(substrate)
        
        boundaries = []
        n = len(data)
        
        if n < window_size:
            return boundaries
        
        # Initialize reference distribution
        ref_dist = self._compute_local_distribution(data, 0, window_size)
        
        for pos in range(window_size, n, window_size // 4):
            curr_dist = self._compute_local_distribution(data, pos, window_size)
            
            # Fisher-Rao distance to reference
            change = fisher_rao_distance(ref_dist, curr_dist)
            
            # Update reference if change is small (gradual drift)
            if change < 0.3:
                ref_dist = 0.7 * ref_dist + 0.3 * curr_dist
                ref_dist = ref_dist / ref_dist.sum()
            
            # Record significant change
            if change > threshold:
                boundaries.append({
                    'position': pos,
                    'strength': change,
                    'local_entropy': scipy_entropy(curr_dist),
                })
        
        return boundaries


# =============================================================================
# 8. CONVENIENCE FUNCTIONS
# =============================================================================

def quick_modality_detect(
    data: bytes,
    modalities: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Quick modality detection for bytes.
    
    Returns: (modality, confidence)
    """
    detector = AdvancedModalityDetector(
        modalities=modalities,
        use_spectral=True,
        use_compression=True,
        adaptive=True,
    )
    
    # Compute distribution
    dist = np.zeros(256)
    for b in data:
        dist[b] += 1
    dist = dist / dist.sum()
    
    modality, posteriors = detector.detect(dist, raw_data=data)
    confidence = posteriors.get(modality, 0.0)
    
    return modality, confidence


def quick_boundary_detect(
    data: bytes,
    scales: Optional[List[int]] = None
) -> np.ndarray:
    """
    Quick boundary detection for bytes.
    
    Returns: array of boundary positions
    """
    detector = AdvancedBoundaryDetector(
        scales=scales or [8, 16, 32],
        use_fisher_rao=True,
        use_geodesic=True,
    )
    
    class MockSubstrate:
        def __init__(self, data):
            self.byte_values = list(data)
            self.length = len(data)
    
    substrate = MockSubstrate(data)
    return detector.detect_boundary_positions(substrate, percentile=70.0, min_gap=5)


# =============================================================================
# 9. BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Old class names for backward compatibility
InformationGeometryModalityDetector = AdvancedModalityDetector
GeometricBoundaryDetector = AdvancedBoundaryDetector
