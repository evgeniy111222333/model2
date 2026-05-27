import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _softmax, _sigmoid, _safe_normalize
from bcs.cognition.credit import AdaptiveCalibration, AdaptiveEvidenceScorer, EpisodeCreditEngine

class SemanticLatentDynamics:
    """Learned semantic latent dynamics grounded in the byte-field substrate.

    This layer is not a rule parser. It learns an episode latent z_sem from
    field states, byte distributions, discovered clusters, token evidence, and
    dynamic embeddings. A semantic crystal is an attractor in this learned
    latent space, not a hand-written subject/relation/object record.
    """

    def __init__(
        self,
        d_input: int = 2048,
        d_sem: int = 256,
        learning_rate: float = 0.015,
        memory_threshold: float = 0.93,
        max_crystals: int = 4096,
        seed: int = 1337,
    ):
        self.d_input = int(d_input)
        self.d_sem = int(d_sem)
        self.learning_rate = float(learning_rate)
        self.memory_threshold = float(memory_threshold)
        self.max_crystals = int(max_crystals)
        self.rng = np.random.default_rng(seed)

        self.W_enc = (self.rng.standard_normal((self.d_sem, self.d_input)).astype(np.float32)
                      / np.sqrt(max(self.d_input, 1)))
        self.W_dec = (self.rng.standard_normal((self.d_input, self.d_sem)).astype(np.float32)
                      * 0.02)
        self.W_pred = (self.rng.standard_normal((self.d_sem, self.d_sem)).astype(np.float32)
                       * 0.02)

        self.crystals = []
        self.prev_z = None
        self.history = []
        self.feature_blocks = [
            ('sequence', 0, 768),
            ('byte_distribution', 768, 1024),
            ('field_concept', 1024, 1152),
            ('embedding_summary', 1152, 1280),
            ('cluster_distribution', 1280, 1536),
            ('cluster_stats', 1536, 1568),
            ('token_distribution', 1568, 1824),
        ]

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(x))
        if norm < 1e-10:
            return np.zeros_like(x, dtype=np.float32)
        return (x / norm).astype(np.float32)

    def _resize(self, x: np.ndarray, dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(arr) >= dim:
            return arr[:dim].copy()
        return np.pad(arr, (0, dim - len(arr))).astype(np.float32)

    def _condition_block(self, x: np.ndarray, dim: int) -> np.ndarray:
        """Whiten one evidence channel so scalar metadata cannot dominate."""
        return self._unit(self._resize(x, dim))

    @staticmethod
    def _deterministic_digest(raw_data: bytes) -> str:
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8', errors='ignore')
        h = 1469598103934665603
        for b in bytes(raw_data[:4096]):
            h ^= int(b)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return f"{h:016x}"

    def _token_distribution(self, tokens: List[Dict]) -> np.ndarray:
        dist = np.zeros(256, dtype=np.float32)
        if not tokens:
            return dist
        for token in tokens:
            token_str = token.get('token_str', '')
            if isinstance(token_str, str):
                raw = token_str.encode('utf-8', errors='ignore')
            else:
                raw = bytes(token_str)
            weight = float(token.get('quality', 0.5)) * max(float(token.get('frequency', 1)), 1.0)
            for b in raw:
                dist[int(b)] += weight
        return _safe_normalize(dist).astype(np.float32) if dist.sum() > 0 else dist

    def _sequence_sketch(self, raw_data: bytes, dims: int = 768) -> np.ndarray:
        """Local byte n-gram sketch with coarse position evidence.

        Each window is hashed independently. This keeps the channel
        order-sensitive without the cascade error of a rolling hash where one
        corrupted byte changes every later window.
        """
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')
        data = bytes(raw_data[:4096])
        sketch = np.zeros(dims, dtype=np.float32)
        if len(data) < 2:
            return sketch

        for n in (2, 3, 5, 8, 13):
            if len(data) < n:
                continue
            n_windows = len(data) - n + 1
            weight = 1.0 + np.log1p(n)
            for i in range(n_windows):
                h = (2166136261 + n * 16777619) & 0xFFFFFFFF
                for j, b in enumerate(data[i:i + n]):
                    h ^= (int(b) + 257 * (j + 1) + 131 * n) & 0xFFFFFFFF
                    h = (h * 16777619) & 0xFFFFFFFF

                pos_bucket = int((i * 32) / max(n_windows, 1))
                pos_mix = (pos_bucket * 2654435761 + n * 97531) & 0xFFFFFFFF

                for h_val, w_mul in ((h, 0.55), (h ^ pos_mix, 0.45)):
                    sign = 1.0 if ((h_val >> 31) & 1) else -1.0
                    sketch[h_val % dims] += sign * weight * w_mul
        return self._unit(sketch)

    def episode_features(
        self,
        substrate,
        field_system,
        embeddings: Optional[np.ndarray],
        clusters: List[Dict],
        tokens: List[Dict],
    ) -> np.ndarray:
        parts = []

        parts.append(self._condition_block(self._sequence_sketch(substrate.raw_data, dims=768), 768))

        byte_dist = substrate.byte_distribution.astype(np.float32)
        parts.append(self._condition_block(byte_dist, 256))

        if field_system is not None:
            concept = field_system.get_concept_activation()
            phi_summary = np.concatenate([
                np.mean(concept, axis=0),
                np.std(concept, axis=0),
            ]).astype(np.float32)
            parts.append(self._condition_block(phi_summary, 128))
        else:
            parts.append(np.zeros(128, dtype=np.float32))

        if embeddings is not None and len(embeddings) > 0:
            emb_summary = np.concatenate([
                np.mean(embeddings, axis=0),
                np.std(embeddings, axis=0),
            ]).astype(np.float32)
            parts.append(self._condition_block(emb_summary, 128))
        else:
            parts.append(np.zeros(128, dtype=np.float32))

        if clusters:
            cluster_dist = np.zeros(256, dtype=np.float32)
            quality = []
            sizes = []
            for c in clusters:
                w = max(float(c.get('size', 1)), 1.0)
                cluster_dist += w * c.get('distribution', np.zeros(256, dtype=np.float32))
                quality.append(float(c.get('quality_score', 0.0)))
                sizes.append(w)
            cluster_dist = self._condition_block(cluster_dist, 256)
            cluster_stats = np.array([
                np.log1p(len(clusters)),
                np.mean(quality) if quality else 0.0,
                np.std(quality) if quality else 0.0,
                np.log1p(np.mean(sizes)) if sizes else 0.0,
                np.log1p(np.std(sizes)) if sizes else 0.0,
            ], dtype=np.float32)
            parts.append(cluster_dist)
            parts.append(self._condition_block(cluster_stats, 32))
        else:
            parts.append(np.zeros(256, dtype=np.float32))
            parts.append(np.zeros(32, dtype=np.float32))

        parts.append(self._condition_block(self._token_distribution(tokens), 256))

        x = np.concatenate(parts).astype(np.float32)
        return self._unit(self._resize(x, self.d_input))

    def get_modality_dim(self, substrate) -> int:
        if substrate is None:
            return self.d_sem
        
        # Detect modality
        if hasattr(substrate, 'detect_modality'):
            mod = substrate.detect_modality()
        else:
            # Check if it's raw bytes or string
            raw_bytes = b''
            if isinstance(substrate, (bytes, bytearray)):
                raw_bytes = bytes(substrate)
            elif isinstance(substrate, str):
                raw_bytes = substrate.encode('utf-8')
            elif isinstance(substrate, np.ndarray):
                if substrate.ndim == 1 and substrate.dtype == np.uint8:
                    raw_bytes = substrate.tobytes()
            
            if not raw_bytes:
                return 64  # Fallback for empty or unknown
                
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            counts = np.bincount(arr, minlength=256)
            dist = counts.astype(np.float64) / len(raw_bytes)
            
            ascii_range = float(np.sum(dist[0x20:0x7F]))
            null_ratio = float(dist[0x00])
            
            p = dist[dist > 0]
            entropy = float(-np.sum(p * np.log2(p))) if p.size > 0 else 0.0
            unique_bytes = int(np.count_nonzero(dist))
            
            if ascii_range > 0.85 and entropy < 5.0:
                mod = "text_ascii"
            elif null_ratio > 0.3:
                mod = "sparse_binary"
            elif unique_bytes < 32 and entropy < 3.0:
                mod = "structured_data"
            else:
                mod = "mixed"

        if mod in ('text_utf8', 'image', 'audio', 'mixed'):
            return 256
        elif mod in ('text_ascii', 'structured', 'structured_data'):
            return 128
        elif mod in ('binary', 'sparse_binary'):
            return 64
        else:
            return 128

    def get_gating_mask(self, substrate) -> np.ndarray:
        dim = self.get_modality_dim(substrate)
        mask = np.zeros(self.d_sem, dtype=np.float32)
        mask[:dim] = 1.0
        return mask

    def encode(self, x: np.ndarray, substrate=None) -> np.ndarray:
        x = self._resize(x, self.d_input)
        z = self._unit(np.tanh(self.W_enc @ x))
        if substrate is not None:
            mask = self.get_gating_mask(substrate)
            z = self._unit(z * mask)
        return z

    def _encode_cache(self, x: np.ndarray, substrate=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._resize(x, self.d_input)
        h = np.tanh(self.W_enc @ x)
        z = self._unit(h)
        if substrate is not None:
            mask = self.get_gating_mask(substrate)
            z = self._unit(z * mask)
        return x, h.astype(np.float32), z.astype(np.float32)

    def _contrastive_feature_loss(
        self,
        anchor_x: np.ndarray,
        positive_x: np.ndarray,
        negative_xs: List[np.ndarray],
        margin: float = 0.22,
        substrate=None,
    ) -> Dict:
        _, _, za = self._encode_cache(anchor_x, substrate=substrate)
        _, _, zp = self._encode_cache(positive_x, substrate=substrate)
        pos_sim = float(np.dot(za, zp))
        neg_sims = []
        for neg_x in negative_xs:
            _, _, zn = self._encode_cache(neg_x, substrate=substrate)
            neg_sims.append(float(np.dot(za, zn)))
        hardest = max(neg_sims) if neg_sims else -1.0
        loss = max(0.0, float(margin) - pos_sim + hardest) if neg_sims else max(0.0, 1.0 - pos_sim)
        return {
            'loss': float(loss),
            'positive_similarity': pos_sim,
            'hardest_negative_similarity': float(hardest),
            'n_negatives': len(negative_xs),
        }

    def train_contrastive_features(
        self,
        anchor_x: np.ndarray,
        positive_x: np.ndarray,
        negative_xs: Optional[List[np.ndarray]] = None,
        epochs: int = 8,
        lr: Optional[float] = None,
        margin: float = 0.22,
        substrate=None,
    ) -> Dict:
        """Train semantic encoder geometry from grounded feature positives/negatives."""
        negative_xs = list(negative_xs or [])
        step_lr = float(self.learning_rate if lr is None else lr)
        before = self._contrastive_feature_loss(anchor_x, positive_x, negative_xs, margin=margin, substrate=substrate)

        # Convert to torch tensor with autograd enabled and enforce float32
        W_enc_t = torch.from_numpy(self.W_enc.copy()).float().requires_grad_(True)
        xa_t = torch.from_numpy(self._resize(anchor_x, self.d_input)).float()
        xp_t = torch.from_numpy(self._resize(positive_x, self.d_input)).float()
        neg_xs_t = [torch.from_numpy(self._resize(nx, self.d_input)).float() for nx in negative_xs]

        # Get modality dynamic gating mask
        mask_np = self.get_gating_mask(substrate)
        mask_t = torch.from_numpy(mask_np).float().to(W_enc_t.device)
        xa_t = torch.from_numpy(self._resize(anchor_x, self.d_input)).float()
        xp_t = torch.from_numpy(self._resize(positive_x, self.d_input)).float()
        neg_xs_t = [torch.from_numpy(self._resize(nx, self.d_input)).float() for nx in negative_xs]

        for _ in range(max(1, int(epochs))):
            if W_enc_t.grad is not None:
                W_enc_t.grad.zero_()

            # Forward pass: anchor and positive
            ha = torch.tanh(W_enc_t @ xa_t)
            ha = ha * mask_t
            norm_a = torch.linalg.norm(ha)
            za = ha / norm_a if norm_a > 1e-10 else torch.zeros_like(ha)

            hp = torch.tanh(W_enc_t @ xp_t)
            hp = hp * mask_t
            norm_p = torch.linalg.norm(hp)
            zp = hp / norm_p if norm_p > 1e-10 else torch.zeros_like(hp)

            pos_sim = torch.dot(za, zp)

            loss = 0.35 * (1.0 - pos_sim)

            if neg_xs_t:
                neg_sims = []
                for xn_t in neg_xs_t:
                    hn = torch.tanh(W_enc_t @ xn_t)
                    hn = hn * mask_t
                    norm_n = torch.linalg.norm(hn)
                    zn = hn / norm_n if norm_n > 1e-10 else torch.zeros_like(hn)
                    neg_sims.append(torch.dot(za, zn))
                neg_sims_t = torch.stack(neg_sims)
                hard_sim = torch.max(neg_sims_t)

                with torch.no_grad():
                    violation = float(margin) - float(pos_sim.item()) + float(hard_sim.item())

                if violation > 0.0:
                    loss = loss + 0.55 * hard_sim

            loss.backward()

            with torch.no_grad():
                W_enc_t.data -= step_lr * W_enc_t.grad
                # Normalization constraint
                row_norms = torch.linalg.norm(W_enc_t, axis=1, keepdim=True)
                scale = torch.clamp(3.0 / torch.clamp(row_norms, min=1e-8), max=1.0)
                W_enc_t.multiply_(scale)

        self.W_enc = W_enc_t.detach().cpu().numpy().astype(np.float32)
        after = self._contrastive_feature_loss(anchor_x, positive_x, negative_xs, margin=margin, substrate=substrate)
        return {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': step_lr,
            'margin': float(margin),
            'loss_before': before['loss'],
            'loss_after': after['loss'],
            'positive_similarity_before': before['positive_similarity'],
            'positive_similarity_after': after['positive_similarity'],
            'hardest_negative_before': before['hardest_negative_similarity'],
            'hardest_negative_after': after['hardest_negative_similarity'],
            'n_negatives': len(negative_xs),
        }

    def _update_autoencoder(self, x: np.ndarray, z: np.ndarray) -> float:
        lr = self.learning_rate
        recon = np.tanh(self.W_dec @ z)
        err = recon - x
        loss = float(np.mean(err ** 2))
        d_recon = (2.0 / max(len(err), 1)) * err * (1.0 - recon ** 2)
        d_z = (self.W_dec.T @ d_recon) * (1.0 - z ** 2)
        
        # Mask out gradient updates for inactive dimensions
        active_mask = (z != 0.0).astype(np.float32)
        d_z = d_z * active_mask
        
        self.W_dec -= lr * np.outer(d_recon, z).astype(np.float32)
        self.W_enc -= (lr * 0.25) * np.outer(d_z, x).astype(np.float32)
        return loss

    def _update_prediction(self, z: np.ndarray) -> float:
        if self.prev_z is None:
            self.prev_z = z.copy()
            return 0.0
        pred = np.tanh(self.W_pred @ self.prev_z)
        err = pred - z
        loss = float(np.mean(err ** 2))
        d_pred = (2.0 / max(len(err), 1)) * err * (1.0 - pred ** 2)
        
        # Mask out gradient updates for inactive dimensions
        active_mask = (z != 0.0).astype(np.float32)
        d_pred = d_pred * active_mask
        
        self.W_pred -= self.learning_rate * np.outer(d_pred, self.prev_z).astype(np.float32)
        self.prev_z = z.copy()
        return loss

    def _update_invariance(self, x: np.ndarray, z: np.ndarray, substrate=None) -> float:
        x_aug = x.copy()
        if substrate is not None:
            # Extract raw bytes from substrate
            raw = b''
            if hasattr(substrate, 'raw_data'):
                raw = bytes(substrate.raw_data)
            elif isinstance(substrate, (bytes, bytearray)):
                raw = bytes(substrate)
            elif isinstance(substrate, str):
                raw = substrate.encode('utf-8')
            elif isinstance(substrate, np.ndarray):
                if substrate.ndim == 1 and substrate.dtype == np.uint8:
                    raw = substrate.tobytes()
                    
            if raw:
                # Detect active dimension/modality
                dim = self.get_modality_dim(substrate)
                
                # Perform physical mutation on bytes
                raw_arr = np.frombuffer(raw, dtype=np.uint8).copy()
                n_bytes = len(raw_arr)
                if n_bytes > 0:
                    if dim == 64: # binary
                        # Bit flip
                        flip_mask = (self.rng.random(n_bytes) < 0.03)
                        if np.any(flip_mask):
                            raw_arr[flip_mask] ^= self.rng.choice([1, 2, 4, 8, 16, 32, 64, 128], size=np.sum(flip_mask)).astype(np.uint8)
                    elif dim == 128: # ascii/structured
                        # Typo simulation: swap adjacent bytes
                        if n_bytes > 1:
                            swap_mask = (self.rng.random(n_bytes - 1) < 0.05)
                            for idx in np.where(swap_mask)[0]:
                                raw_arr[idx], raw_arr[idx+1] = raw_arr[idx+1], raw_arr[idx]
                    else: # mixed / utf8 / media
                        # Add small noise
                        noise = self.rng.choice([-1, 0, 1], size=n_bytes, p=[0.1, 0.8, 0.1])
                        raw_arr = np.clip(raw_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        
                    raw_aug = raw_arr.tobytes()
                    
                    # Recompute physical features
                    seq_aug = self._condition_block(self._sequence_sketch(raw_aug, dims=768), 768)
                    
                    counts = np.bincount(raw_arr, minlength=256)
                    dist_aug = counts.astype(np.float32) / max(len(raw_aug), 1)
                    dist_aug = self._condition_block(dist_aug, 256)
                    
                    # Store back in x_aug
                    x_aug[0:768] = seq_aug
                    x_aug[768:1024] = dist_aug
        else:
            # Fallback random mask for physical dimensions if substrate is not available
            mask = (self.rng.random(1024) > 0.08).astype(np.float32)
            x_aug[0:1024] *= mask
            
        # Apply semantic block modifications
        semantic_slices = [
            (1024, 1152), # field_concept
            (1152, 1280), # embedding_summary
            (1280, 1536), # cluster_distribution
            (1536, 1568), # cluster_stats
            (1568, 1824), # token_distribution
        ]
        for start_idx, end_idx in semantic_slices:
            if self.rng.random() < 0.05:
                x_aug[start_idx:end_idx] = 0.0  # Channel Dropout
            else:
                scale = float(self.rng.uniform(0.9, 1.1))
                x_aug[start_idx:end_idx] *= scale  # Scale Jittering
                
        # Normalize and encode augmented feature vector
        x_aug = self._unit(self._resize(x_aug, self.d_input))
        z_aug = self.encode(x_aug, substrate=substrate)
        
        target = self._unit(0.5 * (z + z_aug))
        inv_loss = float(np.mean((z - z_aug) ** 2))
        dz = (z - target) * (1.0 - z ** 2)
        dz_aug = (z_aug - target) * (1.0 - z_aug ** 2)
        
        # Mask out gradient updates for inactive dimensions
        active_mask = (z != 0.0).astype(np.float32)
        dz = dz * active_mask
        dz_aug = dz_aug * active_mask
        
        self.W_enc -= (self.learning_rate * 0.15) * np.outer(dz, x).astype(np.float32)
        self.W_enc -= (self.learning_rate * 0.15) * np.outer(dz_aug, x_aug).astype(np.float32)
        return inv_loss

    def _block_evidence(self, x: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], float, float]:
        sims = {}
        valid = []
        x = self._resize(x, self.d_input)
        y = self._resize(y, self.d_input)
        for name, start, end in self.feature_blocks:
            a = x[start:end]
            b = y[start:end]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-10 or nb < 1e-10:
                continue
            sim = float(np.dot(a / na, b / nb))
            sims[name] = sim
            valid.append(sim)
        if not valid:
            return sims, 0.0, 0.0
        arr = np.array(valid, dtype=np.float32)
        k = max(1, int(np.ceil(0.4 * len(arr))))
        lower_tail = float(np.mean(np.sort(arr)[:k]))
        return sims, float(np.mean(arr)), lower_tail

    def _nearest_crystal(self, z: np.ndarray, x: Optional[np.ndarray] = None) -> Tuple[int, float, float, float, Dict[str, float], float, float]:
        best_idx = -1
        best_score = -1.0
        best_latent = -1.0
        best_feature = -1.0
        best_blocks = {}
        best_block_mean = -1.0
        best_block_floor = -1.0
        for idx, crystal in enumerate(self.crystals):
            latent_sim = float(np.dot(z, crystal['z']))
            if x is not None:
                feature_sim = float(np.dot(self._unit(x), self._unit(crystal['feature_mean'])))
                block_sims, block_mean, block_floor = self._block_evidence(x, crystal['feature_mean'])
                score = min(latent_sim, feature_sim, block_floor)
            else:
                feature_sim = latent_sim
                block_sims = {}
                block_mean = latent_sim
                block_floor = latent_sim
                score = latent_sim
            if score > best_score:
                best_score = score
                best_latent = latent_sim
                best_feature = feature_sim
                best_blocks = block_sims
                best_block_mean = block_mean
                best_block_floor = block_floor
                best_idx = idx
        return best_idx, best_score, best_latent, best_feature, best_blocks, best_block_mean, best_block_floor

    def consolidate(self, z: np.ndarray, x: np.ndarray, source_digest: str) -> Dict:
        idx, sim, latent_sim, feature_sim, block_sims, block_mean, block_floor = self._nearest_crystal(z, x)
        if idx >= 0 and sim >= self.memory_threshold:
            crystal = self.crystals[idx]
            n = int(crystal['n']) + 1
            crystal['z'] = self._unit((crystal['z'] * crystal['n'] + z) / n)
            crystal['feature_mean'] = ((crystal['feature_mean'] * crystal['n'] + x) / n).astype(np.float32)
            crystal['n'] = n
            crystal['last_similarity'] = sim
            crystal['source_digests'].add(source_digest)
            return {
                'status': 'recognized',
                'idx': idx,
                'similarity': sim,
                'latent_similarity': latent_sim,
                'feature_similarity': feature_sim,
                'block_mean_similarity': block_mean,
                'block_floor_similarity': block_floor,
                'block_similarities': block_sims,
                'n': n,
            }

        if len(self.crystals) >= self.max_crystals:
            weakest = min(range(len(self.crystals)), key=lambda i: self.crystals[i]['n'])
            self.crystals.pop(weakest)
        self.crystals.append({
            'z': z.copy(),
            'feature_mean': x.copy(),
            'n': 1,
            'last_similarity': sim,
            'source_digests': {source_digest},
        })
        return {
            'status': 'novel',
            'idx': len(self.crystals) - 1,
            'similarity': sim,
            'latent_similarity': latent_sim,
            'feature_similarity': feature_sim,
            'block_mean_similarity': block_mean,
            'block_floor_similarity': block_floor,
            'block_similarities': block_sims,
            'n': 1,
        }

    def observe_episode(
        self,
        substrate,
        field_system,
        embeddings: Optional[np.ndarray],
        clusters: List[Dict],
        tokens: List[Dict],
    ) -> Dict:
        x = self.episode_features(substrate, field_system, embeddings, clusters, tokens)
        z = self.encode(x, substrate=substrate)
        recon_loss = self._update_autoencoder(x, z)
        pred_loss = self._update_prediction(z)
        inv_loss = self._update_invariance(x, z, substrate=substrate)
        source_digest = self._deterministic_digest(substrate.raw_data[:4096])
        mem = self.consolidate(z, x, source_digest)
        separation = 0.0
        if len(self.crystals) > 1:
            sims = [
                min(
                    float(np.dot(z, c['z'])),
                    float(np.dot(self._unit(x), self._unit(c['feature_mean']))),
                    self._block_evidence(x, c['feature_mean'])[2],
                )
                for i, c in enumerate(self.crystals)
                if i != mem['idx']
            ]
            separation = float(mem['similarity'] - max(sims)) if sims else 0.0
        record = {
            'reconstruction_loss': recon_loss,
            'prediction_loss': pred_loss,
            'invariance_loss': inv_loss,
            'memory_status': mem['status'],
            'memory_similarity': float(mem['similarity']),
            'latent_similarity': float(mem.get('latent_similarity', mem['similarity'])),
            'feature_similarity': float(mem.get('feature_similarity', mem['similarity'])),
            'block_mean_similarity': float(mem.get('block_mean_similarity', mem['similarity'])),
            'block_floor_similarity': float(mem.get('block_floor_similarity', mem['similarity'])),
            'block_similarities': {
                k: float(v) for k, v in mem.get('block_similarities', {}).items()
            },
            'memory_index': int(mem['idx']),
            'memory_count': int(mem['n']),
            'n_semantic_crystals': len(self.crystals),
            'separation_margin': separation,
            'z_norm': float(np.linalg.norm(z)),
        }
        self.history.append(record)
        return record

    def compare_episode(
        self,
        substrate,
        field_system,
        embeddings: Optional[np.ndarray],
        clusters: List[Dict],
        tokens: List[Dict],
    ) -> Dict:
        x = self.episode_features(substrate, field_system, embeddings, clusters, tokens)
        z = self.encode(x, substrate=substrate)
        idx, sim, latent_sim, feature_sim, block_sims, block_mean, block_floor = self._nearest_crystal(z, x)
        return {
            'nearest_index': int(idx),
            'similarity': float(sim),
            'latent_similarity': float(latent_sim),
            'feature_similarity': float(feature_sim),
            'block_mean_similarity': float(block_mean),
            'block_floor_similarity': float(block_floor),
            'block_similarities': {k: float(v) for k, v in block_sims.items()},
            'recognized': bool(idx >= 0 and sim >= self.memory_threshold),
        }




class SemanticByteDecoder:
    """Byte-level generative decoder trained only from observed byte streams."""

    def __init__(self, max_context: int = 8, max_entries: int = 200000, latent_dim: int = 96, seed: int = 3031):
        self.max_context = int(max_context)
        self.max_entries = int(max_entries)
        self.latent_dim = int(latent_dim)
        self.counts = {}
        self.unigram = np.zeros(256, dtype=np.float32)
        self.total = 0
        rng = np.random.default_rng(seed)
        self.W_condition = (rng.standard_normal((256, self.latent_dim)).astype(np.float32)
                            / np.sqrt(max(self.latent_dim, 1)) * 0.02)
        self.condition_bias = np.zeros(256, dtype=np.float32)

    @staticmethod
    def _to_bytes(data) -> bytes:
        if data is None:
            return b''
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, str):
            return data.encode('utf-8', errors='ignore')
        return bytes(data)

    def observe(self, data) -> Dict:
        raw = self._to_bytes(data)
        if not raw:
            return {'observed': 0, 'contexts': len(self.counts)}
        arr = np.frombuffer(raw, dtype=np.uint8)
        self.unigram += np.bincount(arr, minlength=256).astype(np.float32)
        self.total += len(raw)
        for i, b in enumerate(raw):
            max_ctx = min(self.max_context, i)
            for width in range(1, max_ctx + 1):
                ctx = raw[i - width:i]
                if ctx not in self.counts:
                    if len(self.counts) >= self.max_entries:
                        break
                    self.counts[ctx] = np.zeros(256, dtype=np.float32)
                self.counts[ctx][int(b)] += 1.0
        return {'observed': len(raw), 'contexts': len(self.counts)}

    def distribution(self, context: bytes) -> Tuple[np.ndarray, int]:
        ctx_raw = self._to_bytes(context)
        for width in range(min(self.max_context, len(ctx_raw)), 0, -1):
            ctx = ctx_raw[-width:]
            if ctx in self.counts:
                dist = self.counts[ctx].copy()
                total = float(dist.sum())
                if total > 0:
                    return (dist / total).astype(np.float32), width
        if self.total > 0:
                return (self.unigram / max(float(self.unigram.sum()), 1.0)).astype(np.float32), 0
        return (np.ones(256, dtype=np.float32) / 256.0), 0

    def _condition_vector(self, condition: Optional[np.ndarray]) -> np.ndarray:
        if condition is None:
            return np.zeros(self.latent_dim, dtype=np.float32)
        arr = np.asarray(condition, dtype=np.float32).reshape(-1)
        if len(arr) >= self.latent_dim:
            arr = arr[:self.latent_dim].copy()
        else:
            arr = np.pad(arr, (0, self.latent_dim - len(arr))).astype(np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-10:
            arr = arr / norm
        return arr.astype(np.float32)

    @staticmethod
    def _prob_softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        centered = logits - float(np.max(logits))
        exp_vals = np.exp(np.clip(centered, -60.0, 60.0))
        return (exp_vals / max(float(exp_vals.sum()), 1e-12)).astype(np.float32)

    def conditioned_distribution(
        self,
        condition: Optional[np.ndarray],
        context: bytes = b'',
        latent_strength: float = 0.45,
    ) -> Tuple[np.ndarray, Dict]:
        base, width = self.distribution(context)
        cond = self._condition_vector(condition)
        latent = self._prob_softmax(self.W_condition @ cond + self.condition_bias)
        strength = float(np.clip(latent_strength, 0.0, 1.0))
        dist = (1.0 - strength) * base + strength * latent
        dist = dist / max(float(dist.sum()), 1e-12)
        return dist.astype(np.float32), {
            'context_width': int(width),
            'latent_strength': strength,
            'condition_norm': float(np.linalg.norm(cond)),
        }

    def nll(self, data) -> Dict:
        raw = self._to_bytes(data)
        if not raw:
            return {'mean_nll': 0.0, 'perplexity': 1.0, 'bytes': 0, 'contexts': len(self.counts)}
        context = bytearray()
        losses = []
        for b in raw:
            dist, _ = self.distribution(bytes(context))
            p = max(float(dist[int(b)]), 1e-12)
            losses.append(-float(np.log(p)))
            context.append(int(b))
            if len(context) > self.max_context:
                del context[0]
        mean_nll = float(np.mean(losses)) if losses else 0.0
        return {
            'mean_nll': mean_nll,
            'perplexity': float(np.exp(min(mean_nll, 20.0))),
            'bytes': len(raw),
            'contexts': len(self.counts),
        }

    def conditioned_nll(
        self,
        data,
        condition: Optional[np.ndarray],
        seed: bytes = b'',
        latent_strength: float = 0.45,
    ) -> Dict:
        raw = self._to_bytes(data)
        if not raw:
            return {'mean_nll': 0.0, 'perplexity': 1.0, 'bytes': 0, 'contexts': len(self.counts)}
        context = bytearray(self._to_bytes(seed)[-self.max_context:])
        losses = []
        widths = []
        for b in raw:
            dist, meta = self.conditioned_distribution(condition, bytes(context), latent_strength=latent_strength)
            p = max(float(dist[int(b)]), 1e-12)
            losses.append(-float(np.log(p)))
            widths.append(int(meta.get('context_width', 0)))
            context.append(int(b))
            if len(context) > self.max_context:
                del context[0]
        mean_nll = float(np.mean(losses)) if losses else 0.0
        return {
            'mean_nll': mean_nll,
            'perplexity': float(np.exp(min(mean_nll, 20.0))),
            'bytes': len(raw),
            'contexts': len(self.counts),
            'mean_context_width': float(np.mean(widths)) if widths else 0.0,
        }

    def observe_conditioned(
        self,
        condition: Optional[np.ndarray],
        target,
        seed: bytes = b'',
        epochs: int = 6,
        lr: float = 0.035,
        latent_strength: float = 0.45,
    ) -> Dict:
        raw = self._to_bytes(target)
        if not raw:
            return {'status': 'empty', 'loss_before': 0.0, 'loss_after': 0.0}
        cond = self._condition_vector(condition)
        before = self.conditioned_nll(raw, cond, seed=seed, latent_strength=latent_strength)
        arr = np.frombuffer(raw, dtype=np.uint8)
        target_dist = np.bincount(arr, minlength=256).astype(np.float32)
        target_dist = target_dist / max(float(target_dist.sum()), 1.0)

        for _ in range(max(1, int(epochs))):
            logits = self.W_condition @ cond + self.condition_bias
            pred = self._prob_softmax(logits)
            grad = (pred - target_dist).astype(np.float32)
            self.W_condition -= np.float32(lr) * np.outer(grad, cond).astype(np.float32)
            self.condition_bias -= np.float32(lr * 0.25) * grad
            self.condition_bias = np.clip(self.condition_bias, -4.0, 4.0).astype(np.float32)
            row_norms = np.linalg.norm(self.W_condition, axis=1, keepdims=True)
            scale = np.minimum(1.0, 4.0 / np.maximum(row_norms, 1e-8))
            self.W_condition = (self.W_condition * scale).astype(np.float32)

        self.observe(seed + b' ' + raw if seed else raw)
        after = self.conditioned_nll(raw, cond, seed=seed, latent_strength=latent_strength)
        return {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'latent_strength': float(latent_strength),
            'loss_before': before['mean_nll'],
            'loss_after': after['mean_nll'],
            'perplexity_before': before['perplexity'],
            'perplexity_after': after['perplexity'],
            'bytes': len(raw),
        }

    def generate(
        self,
        seed,
        evidence: bytes = b'',
        max_bytes: int = 160,
        evidence_strength: float = 1.75,
        condition: Optional[np.ndarray] = None,
        latent_strength: float = 0.35,
    ) -> Dict:
        seed_raw = self._to_bytes(seed)
        evidence_raw = self._to_bytes(evidence)
        output = bytearray()
        context = bytearray(seed_raw[-self.max_context:])
        confidences = []

        for t in range(max(1, int(max_bytes))):
            if condition is None:
                dist, width = self.distribution(bytes(context))
            else:
                dist, meta = self.conditioned_distribution(
                    condition,
                    bytes(context),
                    latent_strength=latent_strength,
                )
                width = int(meta.get('context_width', 0))
            if evidence_raw and t < len(evidence_raw):
                ev = int(evidence_raw[t])
                dist = dist.copy()
                dist[ev] += evidence_strength
                dist = dist / max(float(dist.sum()), 1e-12)
            next_b = int(np.argmax(dist))
            confidences.append(float(dist[next_b]))
            output.append(next_b)
            context.append(next_b)
            if len(context) > self.max_context:
                del context[0]
            if next_b in (0, 10) and t > 16 and width == 0:
                break

        return {
            'bytes': bytes(output),
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'contexts': len(self.counts),
            'observed_bytes': int(self.total),
            'latent_conditioned': bool(condition is not None),
        }




class SemanticAssociativeReasoner:
    """Multi-step associative reasoning over byte-grounded evidence."""

    FEATURE_NAMES = (
        'state_hidden_similarity',
        'query_feature_similarity',
        'base_evidence_support',
        'transition_continuity',
        'repeat_seen',
        'byte_evidence_overlap',
        'memory_strength',
        'source_coherence',
    )

    def __init__(
        self,
        n_steps: int = 4,
        damping: float = 0.35,
        transition_strength: float = 0.18,
        repeat_penalty: float = 0.04,
    ):
        self.n_steps = int(n_steps)
        self.damping = float(damping)
        self.transition_strength = float(transition_strength)
        self.repeat_penalty = float(repeat_penalty)
        self.score_weights = np.array([
            0.34,
            0.24,
            0.22,
            self.transition_strength,
            -self.repeat_penalty,
            0.30,
            0.06,
            0.04,
        ], dtype=np.float32)
        init_mix = np.array([self.damping, 0.45, 0.20], dtype=np.float32)
        init_mix = init_mix / max(float(init_mix.sum()), 1e-8)
        self.state_mix_logits = np.log(init_mix + 1e-6).astype(np.float32)
        self.state_mix = init_mix.astype(np.float32)

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(x))
        if norm < 1e-10:
            return np.zeros_like(x, dtype=np.float32)
        return (x / norm).astype(np.float32)

    def _transition(self, prev: Optional[Dict], cand: Dict) -> float:
        if prev is None:
            return 0.0
        if prev.get('candidate_id') == cand.get('candidate_id'):
            return -0.25
        if prev.get('source_digest') and prev.get('source_digest') == cand.get('source_digest'):
            prev_mid = 0.5 * (float(prev.get('start', 0)) + float(prev.get('end', 0)))
            cand_mid = 0.5 * (float(cand.get('start', 0)) + float(cand.get('end', 0)))
            scale = max(float(prev.get('end', 0)) - float(prev.get('start', 0)), 32.0)
            distance = abs(cand_mid - prev_mid)
            return float(np.exp(-distance / max(scale, 1.0)))
        if prev.get('kind') == 'qa_memory' and cand.get('kind') == 'qa_answer':
            return 0.12
        return 0.0

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return scores
        temp = max(float(temperature), 1e-4)
        centered = (scores - float(np.max(scores))) / temp
        exp_scores = np.exp(np.clip(centered, -60.0, 60.0))
        total = max(float(exp_scores.sum()), 1e-12)
        return (exp_scores / total).astype(np.float32)

    def _candidate_features(
        self,
        state: np.ndarray,
        intent_feature: np.ndarray,
        prev: Optional[Dict],
        visited: set,
        cand: Dict,
    ) -> np.ndarray:
        hidden_sim = float(np.dot(state, cand['hidden']))
        feature_sim = float(np.dot(intent_feature, cand['feature']))
        base = float(cand.get('base_score', cand.get('score', 0.0)))
        transition = self._transition(prev, cand)
        repeat_seen = 1.0 if cand.get('candidate_id') in visited else 0.0
        byte_overlap = float(cand.get('byte_overlap', 0.0))
        memory_strength = float(min(1.0, np.log1p(max(float(cand.get('n', 1)), 1.0)) / np.log1p(16.0)))
        source_coherence = 0.0
        if prev is not None and prev.get('source_digest') and prev.get('source_digest') == cand.get('source_digest'):
            source_coherence = 1.0
        return np.array([
            hidden_sim,
            feature_sim,
            base,
            transition,
            repeat_seen,
            byte_overlap,
            memory_strength,
            source_coherence,
        ], dtype=np.float32)

    def _score(self, features: np.ndarray) -> float:
        return float(np.dot(self.score_weights, np.asarray(features, dtype=np.float32)))

    def _advance_state(self, state: np.ndarray, cand: Dict, intent_hidden: np.ndarray) -> np.ndarray:
        mix = self._softmax(self.state_mix_logits, temperature=1.0)
        self.state_mix = mix.astype(np.float32)
        return self._unit(mix[0] * state + mix[1] * cand['hidden'] + mix[2] * intent_hidden)

    def _regularize_policy(self):
        self.score_weights = np.nan_to_num(self.score_weights, nan=0.0, posinf=2.0, neginf=-2.0)
        if self.score_weights.shape[0] != len(self.FEATURE_NAMES):
            resized = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
            n = min(len(resized), int(self.score_weights.shape[0]))
            resized[:n] = self.score_weights[:n]
            self.score_weights = resized
        lower = np.array([-0.75, -0.75, -0.75, -1.00, -2.00, -0.50, -0.50, -0.50], dtype=np.float32)
        upper = np.array([2.25, 2.25, 2.25, 1.25, 0.00, 2.25, 1.25, 1.25], dtype=np.float32)
        self.score_weights = np.clip(self.score_weights, lower, upper).astype(np.float32)
        norm = float(np.linalg.norm(self.score_weights[[0, 1, 2, 3, 5, 6, 7]]))
        if norm > 3.0:
            self.score_weights[[0, 1, 2, 3, 5, 6, 7]] *= np.float32(3.0 / norm)
        self.state_mix_logits = np.nan_to_num(self.state_mix_logits, nan=0.0, posinf=2.0, neginf=-2.0)
        self.state_mix_logits = np.clip(self.state_mix_logits, -4.0, 4.0).astype(np.float32)
        self.state_mix = self._softmax(self.state_mix_logits, temperature=1.0)

    def policy_stats(self) -> Dict:
        return {
            'feature_names': list(self.FEATURE_NAMES),
            'score_weights': [float(x) for x in self.score_weights],
            'state_mix': [float(x) for x in self.state_mix],
            'state_mix_logits': [float(x) for x in self.state_mix_logits],
            'n_steps': int(self.n_steps),
        }

    def _resolve_step_targets(
        self,
        step: int,
        target_ids: List[str],
        candidates: List[Dict],
    ) -> List[int]:
        if not target_ids:
            return []
        tid = str(target_ids[min(step, len(target_ids) - 1)])
        out = [i for i, cand in enumerate(candidates) if str(cand.get('candidate_id')) == tid]
        if out:
            return out
        target_raw = None
        for cand in candidates:
            if str(cand.get('candidate_id')) == tid:
                target_raw = cand.get('raw')
                break
        if target_raw is not None:
            out = [i for i, cand in enumerate(candidates) if cand.get('raw') == target_raw]
        return out

    def train_policy(
        self,
        intent_feature: np.ndarray,
        intent_hidden: np.ndarray,
        candidates: List[Dict],
        target_id: Optional[str] = None,
        target_ids: Optional[List[str]] = None,
        epochs: int = 8,
        lr: float = 0.05,
        temperature: float = 0.85,
        counterfactual_weight: float = 0.35,
        train_state: bool = True,
    ) -> Dict:
        usable = [c for c in candidates if c.get('candidate_id') is not None]
        if target_ids is None:
            target_path = [str(target_id)] if target_id is not None else []
        else:
            target_path = [str(t) for t in target_ids if t is not None]
        known_ids = {str(c.get('candidate_id')) for c in usable}
        missing_targets = [tid for tid in target_path if tid not in known_ids]
        if not usable or not target_path or len(missing_targets) == len(target_path):
            return {
                'status': 'no_target_candidate',
                'target_id': target_path[0] if target_path else None,
                'target_path': target_path,
                'loss_before': None,
                'loss_after': None,
                'target_probability_before': 0.0,
                'target_probability_after': 0.0,
                'counterfactual_margin_before': 0.0,
                'counterfactual_margin_after': 0.0,
                'policy': self.policy_stats(),
            }

        intent_feature = self._unit(intent_feature)
        intent_hidden = self._unit(intent_hidden)

        def rollout(update: bool = False) -> Dict:
            state = self._unit(intent_hidden)
            visited = set()
            prev = None
            losses = []
            target_probs = []
            margins = []
            steps = []

            for step in range(max(1, self.n_steps)):
                feats = np.stack([
                    self._candidate_features(state, intent_feature, prev, visited, cand)
                    for cand in usable
                ]).astype(np.float32)
                scores = feats @ self.score_weights
                probs = self._softmax(scores, temperature=temperature)
                target_indices = self._resolve_step_targets(step, target_path, usable)
                if not target_indices:
                    target_indices = [
                        i for i, cand in enumerate(usable)
                        if str(cand.get('candidate_id')) in target_path
                    ]
                if not target_indices:
                    continue
                target_prob = max(float(np.sum(probs[target_indices])), 1e-12)
                target_weights = probs[target_indices] / max(float(np.sum(probs[target_indices])), 1e-12)
                target_feat = target_weights @ feats[target_indices]
                target_score = float(np.mean(scores[target_indices]))
                non_target = [i for i in range(len(usable)) if i not in set(target_indices)]
                best_other_idx = max(non_target, key=lambda i: float(scores[i])) if non_target else target_indices[0]
                margin = float(target_score - scores[best_other_idx]) if non_target else float(target_score)
                margin_loss = max(0.0, 0.18 - margin)
                loss = -float(np.log(target_prob)) + counterfactual_weight * margin_loss

                if update:
                    expected = probs @ feats
                    grad = target_feat - expected
                    if non_target and margin_loss > 0.0:
                        grad += counterfactual_weight * (target_feat - feats[best_other_idx])
                    self.score_weights += np.float32(lr) * grad.astype(np.float32)
                    self._regularize_policy()
                    scores = feats @ self.score_weights
                    probs = self._softmax(scores, temperature=temperature)

                chosen_idx = int(np.argmax(scores))
                chosen = usable[chosen_idx]
                visited.add(chosen.get('candidate_id'))
                prev = chosen
                state = self._advance_state(state, chosen, intent_hidden)
                losses.append(loss)
                target_probs.append(target_prob)
                margins.append(margin)
                steps.append({
                    'step': int(step),
                    'chosen_id': chosen.get('candidate_id'),
                    'target_probability': float(target_prob),
                    'target_margin': float(margin),
                    'target_ids': [usable[i].get('candidate_id') for i in target_indices],
                    'loss': float(loss),
                })

            return {
                'loss': float(np.mean(losses)) if losses else 0.0,
                'target_probability': float(np.mean(target_probs)) if target_probs else 0.0,
                'counterfactual_margin': float(np.mean(margins)) if margins else 0.0,
                'steps': steps,
            }

        before = rollout(update=False)
        for _ in range(max(1, int(epochs))):
            rollout(update=True)
            if train_state:
                base_logits = self.state_mix_logits.copy()
                grad_logits = np.zeros_like(base_logits, dtype=np.float32)
                eps = 0.035
                for k in range(len(base_logits)):
                    self.state_mix_logits = base_logits.copy()
                    self.state_mix_logits[k] += np.float32(eps)
                    self._regularize_policy()
                    plus = rollout(update=False)['loss']
                    self.state_mix_logits = base_logits.copy()
                    self.state_mix_logits[k] -= np.float32(eps)
                    self._regularize_policy()
                    minus = rollout(update=False)['loss']
                    grad_logits[k] = np.float32((plus - minus) / (2.0 * eps))
                self.state_mix_logits = base_logits - np.float32(lr * 0.35) * grad_logits
                self._regularize_policy()
        after = rollout(update=False)
        return {
            'status': 'trained',
            'target_id': target_path[0] if target_path else None,
            'target_path': target_path,
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'train_state': bool(train_state),
            'loss_before': before['loss'],
            'loss_after': after['loss'],
            'target_probability_before': before['target_probability'],
            'target_probability_after': after['target_probability'],
            'counterfactual_margin_before': before['counterfactual_margin'],
            'counterfactual_margin_after': after['counterfactual_margin'],
            'trace_before': before['steps'],
            'trace_after': after['steps'],
            'policy': self.policy_stats(),
        }

    def reason(
        self,
        intent_feature: np.ndarray,
        intent_hidden: np.ndarray,
        candidates: List[Dict],
        ablate_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict:
        usable = [c for c in candidates if c.get('candidate_id') != ablate_id]
        if not usable:
            return {
                'best': None,
                'trace': [],
                'best_score': -1.0,
                'path_score': -1.0,
                'n_steps': 0,
            }

        state = self._unit(intent_hidden)
        intent_feature = self._unit(intent_feature)
        intent_hidden = self._unit(intent_hidden)
        trace = []
        visited = set()
        prev = None

        for step in range(max(1, self.n_steps)):
            scored = []
            for cand in usable:
                feat = self._candidate_features(state, intent_feature, prev, visited, cand)
                score = self._score(feat)
                scored.append({
                    'score': float(score),
                    'features': feat,
                    'candidate': cand,
                })
            probs = self._softmax(np.array([row['score'] for row in scored], dtype=np.float32), temperature=0.85)
            for row, prob in zip(scored, probs):
                row['policy_probability'] = float(prob)
            scored.sort(key=lambda row: row['score'], reverse=True)
            top_row = scored[0]
            score = float(top_row['score'])
            feat = top_row['features']
            cand = top_row['candidate']
            hidden_sim = float(feat[0])
            feature_sim = float(feat[1])
            transition = float(feat[3])
            contributions = feat * self.score_weights
            visited.add(cand.get('candidate_id'))
            trace.append({
                'step': int(step),
                'candidate_id': cand.get('candidate_id'),
                'kind': cand.get('kind', 'episode'),
                'score': float(score),
                'policy_probability': float(top_row['policy_probability']),
                'base_score': float(cand.get('base_score', cand.get('score', 0.0))),
                'hidden_similarity': float(hidden_sim),
                'feature_similarity': float(feature_sim),
                'transition': float(transition),
                'byte_overlap': float(feat[5]),
                'memory_strength': float(feat[6]),
                'source_coherence': float(feat[7]),
                'feature_contributions': {
                    name: float(val)
                    for name, val in zip(self.FEATURE_NAMES, contributions)
                },
                'start': int(cand.get('start', 0)),
                'end': int(cand.get('end', len(cand.get('raw', b'')))),
                'text': cand.get('raw', b'')[:180].decode('utf-8', errors='replace'),
            })
            prev = cand
            state = self._advance_state(state, cand, intent_hidden)

        by_id = {}
        for item in trace:
            cid = item['candidate_id']
            if cid not in by_id:
                by_id[cid] = {'score_sum': 0.0, 'visits': 0, 'candidate': None}
            by_id[cid]['score_sum'] += float(item['score'])
            by_id[cid]['visits'] += 1
        cand_map = {c.get('candidate_id'): c for c in usable}
        best_id = max(by_id, key=lambda cid: by_id[cid]['score_sum'] / max(by_id[cid]['visits'], 1))
        best = cand_map[best_id]
        path_score = by_id[best_id]['score_sum'] / max(by_id[best_id]['visits'], 1)

        ranked = sorted(
            [
                {
                    'candidate_id': cid,
                    'score': vals['score_sum'] / max(vals['visits'], 1),
                    'visits': vals['visits'],
                }
                for cid, vals in by_id.items()
            ],
            key=lambda row: row['score'],
            reverse=True,
        )

        return {
            'best': best,
            'trace': trace,
            'ranked_path': ranked[:top_k],
            'support_base_score': float(max(t['base_score'] for t in trace)) if trace else -1.0,
            'best_score': float(path_score),
            'path_score': float(np.mean([t['score'] for t in trace])) if trace else float(path_score),
            'n_steps': len(trace),
            'policy': self.policy_stats(),
        }




class SemanticDifferentiableMemoryGraph(nn.Module):
    """Matrix memory graph with long counterfactual rollouts over grounded evidence nodes."""

    EDGE_FEATURE_NAMES = (
        'key_affinity',
        'source_coherence',
        'temporal_coherence',
        'target_base_support',
        'target_byte_overlap',
        'target_memory_strength',
    )

    NODE_FEATURE_NAMES = (
        'query_key_similarity',
        'base_support',
        'byte_overlap',
        'memory_strength',
    )

    def __init__(
        self,
        d_hidden: int = 96,
        d_graph: Optional[int] = None,
        rollout_steps: int = 8,
        max_nodes: int = 256,
        seed: int = 4043,
    ):
        super().__init__()
        self.d_hidden = int(d_hidden)
        self.d_graph = int(d_graph if d_graph is not None else d_hidden)
        self.rollout_steps = int(max(2, rollout_steps))
        self.max_nodes = int(max_nodes)
        
        # Use seed for initialization
        rng = np.random.default_rng(seed)
        eye_q = np.eye(self.d_hidden, self.d_graph, dtype=np.float32)
        
        w_query_init = eye_q + (rng.standard_normal((self.d_hidden, self.d_graph)).astype(np.float32) * 0.01)
        w_key_init = eye_q.copy() + (rng.standard_normal((self.d_hidden, self.d_graph)).astype(np.float32) * 0.01)
        w_value_init = eye_q.copy() + (rng.standard_normal((self.d_hidden, self.d_graph)).astype(np.float32) * 0.01)
        
        self._W_query = nn.Parameter(torch.from_numpy(w_query_init))
        self._W_key = nn.Parameter(torch.from_numpy(w_key_init))
        self._W_value = nn.Parameter(torch.from_numpy(w_value_init))
        
        self._edge_weights = nn.Parameter(torch.tensor([0.42, 0.14, 0.18, 0.18, 0.28, 0.06], dtype=torch.float32))
        self._node_weights = nn.Parameter(torch.tensor([0.46, 0.22, 0.30, 0.06], dtype=torch.float32))
        self._state_mix_logits = nn.Parameter(torch.log(torch.tensor([0.58, 0.42], dtype=torch.float32)))
        self._flow_strength = nn.Parameter(torch.tensor(0.70, dtype=torch.float32))
        
        self.training_history = []

    @property
    def device(self) -> torch.device:
        """Повернути пристрій, на якому розташовані параметри."""
        return self._W_query.device

    def _to_tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        """Допоміжний метод для конвертації NumPy в PyTorch тензор."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, dtype=dtype, device=self.device)

    # Властивості для зворотної сумісності (Legacy NumPy Interface)
    @property
    def W_query(self):
        return self._W_query.detach().cpu().numpy()

    @W_query.setter
    def W_query(self, value):
        with torch.no_grad():
            self._W_query.copy_(self._to_tensor(value))

    @property
    def W_key(self):
        return self._W_key.detach().cpu().numpy()

    @W_key.setter
    def W_key(self, value):
        with torch.no_grad():
            self._W_key.copy_(self._to_tensor(value))

    @property
    def W_value(self):
        return self._W_value.detach().cpu().numpy()

    @W_value.setter
    def W_value(self, value):
        with torch.no_grad():
            self._W_value.copy_(self._to_tensor(value))

    @property
    def edge_weights(self):
        return self._edge_weights.detach().cpu().numpy()

    @edge_weights.setter
    def edge_weights(self, value):
        with torch.no_grad():
            self._edge_weights.copy_(self._to_tensor(value))

    @property
    def node_weights(self):
        return self._node_weights.detach().cpu().numpy()

    @node_weights.setter
    def node_weights(self, value):
        with torch.no_grad():
            self._node_weights.copy_(self._to_tensor(value))

    @property
    def state_mix_logits(self):
        return self._state_mix_logits.detach().cpu().numpy()

    @state_mix_logits.setter
    def state_mix_logits(self, value):
        with torch.no_grad():
            self._state_mix_logits.copy_(self._to_tensor(value))

    @property
    def flow_strength(self):
        return np.float32(self._flow_strength.detach().cpu().item())

    @flow_strength.setter
    def flow_strength(self, value):
        with torch.no_grad():
            self._flow_strength.copy_(self._to_tensor(value))

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        """Нормалізація вектора або матриці."""
        if x.ndim == 1:
            norm = torch.norm(x, p=2)
            if norm < 1e-10:
                return torch.zeros_like(x)
            return x / norm
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / torch.clamp(norms, min=1e-10)

    @staticmethod
    def _softmax(x: torch.Tensor, axis: int = -1, temperature: float = 1.0) -> torch.Tensor:
        """Softmax з температурним масштабуванням."""
        temp = max(float(temperature), 1e-4)
        max_val = torch.max(x, dim=axis, keepdim=True)[0]
        centered = (x - max_val) / temp
        exp_vals = torch.exp(torch.clamp(centered, min=-60.0, max=60.0))
        sum_exp = torch.sum(exp_vals, dim=axis, keepdim=True)
        return exp_vals / torch.clamp(sum_exp, min=1e-12)

    def _project_query(self, h: torch.Tensor) -> torch.Tensor:
        h_tensor = self._to_tensor(h)
        h_flat = h_tensor.view(-1)
        return self._unit(torch.tanh(torch.matmul(h_flat, self._W_query)))

    def _project_key(self, H: torch.Tensor) -> torch.Tensor:
        H_tensor = self._to_tensor(H)
        return self._unit(torch.tanh(torch.matmul(H_tensor, self._W_key)))

    def _project_value(self, H: torch.Tensor) -> torch.Tensor:
        H_tensor = self._to_tensor(H)
        return self._unit(torch.tanh(torch.matmul(H_tensor, self._W_value)))

    @staticmethod
    def _memory_strength(cand: Dict) -> float:
        return float(min(1.0, np.log1p(max(float(cand.get('n', 1)), 1.0)) / np.log1p(16.0)))

    def _select_nodes(self, candidates: List[Dict], target_ids: Optional[List[str]] = None) -> List[Dict]:
        target_set = {str(t) for t in (target_ids or [])}
        if len(candidates) <= self.max_nodes:
            return list(candidates)
        targets = [c for c in candidates if str(c.get('candidate_id')) in target_set]
        rest = [c for c in candidates if str(c.get('candidate_id')) not in target_set]
        rest.sort(key=lambda row: float(row.get('base_score', row.get('score', 0.0))), reverse=True)
        out = []
        seen = set()
        for row in targets + rest:
            cid = str(row.get('candidate_id'))
            if cid in seen:
                continue
            out.append(row)
            seen.add(cid)
            if len(out) >= self.max_nodes and target_set.issubset(seen):
                break
        return out[:self.max_nodes]

    def _node_features(self, q: torch.Tensor, K: torch.Tensor, candidates: List[Dict]) -> torch.Tensor:
        n = len(candidates)
        device = self.device
        
        # Ознака 0: схожість запиту та ключа (диференційовна)
        feat0 = torch.matmul(K, q)  # (n,)
        
        # Інші ознаки кандидатів (фіксовані метадані)
        feat1 = torch.tensor([float(c.get('base_score', c.get('score', 0.0))) for c in candidates], dtype=torch.float32, device=device)
        feat2 = torch.tensor([float(c.get('byte_overlap', 0.0)) for c in candidates], dtype=torch.float32, device=device)
        feat3 = torch.tensor([self._memory_strength(c) for c in candidates], dtype=torch.float32, device=device)
        
        return torch.stack([feat0, feat1, feat2, feat3], dim=1)

    def _edge_features(self, K: torch.Tensor, candidates: List[Dict]) -> torch.Tensor:
        n = len(candidates)
        device = self.device
        
        # Ознака 0: подібність ключів (диференційовна)
        feat0 = torch.mm(K, K.t())
        
        # Порівняння source_digest
        digests = [c.get('source_digest') for c in candidates]
        dig_arr = np.array(digests, dtype=object)
        mask_valid = (dig_arr != None)[:, None] & (dig_arr != None)[None, :]
        same_source_np = (dig_arr[:, None] == dig_arr[None, :]) & mask_valid
        same_source = torch.tensor(same_source_np, dtype=torch.float32, device=device)
        
        # Ознака 1: те саме джерело
        feat1 = same_source
        
        # Позиції та масштаби для тимчасової когерентності
        starts = torch.tensor([float(c.get('start', 0)) for c in candidates], dtype=torch.float32, device=device)
        ends = torch.tensor([float(c.get('end', c.get('start', 0) + len(c.get('raw', b'')))) for c in candidates], dtype=torch.float32, device=device)
        mids = 0.5 * (starts + ends)
        spans = torch.clamp(ends - starts, min=32.0)
        
        mid_diff = torch.abs(mids.unsqueeze(1) - mids.unsqueeze(0))
        scale = torch.max(spans.unsqueeze(1), spans.unsqueeze(0))
        
        # Ознака 2: тимчасова когерентність
        feat2 = torch.exp(-mid_diff / scale) * same_source
        
        # Ознаки 3, 4, 5: базовий бал, перекриття байтів, стабільність пам'яті цілі
        base = torch.tensor([float(c.get('base_score', c.get('score', 0.0))) for c in candidates], dtype=torch.float32, device=device)
        byte = torch.tensor([float(c.get('byte_overlap', 0.0)) for c in candidates], dtype=torch.float32, device=device)
        mem = torch.tensor([self._memory_strength(c) for c in candidates], dtype=torch.float32, device=device)
        
        feat3 = base.unsqueeze(0).expand(n, n)
        feat4 = byte.unsqueeze(0).expand(n, n)
        feat5 = mem.unsqueeze(0).expand(n, n)
        
        return torch.stack([feat0, feat1, feat2, feat3, feat4, feat5], dim=2)

    def _build(self, intent_hidden: Union[np.ndarray, torch.Tensor], candidates: List[Dict], target_ids: Optional[List[str]] = None) -> Dict:
        nodes = self._select_nodes(candidates, target_ids=target_ids)
        if not nodes:
            return {
                'nodes': [], 'H': None, 'K': None, 'V': None, 'q': None, 'A': None, 
                'node_logits': None, 'edge_logits': None, 'node_features': None, 'edge_features': None
            }
        
        hiddens = []
        for c in nodes:
            h = c['hidden']
            if isinstance(h, torch.Tensor):
                hiddens.append(h.to(self.device).view(-1))
            else:
                hiddens.append(torch.tensor(h, dtype=torch.float32, device=self.device).view(-1))
        H = self._unit(torch.stack(hiddens))
        
        q = self._project_query(intent_hidden)
        K = self._project_key(H)
        V = self._project_value(H)
        
        node_features = self._node_features(q, K, nodes)
        edge_features = self._edge_features(K, nodes)
        
        edge_logits = torch.matmul(edge_features, self._edge_weights)
        
        diag_indices = torch.arange(len(nodes), device=self.device)
        edge_logits[diag_indices, diag_indices] = edge_logits[diag_indices, diag_indices] - 0.20
        
        A = self._softmax(edge_logits, axis=1, temperature=0.82)
        
        return {
            'nodes': nodes,
            'H': H,
            'K': K,
            'V': V,
            'q': q,
            'A': A,
            'node_logits': torch.matmul(node_features, self._node_weights),
            'edge_logits': edge_logits,
            'node_features': node_features,
            'edge_features': edge_features,
        }

    def rollout(
        self,
        intent_hidden: Union[np.ndarray, torch.Tensor],
        candidates: List[Dict],
        target_ids: Optional[List[str]] = None,
        ablate_ids: Optional[List[str]] = None,
        rollout_steps: Optional[int] = None,
        top_k: int = 5,
        return_cache: bool = False,
    ) -> Dict:
        ablate_set = {str(x) for x in (ablate_ids or [])}
        usable = [c for c in candidates if str(c.get('candidate_id')) not in ablate_set]
        graph = self._build(intent_hidden, usable, target_ids=target_ids)
        nodes = graph['nodes']
        if not nodes:
            return {
                'best': None,
                'node_scores': {},
                'target_probability': torch.tensor(0.0, device=self.device),
                'target_prior_probability': torch.tensor(0.0, device=self.device),
                'target_path_probability': torch.tensor(0.0, device=self.device),
                'target_lift_over_prior': torch.tensor(0.0, device=self.device),
                'trace': [],
                'counterfactual_ready': False,
            }

        q = graph['q']
        K = graph['K']
        V = graph['V']
        A = graph['A']
        node_features = graph['node_features']
        node_logits = graph['node_logits']
        
        prior = self._softmax(node_logits, axis=0, temperature=0.82)
        p = prior.clone()
        mix = self._softmax(self._state_mix_logits, axis=0, temperature=1.0)
        steps = max(2, int(rollout_steps if rollout_steps is not None else self.rollout_steps))
        probs_over_time = [prior.clone()]
        trace = []

        for step in range(steps):
            message = torch.matmul(p, V)
            state = self._unit(mix[0] * q + mix[1] * message)
            compat = torch.matmul(K, state)
            flow = torch.matmul(p, A)
            logits = compat + self._flow_strength * torch.log(torch.clamp(flow, min=1e-12)) + node_logits
            p = self._softmax(logits, axis=0, temperature=0.74)
            probs_over_time.append(p.clone())
            
            top_idx = int(torch.argmax(p).item())
            
            # Decode raw text safely
            raw_text = nodes[top_idx].get('raw', b'')
            if isinstance(raw_text, bytes):
                decoded_text = raw_text[:180].decode('utf-8', errors='replace')
            else:
                decoded_text = str(raw_text)[:180]
                
            trace.append({
                'step': int(step),
                'candidate_id': nodes[top_idx].get('candidate_id'),
                'kind': nodes[top_idx].get('kind', 'episode'),
                'probability': float(p[top_idx].item()),
                'base_score': float(nodes[top_idx].get('base_score', nodes[top_idx].get('score', 0.0))),
                'byte_overlap': float(nodes[top_idx].get('byte_overlap', 0.0)),
                'text': decoded_text,
            })

        score_vec = torch.mean(torch.stack(probs_over_time), dim=0) if probs_over_time else p
        best_idx = int(torch.argmax(score_vec).item())
        
        target_set = {str(x) for x in (target_ids or [])}
        target_indices = [i for i, c in enumerate(nodes) if str(c.get('candidate_id')) in target_set]
        
        if target_indices:
            target_mask = torch.zeros(len(nodes), device=self.device)
            target_mask[target_indices] = 1.0
            
            target_probability = torch.sum(score_vec * target_mask)
            target_prior_probability = torch.sum(prior * target_mask)
            target_path_probability = torch.mean(torch.stack([torch.sum(pt * target_mask) for pt in probs_over_time]))
            target_lift_over_prior = target_probability / torch.clamp(target_prior_probability, min=1e-9)
        else:
            target_probability = torch.tensor(0.0, device=self.device)
            target_prior_probability = torch.tensor(0.0, device=self.device)
            target_path_probability = torch.tensor(0.0, device=self.device)
            target_lift_over_prior = torch.tensor(0.0, device=self.device)

        ranked_idx = torch.argsort(score_vec, descending=True)[:max(1, int(top_k))]
        ranked_idx_list = [idx.item() for idx in ranked_idx]
        
        node_scores = {
            str(nodes[i].get('candidate_id')): float(score_vec[i].item())
            for i in range(len(nodes))
        }
        
        result = {
            'best': nodes[best_idx],
            'best_score': float(score_vec[best_idx].item()),
            'best_probability': float(score_vec[best_idx].item()),
            'node_scores': node_scores,
            'target_probability': target_probability,
            'target_prior_probability': target_prior_probability,
            'target_path_probability': target_path_probability,
            'target_lift_over_prior': target_lift_over_prior,
            'trace': trace,
            'ranked_path': [
                {
                    'candidate_id': nodes[i].get('candidate_id'),
                    'score': float(score_vec[i].item()),
                    'kind': nodes[i].get('kind', 'episode'),
                }
                for i in ranked_idx_list
            ],
            'n_nodes': len(nodes),
            'n_steps': steps,
            'counterfactual_ready': bool(len(nodes) > 1),
        }
        
        if return_cache:
            result['cache'] = {
                'probabilities': [pt.clone() for pt in probs_over_time],
                'score_vec': score_vec.clone(),
                'prior': prior.clone(),
                'A': A.clone(),
                'node_logits': node_logits.clone(),
                'edge_logits': graph['edge_logits'].clone(),
                'node_features': node_features.clone(),
                'edge_features': graph['edge_features'].clone(),
                'nodes': nodes,
                'target_indices': list(target_indices),
            }
        return result

    def _regularize(self):
        with torch.no_grad():
            for param in self.parameters():
                param.data.nan_to_num_(nan=0.0)

            self._edge_weights.clamp_(min=-2.0, max=2.5)
            self._node_weights.clamp_(min=-2.0, max=2.5)
            self._state_mix_logits.clamp_(min=-4.0, max=4.0)
            self._flow_strength.clamp_(min=0.05, max=2.0)

            for name in ('_W_query', '_W_key', '_W_value'):
                W = getattr(self, name)
                row_norms = torch.norm(W, p=2, dim=1, keepdim=True)
                scale = torch.minimum(torch.tensor(1.0, device=W.device), 3.5 / torch.clamp(row_norms, min=1e-8))
                W.mul_(scale)

    def train(
        self,
        intent_hidden: Union[np.ndarray, torch.Tensor],
        candidates: List[Dict],
        target_ids: List[str],
        epochs: int = 12,
        lr: float = 0.035,
        counterfactual_weight: float = 0.35,
    ) -> Dict:
        target_path = [str(t) for t in target_ids if t is not None]
        
        # Розрахунок стану "до навчання"
        before = self.rollout(intent_hidden, candidates, target_ids=target_path)
        if not target_path or before.get('n_nodes', 0) == 0:
            return {
                'status': 'no_target_candidate',
                'loss_before': None,
                'loss_after': None,
                'target_probability_before': 0.0,
                'target_probability_after': 0.0,
            }

        def loss_from_rollout(result: Dict) -> torch.Tensor:
            target_p = torch.clamp(result.get('target_probability'), min=1e-12)
            path_p = torch.clamp(result.get('target_path_probability'), min=1e-12)
            prior_p = torch.clamp(result.get('target_prior_probability'), min=0.0)
            
            best_other = torch.tensor(0.0, device=self.device)
            for cid, score in result.get('node_scores', {}).items():
                if cid not in target_path:
                    best_other = torch.max(best_other, torch.tensor(score, device=self.device))
            
            margin_loss = torch.clamp(0.18 - result.get('target_probability') + best_other, min=0.0)
            prior_regression = torch.clamp(result.get('target_prior_probability') - result.get('target_probability'), min=0.0)
            
            return (
                0.58 * -torch.log(target_p)
                + 0.42 * -torch.log(path_p)
                + counterfactual_weight * margin_loss
                + 0.12 * prior_regression
            )

        before_loss = float(loss_from_rollout(before).item())

        # Цикл градієнтного спуску через PyTorch Autograd
        for epoch in range(max(1, int(epochs))):
            current = self.rollout(intent_hidden, candidates, target_ids=target_path)
            if current.get('n_nodes', 0) == 0:
                break
                
            loss = loss_from_rollout(current)
            
            self.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad
            
            self._regularize()

        # Розрахунок стану "після навчання"
        after = self.rollout(intent_hidden, candidates, target_ids=target_path)
        target_id = target_path[0] if target_path else None
        cf = self.rollout(intent_hidden, candidates, target_ids=target_path, ablate_ids=[target_id] if target_id else None)
        
        after_loss = float(loss_from_rollout(after).item())
        cf_drop = float(max(0.0, after.get('target_probability').item() - cf.get('target_probability').item()))
        
        result = {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'loss_before': before_loss,
            'loss_after': after_loss,
            'target_probability_before': float(before.get('target_probability').item()),
            'target_probability_after': float(after.get('target_probability').item()),
            'target_prior_probability_before': float(before.get('target_prior_probability').item()),
            'target_prior_probability_after': float(after.get('target_prior_probability').item()),
            'target_path_probability_before': float(before.get('target_path_probability').item()),
            'target_path_probability_after': float(after.get('target_path_probability').item()),
            'target_lift_before': float(before.get('target_lift_over_prior').item()),
            'target_lift_after': float(after.get('target_lift_over_prior').item()),
            'counterfactual_probability_after': float(cf.get('target_probability').item()),
            'counterfactual_drop': cf_drop,
            'n_nodes': int(after.get('n_nodes')),
            'n_steps': int(after.get('n_steps')),
            'trace_before': before.get('trace', []),
            'trace_after': after.get('trace', []),
            'ranked_after': after.get('ranked_path', []),
            'edge_weights': [float(x) for x in self.edge_weights],
            'node_weights': [float(x) for x in self.node_weights],
            'flow_strength': float(self.flow_strength),
        }
        self.training_history.append(result)
        return result




class SemanticStateConstructor:
    """Learned concept-state constructor for intent/evidence retrieval.

    It converts raw semantic feature vectors into role-aware latent states and
    stores learned intent -> evidence associations. Byte overlap becomes one
    evidence channel; candidate generation is driven by this learned state.
    """

    def __init__(
        self,
        d_input: int = 2048,
        d_state: int = 96,
        max_associations: int = 8192,
        seed: int = 5051,
    ):
        self.d_input = int(d_input)
        self.d_state = int(d_state)
        self.max_associations = int(max(64, max_associations))
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(max(self.d_input, 1))
        self.W_query_state = (rng.standard_normal((self.d_state, self.d_input)).astype(np.float32) * scale)
        self.W_segment_state = (rng.standard_normal((self.d_state, self.d_input)).astype(np.float32) * scale)
        self.W_context_state = (rng.standard_normal((self.d_state, self.d_state)).astype(np.float32) * 0.03)
        self.memory_gate_logit = np.float32(-0.25)
        self.associations = []
        self.training_history = []

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-10:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr / norm).astype(np.float32)

    def _resize(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(arr) >= self.d_input:
            return arr[:self.d_input].copy()
        return np.pad(arr, (0, self.d_input - len(arr))).astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        temp = max(float(temperature), 1e-4)
        centered = (arr - np.max(arr)) / temp
        exp_vals = np.exp(np.clip(centered, -60.0, 60.0))
        return (exp_vals / max(float(np.sum(exp_vals)), 1e-12)).astype(np.float32)

    def _project_query_base(self, feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._resize(feature)
        pre = np.tanh(self.W_query_state @ x)
        return x, pre.astype(np.float32), self._unit(pre)

    def _project_segment_base(self, feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._resize(feature)
        pre = np.tanh(self.W_segment_state @ x)
        return x, pre.astype(np.float32), self._unit(pre)

    def _memory_context_from_base(
        self,
        q_base: np.ndarray,
        query_feature: Optional[np.ndarray] = None,
        top_k: int = 16,
    ) -> Tuple[np.ndarray, float, List[Dict]]:
        if not self.associations:
            return np.zeros(self.d_state, dtype=np.float32), 0.0, []
        q = self._unit(q_base)
        q_feature = self._unit(self._resize(query_feature)) if query_feature is not None else None
        rows = []
        for idx, item in enumerate(self.associations):
            state_sim = float(np.dot(q, item['intent_state']))
            feature_sim = (
                float(np.dot(q_feature, item.get('intent_feature', q_feature)))
                if q_feature is not None else state_sim
            )
            pos_sim = float(0.58 * state_sim + 0.42 * feature_sim)
            neg_sim = 0.0
            for neg_state in item.get('negative_intent_states', []):
                neg_sim = max(neg_sim, float(np.dot(q, neg_state)))
            if q_feature is not None:
                for neg_feature in item.get('negative_intent_features', []):
                    neg_sim = max(neg_sim, float(np.dot(q_feature, neg_feature)))
            sim = float(pos_sim - max(0.0, neg_sim))
            rows.append((idx, sim))
        rows.sort(key=lambda row: row[1], reverse=True)
        chosen = rows[:max(1, min(int(top_k), len(rows)))]
        scores = np.array([row[1] for row in chosen], dtype=np.float32)
        probs = self._softmax(scores, temperature=0.55)
        context = np.zeros(self.d_state, dtype=np.float32)
        trace = []
        for (idx, score), prob in zip(chosen, probs):
            item = self.associations[idx]
            context += np.float32(prob * max(score, 0.0)) * item['target_state']
            trace.append({
                'association_id': item.get('association_id'),
                'target_id': item.get('target_id'),
                'similarity': float(score),
                'probability': float(prob),
                'target_state': item['target_state'].copy(),
                'n': int(item.get('n', 1)),
            })
        support = float(max(scores)) if scores.size else 0.0
        return self._unit(context), support, trace

    def query_state(self, feature: np.ndarray, return_meta: bool = False):
        _, _, q_base = self._project_query_base(feature)
        context, support, trace = self._memory_context_from_base(q_base, query_feature=feature)
        gate = float(_sigmoid(np.array([
            float(self.memory_gate_logit) + 3.0 * (support - 0.42)
        ], dtype=np.float32))[0])
        transformed_context = (
            self._unit(context + 0.20 * np.tanh(self.W_context_state @ context))
            if np.linalg.norm(context) > 1e-10 else context
        )
        state = self._unit((1.0 - gate) * q_base + gate * transformed_context)
        if return_meta:
            return state, {
                'memory_support': float(support),
                'memory_gate': float(gate),
                'memory_trace': [
                    {k: v for k, v in row.items() if k != 'target_state'}
                    for row in trace
                ],
            }
        return state

    def segment_state(self, feature: np.ndarray) -> np.ndarray:
        _, _, state = self._project_segment_base(feature)
        return state

    def candidate_score(
        self,
        query_feature: np.ndarray,
        candidate_feature: np.ndarray,
        candidate_id: Optional[str] = None,
    ) -> Dict:
        q_state, meta = self.query_state(query_feature, return_meta=True)
        s_state = self.segment_state(candidate_feature)
        direct = float(np.dot(q_state, s_state))
        assoc = 0.0
        _, _, q_base = self._project_query_base(query_feature)
        _, _, trace = self._memory_context_from_base(q_base, query_feature=query_feature)
        for row in trace:
            target_affinity = float(np.dot(s_state, row['target_state']))
            id_match = candidate_id is not None and str(row.get('target_id')) == str(candidate_id)
            support = float(row['probability']) * max(0.0, float(row['similarity'])) * max(0.0, target_affinity)
            if id_match:
                support = max(support, float(row['probability']) * max(0.0, float(row['similarity'])))
            assoc = max(assoc, support)
        return {
            'state_similarity': direct,
            'associative_support': float(assoc),
            'memory_support': float(meta['memory_support']),
            'memory_gate': float(meta['memory_gate']),
        }

    def _loss(
        self,
        q_feature: np.ndarray,
        target_feature: np.ndarray,
        negative_features: List[np.ndarray],
        negative_query_features: Optional[List[np.ndarray]] = None,
        margin: float = 0.24,
    ) -> Dict:
        negative_query_features = list(negative_query_features or [])
        q = self.query_state(q_feature)
        t = self.segment_state(target_feature)
        pos = float(np.dot(q, t))
        neg_sims = [float(np.dot(q, self.segment_state(nf))) for nf in negative_features]
        neg_query_sims = [float(np.dot(self.query_state(nq), t)) for nq in negative_query_features]
        hard = max(neg_sims) if neg_sims else -1.0
        hard_query = max(neg_query_sims) if neg_query_sims else -1.0
        hard_any = max(hard, hard_query)
        loss = max(0.0, float(margin) - pos + hard_any) if (neg_sims or neg_query_sims) else max(0.0, 1.0 - pos)
        return {
            'loss': float(loss),
            'positive_similarity': pos,
            'hardest_negative_similarity': float(hard),
            'hardest_negative_query_similarity': float(hard_query),
            'n_negatives': int(len(negative_features)),
            'n_negative_queries': int(len(negative_query_features)),
        }

    def _regularize(self):
        for name in ('W_query_state', 'W_segment_state', 'W_context_state'):
            W = getattr(self, name)
            W = np.nan_to_num(W, nan=0.0, posinf=2.0, neginf=-2.0)
            row_norms = np.linalg.norm(W, axis=1, keepdims=True)
            scale = np.minimum(1.0, 4.0 / np.maximum(row_norms, 1e-8))
            setattr(self, name, (W * scale).astype(np.float32))
        self.memory_gate_logit = np.float32(np.clip(float(self.memory_gate_logit), -4.0, 4.0))

    def _store_association(
        self,
        q_feature: np.ndarray,
        target_feature: np.ndarray,
        target_id: Optional[str] = None,
        association_id: Optional[str] = None,
        negative_query_features: Optional[List[np.ndarray]] = None,
    ):
        _, _, q_base = self._project_query_base(q_feature)
        _, _, t_base = self._project_segment_base(target_feature)
        negative_query_features = list(negative_query_features or [])
        neg_states = []
        neg_features = []
        for nqf in negative_query_features[:16]:
            _, _, nq_base = self._project_query_base(nqf)
            neg_states.append(nq_base.copy())
            neg_features.append(self._unit(self._resize(nqf)))
        association_id = str(association_id or target_id or len(self.associations))
        for item in self.associations:
            if item.get('association_id') == association_id:
                n = int(item.get('n', 1)) + 1
                item['intent_state'] = self._unit((item['intent_state'] * item.get('n', 1) + q_base) / n)
                item['target_state'] = self._unit((item['target_state'] * item.get('n', 1) + t_base) / n)
                item['intent_feature'] = self._unit((item.get('intent_feature', self._resize(q_feature)) * item.get('n', 1) + self._resize(q_feature)) / n)
                item['target_feature'] = self._unit((item.get('target_feature', self._resize(target_feature)) * item.get('n', 1) + self._resize(target_feature)) / n)
                if neg_states:
                    item['negative_intent_states'] = (item.get('negative_intent_states', []) + neg_states)[-16:]
                    item['negative_intent_features'] = (item.get('negative_intent_features', []) + neg_features)[-16:]
                item['n'] = n
                item['target_id'] = target_id
                return
        if len(self.associations) >= self.max_associations:
            weakest = min(range(len(self.associations)), key=lambda i: self.associations[i].get('n', 1))
            self.associations.pop(weakest)
        self.associations.append({
            'association_id': association_id,
            'target_id': target_id,
            'intent_feature': self._unit(self._resize(q_feature)),
            'target_feature': self._unit(self._resize(target_feature)),
            'intent_state': q_base.copy(),
            'target_state': t_base.copy(),
            'negative_intent_states': neg_states,
            'negative_intent_features': neg_features,
            'n': 1,
        })

    def train_episode(
        self,
        q_feature: np.ndarray,
        target_feature: np.ndarray,
        negative_features: Optional[List[np.ndarray]] = None,
        target_id: Optional[str] = None,
        association_id: Optional[str] = None,
        epochs: int = 10,
        lr: float = 0.04,
        margin: float = 0.24,
        negative_query_features: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        negative_features = list(negative_features or [])
        negative_query_features = list(negative_query_features or [])
        before = self._loss(q_feature, target_feature, negative_features, negative_query_features, margin=margin)
        xq = self._resize(q_feature)
        xt = self._resize(target_feature)

        xq_t = torch.from_numpy(xq).float()
        xt_t = torch.from_numpy(xt).float()
        neg_xs_t = [torch.from_numpy(self._resize(nf)).float() for nf in negative_features]
        neg_q_xs_t = [torch.from_numpy(self._resize(nqf)).float() for nqf in negative_query_features]

        for _ in range(max(1, int(epochs))):
            W_q_t = torch.from_numpy(self.W_query_state.copy()).float().requires_grad_(True)
            W_s_t = torch.from_numpy(self.W_segment_state.copy()).float().requires_grad_(True)

            # Forward pass: query and target
            q_pre = W_q_t @ xq_t
            qh = torch.tanh(q_pre)
            norm_q = torch.linalg.norm(qh)
            q = qh / norm_q if norm_q > 1e-10 else torch.zeros_like(qh)

            t_pre = W_s_t @ xt_t
            th = torch.tanh(t_pre)
            norm_t = torch.linalg.norm(th)
            t = th / norm_t if norm_t > 1e-10 else torch.zeros_like(th)

            pos_sim = torch.dot(q, t)

            loss = 0.70 * (1.0 - pos_sim)

            # Check negative features violation
            violation_val = 0.0
            if neg_xs_t:
                neg_sims = []
                for xn_t in neg_xs_t:
                    n_pre = W_s_t @ xn_t
                    nh_pre = torch.tanh(n_pre)
                    norm_n = torch.linalg.norm(nh_pre)
                    nh = nh_pre / norm_n if norm_n > 1e-10 else torch.zeros_like(nh_pre)
                    neg_sims.append(torch.dot(q, nh))
                neg_sims_t = torch.stack(neg_sims)
                hard_sim = torch.max(neg_sims_t)

                with torch.no_grad():
                    violation_val = float(margin) - float(pos_sim.item()) + float(hard_sim.item())

                if violation_val > 0.0:
                    loss = loss + 0.50 * hard_sim
            else:
                with torch.no_grad():
                    violation_val = 1.0 - float(pos_sim.item())

            # Check negative query features violation
            if neg_q_xs_t:
                nq_sims_target = []
                nq_sims_intent = []
                for xnq_t in neg_q_xs_t:
                    nq_pre = W_q_t @ xnq_t
                    nqh_pre = torch.tanh(nq_pre)
                    norm_nq = torch.linalg.norm(nqh_pre)
                    nq = nqh_pre / norm_nq if norm_nq > 1e-10 else torch.zeros_like(nqh_pre)
                    nq_sims_target.append(torch.dot(nq, t))
                    nq_sims_intent.append(torch.dot(nq, q))
                
                nq_sims_target_t = torch.stack(nq_sims_target)
                nq_sims_intent_t = torch.stack(nq_sims_intent)
                
                with torch.no_grad():
                    max_sims = torch.maximum(nq_sims_target_t, nq_sims_intent_t)
                    hard_idx = torch.argmax(max_sims)
                
                nq_target_sim = nq_sims_target_t[hard_idx]
                nq_intent_sim = nq_sims_intent_t[hard_idx]

                with torch.no_grad():
                    query_violation = float(margin) - float(pos_sim.item()) + max(float(nq_target_sim.item()), float(nq_intent_sim.item()))

                if query_violation > 0.0:
                    loss = loss + 0.45 * nq_target_sim + 0.2925 * nq_intent_sim

            loss.backward()

            with torch.no_grad():
                self.W_query_state = (W_q_t.data - lr * W_q_t.grad).numpy()
                self.W_segment_state = (W_s_t.data - lr * W_s_t.grad).numpy()

            self.memory_gate_logit += np.float32(lr * 0.05 * (1.0 if violation_val > 0.0 else 0.25))
            self._regularize()

        self._store_association(
            q_feature,
            target_feature,
            target_id=target_id,
            association_id=association_id,
            negative_query_features=negative_query_features,
        )
        after = self._loss(q_feature, target_feature, negative_features, negative_query_features, margin=margin)
        record = {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'margin': float(margin),
            'loss_before': before['loss'],
            'loss_after': after['loss'],
            'positive_similarity_before': before['positive_similarity'],
            'positive_similarity_after': after['positive_similarity'],
            'hardest_negative_before': before['hardest_negative_similarity'],
            'hardest_negative_after': after['hardest_negative_similarity'],
            'hardest_negative_query_before': before['hardest_negative_query_similarity'],
            'hardest_negative_query_after': after['hardest_negative_query_similarity'],
            'n_negatives': len(negative_features),
            'n_negative_queries': len(negative_query_features),
            'associations': len(self.associations),
            'memory_gate_logit': float(self.memory_gate_logit),
        }
        self.training_history.append(record)
        return record

    def snapshot(self) -> Dict:
        return {
            'W_query_state': self.W_query_state.copy(),
            'W_segment_state': self.W_segment_state.copy(),
            'W_context_state': self.W_context_state.copy(),
            'memory_gate_logit': np.float32(self.memory_gate_logit),
            'associations': [
                {
                    'association_id': item.get('association_id'),
                    'target_id': item.get('target_id'),
                    'intent_feature': item['intent_feature'].copy(),
                    'target_feature': item['target_feature'].copy(),
                    'intent_state': item['intent_state'].copy(),
                    'target_state': item['target_state'].copy(),
                    'negative_intent_states': [x.copy() for x in item.get('negative_intent_states', [])],
                    'negative_intent_features': [x.copy() for x in item.get('negative_intent_features', [])],
                    'n': int(item.get('n', 1)),
                }
                for item in self.associations
            ],
        }

    def restore(self, snap: Dict):
        self.W_query_state = snap['W_query_state'].copy()
        self.W_segment_state = snap['W_segment_state'].copy()
        self.W_context_state = snap['W_context_state'].copy()
        self.memory_gate_logit = np.float32(snap['memory_gate_logit'])
        self.associations = [
            {
                'association_id': item.get('association_id'),
                'target_id': item.get('target_id'),
                'intent_feature': item['intent_feature'].copy(),
                'target_feature': item['target_feature'].copy(),
                'intent_state': item['intent_state'].copy(),
                'target_state': item['target_state'].copy(),
                'negative_intent_states': [x.copy() for x in item.get('negative_intent_states', [])],
                'negative_intent_features': [x.copy() for x in item.get('negative_intent_features', [])],
                'n': int(item.get('n', 1)),
            }
            for item in snap.get('associations', [])
        ]

    def report(self) -> Dict:
        return {
            'type': 'semantic_state_constructor',
            'associations': int(len(self.associations)),
            'memory_gate_logit': float(self.memory_gate_logit),
            'training_steps': int(len(self.training_history)),
        }




class SemanticQueryReadout:
    """Query-conditioned semantic readout grounded in byte evidence.

    The readout does not parse grammar and does not contain question templates.
    It indexes byte windows from observed episodes, learns optional query-answer
    associations as experience, and answers by retrieving byte evidence through
    the same semantic feature geometry used by SemanticLatentDynamics.
    """

    def __init__(
        self,
        d_input: int = 2048,
        d_hidden: int = 256,
        segment_length: int = 160,
        stride: int = 80,
        max_segments: int = 8192,
        max_qa_memories: int = 4096,
        min_confidence: float = 0.58,
        seed: int = 2027,
    ):
        self.d_input = int(d_input)
        self.d_hidden = int(d_hidden)
        self.segment_length = int(segment_length)
        self.stride = int(stride)
        self.max_segments = int(max_segments)
        self.max_qa_memories = int(max_qa_memories)
        self.min_confidence = float(min_confidence)
        self.rng = np.random.default_rng(seed)

        self.W_query = (self.rng.standard_normal((self.d_hidden, self.d_input)).astype(np.float32)
                        / np.sqrt(max(self.d_input, 1)))
        self.W_segment = (self.rng.standard_normal((self.d_hidden, self.d_input)).astype(np.float32)
                          / np.sqrt(max(self.d_input, 1)))
        self.field_byte_coupling = np.zeros(256, dtype=np.float32)
        self.deep_training_history = []

        self.segments = []
        self.qa_memories = []
        self.byte_decoder = SemanticByteDecoder(latent_dim=self.d_hidden)
        self.state_constructor = SemanticStateConstructor(
            d_input=self.d_input,
            d_state=self.d_hidden,
            max_associations=max(self.max_segments, self.max_qa_memories),
        )
        self.evidence_scorer = AdaptiveEvidenceScorer()
        self.reasoner = SemanticAssociativeReasoner()
        self.memory_graph = SemanticDifferentiableMemoryGraph(d_hidden=self.d_hidden)
        self.calibration = AdaptiveCalibration()
        self.credit_engine = EpisodeCreditEngine()
        self.observations = 0

    @staticmethod
    def _to_bytes(data) -> bytes:
        if data is None:
            return b''
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, str):
            return data.encode('utf-8', errors='ignore')
        return bytes(data)

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(x))
        if norm < 1e-10:
            return np.zeros_like(x, dtype=np.float32)
        return (x / norm).astype(np.float32)

    def _resize(self, x: np.ndarray, dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(arr) >= dim:
            return arr[:dim].copy()
        return np.pad(arr, (0, dim - len(arr))).astype(np.float32)

    def _byte_distribution(self, raw_data: bytes) -> np.ndarray:
        raw = self._to_bytes(raw_data)
        dist = np.zeros(256, dtype=np.float32)
        if not raw:
            return dist
        arr = np.frombuffer(raw, dtype=np.uint8)
        counts = np.bincount(arr, minlength=256).astype(np.float32)
        return self._unit(counts)

    def feature_from_bytes(self, raw_data: bytes, semantic: Optional[SemanticLatentDynamics] = None) -> np.ndarray:
        raw = self._to_bytes(raw_data)
        sem = semantic if semantic is not None else SemanticLatentDynamics(d_input=self.d_input)

        parts = [
            sem._condition_block(sem._sequence_sketch(raw, dims=768), 768),
            sem._condition_block(self._byte_distribution(raw), 256),
            np.zeros(128, dtype=np.float32),
            np.zeros(128, dtype=np.float32),
            sem._condition_block(self._byte_distribution(raw), 256),
            sem._condition_block(np.array([
                np.log1p(len(raw)),
                float(len(set(raw))) / 256.0 if raw else 0.0,
                float(np.std(np.frombuffer(raw, dtype=np.uint8))) / 128.0 if raw else 0.0,
            ], dtype=np.float32), 32),
            sem._condition_block(self._byte_distribution(raw), 256),
        ]
        base = self._resize(np.concatenate(parts).astype(np.float32), self.d_input)
        semantic_z = sem.encode(base, substrate=raw)
        parts.append(sem._condition_block(semantic_z, 128))
        return self._unit(self._resize(np.concatenate(parts).astype(np.float32), self.d_input))

    def _project_query(self, x: np.ndarray) -> np.ndarray:
        legacy = self._unit(np.tanh(self.W_query @ self._resize(x, self.d_input)))
        learned = self.state_constructor.query_state(x) if self.state_constructor is not None else legacy
        return self._unit(0.42 * legacy + 0.58 * learned)

    def _project_segment(self, x: np.ndarray) -> np.ndarray:
        legacy = self._unit(np.tanh(self.W_segment @ self._resize(x, self.d_input)))
        learned = self.state_constructor.segment_state(x) if self.state_constructor is not None else legacy
        return self._unit(0.42 * legacy + 0.58 * learned)

    def _projection_loss(
        self,
        q_feature: np.ndarray,
        target_feature: np.ndarray,
        negative_features: List[np.ndarray],
        margin: float = 0.20,
    ) -> Dict:
        q_hidden = self._project_query(q_feature)
        target_hidden = self._project_segment(target_feature)
        pos = float(np.dot(q_hidden, target_hidden))
        neg_sims = [float(np.dot(q_hidden, self._project_segment(nf))) for nf in negative_features]
        hard = max(neg_sims) if neg_sims else -1.0
        loss = max(0.0, float(margin) - pos + hard) if neg_sims else max(0.0, 1.0 - pos)
        return {
            'loss': float(loss),
            'positive_similarity': pos,
            'hardest_negative_similarity': float(hard),
            'n_negatives': len(negative_features),
        }

    def _regularize_projection(self):
        for name in ('W_query', 'W_segment'):
            W = getattr(self, name)
            W = np.nan_to_num(W, nan=0.0, posinf=2.0, neginf=-2.0)
            row_norms = np.linalg.norm(W, axis=1, keepdims=True)
            scale = np.minimum(1.0, 4.0 / np.maximum(row_norms, 1e-8))
            setattr(self, name, (W * scale).astype(np.float32))

    def _refresh_feature_cache(self, semantic: Optional[SemanticLatentDynamics] = None):
        for item in self.segments:
            item['feature'] = self.feature_from_bytes(item['raw'], semantic)
            item['hidden'] = self._project_segment(item['feature'])
        for item in self.qa_memories:
            item['query_feature'] = self.feature_from_bytes(item['query_raw'], semantic)
            item['answer_feature'] = self.feature_from_bytes(item['answer_raw'], semantic)
            item['query_hidden'] = self._project_query(item['query_feature'])
            item['answer_hidden'] = self._project_segment(item['answer_feature'])

    def _refresh_projection_cache(self):
        for item in self.segments:
            item['hidden'] = self._project_segment(item['feature'])
        for item in self.qa_memories:
            item['query_hidden'] = self._project_query(item['query_feature'])
            item['answer_hidden'] = self._project_segment(item['answer_feature'])

    def _train_projection_contrast(
        self,
        q_feature: np.ndarray,
        target_feature: np.ndarray,
        negative_features: List[np.ndarray],
        epochs: int = 8,
        lr: float = 0.035,
        margin: float = 0.20,
    ) -> Dict:
        negative_features = list(negative_features or [])
        before = self._projection_loss(q_feature, target_feature, negative_features, margin=margin)
        xq = self._resize(q_feature, self.d_input)
        xt = self._resize(target_feature, self.d_input)

        xq_t = torch.from_numpy(xq).float()
        xt_t = torch.from_numpy(xt).float()
        neg_xs_t = [torch.from_numpy(self._resize(nf, self.d_input)).float() for nf in negative_features]

        for _ in range(max(1, int(epochs))):
            W_q_t = torch.from_numpy(self.W_query.copy()).float().requires_grad_(True)
            W_s_t = torch.from_numpy(self.W_segment.copy()).float().requires_grad_(True)

            # Forward pass: query and target
            q_pre = W_q_t @ xq_t
            qh_pre = torch.tanh(q_pre)
            norm_q = torch.linalg.norm(qh_pre)
            qh = qh_pre / norm_q if norm_q > 1e-10 else torch.zeros_like(qh_pre)

            t_pre = W_s_t @ xt_t
            th_pre = torch.tanh(t_pre)
            norm_t = torch.linalg.norm(th_pre)
            th = th_pre / norm_t if norm_t > 1e-10 else torch.zeros_like(th_pre)

            pos_sim = torch.dot(qh, th)

            with torch.no_grad():
                pos_sim_val = float(pos_sim.item())
                pull_scale = 0.75 if pos_sim_val < 0.45 else 0.50

            loss = pull_scale * (1.0 - pos_sim)

            # Check negative features violation
            if neg_xs_t:
                neg_sims = []
                for xn_t in neg_xs_t:
                    n_pre = W_s_t @ xn_t
                    nh_pre = torch.tanh(n_pre)
                    norm_n = torch.linalg.norm(nh_pre)
                    nh = nh_pre / norm_n if norm_n > 1e-10 else torch.zeros_like(nh_pre)
                    neg_sims.append(torch.dot(qh, nh))
                neg_sims_t = torch.stack(neg_sims)
                hard_sim = torch.max(neg_sims_t)

                with torch.no_grad():
                    violation = float(margin) - pos_sim_val + float(hard_sim.item())

                if violation > 0.0:
                    loss = loss + 0.42 * hard_sim

            loss.backward()

            with torch.no_grad():
                self.W_query = (W_q_t.data - lr * W_q_t.grad).numpy()
                self.W_segment = (W_s_t.data - lr * W_s_t.grad).numpy()

            self._regularize_projection()

        self._refresh_projection_cache()
        after = self._projection_loss(q_feature, target_feature, negative_features, margin=margin)
        return {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'margin': float(margin),
            'loss_before': before['loss'],
            'loss_after': after['loss'],
            'positive_similarity_before': before['positive_similarity'],
            'positive_similarity_after': after['positive_similarity'],
            'hardest_negative_before': before['hardest_negative_similarity'],
            'hardest_negative_after': after['hardest_negative_similarity'],
            'n_negatives': len(negative_features),
        }

    def _digest(self, raw_data: bytes, semantic: Optional[SemanticLatentDynamics] = None) -> str:
        sem = semantic if semantic is not None else SemanticLatentDynamics(d_input=self.d_input)
        return sem._deterministic_digest(self._to_bytes(raw_data))

    def _bytegram_overlap(self, query: bytes, evidence: bytes) -> float:
        q = self._to_bytes(query)
        e = self._to_bytes(evidence)
        if len(q) < 3 or len(e) < 3:
            return 0.0
        coverages = []
        for n in (3, 4, 5, 6):
            if len(q) < n or len(e) < n:
                continue
            qgrams = {q[i:i + n] for i in range(0, len(q) - n + 1)}
            egrams = {e[i:i + n] for i in range(0, len(e) - n + 1)}
            if not qgrams or not egrams:
                continue
            inter = len(qgrams & egrams)
            coverage = inter / max(len(qgrams), 1)
            precision = inter / max(len(egrams), 1)
            coverages.append(0.82 * coverage + 0.18 * np.sqrt(coverage * precision))
        return float(np.mean(coverages)) if coverages else 0.0

    def _byte_prob_distribution(self, raw_data: bytes) -> np.ndarray:
        raw = self._to_bytes(raw_data)
        dist = np.zeros(256, dtype=np.float32)
        if not raw:
            return dist
        arr = np.frombuffer(raw, dtype=np.uint8)
        dist += np.bincount(arr, minlength=256).astype(np.float32)
        return (dist / max(float(dist.sum()), 1.0)).astype(np.float32)

    def _field_alignment(self, field_system, target_raw: bytes) -> float:
        if field_system is None or not hasattr(field_system, 'active_byte_indices'):
            return 0.0
        target_dist = self._byte_prob_distribution(target_raw)
        active = np.asarray(field_system.active_byte_indices, dtype=np.int64)
        target_active = target_dist[active].astype(np.float32)
        if float(target_active.sum()) <= 1e-10:
            return 0.0
        target_active = target_active / max(float(target_active.sum()), 1e-10)
        if hasattr(field_system, 'get_concept_activation'):
            concept = field_system.get_concept_activation()
            field_active = np.mean(concept, axis=0).astype(np.float32)
        else:
            field_active = np.maximum(np.mean(field_system.Phi, axis=0), 0.0).astype(np.float32)
        field_active = field_active / max(float(field_active.sum()), 1e-10)
        return float(np.dot(self._unit(target_active), self._unit(field_active)))

    def train_field_coupling(
        self,
        intent: bytes,
        target: bytes,
        field_system=None,
        epochs: int = 4,
        lr: float = 0.05,
        relax_steps: int = 3,
    ) -> Dict:
        """Learn byte-grounded field coupling and optionally inject it into FieldSystemV6."""
        intent_raw = self._to_bytes(intent)
        target_raw = self._to_bytes(target)
        target_dist = self._byte_prob_distribution(target_raw)
        intent_dist = self._byte_prob_distribution(intent_raw)
        before = self._field_alignment(field_system, target_raw)

        for _ in range(max(1, int(epochs))):
            delta = target_dist - 0.35 * intent_dist
            delta -= float(np.mean(delta))
            self.field_byte_coupling += np.float32(lr) * delta.astype(np.float32)
            self.field_byte_coupling = np.clip(self.field_byte_coupling, -2.0, 2.0).astype(np.float32)

        applied = 0
        if field_system is not None and hasattr(field_system, 'active_byte_indices'):
            phi_orig = field_system.Phi.copy() if hasattr(field_system, 'Phi') else None
            u_orig = field_system.u.copy() if hasattr(field_system, 'u') else None
            v_orig = field_system.v.copy() if hasattr(field_system, 'v') else None
            ctx_orig = field_system.context_injection_vector.copy() if hasattr(field_system, 'context_injection_vector') else None
            kappa_orig = float(getattr(field_system, 'context_injection_kappa', 0.0))
            active = np.asarray(field_system.active_byte_indices, dtype=np.int64)
            target_active = target_dist[active].astype(np.float32)
            learned_active = self.field_byte_coupling[active].astype(np.float32)
            vector = np.maximum(target_active + 0.35 * learned_active, 0.0)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-10:
                vector = vector / norm
                if not hasattr(field_system, 'context_injection_vector'):
                    field_system.context_injection_vector = np.zeros(len(active), dtype=np.float32)
                    field_system.context_injection_kappa = 0.0
                field_system.context_injection_vector = (
                    0.70 * field_system.context_injection_vector
                    + 0.30 * vector.astype(np.float32)
                ).astype(np.float32)
                field_system.context_injection_kappa = max(
                    float(getattr(field_system, 'context_injection_kappa', 0.0)),
                    min(0.12, 0.025 + 0.012 * max(1, int(epochs))),
                )
                for _ in range(max(0, int(relax_steps))):
                    field_system.step()
                applied = 1

        after = self._field_alignment(field_system, target_raw)
        if applied and after + 1e-6 < before:
            if phi_orig is not None:
                field_system.Phi = phi_orig
            if u_orig is not None:
                field_system.u = u_orig
            if v_orig is not None:
                field_system.v = v_orig
            if ctx_orig is not None:
                field_system.context_injection_vector = ctx_orig
            field_system.context_injection_kappa = kappa_orig
            after = before
            applied = 0
        return {
            'status': 'trained',
            'epochs': int(max(1, epochs)),
            'lr': float(lr),
            'applied_to_field': bool(applied),
            'alignment_before': float(before),
            'alignment_after': float(after),
            'coupling_norm': float(np.linalg.norm(self.field_byte_coupling)),
        }

    def _iter_segments(self, raw_data: bytes) -> List[Tuple[int, int, bytes]]:
        raw = self._to_bytes(raw_data)
        if not raw:
            return []
        if len(raw) <= self.segment_length:
            return [(0, len(raw), raw)]
        out = []
        stride = max(1, self.stride)
        for start in range(0, len(raw), stride):
            end = min(len(raw), start + self.segment_length)
            if end - start < max(16, self.segment_length // 4) and out:
                break
            out.append((start, end, raw[start:end]))
            if end >= len(raw):
                break
        return out

    def _add_segment(
        self,
        raw_segment: bytes,
        semantic: Optional[SemanticLatentDynamics],
        source_digest: str,
        start: int = 0,
        end: Optional[int] = None,
        context_z: Optional[np.ndarray] = None,
        kind: str = 'episode',
    ) -> int:
        raw = self._to_bytes(raw_segment)
        if not raw:
            return -1
        digest = self._digest(raw, semantic)
        for idx, item in enumerate(self.segments):
            if item['digest'] == digest:
                item['n'] += 1
                return idx
        feature = self.feature_from_bytes(raw, semantic)
        hidden = self._project_segment(feature)
        if len(self.segments) >= self.max_segments:
            weakest = min(range(len(self.segments)), key=lambda i: self.segments[i]['n'])
            self.segments.pop(weakest)
        self.segments.append({
            'raw': raw,
            'feature': feature,
            'hidden': hidden,
            'digest': digest,
            'source_digest': source_digest,
            'start': int(start),
            'end': int(end if end is not None else start + len(raw)),
            'context_z': None if context_z is None else np.asarray(context_z, dtype=np.float32).copy(),
            'kind': kind,
            'n': 1,
        })
        self.byte_decoder.observe(raw)
        return len(self.segments) - 1

    def observe_episode(
        self,
        raw_data: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        context_z: Optional[np.ndarray] = None,
    ) -> Dict:
        raw = self._to_bytes(raw_data)
        source_digest = self._digest(raw[:4096], semantic)
        before = len(self.segments)
        for start, end, segment in self._iter_segments(raw):
            self._add_segment(segment, semantic, source_digest, start, end, context_z, kind='episode')
        self.observations += 1
        return {
            'segments_added': int(max(0, len(self.segments) - before)),
            'segments_total': int(len(self.segments)),
            'qa_memories': int(len(self.qa_memories)),
            'observations': int(self.observations),
        }

    def learn_pair(
        self,
        query: bytes,
        answer: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        context: Optional[bytes] = None,
        repetitions: int = 1,
    ) -> Dict:
        query_raw = self._to_bytes(query)
        answer_raw = self._to_bytes(answer)
        if context is not None:
            self.observe_episode(context, semantic)

        q_feature = self.feature_from_bytes(query_raw, semantic)
        a_feature = self.feature_from_bytes(answer_raw, semantic)
        q_hidden = self._project_query(q_feature)
        a_hidden = self._project_segment(a_feature)
        digest = self._digest(query_raw + b'\x00' + answer_raw, semantic)
        decoder_before = self.byte_decoder.nll(answer_raw)

        for idx, item in enumerate(self.qa_memories):
            if item['digest'] == digest:
                n = int(item['n']) + int(max(1, repetitions))
                item['query_feature'] = self._unit((item['query_feature'] * item['n'] + q_feature * repetitions) / n)
                item['answer_feature'] = self._unit((item['answer_feature'] * item['n'] + a_feature * repetitions) / n)
                item['query_hidden'] = self._unit((item['query_hidden'] * item['n'] + q_hidden * repetitions) / n)
                item['answer_hidden'] = self._unit((item['answer_hidden'] * item['n'] + a_hidden * repetitions) / n)
                item['n'] = n
                segment_idx = self._add_segment(answer_raw, semantic, digest, 0, len(answer_raw), None, kind='qa_answer')
                self.byte_decoder.observe(query_raw + b' ' + answer_raw)
                target_path = [f'qa:{idx}']
                if segment_idx >= 0:
                    target_path.append(f'segment:{segment_idx}')
                state_training = self._train_state_constructor_target(
                    q_feature,
                    a_feature,
                    query_raw,
                    target_path,
                    epochs=max(6, int(repetitions) * 4),
                    lr=0.045,
                )
                q_hidden = self._project_query(q_feature)
                evidence_training = self._train_evidence_scorer_target(
                    q_feature,
                    q_hidden,
                    query_raw,
                    target_path,
                    epochs=max(6, int(repetitions) * 4),
                    lr=0.040,
                )
                policy_training = self._train_reasoner_target(
                    q_feature,
                    q_hidden,
                    query_raw,
                    target_path,
                    epochs=max(6, int(repetitions) * 4),
                    lr=0.045,
                )
                decoder_after = self.byte_decoder.nll(answer_raw)
                return {
                    'status': 'updated',
                    'qa_memories': len(self.qa_memories),
                    'n': n,
                    'state_constructor_training': state_training,
                    'evidence_scorer_training': evidence_training,
                    'policy_training': policy_training,
                    'decoder_nll_before': decoder_before,
                    'decoder_nll_after': decoder_after,
                }

        if len(self.qa_memories) >= self.max_qa_memories:
            weakest = min(range(len(self.qa_memories)), key=lambda i: self.qa_memories[i]['n'])
            self.qa_memories.pop(weakest)
        self.qa_memories.append({
            'query_raw': query_raw,
            'answer_raw': answer_raw,
            'query_feature': q_feature,
            'answer_feature': a_feature,
            'query_hidden': q_hidden,
            'answer_hidden': a_hidden,
            'digest': digest,
            'n': int(max(1, repetitions)),
        })
        qa_idx = len(self.qa_memories) - 1
        segment_idx = self._add_segment(answer_raw, semantic, digest, 0, len(answer_raw), None, kind='qa_answer')
        self.byte_decoder.observe(query_raw + b' ' + answer_raw)
        target_path = [f'qa:{qa_idx}']
        if segment_idx >= 0:
            target_path.append(f'segment:{segment_idx}')
        state_training = self._train_state_constructor_target(
            q_feature,
            a_feature,
            query_raw,
            target_path,
            epochs=max(6, int(repetitions) * 4),
            lr=0.045,
        )
        q_hidden = self._project_query(q_feature)
        evidence_training = self._train_evidence_scorer_target(
            q_feature,
            q_hidden,
            query_raw,
            target_path,
            epochs=max(6, int(repetitions) * 4),
            lr=0.040,
        )
        policy_training = self._train_reasoner_target(
            q_feature,
            q_hidden,
            query_raw,
            target_path,
            epochs=max(6, int(repetitions) * 4),
            lr=0.045,
        )
        decoder_after = self.byte_decoder.nll(answer_raw)
        return {
            'status': 'stored',
            'qa_memories': len(self.qa_memories),
            'n': int(max(1, repetitions)),
            'state_constructor_training': state_training,
            'evidence_scorer_training': evidence_training,
            'policy_training': policy_training,
            'decoder_nll_before': decoder_before,
            'decoder_nll_after': decoder_after,
        }

    def _score_qa(self, q_feature: np.ndarray, q_hidden: np.ndarray, query_raw: bytes, item: Dict, candidate_id: Optional[str] = None) -> Dict:
        query_sim = float(np.dot(q_feature, item['query_feature']))
        legacy_q = self._unit(np.tanh(self.W_query @ self._resize(q_feature, self.d_input)))
        legacy_a = self._unit(np.tanh(self.W_segment @ self._resize(item['answer_feature'], self.d_input)))
        hidden_sim = float(np.dot(q_hidden, item['query_hidden']))
        legacy_hidden_sim = float(np.dot(legacy_q, legacy_a))
        byte_overlap = self._bytegram_overlap(query_raw, item['query_raw'] + b' ' + item['answer_raw'])
        state = self.state_constructor.candidate_score(q_feature, item['answer_feature'], candidate_id=candidate_id)
        state_sim = float(state.get('state_similarity', hidden_sim))
        assoc = float(state.get('associative_support', 0.0))
        channels = {
            'semantic_similarity': query_sim,
            'projection_similarity': hidden_sim,
            'legacy_similarity': legacy_hidden_sim,
            'semantic_state_similarity': state_sim,
            'associative_support': assoc,
            'memory_support': float(state.get('memory_support', 0.0)),
            'memory_gate': float(state.get('memory_gate', 0.0)),
            'byte_overlap': byte_overlap,
        }
        evidence = self.evidence_scorer.score('qa_memory', channels)
        score = float(evidence['score'])
        return {
            'kind': 'qa_memory',
            'score': score,
            'query_similarity': query_sim,
            'hidden_similarity': hidden_sim,
            'legacy_hidden_similarity': legacy_hidden_sim,
            'state_reliability': float(evidence['evidence_features'].get('memory_focus', 0.0)),
            'semantic_state_similarity': state_sim,
            'associative_support': assoc,
            'memory_support': float(state.get('memory_support', 0.0)),
            'memory_gate': float(state.get('memory_gate', 0.0)),
            'byte_overlap': byte_overlap,
            'evidence_role': evidence['evidence_role'],
            'evidence_channels': channels,
            'evidence_features': evidence['evidence_features'],
            'evidence_weights': evidence['evidence_weights'],
            'evidence_sharpness': evidence['evidence_sharpness'],
            'raw': item['answer_raw'],
            'n': int(item['n']),
        }

    def _score_segment(self, q_feature: np.ndarray, q_hidden: np.ndarray, query_raw: bytes, item: Dict, candidate_id: Optional[str] = None) -> Dict:
        feature_sim = float(np.dot(q_feature, item['feature']))
        legacy_q = self._unit(np.tanh(self.W_query @ self._resize(q_feature, self.d_input)))
        legacy_s = self._unit(np.tanh(self.W_segment @ self._resize(item['feature'], self.d_input)))
        hidden_sim = float(np.dot(q_hidden, item['hidden']))
        legacy_hidden_sim = float(np.dot(legacy_q, legacy_s))
        byte_overlap = self._bytegram_overlap(query_raw, item['raw'])
        state = self.state_constructor.candidate_score(q_feature, item['feature'], candidate_id=candidate_id)
        state_sim = float(state.get('state_similarity', hidden_sim))
        assoc = float(state.get('associative_support', 0.0))
        channels = {
            'semantic_similarity': feature_sim,
            'projection_similarity': hidden_sim,
            'legacy_similarity': legacy_hidden_sim,
            'semantic_state_similarity': state_sim,
            'associative_support': assoc,
            'memory_support': float(state.get('memory_support', 0.0)),
            'memory_gate': float(state.get('memory_gate', 0.0)),
            'byte_overlap': byte_overlap,
        }
        evidence = self.evidence_scorer.score(item.get('kind', 'episode'), channels)
        score = float(evidence['score'])
        return {
            'kind': item.get('kind', 'episode'),
            'score': score,
            'feature_similarity': feature_sim,
            'hidden_similarity': hidden_sim,
            'legacy_hidden_similarity': legacy_hidden_sim,
            'state_reliability': float(evidence['evidence_features'].get('memory_focus', 0.0)),
            'semantic_state_similarity': state_sim,
            'associative_support': assoc,
            'memory_support': float(state.get('memory_support', 0.0)),
            'memory_gate': float(state.get('memory_gate', 0.0)),
            'byte_overlap': byte_overlap,
            'evidence_role': evidence['evidence_role'],
            'evidence_channels': channels,
            'evidence_features': evidence['evidence_features'],
            'evidence_weights': evidence['evidence_weights'],
            'evidence_sharpness': evidence['evidence_sharpness'],
            'raw': item['raw'],
            'start': int(item.get('start', 0)),
            'end': int(item.get('end', len(item['raw']))),
            'n': int(item.get('n', 1)),
        }

    def _candidate_records(self, q_feature: np.ndarray, q_hidden: np.ndarray, query_raw: bytes) -> List[Dict]:
        candidates = []
        for idx, item in enumerate(self.qa_memories):
            candidate_id = f'qa:{idx}'
            scored = self._score_qa(q_feature, q_hidden, query_raw, item, candidate_id=candidate_id)
            candidates.append({
                **scored,
                'candidate_id': candidate_id,
                'base_score': float(scored['score']),
                'feature': item['answer_feature'],
                'hidden': item['answer_hidden'],
                'source_digest': item.get('digest'),
                'start': 0,
                'end': len(item['answer_raw']),
            })
        for idx, item in enumerate(self.segments):
            candidate_id = f'segment:{idx}'
            scored = self._score_segment(q_feature, q_hidden, query_raw, item, candidate_id=candidate_id)
            candidates.append({
                **scored,
                'candidate_id': candidate_id,
                'base_score': float(scored['score']),
                'feature': item['feature'],
                'hidden': item['hidden'],
                'source_digest': item.get('source_digest'),
                'digest': item.get('digest'),
            })
        return candidates

    def _training_candidate_pool(
        self,
        candidates: List[Dict],
        target_ids: List[str],
        max_candidates: int = 192,
    ) -> List[Dict]:
        target_set = {str(t) for t in target_ids}
        target_rows = [c for c in candidates if str(c.get('candidate_id')) in target_set]
        hard_rows = sorted(
            [c for c in candidates if str(c.get('candidate_id')) not in target_set],
            key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
            reverse=True,
        )
        pool = []
        seen = set()
        for row in target_rows + hard_rows:
            cid = str(row.get('candidate_id'))
            if cid in seen:
                continue
            pool.append(row)
            seen.add(cid)
            if len(pool) >= max_candidates and target_set.issubset(seen):
                break
        return pool

    def _train_state_constructor_target(
        self,
        q_feature: np.ndarray,
        target_feature: np.ndarray,
        query_raw: bytes,
        target_ids: List[str],
        epochs: int = 12,
        lr: float = 0.045,
        margin: float = 0.24,
        negative_query_features: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        q_hidden = self._project_query(q_feature)
        candidates = self._candidate_records(q_feature, q_hidden, query_raw)
        target_set = {str(t) for t in target_ids}
        hard_rows = sorted(
            [c for c in candidates if str(c.get('candidate_id')) not in target_set],
            key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
            reverse=True,
        )
        negative_features = [row['feature'] for row in hard_rows[:24]]
        stats = self.state_constructor.train_episode(
            q_feature,
            target_feature,
            negative_features,
            target_id=target_ids[0] if target_ids else None,
            association_id=self._digest(query_raw + b'\x08' + b'|'.join(t.encode('utf-8') for t in target_ids)),
            epochs=epochs,
            lr=lr,
            margin=margin,
            negative_query_features=negative_query_features,
        )
        self._refresh_projection_cache()
        stats['candidate_total'] = len(candidates)
        stats['candidate_pool_size'] = len(hard_rows) + len(target_ids)
        return stats

    def _train_evidence_scorer_target(
        self,
        q_feature: np.ndarray,
        q_hidden: np.ndarray,
        query_raw: bytes,
        target_ids: List[str],
        epochs: int = 10,
        lr: float = 0.035,
        margin: float = 0.18,
    ) -> Dict:
        candidates = self._candidate_records(q_feature, q_hidden, query_raw)
        pool = self._training_candidate_pool(candidates, target_ids, max_candidates=192)
        stats = self.evidence_scorer.train_candidates(
            pool,
            target_ids=target_ids,
            epochs=epochs,
            lr=lr,
            margin=margin,
        )
        stats['candidate_pool_size'] = len(pool)
        stats['candidate_total'] = len(candidates)
        if stats.get('status') == 'trained':
            self.calibration.observe('evidence_margin', max(0.0, float(stats.get('target_score_after', 0.0)) - float(stats.get('hardest_negative_after', 0.0))))
            self.calibration.observe('positive_evidence_support', stats.get('target_score_after', 0.0))
            self.calibration.observe('negative_evidence_support', stats.get('hardest_negative_after', 0.0))
        return stats

    def _train_reasoner_target(
        self,
        q_feature: np.ndarray,
        q_hidden: np.ndarray,
        query_raw: bytes,
        target_ids: List[str],
        epochs: int = 12,
        lr: float = 0.05,
    ) -> Dict:
        candidates = self._candidate_records(q_feature, q_hidden, query_raw)
        pool = self._training_candidate_pool(candidates, target_ids)
        stats = self.reasoner.train_policy(
            q_feature,
            q_hidden,
            pool,
            target_ids=target_ids,
            epochs=epochs,
            lr=lr,
            temperature=0.82,
            counterfactual_weight=0.42,
            train_state=True,
        )
        stats['candidate_pool_size'] = len(pool)
        stats['candidate_total'] = len(candidates)
        return stats

    def _train_memory_graph_target(
        self,
        q_feature: np.ndarray,
        q_hidden: np.ndarray,
        query_raw: bytes,
        target_ids: List[str],
        epochs: int = 12,
        lr: float = 0.035,
    ) -> Dict:
        candidates = self._candidate_records(q_feature, q_hidden, query_raw)
        pool = self._training_candidate_pool(
            candidates,
            target_ids,
            max_candidates=min(self.memory_graph.max_nodes, 256),
        )
        stats = self.memory_graph.train(
            q_hidden,
            pool,
            target_ids=target_ids,
            epochs=epochs,
            lr=lr,
            counterfactual_weight=0.38,
        )
        stats['candidate_pool_size'] = len(pool)
        stats['candidate_total'] = len(candidates)
        return stats

    def learn_trajectory(
        self,
        intent: bytes,
        target: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        context: Optional[bytes] = None,
        repetitions: int = 1,
        epochs: int = 16,
        lr: float = 0.055,
    ) -> Dict:
        """Train free-form intent -> evidence trajectory without grammar templates."""
        intent_raw = self._to_bytes(intent)
        target_raw = self._to_bytes(target)
        if context is not None:
            self.observe_episode(context, semantic)
        q_feature = self.feature_from_bytes(intent_raw, semantic)
        q_hidden = self._project_query(q_feature)
        decoder_before = self.byte_decoder.nll(target_raw)
        digest = self._digest(intent_raw + b'\x01' + target_raw, semantic)
        segment_idx = self._add_segment(target_raw, semantic, digest, 0, len(target_raw), None, kind='trajectory_target')
        for _ in range(max(1, int(repetitions))):
            self.byte_decoder.observe(intent_raw + b' ' + target_raw)
        target_path = [f'segment:{segment_idx}'] if segment_idx >= 0 else []
        target_feature = self.feature_from_bytes(target_raw, semantic)
        state_training = self._train_state_constructor_target(
            q_feature,
            target_feature,
            intent_raw,
            target_path,
            epochs=max(6, int(epochs)),
            lr=lr * 0.90,
        )
        q_hidden = self._project_query(q_feature)
        evidence_training = self._train_evidence_scorer_target(
            q_feature,
            q_hidden,
            intent_raw,
            target_path,
            epochs=max(6, int(epochs)),
            lr=lr * 0.80,
        )
        policy_training = self._train_reasoner_target(
            q_feature,
            q_hidden,
            intent_raw,
            target_path,
            epochs=max(1, int(epochs)),
            lr=lr,
        )
        graph_training = self._train_memory_graph_target(
            q_feature,
            q_hidden,
            intent_raw,
            target_path,
            epochs=max(4, int(epochs)),
            lr=lr * 0.75,
        )
        decoder_after = self.byte_decoder.nll(target_raw)
        after = self.respond(intent_raw, semantic=semantic, mode='auto', max_bytes=min(160, max(32, len(target_raw) + 16)), top_k=5)
        return {
            'status': 'trained',
            'target_ids': target_path,
            'segments_total': len(self.segments),
            'state_constructor_training': state_training,
            'evidence_scorer_training': evidence_training,
            'policy_training': policy_training,
            'graph_training': graph_training,
            'decoder_nll_before': decoder_before,
            'decoder_nll_after': decoder_after,
            'after_response': {
                'responded': bool(after.get('responded', False)),
                'response_kind': after.get('response_kind'),
                'confidence': float(after.get('confidence', 0.0)),
                'response_text': after.get('response_text', '')[:240],
                'counterfactual_sensitivity': float(after.get('counterfactual_sensitivity', 0.0)),
            },
        }

    def learn_grounded_end_to_end(
        self,
        intent: bytes,
        target: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        context: Optional[bytes] = None,
        field_system=None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
        max_negatives: int = 24,
    ) -> Dict:
        """Compatibility wrapper for global episode-level credit assignment."""
        return self.learn_credit_episode(
            intent,
            target,
            semantic=semantic,
            context=context,
            field_system=field_system,
            negative_probes=None,
            repetitions=repetitions,
            epochs=epochs,
            lr=lr,
            max_negatives=max_negatives,
        )

    def learn_credit_episode(
        self,
        intent: bytes,
        target: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        context: Optional[bytes] = None,
        field_system=None,
        negative_probes: Optional[List[bytes]] = None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
        max_negatives: int = 24,
    ) -> Dict:
        """Train one grounded episode through unified causal credit routing."""
        record = self.credit_engine.learn(
            self,
            intent,
            target,
            semantic=semantic,
            context=context,
            field_system=field_system,
            negative_probes=negative_probes,
            repetitions=repetitions,
            epochs=epochs,
            lr=lr,
            max_negatives=max_negatives,
        )
        self.deep_training_history.append(record)
        return record

    def _learn_grounded_end_to_end_legacy(
        self,
        intent: bytes,
        target: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        context: Optional[bytes] = None,
        field_system=None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
        max_negatives: int = 24,
    ) -> Dict:
        """Legacy sequential updater kept for comparison, not used by public API."""
        intent_raw = self._to_bytes(intent)
        target_raw = self._to_bytes(target)
        if context is not None:
            self.observe_episode(context, semantic)

        q_feature = self.feature_from_bytes(intent_raw, semantic)
        target_feature = self.feature_from_bytes(target_raw, semantic)
        q_hidden = self._project_query(q_feature)
        decoder_before = self.byte_decoder.nll(target_raw)
        conditioned_before = self.byte_decoder.conditioned_nll(target_raw, q_hidden, seed=intent_raw)
        before_response = self.respond(
            intent_raw,
            semantic=semantic,
            mode='auto',
            max_bytes=min(220, max(48, len(target_raw) + 32)),
            top_k=7,
        )

        digest = self._digest(intent_raw + b'\x02' + target_raw, semantic)
        segment_idx = self._add_segment(target_raw, semantic, digest, 0, len(target_raw), None, kind='deep_target')
        target_path = [f'segment:{segment_idx}'] if segment_idx >= 0 else []

        candidates = self._candidate_records(q_feature, q_hidden, intent_raw)
        hard_rows = sorted(
            [
                c for c in candidates
                if str(c.get('candidate_id')) not in set(target_path)
            ],
            key=lambda row: float(row.get('base_score', row.get('score', 0.0))),
            reverse=True,
        )
        negative_features = [row['feature'] for row in hard_rows[:max(1, int(max_negatives))]]

        semantic_training = None
        if semantic is not None:
            semantic_training = semantic.train_contrastive_features(
                q_feature,
                target_feature,
                negative_features,
                epochs=max(4, int(epochs) // 2),
                lr=lr * 0.65,
                margin=0.24,
            )
            self._refresh_feature_cache(semantic)
            q_feature = self.feature_from_bytes(intent_raw, semantic)
            target_feature = self.feature_from_bytes(target_raw, semantic)
            q_hidden = self._project_query(q_feature)

        projection_training = self._train_projection_contrast(
            q_feature,
            target_feature,
            negative_features,
            epochs=max(6, int(epochs)),
            lr=lr,
            margin=0.22,
        )
        self._refresh_projection_cache()
        q_hidden = self._project_query(q_feature)

        field_training = self.train_field_coupling(
            intent_raw,
            target_raw,
            field_system=field_system,
            epochs=max(3, int(epochs) // 3),
            lr=lr,
            relax_steps=4 if field_system is not None else 0,
        )

        for _ in range(max(1, int(repetitions))):
            self.byte_decoder.observe(intent_raw + b' ' + target_raw)
        conditioned_decoder_training = self.byte_decoder.observe_conditioned(
            q_hidden,
            target_raw,
            seed=intent_raw,
            epochs=max(6, int(epochs)),
            lr=lr * 0.80,
            latent_strength=0.48,
        )

        graph_training = self._train_memory_graph_target(
            q_feature,
            q_hidden,
            intent_raw,
            target_path,
            epochs=max(8, int(epochs)),
            lr=lr * 0.78,
        )

        policy_training = self._train_reasoner_target(
            q_feature,
            q_hidden,
            intent_raw,
            target_path,
            epochs=max(8, int(epochs)),
            lr=lr,
        )
        decoder_after = self.byte_decoder.nll(target_raw)
        conditioned_after = self.byte_decoder.conditioned_nll(target_raw, q_hidden, seed=intent_raw)
        after_response = self.respond(
            intent_raw,
            semantic=semantic,
            mode='auto',
            max_bytes=min(220, max(48, len(target_raw) + 32)),
            top_k=7,
        )
        negative_probe = self.respond(
            b'zxq unrelated counterfactual bytes',
            semantic=semantic,
            mode='auto',
            max_bytes=96,
            top_k=7,
        )

        record = {
            'status': 'trained',
            'target_ids': target_path,
            'segments_total': len(self.segments),
            'semantic_training': semantic_training,
            'projection_training': projection_training,
            'field_training': field_training,
            'policy_training': policy_training,
            'graph_training': graph_training,
            'conditioned_decoder_training': conditioned_decoder_training,
            'decoder_nll_before': decoder_before,
            'decoder_nll_after': decoder_after,
            'conditioned_nll_before': conditioned_before,
            'conditioned_nll_after': conditioned_after,
            'before_response': {
                'responded': bool(before_response.get('responded', False)),
                'response_kind': before_response.get('response_kind'),
                'confidence': float(before_response.get('confidence', 0.0)),
                'response_text': before_response.get('response_text', '')[:240],
                'counterfactual_sensitivity': float(before_response.get('counterfactual_sensitivity', 0.0)),
            },
            'after_response': {
                'responded': bool(after_response.get('responded', False)),
                'response_kind': after_response.get('response_kind'),
                'confidence': float(after_response.get('confidence', 0.0)),
                'response_text': after_response.get('response_text', '')[:240],
                'counterfactual_sensitivity': float(after_response.get('counterfactual_sensitivity', 0.0)),
            },
            'negative_probe': {
                'responded': bool(negative_probe.get('responded', False)),
                'confidence': float(negative_probe.get('confidence', 0.0)),
                'response_kind': negative_probe.get('response_kind'),
            },
        }
        self.deep_training_history.append(record)
        return record

    def answer(
        self,
        query: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        top_k: int = 3,
    ) -> Dict:
        query_raw = self._to_bytes(query)
        q_feature = self.feature_from_bytes(query_raw, semantic)
        q_hidden = self._project_query(q_feature)

        candidates = self._candidate_records(q_feature, q_hidden, query_raw)

        if not candidates:
            return {
                'answered': False,
                'confidence': 0.0,
                'answer_bytes': b'',
                'answer_text': '',
                'top_candidates': [],
                'reason': 'empty_memory',
            }

        candidates.sort(key=lambda x: x['base_score'], reverse=True)
        reasoning = self.reasoner.reason(q_feature, q_hidden, candidates, top_k=top_k)
        graph_reasoning = self.memory_graph.rollout(q_hidden, candidates, top_k=top_k)
        top = reasoning['best'] if reasoning.get('best') is not None else candidates[0]
        graph_top = graph_reasoning.get('best')
        graph_best_score = float(graph_reasoning.get('best_probability', 0.0))
        policy = self.calibration.answer_policy(
            self.min_confidence,
            n_candidates=len(candidates),
            n_graph_nodes=int(graph_reasoning.get('n_nodes', len(candidates))),
        )
        if graph_top is not None:
            graph_base = float(graph_top.get('base_score', graph_top.get('score', 0.0)))
            graph_byte = float(graph_top.get('byte_overlap', 0.0))
            top_base = float(top.get('base_score', top.get('score', 0.0)))
            if (
                graph_best_score >= float(policy['graph_rerank_probability'])
                and graph_base + float(policy['graph_rerank_weight']) * graph_best_score >= top_base - float(policy['answer_margin_center'])
                and max(graph_base, graph_byte) >= float(policy['graph_base_support_floor'])
            ):
                top = graph_top
        counterfactual = self.reasoner.reason(
            q_feature,
            q_hidden,
            candidates,
            ablate_id=top.get('candidate_id'),
            top_k=top_k,
        )
        graph_counterfactual = self.memory_graph.rollout(
            q_hidden,
            candidates,
            ablate_ids=[top.get('candidate_id')],
            top_k=top_k,
        )
        second_score = float(counterfactual.get('best_score', -1.0))
        support_base = float(reasoning.get('support_base_score', top.get('base_score', top.get('score', 0.0))))
        reason_score = float(max(
            float(reasoning.get('best_score', 0.0)),
            0.65 * graph_best_score + 0.35 * float(top.get('base_score', top.get('score', 0.0))),
        ))
        top_byte_support = float(top.get('byte_overlap', 0.0))
        evidence_support = float(max(support_base, float(top.get('base_score', top.get('score', 0.0)))))
        support_gate = float(_sigmoid(np.array([
            (evidence_support - float(policy['evidence_support_center']))
            * float(policy['support_gate_slope'])
        ], dtype=np.float32))[0])
        byte_gate = float(_sigmoid(np.array([
            (top_byte_support - float(policy['byte_support_center']))
            * float(policy['byte_gate_slope'])
        ], dtype=np.float32))[0])
        evidence_gate = float(
            float(policy['evidence_gate_floor'])
            + float(policy['evidence_gate_gain']) * max(support_gate, byte_gate)
        )
        evidence_gate = float(np.clip(evidence_gate, 0.0, 1.0))
        top_conf_score = (
            float(policy['support_conf_weight']) * support_base
            + float(policy['reason_conf_weight']) * reason_score
        )
        score_conf = float(_sigmoid(np.array([
            (top_conf_score - float(policy['answer_score_center']))
            * float(policy['score_gate_slope'])
        ], dtype=np.float32))[0])
        margin_conf = float(_sigmoid(np.array([
            ((top['score'] - second_score) - float(policy['answer_margin_center']))
            * float(policy['margin_gate_slope'])
        ], dtype=np.float32))[0])
        confidence = float(
            score_conf
            * (float(policy['margin_conf_base']) + float(policy['margin_conf_gain']) * margin_conf)
            * evidence_gate
        )
        answered = bool(confidence >= float(policy['answer_confidence_threshold']))

        def summarize(c):
            raw = c['raw']
            return {
                'kind': c['kind'],
                'score': float(c['score']),
                'byte_overlap': float(c.get('byte_overlap', 0.0)),
                'semantic_state_similarity': float(c.get('semantic_state_similarity', 0.0)),
                'associative_support': float(c.get('associative_support', 0.0)),
                'evidence_role': c.get('evidence_role'),
                'evidence_sharpness': float(c.get('evidence_sharpness', 0.0)),
                'evidence_weights': c.get('evidence_weights', {}),
                'text': raw[:240].decode('utf-8', errors='replace'),
                'start': int(c.get('start', 0)),
                'end': int(c.get('end', len(raw))),
                'n': int(c.get('n', 1)),
            }

        answer_bytes = top['raw'] if answered else b''
        cf_sensitivity = float(max(0.0, float(reasoning.get('best_score', top.get('score', 0.0))) - second_score))
        self.calibration.observe_many({
            'answer_confidence': confidence,
            'evidence_support': evidence_support,
            'byte_support': top_byte_support,
            'answer_score': top_conf_score,
            'answer_margin': float(top['score'] - second_score),
            'graph_support': graph_best_score,
            'graph_base_support': float(top.get('base_score', top.get('score', 0.0))),
        })
        if not answered:
            self.calibration.observe_negative_probe(confidence)
        return {
            'answered': answered,
            'confidence': confidence,
            'uncertainty': float(1.0 - confidence),
            'counterfactual_sensitivity': cf_sensitivity,
            'score': float(top.get('base_score', top.get('score', 0.0))),
            'reasoning_score': float(reasoning.get('best_score', top.get('score', 0.0))),
            'answer_bytes': answer_bytes,
            'answer_text': answer_bytes.decode('utf-8', errors='replace'),
            'best_candidate': summarize(top),
            'top_candidates': [summarize(c) for c in candidates[:top_k]],
            'reasoning': {
                'n_steps': int(reasoning.get('n_steps', 0)),
                'trace': reasoning.get('trace', []),
                'ranked_path': reasoning.get('ranked_path', []),
                'support_base_score': float(reasoning.get('support_base_score', -1.0)),
                'evidence_support': evidence_support,
                'evidence_gate': evidence_gate,
                'byte_gate': byte_gate,
                'support_gate': support_gate,
                'calibration_policy': policy,
                'counterfactual_best_score': float(second_score),
                'graph': {
                    'n_nodes': int(graph_reasoning.get('n_nodes', 0)),
                    'n_steps': int(graph_reasoning.get('n_steps', 0)),
                    'best_probability': graph_best_score,
                    'target_probability': float(graph_reasoning.get('target_probability', 0.0)),
                    'counterfactual_best_probability': float(graph_counterfactual.get('best_probability', 0.0)),
                    'trace': graph_reasoning.get('trace', []),
                    'ranked_path': graph_reasoning.get('ranked_path', []),
                },
            },
            'memory_counts': {
                'segments': len(self.segments),
                'qa_memories': len(self.qa_memories),
                'state_associations': len(self.state_constructor.associations),
            },
            'state_constructor': self.state_constructor.report(),
            'evidence_scorer': self.evidence_scorer.report(),
        }

    def respond(
        self,
        intent: bytes,
        semantic: Optional[SemanticLatentDynamics] = None,
        mode: str = 'auto',
        max_bytes: int = 160,
        top_k: int = 3,
    ) -> Dict:
        """Free-form response to any byte intent, not only question-answer pairs."""
        retrieval = self.answer(intent, semantic=semantic, top_k=top_k)
        mode = str(mode or 'auto').lower()
        top = retrieval.get('best_candidate') or (retrieval['top_candidates'][0] if retrieval.get('top_candidates') else None)
        evidence_text = top.get('text', '') if top else ''
        evidence_bytes = evidence_text.encode('utf-8', errors='ignore')

        if retrieval['answered'] and mode != 'generate':
            kind = 'learned_answer' if top and top.get('kind') == 'qa_memory' else 'evidence'
            return {
                'responded': True,
                'response_kind': kind,
                'response_bytes': retrieval['answer_bytes'],
                'response_text': retrieval['answer_text'],
                'confidence': retrieval['confidence'],
                'uncertainty': retrieval['uncertainty'],
                'counterfactual_sensitivity': retrieval['counterfactual_sensitivity'],
                'retrieval': retrieval,
            }

        retrieval_score = float(retrieval.get('score', -1.0))
        top_score = float(top.get('score', retrieval_score)) if top is not None else retrieval_score
        support_score = float(retrieval.get('reasoning', {}).get('support_base_score', -1.0))
        graph_report = retrieval.get('reasoning', {}).get('graph', {})
        graph_support = float(max(
            graph_report.get('best_probability', 0.0),
            graph_report.get('target_probability', 0.0),
        ))
        graph_cf_sensitivity = float(max(
            0.0,
            graph_report.get('best_probability', 0.0)
            - graph_report.get('counterfactual_best_probability', 0.0),
        ))
        n_graph_nodes = max(int(graph_report.get('n_nodes', 0)), 1)
        gen_policy = self.calibration.generation_policy(self.min_confidence, n_graph_nodes=n_graph_nodes)
        graph_floor = max(
            float(gen_policy['graph_probability_floor']),
            float(gen_policy['graph_sparse_floor_scale']) / n_graph_nodes,
        )
        evidence_support = max(retrieval_score, top_score, support_score)
        retrieval_confidence = float(retrieval.get('confidence', 0.0))
        top_byte_support = float(top.get('byte_overlap', 0.0)) if top is not None else 0.0
        retrieval_policy = retrieval.get('reasoning', {}).get('calibration_policy', {})
        byte_grounded_ok = (
            bool(retrieval['answered'])
            or top_byte_support >= float(retrieval_policy.get('byte_support_center', 0.06))
        )
        evidence_ok = evidence_support >= float(gen_policy['evidence_support_floor']) or retrieval['answered']
        retrieval_confidence_ok = (
            retrieval_confidence >= float(gen_policy['generation_confidence_floor']) * 0.75
            or evidence_support >= float(gen_policy['evidence_override_floor'])
            or retrieval['answered']
        )
        graph_ok = (
            graph_support >= graph_floor
            or evidence_support >= float(gen_policy['evidence_override_floor'])
            or retrieval['answered']
        )
        counterfactual_ok = graph_cf_sensitivity >= 0.0 or retrieval.get('counterfactual_sensitivity', 0.0) >= 0.0
        generation_allowed = evidence_ok and retrieval_confidence_ok and graph_ok and counterfactual_ok and byte_grounded_ok
        if mode in ('auto', 'generate', 'free', 'freeform') and top is not None and generation_allowed:
            q_feature = self.feature_from_bytes(intent, semantic)
            q_hidden = self._project_query(q_feature)
            generated = self.byte_decoder.generate(
                intent,
                evidence=evidence_bytes,
                max_bytes=max_bytes,
                condition=q_hidden,
                latent_strength=0.34,
            )
            confidence = float(min(
                1.0,
                float(gen_policy['retrieval_weight']) * retrieval.get('confidence', 0.0)
                + float(gen_policy['score_weight']) * max(0.0, retrieval_score)
                + float(gen_policy['decoder_weight']) * generated['mean_confidence'],
            ))
            decoder_ok = float(generated['mean_confidence']) >= float(gen_policy['decoder_confidence_floor'])
            accepted_generation = (
                confidence >= float(gen_policy['generation_confidence_floor'])
                and evidence_ok
                and retrieval_confidence_ok
                and graph_ok
                and decoder_ok
                and counterfactual_ok
                and byte_grounded_ok
            )
            self.calibration.observe_generation(
                evidence_support,
                graph_support,
                float(generated['mean_confidence']),
                accepted_generation,
            )
            if (
                accepted_generation
            ):
                response_bytes = generated['bytes']
                return {
                    'responded': True,
                    'response_kind': 'generated',
                    'response_bytes': response_bytes,
                    'response_text': response_bytes.decode('utf-8', errors='replace'),
                    'confidence': confidence,
                    'uncertainty': float(1.0 - confidence),
                    'counterfactual_sensitivity': retrieval.get('counterfactual_sensitivity', 0.0),
                    'retrieval': retrieval,
                    'decoder': {
                        'mean_confidence': generated['mean_confidence'],
                        'contexts': generated['contexts'],
                        'observed_bytes': generated['observed_bytes'],
                        'acceptance': {
                            'evidence_support': float(evidence_support),
                            'byte_support': float(top_byte_support),
                            'byte_grounded_ok': bool(byte_grounded_ok),
                            'retrieval_confidence': float(retrieval_confidence),
                            'retrieval_confidence_ok': bool(retrieval_confidence_ok),
                            'graph_support': float(graph_support),
                            'graph_floor': float(graph_floor),
                            'counterfactual_sensitivity': float(graph_cf_sensitivity),
                            'decoder_ok': bool(decoder_ok),
                            'calibration_policy': gen_policy,
                        },
                    },
                }

        return {
            'responded': False,
            'response_kind': 'uncertain',
            'response_bytes': b'',
            'response_text': '',
            'confidence': retrieval.get('confidence', 0.0),
            'uncertainty': float(1.0 - retrieval.get('confidence', 0.0)),
            'counterfactual_sensitivity': retrieval.get('counterfactual_sensitivity', 0.0),
            'retrieval': retrieval,
        }



