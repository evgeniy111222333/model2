import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from bcs.utils import _kl_divergence, _js_divergence, _safe_normalize


class ConversionLayersV3:
    """
    Шари адаптивної конвертації V3 — Рівняння (23) концепції.

    Кожен конвертаційний шар ℓ визначається трійкою C_ℓ = (K_ℓ, T_ℓ, R_ℓ):
    - K_ℓ — множина когнітивних кластерів рівня ℓ
    - T_ℓ : K_ℓ → R^{d_ℓ} — функція перетворення (Рівняння 23)
    - R_ℓ : R^{d_ℓ} × R^{d_ℓ+1} → R — функція релевантності

    Рівняння (23):
    T_ℓ(C) = σ(W_ℓ · Σ_{C'∈N(C)} R(C,C')/|N(C)| · T_{ℓ-1}(C') + b_ℓ)

    Де:
    - W_ℓ ∈ R^{d_ℓ × d_{ℓ-1}} — матриця перетворення, що навчається
    - R(C,C') — функція релевантності (на основі KL-дивергенції)
    - N(C) — сусідні кластери у графі взаємодії
    - σ — функція активації (tanh)
    - b_ℓ ∈ R^{d_ℓ} — зсув

    Рівень 0 (Рівняння 22): T_0(C) = p_C ∈ R^{256}
    """

    def __init__(self, n_levels: int = 4, merge_threshold: float = 0.2,
                 d_representation: int = 64, learning_rate: float = 0.001):
        self.n_levels = n_levels
        self.merge_threshold = merge_threshold
        self.d_representation = d_representation
        self.learning_rate = learning_rate

        # Навчальні параметри для кожного рівня: W_ℓ та b_ℓ (Рівняння 23)
        self.W_transform = []   # W_ℓ ∈ R^{d_ℓ × d_{ℓ-1}}
        self.b_transform = []   # b_ℓ ∈ R^{d_ℓ}

        # Ініціалізація параметрів (відкладена до першого convert())
        self._initialized = False

    def _initialize_params(self, d_input: int):
        """Ініціалізувати W_ℓ та b_ℓ для кожного рівня."""
        d_current = d_input
        for level in range(self.n_levels):
            d_out = max(self.d_representation // (1 + level // 2), 16)
            self.W_transform.append(
                np.random.randn(d_out, d_current).astype(np.float32) * 0.05
            )
            self.b_transform.append(
                np.zeros(d_out, dtype=np.float32)
            )
            d_current = d_out
        self._initialized = True

    def _compute_relevance(self, dist_i: np.ndarray, dist_j: np.ndarray) -> float:
        """
        Функція релевантності R(C, C') — на основі KL-дивергенції.

        R(C, C') = exp(-KL(p_C || p_{C'}))

        Чим схожіші кластери, тим вища релевантність.
        """
        kl = _kl_divergence(dist_i, dist_j)
        return float(np.exp(-kl))

    def _build_cluster_graph(self, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Побудувати граф сусідства кластерів для message passing.

        Ребра: просторова суміжність + семантична спорідненість.
        """
        n = len(items)
        if n <= 1:
            return np.zeros((1, 1)), np.ones((1, 1))

        adjacency = np.zeros((n, n), dtype=np.float32)
        relevance = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Просторова суміжність
                spatial_adj = (
                    items[i]['cluster']['end'] == items[j]['cluster']['start'] or
                    items[j]['cluster']['end'] == items[i]['cluster']['start']
                )
                # Семантична спорідненість через R(C,C')
                r = self._compute_relevance(
                    items[i]['representation'], items[j]['representation']
                )
                if spatial_adj:
                    adjacency[i, j] = 1.0
                    relevance[i, j] = r
                elif r > 0.5:  # Висока семантична спорідненість
                    adjacency[i, j] = 0.5
                    relevance[i, j] = r * 0.5

        return adjacency, relevance

    def _apply_transform(self, items: List[Dict], level: int) -> List[Dict]:
        """
        Застосувати Рівняння (23): T_ℓ(C) = σ(W_ℓ · Σ R(C,C')/|N(C)| · T_{ℓ-1}(C') + b_ℓ)

        Message passing від сусідів з нормалізацією та навчальним перетворенням.
        """
        if level >= len(self.W_transform):
            return items

        n = len(items)
        if n == 0:
            return items

        W = self.W_transform[level]
        b = self.b_transform[level]
        d_in = W.shape[1]
        d_out = W.shape[0]

        # Побудова графа для message passing
        adjacency, relevance = self._build_cluster_graph(items)

        # Для кожного кластера обчислюємо Рівняння (23)
        new_reps = []
        for i in range(n):
            # Збираємо повідомлення від сусідів
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) == 0:
                # Немає сусідів — використовуємо власне представлення
                rep = items[i]['representation'][:d_in]
                if len(rep) < d_in:
                    rep = np.pad(rep, (0, d_in - len(rep)))
                transformed = np.tanh(W @ rep + b)
            else:
                # Σ R(C,C')/|N(C)| · T_{ℓ-1}(C')
                weighted_sum = np.zeros(d_in, dtype=np.float32)
                total_weight = 0.0
                for j in neighbors:
                    rep_j = items[j]['representation'][:d_in]
                    if len(rep_j) < d_in:
                        rep_j = np.pad(rep_j, (0, d_in - len(rep_j)))
                    r_ij = relevance[i, j]
                    weighted_sum += r_ij * rep_j
                    total_weight += r_ij
                if total_weight > 0:
                    weighted_sum /= total_weight

                # σ(W_ℓ · aggregated_message + b_ℓ)
                transformed = np.tanh(W @ weighted_sum + b)

            new_reps.append(transformed)

        return new_reps

    def convert(self, clusters: List[Dict], substrate) -> List[Dict]:
        """
        Ієрархічна конвертація кластерів згідно Рівняння (23).

        Рівень 0: T_0(C) = p_C ∈ R^{256}  (Рівняння 22)
        Рівень ℓ>0: T_ℓ(C) = σ(W_ℓ · Σ R(C,C')/|N(C)| · T_{ℓ-1}(C') + b_ℓ)
        """
        all_levels = []

        # Рівень 0: базові кластери — T_0(C) = p_C (Рівняння 22)
        level0 = []
        for i, c in enumerate(clusters):
            level0.append({
                'id': f'L0_C{i}',
                'cluster': c,
                'representation': c['distribution'].copy(),  # T_0(C) = p_C
                'level': 0,
                'source_ids': [f'L0_C{i}'],
            })

        all_levels.append({
            'level': 0,
            'items': level0,
            'n_clusters': len(level0),
        })

        # Ініціалізація W_ℓ, b_ℓ при першому виклику
        if not self._initialized:
            self._initialize_params(256)

        # Вищі рівні: злиття + Рівняння (23)
        current_items = level0
        for level_idx in range(1, self.n_levels):
            if len(current_items) <= 1:
                break

            # Крок 1: Злиття схожих суміжних кластерів (ієрархічна агрегація)
            new_items = self._merge_level(current_items, level_idx)

            if len(new_items) >= len(current_items):
                break

            # Крок 2: Застосування Рівняння (23) — навчальне перетворення
            new_reps = self._apply_transform(new_items, level_idx)

            # Оновлюємо представлення
            for i, item in enumerate(new_items):
                if i < len(new_reps):
                    item['representation'] = new_reps[i].astype(np.float32)

            all_levels.append({
                'level': level_idx,
                'items': new_items,
                'n_clusters': len(new_items),
            })
            current_items = new_items

        return all_levels

    def update_weights(self, free_energy_grad: np.ndarray, level: int):
        """
        Оновити W_ℓ та b_ℓ через градієнт вільної енергії.

        W_ℓ ← W_ℓ - η · ∂F_free/∂W_ℓ
        b_ℓ ← b_ℓ - η · ∂F_free/∂b_ℓ
        """
        if level >= len(self.W_transform):
            return
        grad_size = self.W_transform[level].size
        if len(free_energy_grad) >= grad_size:
            grad_W = free_energy_grad[:grad_size].reshape(self.W_transform[level].shape)
            self.W_transform[level] -= self.learning_rate * grad_W
        if len(free_energy_grad) > grad_size:
            grad_b_size = self.b_transform[level].size
            self.b_transform[level] -= self.learning_rate * free_energy_grad[grad_size:grad_size + grad_b_size]

    def _merge_level(self, items: List[Dict], level: int) -> List[Dict]:
        """Злиття схожих суміжних кластерів на основі JS-дивергенції."""
        n = len(items)
        if n <= 1:
            return items

        js_values = []
        for i in range(n - 1):
            rep_i = items[i]['representation']
            rep_j = items[i + 1]['representation']
            js = _js_divergence(rep_i, rep_j)
            js_values.append(js)

        threshold = self.merge_threshold * (1.5 ** level)

        merged = [False] * n
        new_items = []

        i = 0
        while i < n:
            if merged[i]:
                i += 1
                continue

            chain = [i]
            j = i + 1
            while j < n and js_values[j - 1] < threshold:
                chain.append(j)
                merged[j] = True
                j += 1

            merged[i] = True

            if len(chain) > 1:
                combined_dist = np.zeros(256, dtype=np.float32)
                total_size = 0
                all_positions = []
                source_ids = []

                for idx in chain:
                    c = items[idx]['cluster']
                    w = c['size']
                    combined_dist += w * items[idx]['representation'][:256] if len(items[idx]['representation']) >= 256 else w * np.pad(items[idx]['representation'], (0, 256 - len(items[idx]['representation'])))[:256]
                    total_size += w
                    all_positions.extend(c['positions'].tolist())
                    source_ids.extend(items[idx].get('source_ids', [items[idx]['id']]))

                combined_dist /= max(total_size, 1)
                combined_dist = np.maximum(combined_dist, 1e-10)
                combined_dist /= combined_dist.sum()

                new_items.append({
                    'id': f'L{level}_C{len(new_items)}',
                    'cluster': {
                        'positions': np.array(sorted(set(all_positions))),
                        'start': min(items[idx]['cluster']['start'] for idx in chain),
                        'end': max(items[idx]['cluster']['end'] for idx in chain),
                        'size': total_size,
                        'distribution': combined_dist,
                        'mean_u': float(np.mean([items[idx]['cluster']['mean_u'] for idx in chain])),
                        'std_u': float(np.mean([items[idx]['cluster']['std_u'] for idx in chain])),
                        'mean_v': float(np.mean([items[idx]['cluster']['mean_v'] for idx in chain])),
                        'std_v': float(np.mean([items[idx]['cluster']['std_v'] for idx in chain])),
                        'dominant_bytes': items[chain[len(chain) // 2]]['cluster'].get('dominant_bytes', []),
                    },
                    'representation': combined_dist,
                    'level': level,
                    'source_ids': source_ids,
                })
            else:
                new_items.append(items[i].copy())

            i = j if j > i + 1 else i + 1

        return new_items



class DifferentiableLinear(nn.Linear):
    """Wrapper over nn.Linear that adds a .copy() method returning weight matrix as numpy array,
    preserving compatibility with legacy verification/profiling scripts."""
    def copy(self) -> np.ndarray:
        return self.weight.detach().cpu().numpy().copy()


class GNNConversionLayers(nn.Module):
    """
    GNN-конвертаційні шари з message passing.

    Рівняння (23): T_ℓ(C) = σ(W^(ℓ) Σ_{C'∈N(C)} R(C,C') · T_{ℓ-1}(C') + b_ℓ)

    Реалізація message passing між кластерами на графі,
    де ребра визначаються просторовою суміжністю та семантичною спорідненістю.

    Відповідність концепції:
    - Розділ 5: Шари конвертації
    - Рівняння (20-21): Ієрархічна агрегація
    - Рівняння (23): Графова нейронна мережа
    """

    def __init__(
        self,
        n_levels: int = 4,
        d_base: int = 256,
        d_hidden: int = 64,
        n_message_passes: int = 2,
        merge_threshold: float = 0.15,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.d_base = d_base
        self.d_hidden = d_hidden
        self.n_message_passes = n_message_passes
        self.merge_threshold = merge_threshold
        self.learning_rate = learning_rate

        # Параметри GNN для кожного рівня (зареєстровані в PyTorch)
        self.W_message = nn.ModuleList()
        self.W_update = nn.ModuleList()
        self.W_readout = nn.ModuleList()

        d_in = d_base
        for level in range(n_levels):
            d_out = max(d_hidden // (1 + level // 2), 16)

            # Message function: [h_i || h_j] → message
            msg_layer = DifferentiableLinear(2 * d_in, d_out)
            nn.init.normal_(msg_layer.weight, mean=0.0, std=0.05)
            nn.init.zeros_(msg_layer.bias)
            self.W_message.append(msg_layer)

            # Update function: [h_i || aggregated_messages] → h_i'
            upd_layer = DifferentiableLinear(d_in + d_out, d_out)
            nn.init.normal_(upd_layer.weight, mean=0.0, std=0.05)
            nn.init.zeros_(upd_layer.bias)
            self.W_update.append(upd_layer)

            # Readout: h_i → representation
            readout_layer = DifferentiableLinear(d_in, d_out, bias=False)
            nn.init.normal_(readout_layer.weight, mean=0.0, std=0.05)
            self.W_readout.append(readout_layer)

            d_in = d_out

    @property
    def device(self) -> torch.device:
        """Повернути пристрій, на якому розташовані параметри."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def build_cluster_graph(
        self,
        clusters: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Побудувати граф кластерів у PyTorch.

        Вузли = кластери, ребра = просторова суміжність + семантична спорідненість.

        Returns:
            adjacency: (n, n) матриця суміжності (Tensor)
            edge_features: (n, n) матриця ваг ребер (Tensor)
        """
        n = len(clusters)
        device = self.device
        if n == 0:
            return torch.zeros((0, 0), device=device), torch.zeros((0, 0), device=device)

        # Збираємо розподіли
        dists = []
        for c in clusters:
            dist = c['distribution']
            if isinstance(dist, torch.Tensor):
                dists.append(dist.to(device))
            else:
                dists.append(torch.tensor(dist, dtype=torch.float32, device=device))
        P = torch.stack(dists)

        # Векторизоване обчислення JS-дивергенції
        p1 = P.unsqueeze(1)
        p2 = P.unsqueeze(0)
        p1 = torch.clamp(p1, min=1e-10)
        p2 = torch.clamp(p2, min=1e-10)
        m = 0.5 * (p1 + p2)
        kl_1 = p1 * torch.log(p1 / m)
        kl_2 = p2 * torch.log(p2 / m)
        js_matrix = 0.5 * (kl_1.sum(dim=-1) + kl_2.sum(dim=-1))

        # Просторова суміжність: кластери, що дотикаються в послідовності
        starts = torch.tensor([c['start'] for c in clusters], device=device, dtype=torch.float32)
        ends = torch.tensor([c['end'] for c in clusters], device=device, dtype=torch.float32)
        spatial_adjacent = (ends.unsqueeze(1) == starts.unsqueeze(0)) | (starts.unsqueeze(1) == ends.unsqueeze(0))
        spatial_adjacent.fill_diagonal_(False)

        # Матриці суміжності та ребер
        adjacency = torch.zeros((n, n), device=device, dtype=torch.float32)
        edge_features = torch.zeros((n, n), device=device, dtype=torch.float32)

        # Просторові зв'язки
        adjacency = torch.where(spatial_adjacent, torch.tensor(1.0, device=device), adjacency)
        edge_features = torch.where(spatial_adjacent, 1.0 / (1.0 + js_matrix), edge_features)

        # Semantics (for spatially non-adjacent, where JS-divergence < threshold)
        semantic_mask = ~spatial_adjacent & (js_matrix < self.merge_threshold)
        semantic_mask.fill_diagonal_(False)
        adjacency = torch.where(semantic_mask, torch.tensor(0.5, device=device), adjacency)
        edge_features = torch.where(semantic_mask, 0.3 / (1.0 + js_matrix), edge_features)

        return adjacency, edge_features

    def message_passing_step(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """
        Один крок message passing у PyTorch.

        m_i = Σ_{j∈N(i)} W_msg · [h_i || h_j] · e_ij
        h_i' = σ(W_upd · [h_i || m_i] + b)
        """
        n = node_features.shape[0]
        if n == 0:
            return node_features

        level_idx = min(level, len(self.W_message) - 1)
        msg_layer = self.W_message[level_idx]
        upd_layer = self.W_update[level_idx]

        d_in = msg_layer.weight.shape[1] // 2
        d_out = msg_layer.weight.shape[0]

        # Slicing or padding of node_features to expected d_in
        if node_features.shape[1] > d_in:
            node_features = node_features[:, :d_in]
        elif node_features.shape[1] < d_in:
            node_features = F.pad(node_features, (0, d_in - node_features.shape[1]))

        # Split msg weight/bias to compute messages in vectorized form
        W_self = msg_layer.weight[:, :d_in]
        W_neigh = msg_layer.weight[:, d_in:]
        b_msg = msg_layer.bias

        # Linear projections
        node_self = F.linear(node_features, W_self, b_msg)  # (n, d_out)
        node_neigh = F.linear(node_features, W_neigh, None)  # (n, d_out)

        # Degrees
        deg = (adjacency > 0).sum(dim=1, keepdim=True)
        deg_clamped = torch.clamp(deg, min=1.0)

        # Weighted aggregate from neighbors
        S = edge_features.sum(dim=1, keepdim=True)
        aggregated_messages = (S * node_self + torch.mm(edge_features, node_neigh)) / deg_clamped

        # Update node representations
        update_input = torch.cat([node_features, aggregated_messages], dim=1)
        d_upd_in = upd_layer.weight.shape[1]
        if update_input.shape[1] > d_upd_in:
            update_input = update_input[:, :d_upd_in]
        elif update_input.shape[1] < d_upd_in:
            update_input = F.pad(update_input, (0, d_upd_in - update_input.shape[1]))

        new_features = torch.tanh(upd_layer(update_input))
        return new_features

    def convert(
        self,
        clusters: List[Dict],
        substrate=None,
    ) -> List[Dict]:
        """
        Ієрархічна конвертація з GNN message passing у PyTorch.
        """
        if len(clusters) == 0:
            return []

        all_levels = []

        # Рівень 0: базові представлення кластерів
        level0_items = []
        for i, c in enumerate(clusters):
            dist = c['distribution']
            if isinstance(dist, torch.Tensor):
                dist_tensor = dist.to(self.device)
            else:
                dist_tensor = torch.tensor(dist, dtype=torch.float32, device=self.device)

            level0_items.append({
                'id': f'L0_C{i}',
                'cluster': c,
                'representation': dist_tensor,
                'level': 0,
                'source_ids': [f'L0_C{i}'],
            })

        all_levels.append({
            'level': 0,
            'items': level0_items,
            'n_clusters': len(level0_items),
        })

        # Вищі рівні з GNN
        current_items = level0_items
        for level_idx in range(1, self.n_levels):
            if len(current_items) <= 1:
                break

            new_items = self._gnn_aggregate(current_items, level_idx)

            # Перевіряємо зменшення кількості
            if len(new_items) >= len(current_items):
                break

            all_levels.append({
                'level': level_idx,
                'items': new_items,
                'n_clusters': len(new_items),
            })
            current_items = new_items

        return all_levels

    def _gnn_aggregate(
        self,
        items: List[Dict],
        level: int,
    ) -> List[Dict]:
        """
        GNN-агрегація кластерів на одному рівні у PyTorch.
        """
        n = len(items)
        if n <= 1:
            return items

        # Збираємо представлення
        reps = []
        for item in items:
            rep = item['representation']
            if isinstance(rep, torch.Tensor):
                reps.append(rep.to(self.device))
            else:
                reps.append(torch.tensor(rep, dtype=torch.float32, device=self.device))
        node_features = torch.stack(reps)

        if node_features.shape[1] > self.d_base:
            node_features = node_features[:, :self.d_base]
        elif node_features.shape[1] < self.d_base:
            node_features = F.pad(node_features, (0, self.d_base - node_features.shape[1]))

        # Побудова графа
        clusters_for_graph = [item['cluster'] for item in items]
        adjacency, edge_features = self.build_cluster_graph(clusters_for_graph)

        # Message passing (кілька кроків)
        for _ in range(self.n_message_passes):
            node_features = self.message_passing_step(
                node_features, adjacency, edge_features, level
            )

        # Кластеризація представлень → злиття
        new_items = self._merge_by_representation(items, node_features, level)
        return new_items

    def _merge_by_representation(
        self,
        items: List[Dict],
        node_features: torch.Tensor,
        level: int,
    ) -> List[Dict]:
        """Злиття кластерів на основі GNN-представлень."""
        n = len(items)
        if n <= 1:
            return items

        # Косинусна спорідненість між GNN-представленнями
        norms = torch.norm(node_features, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-10)
        normed = node_features / norms
        similarity = torch.mm(normed, normed.t())

        # Конвертуємо подібність у numpy для дискретних рішень на CPU
        similarity_np = similarity.detach().cpu().numpy()

        merged = set()
        new_items = []

        for i in range(n):
            if i in merged:
                continue

            # Знаходимо найбільш схожого сусіда
            best_j = -1
            best_sim = 0.6  # Мінімальний поріг

            # Спочатку перевіряємо просторових сусідів
            for j in [i - 1, i + 1]:
                if 0 <= j < n and j not in merged and similarity_np[i, j] > best_sim:
                    best_j = j
                    best_sim = similarity_np[i, j]

            if best_j >= 0:
                # Злиття i та best_j
                group = [i, best_j]
                merged.add(i)
                merged.add(best_j)

                # Зважена агрегація кластерів
                cluster_i = items[i]['cluster']
                cluster_j = items[best_j]['cluster']

                pos_i = cluster_i['positions']
                pos_j = cluster_j['positions']
                if isinstance(pos_i, torch.Tensor):
                    pos_i = pos_i.cpu().numpy()
                if isinstance(pos_j, torch.Tensor):
                    pos_j = pos_j.cpu().numpy()
                new_positions = np.sort(np.concatenate([pos_i, pos_j]))

                dist_i = cluster_i['distribution']
                dist_j = cluster_j['distribution']
                if isinstance(dist_i, torch.Tensor):
                    dist_i = dist_i.detach().cpu().numpy()
                if isinstance(dist_j, torch.Tensor):
                    dist_j = dist_j.detach().cpu().numpy()

                new_dist = (dist_i * cluster_i['size'] + dist_j * cluster_j['size'])
                new_dist /= max(new_dist.sum(), 1e-10)

                new_cluster = {
                    'positions': new_positions,
                    'start': min(cluster_i['start'], cluster_j['start']),
                    'end': max(cluster_i['end'], cluster_j['end']),
                    'size': len(new_positions),
                    'distribution': new_dist,
                    'mean_u': (cluster_i.get('mean_u', 0) + cluster_j.get('mean_u', 0)) / 2,
                    'std_u': (cluster_i.get('std_u', 0) + cluster_j.get('std_u', 0)) / 2,
                    'mean_v': (cluster_i.get('mean_v', 0) + cluster_j.get('mean_v', 0)) / 2,
                    'std_v': (cluster_i.get('std_v', 0) + cluster_j.get('std_v', 0)) / 2,
                    'dominant_bytes': cluster_i.get('dominant_bytes', []),
                    'quality_score': max(
                        cluster_i.get('quality_score', 0.5),
                        cluster_j.get('quality_score', 0.5)
                    ),
                }

                # Обчислюємо представлення з оновлених GNN-фіч
                new_rep = 0.5 * (node_features[i] + node_features[best_j])

                new_items.append({
                    'id': f'L{level}_M{i}',
                    'cluster': new_cluster,
                    'representation': new_rep,
                    'level': level,
                    'source_ids': items[i].get('source_ids', [items[i]['id']]) +
                                   items[best_j].get('source_ids', [items[best_j]['id']]),
                })
            else:
                # Залишаємо без змін
                merged.add(i)
                new_items.append({
                    'id': f'L{level}_S{i}',
                    'cluster': items[i]['cluster'],
                    'representation': node_features[i],
                    'level': level,
                    'source_ids': items[i].get('source_ids', [items[i]['id']]),
                })

        return new_items

    def update_weights(self, free_energy_grad: Union[np.ndarray, torch.Tensor], level: int):
        """
        Оновити параметри GNN для шару 'level' через градієнт.
        Підтримує як автоматичний backward через PyTorch, так і ручне оновлення градієнтами.
        """
        if level >= self.n_levels:
            return

        # Якщо передано PyTorch loss/тензор з градієнтом, робимо backward і крок оновлення
        if isinstance(free_energy_grad, torch.Tensor) and free_energy_grad.requires_grad:
            free_energy_grad.backward()
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param.data -= self.learning_rate * param.grad
                        param.grad.zero_()
            return

        # Сумісність: ручне оновлення через плаский масив градієнтів (як у ConversionLayersV3)
        if isinstance(free_energy_grad, torch.Tensor):
            grad_np = free_energy_grad.detach().cpu().numpy()
        else:
            grad_np = np.asarray(free_energy_grad, dtype=np.float32)

        msg_layer = self.W_message[level]
        upd_layer = self.W_update[level]
        readout_layer = self.W_readout[level]

        w_msg_size = msg_layer.weight.numel()
        b_msg_size = msg_layer.bias.numel()
        w_upd_size = upd_layer.weight.numel()
        b_upd_size = upd_layer.bias.numel()
        w_readout_size = readout_layer.weight.numel()

        total_size = w_msg_size + b_msg_size + w_upd_size + b_upd_size + w_readout_size
        if len(grad_np) < total_size:
            grad_np = np.pad(grad_np, (0, total_size - len(grad_np)))

        offset = 0
        with torch.no_grad():
            # Оновлення W_message
            g_w_msg = torch.tensor(grad_np[offset : offset + w_msg_size], dtype=torch.float32, device=self.device).view_as(msg_layer.weight)
            msg_layer.weight -= self.learning_rate * g_w_msg
            offset += w_msg_size

            g_b_msg = torch.tensor(grad_np[offset : offset + b_msg_size], dtype=torch.float32, device=self.device).view_as(msg_layer.bias)
            msg_layer.bias -= self.learning_rate * g_b_msg
            offset += b_msg_size

            # Оновлення W_update
            g_w_upd = torch.tensor(grad_np[offset : offset + w_upd_size], dtype=torch.float32, device=self.device).view_as(upd_layer.weight)
            upd_layer.weight -= self.learning_rate * g_w_upd
            offset += w_upd_size

            g_b_upd = torch.tensor(grad_np[offset : offset + b_upd_size], dtype=torch.float32, device=self.device).view_as(upd_layer.bias)
            upd_layer.bias -= self.learning_rate * g_b_upd
            offset += b_upd_size

            # Оновлення W_readout
            g_w_readout = torch.tensor(grad_np[offset : offset + w_readout_size], dtype=torch.float32, device=self.device).view_as(readout_layer.weight)
            readout_layer.weight -= self.learning_rate * g_w_readout


