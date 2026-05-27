import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import torch

from bcs.core.substrate import _make_bytesubstrate, ByteSubstrate
from bcs.core.policy import AdaptiveNumericPolicy
from bcs.core.embedding import DynamicByteEmbedding
from bcs.core.interaction import TorchSpaceValueInteractionV8, FFTSpaceValueInteractionV7, FullTensorInteractionV6
from bcs.core.field import FieldSystemV6, PredictionErrorLoop
from bcs.information.variational import VariationalInference
from bcs.information.bottleneck import IBOptimizer, InformationBottleneck
from bcs.information.modality import BayesianModalityDetector, KnowledgeTransfer
from bcs.information.geometry import InformationGeometryModalityDetector, GeometricBoundaryDetector
from bcs.information.manifold_trajectory import (
    ManifoldTrajectory,
    GeodesicAttention,
    CurvatureNoveltyDetector,
    MemoryAsSubmanifold,
)
from bcs.information.geodesic_integration import TrajectoryContextIntegration
from bcs.optimization.optimization import MultiTimescaleOptimizer, CMAESOptimizer, TimeScaleSystem
from bcs.perception.predictive import HierarchicalPredictiveCoding, PredictiveCoding
from bcs.perception.boundary import MultiScaleBoundaryDetector
from bcs.perception.organization import SelfOrganizerV4, LevelSplitting
from bcs.perception.feedback import FeedbackMechanism
from bcs.perception.geometry import FisherInformationGeometry
from bcs.perception.token import EmergentTokenDiscovery
from bcs.information.conversion import GNNConversionLayers, ConversionLayersV3
from bcs.perception.phase import PhaseTransitionAnalyzer
from bcs.memory.memory import CrystallizedMemory, WorkingMemory, SequenceAssociativeMemory
from bcs.cognition.semantic import SemanticLatentDynamics, SemanticByteDecoder, SemanticAssociativeReasoner, SemanticDifferentiableMemoryGraph, SemanticStateConstructor, SemanticQueryReadout
from bcs.cognition.credit import CreditEpisodeResult, AdaptiveCalibration, AdaptiveEvidenceScorer, EpisodeCreditEngine
from bcs.memory.recognition import ClusterRecognition, ContextResonance

class BCSModelV6:
    """
    Повна модель БКС V6 — максимальна відповідність концепції.

    Покращення над V5:
    1. DynamicByteEmbedding замість статичних векторів
    2. FullTensorInteractionV6 з α, β, γ
    3. FieldSystemV6 з per-byte-value D_k та double-well
    4. VariationalInference для генеративної моделі
    5. IBOptimizer на кожному рівні конвертації
    6. BayesianModalityDetector
    7. MultiTimescaleOptimizer

    Нові компоненти V7 (пам'ять, контекст, перенесення, розщеплення):
    8. TimeScaleSystem — узгоджені часові масштаби
    9. CrystallizedMemory — довгострокова пам'ять (кристалізація)
    10. WorkingMemory — короткострокова пам'ять (кільцевий буфер)
    11. ClusterRecognition — впізнавання через атрактори поля
    12. ContextResonance — послідовний + накопичувальний контекст
    13. KnowledgeTransfer — перенесення знань між модальностями
    14. LevelSplitting — автокаталітичне розщеплення рівнів
    """

    def __init__(
        self,
        # Параметри поля (CONCEPT FIX: D_u та dt адаптовані до Σ-Laplacian)
        D_u: float = 0.008,
        D_v: float = 0.04,
        F_base: float = 0.035,
        k_base: float = 0.060,
        dt: float = 0.1,
        neighborhood_size: int = 5,
        # Тензорні параметри
        d_embedding: int = 64,
        d_beta: int = 32,
        lambda_base: float = 8.0,
        k_neighbors: int = 16,
        use_tensor_interaction: bool = True,
        # Конвертація
        n_conversion_levels: int = 4,
        merge_threshold: float = 0.15,
        # Зворотний зв'язок
        feedback_alpha: float = 0.3,
        # V6 параметри
        n_active_bytes: int = 256,  # FIX: за замовчуванням ВСІ 256 байтів
        use_dynamic_embedding: bool = True,
        use_full_tensor: bool = True,
        use_variational: bool = True,
        use_ib_optimizer: bool = True,
        use_bayesian_modality: bool = True,
        use_multiscale_opt: bool = True,
        use_fisher_geometry: bool = True,
        use_hierarchical_pc: bool = True,
        use_gnn_conversion: bool = True,
        use_token_discovery: bool = True,
        use_phase_analysis: bool = True,
        use_prediction_error_loop: bool = True,
        max_token_length: int = 8,
        min_token_frequency: int = 2,
        use_2d_interaction: bool = True,
        interaction_mode: str = 'dense_bilinear',
        device: str = 'cpu',
        # Нові V7 компоненти
        use_time_scale: bool = True,
        use_crystallized_memory: bool = True,
        use_working_memory: bool = True,
        use_cluster_recognition: bool = True,
        use_context_resonance: bool = True,
        use_knowledge_transfer: bool = True,
        use_level_splitting: bool = True,
        use_sequence_memory: bool = True,
        use_semantic_dynamics: bool = True,
        use_semantic_readout: bool = True,
        use_manifold_trajectory: bool = True,
        use_geodesic_context: bool = True,
    ):
        # Збереження параметрів
        self.D_u = D_u
        self.D_v = D_v
        self.F_base = F_base
        self.k_base = k_base
        self.dt = dt
        self.neighborhood_size = neighborhood_size
        self.d_embedding = d_embedding
        self.d_beta = d_beta
        self.lambda_base = lambda_base
        self.k_neighbors = k_neighbors
        self.use_tensor_interaction = use_tensor_interaction
        self.n_conversion_levels = n_conversion_levels
        self.merge_threshold = merge_threshold
        self.feedback_alpha = feedback_alpha
        self.n_active_bytes = n_active_bytes
        self.use_2d_interaction = use_2d_interaction
        self.interaction_mode = interaction_mode
        self.device = device

        # Флаги
        self.use_dynamic_embedding = use_dynamic_embedding
        self.use_full_tensor = use_full_tensor
        self.use_variational = use_variational
        self.use_ib_optimizer = use_ib_optimizer
        self.use_bayesian_modality = use_bayesian_modality
        self.use_multiscale_opt = use_multiscale_opt
        self.use_fisher_geometry = use_fisher_geometry
        self.use_hierarchical_pc = use_hierarchical_pc
        self.use_gnn_conversion = use_gnn_conversion
        self.use_token_discovery = use_token_discovery
        self.use_phase_analysis = use_phase_analysis
        self.use_prediction_error_loop = use_prediction_error_loop

        # Нові V7 флаги
        self.use_time_scale = use_time_scale
        self.use_crystallized_memory = use_crystallized_memory
        self.use_working_memory = use_working_memory
        self.use_cluster_recognition = use_cluster_recognition
        self.use_context_resonance = use_context_resonance
        self.use_knowledge_transfer = use_knowledge_transfer
        self.use_level_splitting = use_level_splitting
        self.use_sequence_memory = use_sequence_memory
        self.use_semantic_dynamics = use_semantic_dynamics
        self.use_semantic_readout = use_semantic_readout
        self.use_manifold_trajectory = use_manifold_trajectory
        self.use_geodesic_context = use_geodesic_context

        # Компоненти (ініціалізуються пізніше)
        self.substrate = None
        self.field = None
        self.embeddings = None
        self.tensors_v6 = None
        self.W_field = None
        self.prediction_error_loop = None
        self.numeric_policy = AdaptiveNumericPolicy()

        # V7 компоненти
        self.time_scale_system = TimeScaleSystem() if use_time_scale else None
        self.crystal_memory = CrystallizedMemory(
            theta_consolidate=0.15, n_min=2, tau_decay=5e6
        ) if use_crystallized_memory else None
        self.working_memory = WorkingMemory() if use_working_memory else None
        self.cluster_recognition = ClusterRecognition() if use_cluster_recognition else None
        self.context_resonance = ContextResonance() if use_context_resonance else None
        self.knowledge_transfer = KnowledgeTransfer() if use_knowledge_transfer else None
        self.level_splitting = LevelSplitting() if use_level_splitting else None
        self.sequence_memory = SequenceAssociativeMemory() if use_sequence_memory else None
        self.semantic_dynamics = SemanticLatentDynamics() if use_semantic_dynamics else None
        self.semantic_readout = SemanticQueryReadout() if use_semantic_readout else None
        # V8: Manifold Trajectory
        self.manifold_trajectory = ManifoldTrajectory(
            max_length=500,
            decay_rate=0.99,
            novelty_threshold=0.5,
        ) if use_manifold_trajectory else None
        
        # V9: Geodesic Context Engine (ПОВНА ЗАМІНА window-based context)
        self.geodesic_context: Optional[TrajectoryContextIntegration] = None
        if use_geodesic_context:
            try:
                self.geodesic_context = TrajectoryContextIntegration(
                    self,  # передаємо модель
                    config={
                        'temperature': 1.0,
                        'decay_rate': 0.99,
                        'max_trajectory_length': 1000,
                        'novelty_threshold': 0.5,
                    }
                )
                print("   GeodesicContext: УВІМКНЕНО (траєкторія замість вікна)")
            except Exception as e:
                print(f"   GeodesicContext: ПОМИЛКА ініціалізації — {e}")
                self.geodesic_context = None

        # V6 компоненти
        self.dynamic_embedding = DynamicByteEmbedding(
            d_embedding=d_embedding
        ) if use_dynamic_embedding else None

        if use_full_tensor:
            if use_2d_interaction:
                if interaction_mode == 'torch_v8':
                    self.tensors_full = TorchSpaceValueInteractionV8(
                        d_embedding=d_embedding,
                        d_beta=d_beta,
                        lambda_base=lambda_base,
                        k_neighbors=k_neighbors,
                        device=device,
                    )
                else:
                    self.tensors_full = FFTSpaceValueInteractionV7(
                        d_embedding=d_embedding,
                        d_beta=d_beta,
                        lambda_base=lambda_base,
                        k_neighbors=k_neighbors,
                        mode=interaction_mode,
                    )
            else:
                self.tensors_full = FullTensorInteractionV6(
                    d_embedding=d_embedding,
                    d_beta=d_beta,
                    lambda_base=lambda_base,
                    k_neighbors=k_neighbors,
                )
        else:
            self.tensors_full = None

        self.variational = VariationalInference(
            n_levels=n_conversion_levels,
            d_observation=256,
        ) if use_variational else None

        self.ib_optimizer = IBOptimizer() if use_ib_optimizer else None

        # Advanced Modality Detection (Information Geometry)
        self.modality_detector = None
        self.modality_detector_class = 'bayesian'  # default
        if use_bayesian_modality:
            try:
                # Try Information Geometry detector first (more advanced)
                self.modality_detector = InformationGeometryModalityDetector(adaptive=True)
                self.modality_detector_class = 'information_geometry'
                print("   Modality: Information Geometry (advanced)")
            except Exception:
                # Fall back to Bayesian
                self.modality_detector = BayesianModalityDetector()
                self.modality_detector_class = 'bayesian'
                print("   Modality: Bayesian (fallback)")

        self.multiscale_optimizer = MultiTimescaleOptimizer() if use_multiscale_opt else None

        # CONCEPT FIX (Розділ 14.2): CMA-ES мета-оптимізація параметрів D_k, θ_k, μ, a_k
        self.cmaes_optimizer = CMAESOptimizer() if use_multiscale_opt else None

        # V5 компоненти — вбудовані inline (без зовнішніх імпортів)
        self._v5_available = True

        self.fisher_geometry = FisherInformationGeometry() if use_fisher_geometry and self._v5_available else None
        self.hierarchical_pc = None
        self.token_discoverer = EmergentTokenDiscovery(
            min_frequency=min_token_frequency,
            max_token_length=max_token_length,
        ) if use_token_discovery and self._v5_available else None
        self.gnn_conversion = GNNConversionLayers(
            n_levels=n_conversion_levels,
        ) if use_gnn_conversion and self._v5_available else None
        self.phase_analyzer = PhaseTransitionAnalyzer() if use_phase_analysis and self._v5_available else None

        # V4 компоненти
        self.pc = None
        self.organizer = None
        self.ib = None
        self.feedback = None

        self.results = {}

    def ingest(self, data, max_length: Optional[int] = None):
        """Поглинання сирих байтових даних."""
        # V6 FIX: Inline ByteSubstrate (без зовнішніх імпортів)
        ByteSubstrate = _make_bytesubstrate()
        self.substrate = ByteSubstrate(data, max_length=max_length)

        # Баєсівська ідентифікація модальності
        if self.use_bayesian_modality and self.modality_detector is not None:
            modality, posteriors = self.modality_detector.detect(
                self.substrate.byte_distribution,
                N=self.substrate.length
            )
            self.detected_modality = modality
            self.modality_posteriors = posteriors
        else:
            self.detected_modality = self.substrate.detect_modality()
            self.modality_posteriors = {}

        return self

    def build_tensors(self):
        """Побудова тензорів взаємодії."""
        if self.substrate is None:
            raise ValueError("Спершу викличте ingest(data)")

        if self.use_full_tensor and self.tensors_full is not None:
            print("   Обчислення повного тензора взаємодії V6...")
            # CONCEPT FIX: Рівняння (2) каже h_i = φ(b_i, C_i) де C_i — контекстна
            # матриця, що залежить від СТАНУ ПОЛЯ. На цьому етапі поле ще не
            # ініціалізоване, тому використовуємо початкові значення з ByteSubstrate
            # (одногарячне кодування як u-field, байтовий розподіл як v-field).
            # Це краще ніж константи 0.5 та 0.25.
            init_u = np.mean(self.substrate.one_hot, axis=1).astype(np.float32)  # (N,) — початковий u
            init_v = self.substrate.byte_distribution[self.substrate.byte_values].astype(np.float32)  # (N,) — початковий v
            self.embeddings = self.dynamic_embedding.compute_embeddings(
                self.substrate.raw_data,
                init_u,
                init_v,
            ) if self.use_dynamic_embedding and self.dynamic_embedding is not None else None

            self.W_field = self.tensors_full.compute_interaction_field(
                self.substrate, self.embeddings, field=self.field
            )
            self.tensors_v6 = self.tensors_full
        else:
            # Fallback: просте поле взаємодії без тензорів
            self.W_field = np.ones(self.substrate.length, dtype=np.float32) * 0.5

        return self

    def init_field(self):
        """Ініціалізація польової системи V6."""
        if self.substrate is None:
            raise ValueError("Спершу викличте ingest(data)")

        # Польова система V6 з double-well potential
        self.field = FieldSystemV6(
            substrate=self.substrate,
            D_u=self.D_u,
            D_v=self.D_v,
            F_base=self.F_base,
            k_base=self.k_base,
            dt=self.dt,
            neighborhood_size=self.neighborhood_size,
            interaction_field=self.W_field,
            n_active_bytes=self.n_active_bytes,
            numeric_policy=self.numeric_policy,
        )

        # Recompute after FieldSystemV6 exists so 2D value-specific columns follow
        # field.active_byte_indices, not the pre-field fallback order 0..255.
        if self.use_dynamic_embedding and self.dynamic_embedding is not None:
            self.embeddings = self.dynamic_embedding.compute_embeddings(
                self.substrate.raw_data,
                self.field.u,
                self.field.v,
                self.field.Phi,
                self.field.active_byte_indices,
            )

        if self.use_full_tensor and self.tensors_full is not None and self.embeddings is not None:
            self.W_field = self.tensors_full.compute_interaction_field(
                self.substrate, self.embeddings, field=self.field
            )
            self.field.update_feed_rate(self.W_field)

        # Предиктивне кодування (V4) — вбудовані inline (без зовнішніх імпортів)
        self._v4_available = True

        self.pc = PredictiveCoding(context_size=8, learning_rate=0.01)

        boundary_detector = MultiScaleBoundaryDetector(numeric_policy=self.numeric_policy)
        # Try to use advanced GeometricBoundaryDetector
        try:
            geometric_bd = GeometricBoundaryDetector(
                scales=[4, 8, 16, 32, 64, 128],
                use_fisher_rao=True,
                use_geodesic=True,
            )
            # Use geometric detector as primary
            self.organizer = SelfOrganizerV4(
                field_system=self.field,
                predictive_coding=self.pc,
                temperature=1.0,
                boundary_detector=geometric_bd,
            )
            print("   Boundary: Geometric (Information Geometry)")
        except Exception:
            self.organizer = SelfOrganizerV4(
                field_system=self.field,
                predictive_coding=self.pc,
                temperature=1.0,
                boundary_detector=boundary_detector,
            )
            print("   Boundary: MultiScale (fallback)")
        self.ib = InformationBottleneck(n_clusters=10, beta=1.0)
        self.feedback = FeedbackMechanism(alpha=self.feedback_alpha)

        # Ієрархічне предиктивне кодування
        if self.use_hierarchical_pc and self._v5_available:
            d_reps = [256, 128, 64, 32][:self.n_conversion_levels]
            self.hierarchical_pc = HierarchicalPredictiveCoding(
                n_levels=self.n_conversion_levels,
                d_representations=d_reps,
            )

        # Prediction Error Loop (Фрістон) — V8 Byte-Grounded
        if self.use_prediction_error_loop:
            self.prediction_error_loop = PredictionErrorLoop(
                n_active_bytes=self.field.n_active_bytes,
                active_byte_indices=self.field.active_byte_indices,
                context_size=8,
                learning_rate=0.01,
                field_correction_rate=0.001,
                complexity_weight=0.01,
            )

        return self

    def run(
        self,
        n_steps: int = 800,
        record_every: int = 100,
        feedback_every: int = 200,
        window_size: Optional[int] = None,
        window_overlap: int = 0,
    ) -> Dict:
        """Повний цикл обробки БКС V6.

        CONCEPT FIX (Розділ 8.3): Віконна обробка з перетинанням.
        Для великих потоків (N > 10⁴) можна задати window_size,
        і система оброблятиме дані вікнами з перетином δW.
        Це забезпечує:
        1. Обмежене споживання пам'яті: O(W·256) замість O(N·256)
        2. Неперервність контексту: перетин гарантує виявлення
           кластерів на межі вікон обома вікнами
        3. Кристали з попередніх вікон впливають на нове вікно
           через механізм контекстного резонансу
        """
        if self.field is None:
            raise ValueError("Спершу викличте init_field()")

        entropy = self.substrate._shannon_entropy(self.substrate.byte_distribution)
        unique_bytes = int(np.count_nonzero(self.substrate.byte_distribution))

        # CONCEPT FIX (Розділ 8.3): Віконна обробка
        # Якщо window_size не задано, для великих потоків використовуємо
        # автоматичний вибір W = min(N, 10000) з перетином δW = W//5.
        if window_size is None and self.substrate.length > 10000:
            window_size = 10000
            window_overlap = window_size // 5  # δW ≥ λ_max
        use_windowed = (window_size is not None
                        and window_size < self.substrate.length)

        print(f"\n🧠 БКС V6: Обробка {self.substrate.length} байтів ({self.detected_modality})")
        print(f"   Ентропія: {entropy:.2f} біт")
        print(f"   Унікальних байтів: {unique_bytes}")
        print(f"   Dynamic Embedding: {'УВІМКНЕНО' if self.use_dynamic_embedding else 'ВИМКНЕНО'}")
        print(f"   Full Tensor V6: {'УВІМКНЕНО' if self.use_full_tensor else 'ВИМКНЕНО'}")
        print(f"   Double-well Field: УВІМКНЕНО")
        print(f"   Variational: {'УВІМКНЕНО' if self.use_variational else 'ВИМКНЕНО'}")
        print(f"   IB Optimizer: {'УВІМКНЕНО' if self.use_ib_optimizer else 'ВИМКНЕНО'}")
        print(f"   Bayesian Modality: {'УВІМКНЕНО' if self.use_bayesian_modality else 'ВИМКНЕНО'} ({self.modality_detector_class})")
        if use_windowed:
            print(f"   Віконна обробка: УВІМКНЕНО (W={window_size}, δW={window_overlap})")
        else:
            print(f"   Віконна обробка: ВИМКНЕНО (N={self.substrate.length})")

        if self.modality_posteriors:
            top3 = sorted(self.modality_posteriors.items(), key=lambda x: -x[1])[:3]
            print(f"   Модальність: " + ", ".join(f"{m}={p:.2f}" for m, p in top3))

        results = {
            'substrate_info': {
                'length': self.substrate.length,
                'modality': self.detected_modality,
                'modality_posteriors': self.modality_posteriors,
                'entropy': entropy,
                'unique_bytes': unique_bytes,
                'windowed': use_windowed,
                'window_size': window_size if use_windowed else None,
                'window_overlap': window_overlap if use_windowed else None,
            },
            'field_evolution': [],
            'free_energy_over_time': [],
            'final_clusters': [],
            'conversion_levels': [],
            'ib_analysis': {},
            'v6_ib_per_level': {},
            'v6_variational_elbo': [],
            'v6_prediction_error_loop': [],
            'v6_tokens': [],
            'v5_phase_analysis': {},
            'v5_fisher_stats': {},
            'v5_hpc_errors': [],
            'manifold_trajectory': None,  # V8: буде заповнено пізніше
            'geodesic_context': None,  # V9: Geodesic Context Engine
        }

        # V9: Geodesic Context Engine — повна заміна window-based context
        if self.use_geodesic_context and self.geodesic_context is not None:
            # Ініціалізуємо траєкторію з субстрату
            self.geodesic_context.initialize(self.substrate.raw_data)
            print(f"   🌀 GeodesicContext: ініціалізовано {len(self.geodesic_context)} точок")
            
            # Додаємо кожну позицію як окрему точку
            one_hot = self.substrate.one_hot
            n_points = min(100, self.substrate.length)
            step = max(1, self.substrate.length // n_points)
            
            for i in range(0, self.substrate.length, step):
                half_w = 8
                start = max(0, i - half_w)
                end = min(self.substrate.length, i + half_w)
                local_dist = one_hot[start:end].mean(axis=0) if end > start else self.substrate.byte_distribution
                
                if local_dist.sum() > 0:
                    local_dist = local_dist / local_dist.sum()
                
                modality = self._detect_local_modality(local_dist) if hasattr(self, '_detect_local_modality') else 'unknown'
                
                self.geodesic_context.context_engine.push(
                    p=local_dist.astype(np.float64),
                    t=float(i) / self.substrate.length,
                    position=i,
                    modality=modality,
                    metadata={'step': i}
                )
            
            ctx_summary = self.geodesic_context.get_context_summary()
            results['geodesic_context'] = ctx_summary
            print(f"   🌀 GeodesicContext: {ctx_summary.get('n_points', 0)} точок, "
                  f"довжина={ctx_summary.get('total_geodesic_length', 0):.2f}, "
                  f"Betti_1={ctx_summary.get('topology', {}).get('betti_1', 0)}")
        if self.use_manifold_trajectory and self.manifold_trajectory is not None:
            # Створити траєкторію з розподілу байтів субстрату
            substrate_dist = self.substrate.byte_distribution.copy()
            self.manifold_trajectory.push(
                substrate_dist,
                t=0.0,
                metadata={'source': 'initial_substrate'}
            )
            
            # Додати кожну позицію як окрему точку
            one_hot = self.substrate.one_hot
            for i in range(0, self.substrate.length, max(1, self.substrate.length // 100)):
                # Локальний розподіл навколо позиції
                half_w = 8
                start = max(0, i - half_w)
                end = min(self.substrate.length, i + half_w)
                local_dist = one_hot[start:end].mean(axis=0) if end > start else substrate_dist
                self.manifold_trajectory.push(
                    local_dist.astype(np.float64),
                    t=float(i) / self.substrate.length,
                    metadata={'position': i}
                )

        # Навчання предиктивного кодування
        if self.use_sequence_memory and self.sequence_memory is not None:
            seq_prior = self.sequence_memory.apply_to_field(
                self.field,
                self.substrate.raw_data,
                strength=1.25,
                min_confidence=0.55,
                placeholder_only=True,
            )
            results['v8_sequence_memory_prior'] = seq_prior

        pc_mse = self.pc.learn(self.field.u)

        prev_fe = float('inf')
        conv = None

        # === Основний цикл еволюції ===
        for step in range(n_steps):
            self.field.step()

            # Оновлення динамічних ембедингів кожні 50 кроків
            if step % 50 == 0 and self.use_dynamic_embedding and self.dynamic_embedding is not None:
                self.embeddings = self.dynamic_embedding.compute_embeddings(
                    self.substrate.raw_data,
                    self.field.u,
                    self.field.v,
                    self.field.Phi,
                    self.field.active_byte_indices,
                )
                if self.use_full_tensor and self.tensors_full is not None:
                    self.W_field = self.tensors_full.compute_interaction_field(
                        self.substrate, self.embeddings, field=self.field
                    )
                    self.field.update_feed_rate(self.W_field)

            # Навчання PC
            if step % 50 == 0:
                pc_mse = self.pc.learn(self.field.u)

            # Prediction Error Loop (Фрістон) — V8 Byte-Grounded
            # Кожні 5 кроків після 50-крокового warm-up (Allen-Cahn стабілізація)
            if step >= 50 and step % 5 == 0 and self.use_prediction_error_loop and self.prediction_error_loop is not None:
                pel_result = self.prediction_error_loop.step(self.field, self.substrate.byte_values)
                results['v6_prediction_error_loop'].append(pel_result)
                if step % (record_every * 4) == 0:
                    print(f"   PEL: F={pel_result['free_energy']:.4f}, "
                          f"CE={pel_result['mean_error']:.4f}, "
                          f"acc={pel_result.get('byte_accuracy', 0):.1%}, "
                          f"corr={pel_result['corrections_applied']:.6f}")

            # Multi-timescale оптимізація
            if self.use_multiscale_opt and self.multiscale_optimizer is not None:
                fe = self.field.compute_free_energy(1.0)

                if self.multiscale_optimizer.should_update('substrate', step):
                    self.multiscale_optimizer.substrate_step(self.field, fe, prev_fe)

                if self.multiscale_optimizer.should_update('tensor', step):
                    if self.tensors_full is not None and self.embeddings is not None:
                        self.multiscale_optimizer.tensor_step(
                            self.tensors_full, self.embeddings, fe, prev_fe
                        )

                # CONCEPT FIX (Розділ 14.2): CMA-ES мета-оптимізація параметрів
                # на найповільнішому рівні (τ_meta ≈ години, кожні 2000 кроків).
                # Фітнес = швидкість зменшення вільної енергії.
                if (self.cmaes_optimizer is not None
                        and self.multiscale_optimizer.should_update('meta', step)):
                    cmaes_result = self.cmaes_optimizer.step(
                        self.field, fe, prev_fe, n_validation_steps=5
                    )
                    results.setdefault('v7_cmaes_history', []).append(cmaes_result)
                    if cmaes_result.get('applied'):
                        print(f"   CMA-ES gen={cmaes_result['generation']}: "
                              f"fitness={cmaes_result['best_fitness']:.6f}, "
                              f"σ={cmaes_result['sigma']:.4f}, "
                              f"improvement={cmaes_result['improvement']:.6f}")

                prev_fe = fe

            # Варіаційна інференція кожні 50 кроків (CONCEPT FIX: було кожні 100,
            # що давало лише 1 update за весь run — недостатньо для навчання).
            # Також тепер використовуємо ДИНАМІЧНІ спостереження з поля Φ замість
            # статичного byte_distribution — це відповідає концепції: VI моделює
            # поточний стан поля, а не фіксовані вхідні дані.
            if self.use_variational and self.variational is not None and step % 50 == 0 and step > 0:
                # CONCEPT FIX: Динамічні спостереження — середнє Φ по позиціях
                # замість статичного byte_distribution. Рівняння (26):
                # p(s|z^(1)) = Π Cat(b_i; softmax(V z^(1)))
                # Модель має реконструювати ПОТОЧНИЙ стан поля, а не фіксовані дані.
                obs = np.mean(self.field.Phi, axis=0).astype(np.float32)
                obs = np.maximum(obs, 1e-10)
                obs = obs / obs.sum()
                elbo = self.variational.update(obs, lr=0.001)
                results['v6_variational_elbo'].append(elbo)

            # === V7: Кристалізована пам'ять + Впізнавання + Контекст ===
            # Кожні 100 кроків: кластеризація → впізнавання → кристалізація → контекст
            # CONCEPT FIX (Розділ 8.3): Для великих N використовуємо віконну обробку —
            # кластери обробляються по вікнах з перетином, що забезпечує
            # виявлення границь на межі вікон та обмежене споживання пам'яті.
            if step > 0 and step % 100 == 0:
                temp_clusters_v7 = self.organizer.detect_clusters()

                # CONCEPT FIX (Розділ 8.3): Віконна обробка кластерів
                if use_windowed and temp_clusters_v7:
                    windows = self.substrate.windowed_process(
                        window_size, window_overlap
                    )
                    windowed_clusters = []
                    for w_start, w_end in windows:
                        for c in temp_clusters_v7:
                            # Кластер належить вікну якщо є перетин
                            if c['end'] > w_start and c['start'] < w_end:
                                windowed_clusters.append(c)
                    # Унікалізуємо (кластер може належати кільком вікнам)
                    seen = set()
                    unique_clusters = []
                    for c in windowed_clusters:
                        key = (c['start'], c['end'])
                        if key not in seen:
                            seen.add(key)
                            unique_clusters.append(c)
                    temp_clusters_v7 = unique_clusters

                if temp_clusters_v7:
                    for c in temp_clusters_v7:
                        # Визначаємо peak_byte та активацію кластера
                        dist = c.get('distribution', np.zeros(256))
                        peak_byte = int(np.argmax(dist))
                        # Активація: міра концентрації розподілу відносно рівномірного
                        # Якщо max(dist) > 1/256 → є структура
                        uniform = 1.0 / 256.0
                        activation = float(c.get('quality_score', 0.5))

                        # Представлення кластера для впізнавання
                        if self.embeddings is not None and c['start'] < len(self.embeddings):
                            end_idx = min(c['end'], len(self.embeddings))
                            h_cluster = np.mean(self.embeddings[c['start']:end_idx], axis=0)
                        else:
                            h_cluster = dist[:64] if len(dist) >= 64 else np.pad(dist, (0, 64 - len(dist)))

                        # V7: Working Memory — додаємо кластер
                        if self.use_working_memory and self.working_memory is not None:
                            # γ-спорідненість: косинус між кластером та поточним полем
                            wm_ctx = self.working_memory.get_context_vector()
                            if np.linalg.norm(wm_ctx) > 1e-10 and np.linalg.norm(dist) > 1e-10:
                                min_d = min(len(wm_ctx), len(dist))
                                gamma_aff = float(np.dot(wm_ctx[:min_d], dist[:min_d]) /
                                                  (np.linalg.norm(wm_ctx[:min_d]) * np.linalg.norm(dist[:min_d]) + 1e-10))
                            else:
                                gamma_aff = 0.5
                            novelty = 1.0 - activation  # Чим нижча активація, тим новіше
                            self.working_memory.add(c, gamma_affinity=gamma_aff, novelty=novelty)

                        # V7: Cluster Recognition → Crystallization
                        # CONCEPT FIX (Розділ 14.5): Передаємо field_system та позиції
                        # кластера для двоетапної фільтрації шуму:
                        # 1) поріг активації Θ_active
                        # 2) перевірка стійкості через 3 цикли релаксації
                        if (self.use_cluster_recognition and self.cluster_recognition is not None
                                and self.crystal_memory is not None):
                            recognition = self.cluster_recognition.recognize(
                                h_cluster, self.crystal_memory,
                                dist, activation, peak_byte,
                                field_system=self.field,
                                cluster_start=c['start'],
                                cluster_end=c['end'],
                            )
                            results.setdefault('v7_recognition_history', []).append({
                                'step': step,
                                'result': recognition['result'],
                                'cluster_start': c['start'],
                            })

                            # Кристали модифікують параметри тензорів
                            self.crystal_memory.modify_parameters(self.tensors_full, self.embeddings)

                            # CONCEPT FIX (Розділ 14.7): Інкрементальне оновлення LSH індексу
                            # Замість повної перебудови (накопичення stale entries) —
                            # додаємо лише НОВИ кристал за його індексом.
                            # LSH індекс очишається лише при apply_forgetting()
                            # (якщо кристали були видалені).
                            if (self.cluster_recognition is not None
                                    and self.cluster_recognition.use_lsh
                                    and recognition.get('new_crystal_idx') is not None):
                                new_ci = recognition['new_crystal_idx']
                                self.cluster_recognition.update_lsh_index(new_ci, self.crystal_memory)

                    # V7: Context Resonance — резонанс з банком кристалів
                    if (self.use_context_resonance and self.context_resonance is not None
                            and self.crystal_memory is not None
                            and self.crystal_memory.crystals):
                        # Поточне середнє представлення
                        if self.embeddings is not None:
                            h_current = np.mean(self.embeddings, axis=0)
                        else:
                            h_current = np.mean(self.field.Phi, axis=0)[:64]

                        ctx = self.context_resonance.compute_resonance(h_current, self.crystal_memory)
                        injection = self.context_resonance.inject_into_field(self.field, ctx)
                        results.setdefault('v7_context_norms', []).append({
                            'step': step,
                            'ctx_norm': float(np.linalg.norm(ctx)),
                            's_norm': float(np.linalg.norm(self.context_resonance.s)),
                            'injection_mean': float(np.mean(np.abs(injection))),
                        })

                    # V7: Атракторна ін'єкція в поле
                    if (self.use_cluster_recognition and self.cluster_recognition is not None
                            and self.crystal_memory is not None
                            and self.crystal_memory.crystals):
                        self.cluster_recognition.inject_attractor_field(
                            self.field, self.crystal_memory, self.embeddings
                        )

                    # V7: Кристалізована пам'ять — крок забування
                    # CONCEPT FIX (Розділ 14.7): Передаємо cluster_recognition
                    # для перебудови LSH індексу при видаленні кристалів.
                    if self.crystal_memory is not None:
                        self.crystal_memory.step(
                            delta_t=100.0,
                            cluster_recognition=(self.cluster_recognition
                                                 if self.use_cluster_recognition
                                                 else None),
                        )

            # V7: TimeScaleSystem — адаптивний τ_0
            if self.use_time_scale and self.time_scale_system is not None:
                self.time_scale_system.update_tau_0(self.substrate.length)

            # Зворотний зв'язок
            if step > 0 and step % feedback_every == 0 and self.use_tensor_interaction:
                temp_clusters = self.organizer.detect_clusters()
                if len(temp_clusters) > 0:
                    if self.use_gnn_conversion and self.gnn_conversion is not None:
                        conv = self.gnn_conversion.convert(temp_clusters, self.substrate)
                    else:
                        conv_layers = ConversionLayersV3(n_levels=self.n_conversion_levels)
                        conv = conv_layers.convert(temp_clusters, self.substrate)

                if conv is not None and len(conv) > 0 and self.W_field is not None:
                    modified_field, attention = self.feedback.apply(self.W_field, conv)
                    self.W_field = modified_field
                    self.field.update_feed_rate(modified_field)

                # V5/V6 FIX: Навчання ієрархічного предиктивного кодування
                if self.use_hierarchical_pc and self.hierarchical_pc is not None and conv is not None:
                    reps_list = []
                    for lvl_data in conv:
                        items = lvl_data.get('items', [])
                        if items:
                            reps = [item['representation'] for item in items]
                            if torch.is_tensor(reps[0]):
                                lvl_rep = torch.stack(reps).mean(dim=0).detach().cpu().numpy()
                            else:
                                lvl_rep = np.mean(reps, axis=0)
                        else:
                            lvl_idx = lvl_data['level']
                            d_lvl = self.hierarchical_pc.d_representations[min(lvl_idx, len(self.hierarchical_pc.d_representations)-1)]
                            lvl_rep = np.zeros(d_lvl, dtype=np.float32)
                        reps_list.append(lvl_rep)
                    
                    self.hierarchical_pc.n_levels = len(conv)
                    updated_reps, hpc_fe = self.hierarchical_pc.learn_step(reps_list)
                    results['v5_hpc_errors'].append(hpc_fe)

            # Запис статистики
            if step % record_every == 0:
                stats = self.field.get_field_statistics()
                F_free = self.field.compute_free_energy(1.0)
                results['field_evolution'].append({**stats, 'free_energy': F_free})
                results['free_energy_over_time'].append(F_free)

                if step % (record_every * 4) == 0:
                    print(f"   Крок {step}/{n_steps}: "
                          f"u={stats['u_mean']:.4f}, "
                          f"v_std={stats['v_std']:.4f}, "
                          f"phi={stats['phi_mean']:.4f}, "
                          f"F={F_free:.2f}")

        # === Кластеризація ===
        final_clusters = self.organizer.detect_clusters()
        results['final_clusters'] = final_clusters
        results['boundary_indices'] = self.organizer.last_boundaries
        print(f"\n📊 Кластерів: {len(final_clusters)}")
        for i, c in enumerate(final_clusters[:10]):
            print(f"   Кластер {i}: [{c['start']}:{c['end']}], "
                  f"розмір={c['size']}, якість={c.get('quality_score', 0):.3f}")

        # === Конвертаційні шари ===
        if self.use_gnn_conversion and self.gnn_conversion is not None:
            conv = self.gnn_conversion.convert(final_clusters, self.substrate)
        else:
            conv_layers = ConversionLayersV3(n_levels=self.n_conversion_levels)
            conv = conv_layers.convert(final_clusters, self.substrate)

        results['conversion_levels'] = conv
        
        # V9: Geodesic Attention — інтеграція в конвертаційні шари
        if self.use_geodesic_context and self.geodesic_context is not None and conv is not None:
            # Застосовуємо геодезичний attention до конвертованих рівнів
            for level_idx, level_data in enumerate(conv):
                items = level_data.get('items', [])
                if items:
                    # Отримуємо представлення кластерів
                    reprs = [item.get('representation', np.zeros(256)) for item in items]
                    reprs = [r if isinstance(r, np.ndarray) else np.zeros(256) for r in reprs]
                    
                    # Geodesic attention для кожного кластера
                    for item_idx, item in enumerate(items):
                        query = reprs[item_idx] if item_idx < len(reprs) else np.zeros(256)
                        
                        # Викликаємо trajectory attention
                        output, attention = self.geodesic_context.trajectory_attention.forward(
                            query=query,
                            keys=reprs,
                            values=reprs,
                        )
                        
                        # Додаємо геометричну інформацію
                        item['geodesic_attention'] = attention.tolist()
                        item['geodesic_context'] = output.tolist()
                        item['attention_entropy'] = float(-np.sum(attention * np.log(attention + 1e-10)))
                    
                    # Оновлюємо рівень
                    level_data['geodesic_enhanced'] = True
                    level_data['mean_attention_entropy'] = np.mean([
                        item.get('attention_entropy', 0.0) for item in items
                    ])
            
            # Геодезичний summary
            geo_summary = self.geodesic_context.get_summary()
            results['geodesic_attention_summary'] = {
                'n_levels_enhanced': sum(1 for l in conv if l.get('geodesic_enhanced')),
                'mean_entropy': np.mean([l.get('mean_attention_entropy', 0) for l in conv]),
                'context_engine_summary': geo_summary,
            }
            print(f"   🌀 GeodesicAttention: {sum(1 for l in conv if l.get('geodesic_enhanced'))} рівнів з enhanced attention")

        # V5/V6 FIX: Фінальний крок ієрархічного предиктивного кодування
        if self.use_hierarchical_pc and self.hierarchical_pc is not None and conv is not None:
            reps_list = []
            for lvl_data in conv:
                items = lvl_data.get('items', [])
                if items:
                    reps = [item['representation'] for item in items]
                    if torch.is_tensor(reps[0]):
                        lvl_rep = torch.stack(reps).mean(dim=0).detach().cpu().numpy()
                    else:
                        lvl_rep = np.mean(reps, axis=0)
                else:
                    lvl_idx = lvl_data['level']
                    d_lvl = self.hierarchical_pc.d_representations[min(lvl_idx, len(self.hierarchical_pc.d_representations)-1)]
                    lvl_rep = np.zeros(d_lvl, dtype=np.float32)
                reps_list.append(lvl_rep)

            self.hierarchical_pc.n_levels = len(conv)
            updated_reps, hpc_fe = self.hierarchical_pc.learn_step(reps_list)
            results['v5_hpc_errors'].append(hpc_fe)

        # === IB на кожному рівні (V6) ===
        if self.use_ib_optimizer and self.ib_optimizer is not None:
            for level_data in conv:
                level = level_data['level']
                level_clusters = [item['cluster'] for item in level_data.get('items', [])]
                if level_clusters:
                    ib_result = self.ib_optimizer.compute_ib_for_level(
                        level_clusters, self.substrate, level,
                        items=level_data.get('items', [])
                    )
                    results['v6_ib_per_level'][level] = ib_result
                    print(f"   IB Level {level}: I(S;T)={ib_result['I_ST']:.4f}, "
                          f"I(T;Y)={ib_result['I_TY']:.4f}, "
                          f"β*={ib_result['beta_opt']:.2f}")

        # === Information Bottleneck (V4) ===
        local_dist = self.substrate.compute_local_distributions(
            window=max(self.substrate.length // 20, 4)
        )
        ib_labels = self.ib.fit(local_dist)
        ib_analysis = self.ib.compute_ib_objective(local_dist, ib_labels)
        results['ib_analysis'] = ib_analysis

        # === Phase Analysis ===
        if self.use_phase_analysis and self.phase_analyzer is not None:
            phase_results = self.phase_analyzer.full_analysis(self.field)
            results['v5_phase_analysis'] = phase_results
            print(f"\n🌡️ T_c={phase_results['T_c']:.4f}, "
                  f"ψ={phase_results['order_parameter']:.4f}, "
                  f"ξ={phase_results['correlation_length']:.2f}")

        # === Fisher Geometry ===
        if self.use_fisher_geometry and self.fisher_geometry is not None:
            self.fisher_geometry.compute_fisher_matrix_distributions(local_dist)
            stats = self.fisher_geometry.get_stats()
            # Виправлення condition_number
            if stats['condition_number'] == 0.0 and stats['fisher_matrix_trace'] > 0:
                eigvals = np.linalg.eigvalsh(self.fisher_geometry.G)
                positive_eigs = eigvals[eigvals > 1e-10]
                if len(positive_eigs) > 0:
                    stats['condition_number'] = float(positive_eigs[-1] / positive_eigs[0])
            results['v5_fisher_stats'] = stats

        if self.use_token_discovery and self.token_discoverer is not None and len(final_clusters) > 0:
            tokens = self.token_discoverer.discover(self.substrate, final_clusters, self.pc, u_field=self.field.u)
            results['v6_tokens'] = tokens
            print(f"\n🔤 Токенів: {len(tokens)}")
            for t in tokens[:5]:
                print(f"   '{t.get('token_str', '?')[:20]}' "
                      f"(IG={t['info_gain']:.3f}, якість={t['quality']:.3f})")

        if self.use_semantic_dynamics and self.semantic_dynamics is not None:
            semantic_result = self.semantic_dynamics.observe_episode(
                self.substrate,
                self.field,
                self.embeddings,
                final_clusters,
                results.get('v6_tokens', []),
            )
            results['v9_semantic_dynamics'] = semantic_result
            
            # V9: Інтеграція geodesic context в семантичний шар
            if self.use_geodesic_context and self.geodesic_context is not None:
                ctx_vector = self.geodesic_context.get_trajectory_context()
                inj_result = self.geodesic_context.inject_trajectory_into_semantic(
                    self.semantic_dynamics,
                    context_vector=ctx_vector,
                )
                results['geodesic_semantic_injection'] = inj_result
                
                # Додаємо trajectory features до семантичного результату
                if 'trajectory_features' in inj_result:
                    for key, value in inj_result['trajectory_features'].items():
                        semantic_result[f'geo_{key}'] = value
            
            if self.use_semantic_readout and self.semantic_readout is not None:
                context_z = None
                mem_idx = int(semantic_result.get('memory_index', -1))
                if 0 <= mem_idx < len(self.semantic_dynamics.crystals):
                    context_z = self.semantic_dynamics.crystals[mem_idx].get('z')
                
                # V9: Передаємо trajectory context до семантичного read-out
                if self.use_geodesic_context and self.geodesic_context is not None:
                    geo_context = self.geodesic_context.get_trajectory_context()
                    # Додаємо геометричну інформацію до context_z
                    if context_z is not None:
                        if len(context_z.shape) == 1 and len(context_z) > 0:
                            context_z = context_z + 0.1 * geo_context[:len(context_z)]
                
                results['v10_semantic_readout'] = self.semantic_readout.observe_episode(
                    self.substrate.raw_data,
                    self.semantic_dynamics,
                    context_z=context_z,
                )

        # === Variational ELBO ===
        if self.use_variational and self.variational is not None:
            elbo_list = results['v6_variational_elbo']
            if elbo_list:
                print(f"\n📐 Variational: ELBO {elbo_list[0]:.2f} → {elbo_list[-1]:.2f}")

        # === V7: Level Splitting ===
        if self.use_level_splitting and self.level_splitting is not None and conv:
            self.level_splitting.set_levels(conv)
            split_results = []
            for level_idx in range(len(conv)):
                sr = self.level_splitting.attempt_split(level_idx, self)
                split_results.append(sr)
                if sr.get('split_attempted') and sr.get('split_successful'):
                    print(f"   Level {level_idx}: SPLIT → {sr['new_n_clusters']} нових кластерів")
            results['v7_level_splitting'] = split_results
            # Зрощення рівнів
            merge_results = []
            for level_idx in range(len(self.level_splitting.levels) - 1):
                mr = self.level_splitting.attempt_merge(level_idx, self)
                merge_results.append(mr)
                if mr.get('merge_successful'):
                    print(f"   Level {level_idx}: MERGE (ratio={mr['conditional_ratio']:.2f})")
            results['v7_level_merging'] = merge_results

        # === V7: Knowledge Transfer ===
        if self.use_knowledge_transfer and self.knowledge_transfer is not None and self.crystal_memory is not None:
            # Реєструємо модальності кристалів
            modality = self.detected_modality if hasattr(self, 'detected_modality') else 'unknown'
            for idx, crystal in enumerate(self.crystal_memory.crystals):
                self.knowledge_transfer.register_modality(idx, modality)

            # Перевіряємо структурний ізоморфізм між кристалами різних модальностей
            scaffold = self.knowledge_transfer.universal_byte_scaffold_check(
                modality, modality
            )
            results['v7_knowledge_transfer'] = {
                'modality': modality,
                'n_crystals': len(self.crystal_memory.crystals),
                'cross_modal_links': len(self.knowledge_transfer.cross_modal_links),
                'scaffold_compatible': scaffold['compatible'],
            }
            print(f"\n🔗 Knowledge Transfer: {len(self.crystal_memory.crystals)} кристалів, "
                  f"модальність={modality}")

        # === V7: Статистика пам'яті ===
        if self.crystal_memory is not None:
            n_crystals = len(self.crystal_memory.crystals)
            n_recognized = sum(1 for r in self.cluster_recognition.recognition_history
                              if r['result'] == 'recognized') if self.cluster_recognition else 0
            n_ambivalent = sum(1 for r in self.cluster_recognition.recognition_history
                              if r['result'] == 'ambivalent') if self.cluster_recognition else 0
            n_novel = sum(1 for r in self.cluster_recognition.recognition_history
                         if r['result'] == 'novel') if self.cluster_recognition else 0
            print(f"\n💎 Кристалізована пам'ять: {n_crystals} кристалів")
            print(f"   Впізнано: {n_recognized}, Амбівалентно: {n_ambivalent}, Нове: {n_novel}")
            results['v7_crystal_stats'] = {
                'n_crystals': n_crystals,
                'recognized': n_recognized,
                'ambivalent': n_ambivalent,
                'novel': n_novel,
            }

        if self.working_memory is not None:
            wm_relevant = self.working_memory.get_most_relevant(3)
            print(f"🧠 Робоча пам'ять: {len(self.working_memory.buffer)} кластерів")
            results['v7_working_memory'] = {
                'buffer_size': len(self.working_memory.buffer),
                'top_relevant': len(wm_relevant),
            }

        # V7: Time Scale validation
        if self.time_scale_system is not None:
            ts_validation = self.time_scale_system.validate_separation()
            results['v7_time_scales'] = ts_validation
            if not ts_validation['valid']:
                print(f"⚠️ Time scale violations: {ts_validation['violations']}")

        # V7: Context resonance summary
        if self.context_resonance is not None and self.context_resonance.resonance_history:
            rh = self.context_resonance.resonance_history
            print(f"🌊 Контекст: ctx_norm={rh[-1]['ctx_norm']:.4f}, "
                  f"s_norm={rh[-1]['s_norm']:.4f}")

        if self.use_sequence_memory and self.sequence_memory is not None:
            self.sequence_memory.observe(self.substrate.raw_data)
            results['v8_sequence_memory'] = {
                'entries': len(self.sequence_memory.counts),
                'observations': int(self.sequence_memory.total_observations),
            }

        # V8: Manifold Trajectory summary
        if self.use_manifold_trajectory and self.manifold_trajectory is not None:
            traj_summary = self.manifold_trajectory.get_trajectory_summary()
            results['manifold_trajectory'] = traj_summary
            
            print(f"🌀 Траєкторія: {traj_summary['n_points']} точок, "
                  f"довжина={traj_summary['total_length']:.2f}, "
                  f"Betti_1={traj_summary['topology']['betti_1']}")
            
            if traj_summary['topology']['betti_1'] > 0:
                print(f"   🔁 ВИЯВЛЕНО ПЕТЛЮ (= повторення в тексті)")
            
            if traj_summary['curvature_stats']['max'] > 0.3:
                print(f"   ↰ ВИЯВЛЕНО КУТ (= зміна теми)")
        
        # V9: Geodesic Context Engine summary — ПОВНА ЗАМІНА window-based context
        if self.use_geodesic_context and self.geodesic_context is not None:
            geo_summary = self.geodesic_context.get_summary()
            results['geodesic_context_engine'] = geo_summary
            
            if geo_summary:
                n_points = geo_summary.get('n_points', 0)
                geo_length = geo_summary.get('total_geodesic_length', 0.0)
                shapes = geo_summary.get('semantic_shapes', {})
                
                print(f"🌀 GeodesicContextEngine: {n_points} точок, "
                      f"довжина={geo_length:.2f}")
                
                if shapes.get('n_loops', 0) > 0:
                    print(f"   🔁 Петлі (повторення): {shapes['n_loops']}")
                if shapes.get('n_angles', 0) > 0:
                    print(f"   ↰ Кути (зміна теми): {shapes['n_angles']}")
                if shapes.get('n_stops', 0) > 0:
                    print(f"   ⏸ Зупинки (стабільний контекст): {shapes['n_stops']}")
                if shapes.get('n_streams', 0) > 0:
                    print(f"   → Потоки (плавний перехід): {shapes['n_streams']}")
        
        self.results = results
        return results

    def learn_query_answer(self, query, answer, context=None, repetitions: int = 1) -> Dict:
        """Store learned query -> answer experience without a hand-written parser."""
        if self.semantic_readout is None:
            self.semantic_readout = SemanticQueryReadout()
            self.use_semantic_readout = True
        return self.semantic_readout.learn_pair(
            query,
            answer,
            semantic=self.semantic_dynamics,
            context=context,
            repetitions=repetitions,
        )

    def learn_trajectory(self, intent, target, context=None, repetitions: int = 1, epochs: int = 16, lr: float = 0.055) -> Dict:
        """Train a free-form intent -> evidence trajectory through semantic readout."""
        if self.semantic_readout is None:
            self.semantic_readout = SemanticQueryReadout()
            self.use_semantic_readout = True
        return self.semantic_readout.learn_trajectory(
            intent,
            target,
            semantic=self.semantic_dynamics,
            context=context,
            repetitions=repetitions,
            epochs=epochs,
            lr=lr,
        )

    def learn_credit_episode(
        self,
        intent,
        target,
        context=None,
        negative_probes: Optional[List[bytes]] = None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
        max_negatives: int = 24,
    ) -> Dict:
        """Train one episode through global semantic/readout/graph/field/decoder credit routing."""
        if self.semantic_readout is None:
            self.semantic_readout = SemanticQueryReadout()
            self.use_semantic_readout = True
        result = self.semantic_readout.learn_credit_episode(
            intent,
            target,
            semantic=self.semantic_dynamics,
            context=context,
            field_system=self.field,
            negative_probes=negative_probes,
            repetitions=repetitions,
            epochs=epochs,
            lr=lr,
            max_negatives=max_negatives,
        )
        if self.field is not None:
            if self.use_dynamic_embedding and self.dynamic_embedding is not None and self.substrate is not None:
                self.embeddings = self.dynamic_embedding.compute_embeddings(
                    self.substrate.raw_data,
                    self.field.u,
                    self.field.v,
                    self.field.Phi,
                    self.field.active_byte_indices,
                )
            if self.use_full_tensor and self.tensors_full is not None and self.embeddings is not None:
                self.W_field = self.tensors_full.compute_interaction_field(
                    self.substrate,
                    self.embeddings,
                    field=self.field,
                )
                self.field.update_feed_rate(self.W_field)
        return result

    def learn_grounded_end_to_end(
        self,
        intent,
        target,
        context=None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
    ) -> Dict:
        """Compatibility wrapper for episode-level grounded credit assignment."""
        return self.learn_credit_episode(
            intent,
            target,
            context=context,
            repetitions=repetitions,
            epochs=epochs,
            lr=lr,
        )

    def _learn_grounded_end_to_end_legacy(
        self,
        intent,
        target,
        context=None,
        repetitions: int = 1,
        epochs: int = 18,
        lr: float = 0.045,
    ) -> Dict:
        """Legacy sequential updater kept for comparison, not used by public API."""
        if self.semantic_readout is None:
            self.semantic_readout = SemanticQueryReadout()
            self.use_semantic_readout = True
        result = self.semantic_readout.learn_grounded_end_to_end(
            intent,
            target,
            semantic=self.semantic_dynamics,
            context=context,
            field_system=self.field,
            repetitions=repetitions,
            epochs=epochs,
            lr=lr,
        )
        if self.field is not None:
            if self.use_dynamic_embedding and self.dynamic_embedding is not None and self.substrate is not None:
                self.embeddings = self.dynamic_embedding.compute_embeddings(
                    self.substrate.raw_data,
                    self.field.u,
                    self.field.v,
                    self.field.Phi,
                    self.field.active_byte_indices,
                )
            if self.use_full_tensor and self.tensors_full is not None and self.embeddings is not None:
                self.W_field = self.tensors_full.compute_interaction_field(
                    self.substrate,
                    self.embeddings,
                    field=self.field,
                )
                self.field.update_feed_rate(self.W_field)
        return result

    def answer_query(self, query, top_k: int = 3) -> Dict:
        """Answer by retrieving learned byte evidence from semantic readout memory."""
        if self.semantic_readout is None:
            return {
                'answered': False,
                'confidence': 0.0,
                'answer_bytes': b'',
                'answer_text': '',
                'top_candidates': [],
                'reason': 'semantic_readout_disabled',
            }
        return self.semantic_readout.answer(query, semantic=self.semantic_dynamics, top_k=top_k)

    def respond(self, intent, mode: str = 'auto', max_bytes: int = 160, top_k: int = 3) -> Dict:
        """Free-form byte response: evidence, learned answer, generated bytes, or uncertainty."""
        if self.semantic_readout is None:
            return {
                'responded': False,
                'response_kind': 'semantic_readout_disabled',
                'response_bytes': b'',
                'response_text': '',
                'confidence': 0.0,
                'uncertainty': 1.0,
            }
        return self.semantic_readout.respond(
            intent,
            semantic=self.semantic_dynamics,
            mode=mode,
            max_bytes=max_bytes,
            top_k=top_k,
        )

    def learn_end_to_end(self, n_epochs: int = 10, lr: float = 0.001) -> Dict:
        """End-to-end навчання з Fisher природним градієнтом.

        V6 FIX: Euclidean gradient обчислюється з РЕАЛЬНОЇ різниці
        вільної енергії, а НЕ генерується випадково.
        """
        if self.field is None:
            raise ValueError("Спершу викличте init_field()")

        fe_history = []
        initial_fe = self.field.compute_free_energy(1.0)
        fe_history.append(initial_fe)

        for epoch in range(n_epochs):
            # 1. Еволюція поля
            for _ in range(50):
                self.field.step()

            # 2. Free energy
            current_fe = self.field.compute_free_energy(1.0)

            # 3. Fisher natural gradient для тензорів
            #
            # FISHER FIX: ДВІ критичні проблеми виправлено:
            #
            # ПРОБЛЕМА A: Fisher matrix G обчислювалась у просторі розподілів
            # (256x256), але застосовувалась до параметрів W_beta (1024-вим.).
            # ФІКС: діагональне наближення Fisher matrix у просторі W_beta.
            #
            # ПРОБЛЕМА B: Градієнт через perturbation free_energy — НЕ ПРАЦЮЄ,
            # бо FE залежить від Phi, а Phi не змінюється миттєво при зміні W_beta.
            # ФІКС: обчислюємо градієнт ЧЕРЕЗ РІВНЯННЯ ЕВОЛЮЦІЇ:
            #   dPhi/dt = ... + 0.1 * interaction_field * Phi
            #   ∂F/∂W_beta = ∂F/∂dPhi · ∂dPhi/∂interaction_field · ∂interaction_field/∂W_beta
            #
            # ∂F/∂dPhi: напрямок зменшення вільної енергії
            # ∂dPhi/∂interaction_field = 0.1 * Phi (з рівняння еволюції)
            # ∂interaction_field/∂W_beta: через compute_interaction_field
            if self.use_fisher_geometry and self.fisher_geometry is not None:
                if self.tensors_full is not None and self.embeddings is not None:
                    W_beta = self.tensors_full.W_beta
                    is_torch = isinstance(W_beta, torch.Tensor)
                    W_beta_np = W_beta.detach().cpu().numpy() if is_torch else W_beta
                    d1, d2 = W_beta_np.shape

                    # Крок 1: Обчислюємо ∂F/∂dPhi — градієнт вільної енергії
                    # щодо швидкості зміни Phi. Це напрямок, в якому
                    # зміна dPhi зменшує вільну енергію.
                    Phi = self.field.Phi  # (N, n_active_bytes)
                    # F = E_dw + 0.1*E_kinetic - T*S
                    # E_dw = mean(a_k * (Phi^2 - theta_k)^2)
                    # Correct derivative: dE_dw/dPhi_ik = 4*a_k*Phi_ik*(Phi_ik^2 - theta_k)/(N*K)
                    dE_dw_dPhi = self.field.compute_double_well_gradient()

                    # ∂E_kinetic/∂Phi = -D_k * laplacian_Phi (через integration by parts)
                    # Спрощено: використовуємо numerичний градієнт
                    fe_current = self.field.compute_free_energy(1.0)

                    if is_torch:
                        # PyTorch Autograd path
                        W_beta.requires_grad_(True)
                        active_indices = self.field.active_byte_indices
                        
                        # Reconstruct interaction field in PyTorch to compute exact gradient
                        B = torch.matmul(self.tensors_full.u_beta, torch.matmul(W_beta, self.tensors_full.u_beta.T)) + self.tensors_full.B_residual
                        Phi_t = torch.tensor(Phi, device=W_beta.device, dtype=torch.float32)
                        V_t = torch.matmul(
                            torch.clamp(Phi_t, min=0.0),
                            B[active_indices][:, active_indices].T
                        )
                        emb_t = torch.tensor(self.embeddings, device=W_beta.device, dtype=torch.float32)
                        W_field_t = self.tensors_full.forward(V_t, emb_t, use_adaptive_lambda=True)
                        
                        max_w = torch.max(torch.abs(W_field_t))
                        if max_w > 1e-10:
                            W_field_t = 0.5 + 0.5 * (W_field_t / max_w)
                        else:
                            W_field_t = torch.ones_like(W_field_t) * 0.5
                            
                        dE_dw_dPhi_t = torch.tensor(dE_dw_dPhi, device=W_beta.device, dtype=torch.float32)
                        
                        # Loss representing dF/dW_beta
                        if W_field_t.ndim == 1:
                            inner_sum_t = torch.sum(dE_dw_dPhi_t * Phi_t, dim=1)
                            loss = 0.1 * torch.sum(W_field_t * inner_sum_t)
                        else:
                            loss = 0.1 * torch.sum(W_field_t * dE_dw_dPhi_t * Phi_t)
                        
                        if W_beta.grad is not None:
                            W_beta.grad.zero_()
                        
                        if loss.requires_grad:
                            loss.backward()
                            
                            eucl_grad_t = W_beta.grad.detach()
                            if not torch.all(torch.isfinite(eucl_grad_t)):
                                eucl_grad_t = torch.nan_to_num(eucl_grad_t, nan=0.0, posinf=10.0, neginf=-10.0)
                            eucl_grad_t = torch.clamp(eucl_grad_t, -10.0, 10.0)
                            
                            # Track running empirical Fisher diagonal (EMA of grad^2)
                            beta_fisher = 0.95
                            if not hasattr(self, 'fisher_ema_W_beta') or self.fisher_ema_W_beta is None or not isinstance(self.fisher_ema_W_beta, torch.Tensor) or self.fisher_ema_W_beta.device != W_beta.device:
                                self.fisher_ema_W_beta = torch.full_like(W_beta, 1e-5)
                            
                            if not torch.all(torch.isfinite(self.fisher_ema_W_beta)):
                                self.fisher_ema_W_beta = torch.nan_to_num(self.fisher_ema_W_beta, nan=1e-5, posinf=10.0, neginf=1e-5)
                            
                            self.fisher_ema_W_beta.mul_(beta_fisher).addcmul_(eucl_grad_t, eucl_grad_t, value=1.0 - beta_fisher)
                            
                            damping = 1e-3
                            fisher_denom = torch.sqrt(torch.clamp(self.fisher_ema_W_beta, min=1e-8)) + damping
                            nat_grad_t = eucl_grad_t / fisher_denom
                            
                            max_ng = torch.max(torch.abs(nat_grad_t))
                            if not torch.isfinite(max_ng):
                                max_ng = torch.tensor(100.0, device=W_beta.device)
                            if max_ng > 100.0:
                                nat_grad_t.mul_(100.0 / max_ng)
                                
                            with torch.no_grad():
                                W_beta.copy_(W_beta - lr * nat_grad_t)
                    else:
                        # NumPy fallback path: numerical coordinate perturbation
                        block_size = max(1, min(d1, 8))
                        eucl_grad = np.zeros((d1, d2), dtype=np.float32)
                        eps = 1e-4
                        
                        self.tensors_full.W_beta = W_beta_np
                        W_field_base = self.tensors_full.compute_interaction_field(
                            self.substrate, self.embeddings, field=self.field
                        )
                        
                        for bi in range(0, d1, block_size):
                            for bj in range(0, d2, block_size):
                                ci = min(bi + block_size // 2, d1 - 1)
                                cj = min(bj + block_size // 2, d2 - 1)
                                
                                W_pert = W_beta_np.copy()
                                W_pert[ci, cj] += eps
                                self.tensors_full.W_beta = W_pert
                                W_field_pert = self.tensors_full.compute_interaction_field(
                                    self.substrate, self.embeddings, field=self.field
                                )
                                dW_field = (W_field_pert - W_field_base) / eps
                                if not np.all(np.isfinite(dW_field)):
                                    dW_field = np.nan_to_num(dW_field, nan=0.0, posinf=1.0, neginf=-1.0)
                                
                                if dW_field.ndim == 1:
                                    inner_sum = np.sum(dE_dw_dPhi * Phi, axis=1)
                                    grad_center = float(0.1 * np.sum(dW_field * inner_sum))
                                else:
                                    grad_center = float(0.1 * np.sum(dW_field * dE_dw_dPhi * Phi))
                                
                                if not np.isfinite(grad_center):
                                    grad_center = 0.0
                                    
                                bi_end = min(bi + block_size, d1)
                                bj_end = min(bj + block_size, d2)
                                block_W = W_beta_np[bi:bi_end, bj:bj_end]
                                if abs(W_beta_np[ci, cj]) > 1e-10:
                                    scale_block = block_W / W_beta_np[ci, cj]
                                else:
                                    scale_block = np.ones_like(block_W)
                                eucl_grad[bi:bi_end, bj:bj_end] = grad_center * scale_block
                        
                        self.tensors_full.W_beta = W_beta_np
                        
                        if not np.all(np.isfinite(eucl_grad)):
                            eucl_grad = np.nan_to_num(eucl_grad, nan=0.0, posinf=10.0, neginf=-10.0)
                        eucl_grad = np.clip(eucl_grad, -10.0, 10.0)
                        
                        # Track running empirical Fisher diagonal in NumPy
                        beta_fisher = 0.95
                        if not hasattr(self, 'fisher_ema_W_beta') or self.fisher_ema_W_beta is None or isinstance(self.fisher_ema_W_beta, torch.Tensor):
                            self.fisher_ema_W_beta = np.full_like(W_beta_np, 1e-5)
                        
                        if not np.all(np.isfinite(self.fisher_ema_W_beta)):
                            self.fisher_ema_W_beta = np.nan_to_num(self.fisher_ema_W_beta, nan=1e-5, posinf=10.0, neginf=1e-5)
                            
                        self.fisher_ema_W_beta = beta_fisher * self.fisher_ema_W_beta + (1.0 - beta_fisher) * (eucl_grad ** 2)
                        
                        damping = 1e-3
                        nat_grad = eucl_grad / (np.sqrt(np.maximum(self.fisher_ema_W_beta, 1e-8)) + damping)
                        max_ng = np.max(np.abs(nat_grad))
                        if not np.isfinite(max_ng):
                            max_ng = 100.0
                        if max_ng > 100.0:
                            nat_grad *= 100.0 / max_ng
                            
                        self.tensors_full.W_beta -= lr * nat_grad

                    self.W_field = self.tensors_full.compute_interaction_field(
                        self.substrate, self.embeddings, field=self.field
                    )
                    self.field.update_feed_rate(self.W_field)

            # 5. Variational step
            if self.use_variational and self.variational is not None:
                obs = self.substrate.byte_distribution.astype(np.float32)
                self.variational.update(obs, lr=lr)

            # V5/V6 FIX: Навчання ієрархічного предиктивного кодування в learn_end_to_end
            if self.use_hierarchical_pc and self.hierarchical_pc is not None:
                temp_clusters = self.organizer.detect_clusters() if self.organizer is not None else []
                if len(temp_clusters) > 0:
                    if self.use_gnn_conversion and self.gnn_conversion is not None:
                        conv = self.gnn_conversion.convert(temp_clusters, self.substrate)
                    else:
                        conv_layers = ConversionLayersV3(n_levels=self.n_conversion_levels)
                        conv = conv_layers.convert(temp_clusters, self.substrate)
                else:
                    conv = None

                if conv is not None:
                    reps_list = []
                    for lvl_data in conv:
                        items = lvl_data.get('items', [])
                        if items:
                            reps = [item['representation'] for item in items]
                            if torch.is_tensor(reps[0]):
                                lvl_rep = torch.stack(reps).mean(dim=0).detach().cpu().numpy()
                            else:
                                lvl_rep = np.mean(reps, axis=0)
                        else:
                            lvl_idx = lvl_data['level']
                            d_lvl = self.hierarchical_pc.d_representations[min(lvl_idx, len(self.hierarchical_pc.d_representations)-1)]
                            lvl_rep = np.zeros(d_lvl, dtype=np.float32)
                        reps_list.append(lvl_rep)

                    self.hierarchical_pc.n_levels = len(conv)
                    self.hierarchical_pc.learn_step(reps_list, learning_rate=lr)

            # 6. Dynamic embedding update
            if self.use_dynamic_embedding and self.dynamic_embedding is not None:
                self.embeddings = self.dynamic_embedding.compute_embeddings(
                    self.substrate.raw_data,
                    self.field.u,
                    self.field.v,
                    self.field.Phi,
                    self.field.active_byte_indices,
                )
                if self.use_full_tensor and self.tensors_full is not None:
                    self.W_field = self.tensors_full.compute_interaction_field(
                        self.substrate, self.embeddings, field=self.field
                    )
                    self.field.update_feed_rate(self.W_field)

            fe_history.append(current_fe)
            print(f"   Epoch {epoch+1}/{n_epochs}: F_free={current_fe:.4f}")

        return {
            'status': 'completed',
            'initial_fe': float(initial_fe),
            'final_fe': float(fe_history[-1]),
            'fe_history': fe_history,
        }

