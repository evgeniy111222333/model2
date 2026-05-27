"""Quick test for pattern_group issue"""
import sys
import os

# Add E:\arc to path (grandparent of tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bcs.model import BCSModelV6

# Create model with Stage 1 settings
model = BCSModelV6(
    use_variational=False,
    use_ib_optimizer=False,
    use_multiscale_opt=False,
    use_hierarchical_pc=False,
    use_prediction_error_loop=True,
    use_token_discovery=False,
    use_phase_analysis=False,
    use_gnn_conversion=False,
    use_fisher_geometry=False,
    use_crystallized_memory=True,
    use_working_memory=True,
    use_cluster_recognition=True,
    use_context_resonance=True,
    device='cpu',
)

# Test data: XYZW repeated 3 times - should be 3 similar clusters with same pattern_group
data = b'XYZW' * 50 + b'XYZW' * 50 + b'XYZW' * 50
model.ingest(data).build_tensors().init_field()
results = model.run(n_steps=200, record_every=100)

clusters = results.get('final_clusters', [])
print(f'\n=== PATTERN GROUP TEST ===')
print(f'n_clusters: {len(clusters)}')
print(f'pattern_groups: {set(c.get("pattern_group") for c in clusters)}')

# Check which clusters have pattern_group
has_pg = [c.get('pattern_group') is not None for c in clusters]
print(f'clusters with pattern_group: {sum(has_pg)}/{len(clusters)}')

if len(clusters) > 0:
    print(f'\nFirst 3 clusters:')
    for i, c in enumerate(clusters[:3]):
        print(f'  Cluster {i}: [{c["start"]}:{c["end"]}], pattern_group={c.get("pattern_group")}')