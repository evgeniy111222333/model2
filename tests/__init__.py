"""
BCS Stage 1 Tests: Internal Understanding & Processing

Test structure:
- e0_*: Sanity checks
- e1_*: Modality detection
- e2_*: Structure detection (clusters, patterns)
- e3_*: Predictive coding (learning)
- e4_*: Memory and crystallization
- e5_*: Variational inference
"""

import numpy as np
import sys
import os

# Add E:\arc to path so 'bcs' package works
# tests/__init__.py is at E:\arc\bcs\tests\ -> parent is E:\arc\bcs -> grandparent is E:\arc
bcs_grandparent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if bcs_grandparent not in sys.path:
    sys.path.insert(0, bcs_grandparent)

from bcs.model import BCSModelV6


class_colors = {
    'PASS': '[PASS]',
    'FAIL': '[FAIL]',
    'WARN': '[WARN]',
    'INFO': '[INFO]',
    'RESET': '',
}


def print_header(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    status = f"{class_colors['PASS']}PASS{class_colors['RESET']}" if passed else f"{class_colors['FAIL']}FAIL{class_colors['RESET']}"
    print(f"  {status} | {name}")
    if details:
        print(f"        {details}")


def create_model(**kwargs):
    """Create model with Stage 1 relevant settings."""
    # Remove run() params from kwargs - they go to run(), not init()
    kwargs.pop('n_steps', None)
    kwargs.pop('record_every', None)

    defaults = {
        # Flags
        'use_variational': False,  # Start simple
        'use_ib_optimizer': False,
        'use_multiscale_opt': False,
        'use_hierarchical_pc': False,
        'use_prediction_error_loop': True,
        'use_token_discovery': False,
        'use_phase_analysis': False,
        'use_gnn_conversion': False,
        'use_fisher_geometry': False,
        'use_crystallized_memory': True,
        'use_working_memory': True,
        'use_cluster_recognition': True,
        'use_context_resonance': True,
        'use_knowledge_transfer': False,
        'use_level_splitting': False,
        'use_sequence_memory': False,
        'use_semantic_dynamics': False,
        'use_semantic_readout': False,
        'device': 'cpu',
    }
    defaults.update(kwargs)
    return BCSModelV6(**defaults)


def run_model(data, **kwargs):
    """Helper to run model and return results."""
    n_steps = kwargs.pop('n_steps', 200)
    record_every = kwargs.pop('record_every', 100)
    model = create_model(**kwargs)
    model.ingest(data).build_tensors().init_field()
    results = model.run(n_steps=n_steps, record_every=record_every)
    return model, results