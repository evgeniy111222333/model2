"""Memory profiling test: verify chunking implementation with realistic data sizes."""
import sys, os
sys.path.insert(0, 'E:/arc')
os.chdir('E:/arc/bcs')
from bcs.core.field import FieldSystemV6
from bcs.core.interaction import TorchSpaceValueInteractionV8, FFTSpaceValueInteractionV7
from bcs.core.substrate import ByteSubstrate
from bcs.core.embedding import DynamicByteEmbedding
import numpy as np

print("=== MEMORY/CHUNKING TEST ===")

# Test with 100K bytes (realistic text size)
for N in [1000, 10000, 100000]:
    data = b'x' * N
    sub = ByteSubstrate(data)

    for K in [32, 64, 128, 256]:
        field = FieldSystemV6(substrate=sub, n_active_bytes=K)

        # Time step execution (chunked for large N)
        import time
        start = time.time()
        for _ in range(5):
            field.step()
        elapsed = time.time() - start

        phi_mem_mb = N * K * 4 / 1024**2
        print(f"N={N:>7}, K={K}: Phi={phi_mem_mb:>6.1f}MB, 5steps={elapsed:.3f}s, Phi.shape={field.Phi.shape}")

print()

# Test TorchSpaceValueInteractionV8 chunking
print("=== INTERACTION CHUNKING TEST ===")
N = 50000
K = 64
d = 64
data = np.random.randint(0, 256, N, dtype=np.uint8).tobytes()
sub = ByteSubstrate(data)
field = FieldSystemV6(substrate=sub, n_active_bytes=K)

# Create interaction
interaction = TorchSpaceValueInteractionV8(
    d_embedding=d, d_beta=32, lambda_base=8.0, k_neighbors=16
)

# Embeddings
emb_np = np.random.randn(N, d).astype(np.float32)
Phi = field.Phi

# Compute field (uses chunking internally)
import time
start = time.time()
W = interaction.compute_interaction_field(sub, emb_np, field=field)
elapsed = time.time() - start

print(f"N={N}, K={K}, d={d}: interaction field computed in {elapsed:.3f}s")
print(f"W shape: {W.shape}, W range: [{W.min():.3f}, {W.max():.3f}]")

print()
print("=== MEMORY/CHUNKING TEST COMPLETE ===")
