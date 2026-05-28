"""Regression test: verify BCS still works after field.py fix."""
import sys, os
sys.path.insert(0, 'E:/arc')
os.chdir('E:/arc/bcs')
from bcs.core.field import FieldSystemV6
from bcs.core.substrate import ByteSubstrate
import numpy as np

print("=== REGRESSION TEST ===")

# 1. Field initialization
data = b'hello world test text for testing'
sub = ByteSubstrate(data)
field = FieldSystemV6(substrate=sub, n_active_bytes=16)
assert hasattr(field, 'u'), "u field not initialized"
assert hasattr(field, 'v'), "v field not initialized"
assert hasattr(field, 'Phi'), "Phi field not initialized"
assert hasattr(field, 'step_count'), "step_count not initialized"
print("[1] Field init OK")

# 2. Field step runs
field.step()
field.step()
field.step()
assert field.step_count == 3, f"step_count={field.step_count}, expected 3"
print("[2] Field step OK")

# 3. u and v updated
assert field.u is not None and len(field.u) == field.N, "u not updated"
assert field.v is not None and len(field.v) == field.N, "v not updated"
print("[3] u/v fields OK")

# 4. Chunked mode for large N
large_data = b'x' * 20000
large_sub = ByteSubstrate(large_data)
large_field = FieldSystemV6(substrate=large_sub, n_active_bytes=16, neighborhood_size=5)
assert large_field.N == 20000, f"N={large_field.N}, expected 20000"
assert large_field.Phi.shape == (20000, 16), f"Phi shap={large_field.Phi.shape}"
large_field.step()  # uses chunk_size=16384 path
print("[4] Chunked mode OK (N=20000)")

# 5. Free energy computation
fe = large_field.compute_free_energy()
assert np.isfinite(fe), f"free_energy={fe} not finite"
print(f"[5] Free energy OK (FE={fe:.4f})")

# 6. Field statistics
stats = large_field.get_field_statistics()
assert 'u_mean' in stats and 'phi_mean' in stats
print(f"[6] Field stats OK: u_mean={stats['u_mean']:.4f}")

print()
print("=== ALL TESTS PASSED ===")
