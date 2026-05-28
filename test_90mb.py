"""Test 90MB with optimized CharacterManifold."""

import sys, io, time
sys.path.insert(0, 'E:/arc')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from bcs.perception.utf8_segmenter import UTF8Segmenter
from bcs.information.character_manifold_v2 import CharacterManifoldOptimized
from bcs.information.character_continuation import CharacterGeometricContinuation

print("=" * 60)
print("90MB TEST - Optimized CharacterManifold")
print("=" * 60)

# Load 90MB
size = 90 * 1024 * 1024
print("Loading 90MB...")
with open('E:/arc/codesearchnet_python.txt', 'r', encoding='utf-8', errors='replace') as f:
    data = f.read(size).encode('utf-8')
print(f"Loaded: {len(data)/1024/1024:.1f} MB")

# Parse
print("Parsing...")
seg = UTF8Segmenter()
t0 = time.time()
seqs = seg.segment(data)
parse_time = time.time() - t0
print(f"Parsed: {len(seqs):,} sequences in {parse_time:.1f}s")

# Build manifold
print("Building manifold...")
seq_data = [(s.bytes_data, s.start) for s in seqs]
m = CharacterManifoldOptimized()
m.add_sequences(seq_data)

t0 = time.time()
m.learn_from_data(data, seq_data, verbose=True)
learn_time = time.time() - t0

stats = m.get_stats()
print("DONE!")
print(f"Learn time: {learn_time:.1f}s")
print(f"Characters: {stats['n_characters']}")
print(f"Regions: {stats['n_regions']}")
print(f"Transitions: {stats['n_transitions']}")

# Test continuation
print("Testing continuation...")
cont = CharacterGeometricContinuation(m)
traj = [(s.codepoint, s.start) for s in seqs][:100]
probs = cont.continue_from_trajectory(traj)
top = sorted(probs.items(), key=lambda x: -x[1])[:5]

preds = []
for cp, p in top:
    try:
        c = chr(cp)
        if c.isprintable() and c not in ['\n', '\r']:
            preds.append(f"'{c}'")
    except:
        pass
print(f"Top predictions: {preds}")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
before_estimate = 38.28 * (90 / 5)  # linear from 5MB test
print(f"BEFORE (estimated for 90MB): ~{before_estimate:.0f}s")
print(f"AFTER: {learn_time:.1f}s")
print(f"SPEEDUP: ~{before_estimate/learn_time:.1f}x")
