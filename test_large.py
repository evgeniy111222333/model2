"""Test with 20% ~90MB of CodeSearchNet."""

import sys, io, time
sys.path.insert(0, 'E:/arc')  # Add project root to path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("Testing with 20% of CodeSearchNet (~90MB)")
print("=" * 60)

# Import
import numpy as np
from bcs.perception.utf8_segmenter import UTF8Segmenter
from bcs.information.character_manifold import CharacterManifold
from bcs.information.character_continuation import CharacterGeometricContinuation
from bcs.information.character_trajectory import CharacterTrajectory

# Load 20%
print("\n[1] Loading 20% of data...")
with open('E:/arc/codesearchnet_python.txt', 'r', encoding='utf-8', errors='replace') as f:
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0)
    chunk_size = file_size // 5
    raw_data = f.read(chunk_size)

data = raw_data.encode('utf-8')
print(f"  File size: {file_size/1024/1024:.1f} MB")
print(f"  20% chunk: {len(data)/1024/1024:.1f} MB ({len(data):,} bytes)")

# Parse
print("\n[2] Parsing UTF-8 sequences...")
t0 = time.time()
seg = UTF8Segmenter()
seqs = seg.segment(data)
elapsed = time.time() - t0
print(f"  Sequences: {len(seqs):,}")
print(f"  Time: {elapsed:.1f}s")
print(f"  Rate: {len(data)/elapsed/1024/1024:.1f} MB/s")

# Build manifold
print("\n[3] Building character manifold...")
t0 = time.time()
seq_data = [(s.bytes_data, s.start) for s in seqs]
m = CharacterManifold()
m.add_sequences(seq_data)
m.learn_from_data(data, seq_data)
elapsed = time.time() - t0

stats = m.get_stats()
print(f"  Characters: {stats['n_characters']}")
print(f"  Regions: {stats['n_regions']}")
print(f"  Transitions: {stats['n_transitions']}")
print(f"  Time: {elapsed:.1f}s")

# Build continuation
print("\n[4] Building continuation...")
ct = CharacterTrajectory()
ct.segmenter = seg
ct.manifold = m
ct.continuation = CharacterGeometricContinuation(m)

# Build character trajectory
ct.character_points = [(s.codepoint, s.start) for s in seqs]
print(f"  Trajectory points: {len(ct.character_points):,}")

# Test continuation for Python keywords
print("\n[5] Testing continuation quality...")
print("-" * 60)

tests = ['def ', 'class ', 'import ', 'return ', 'self.', 'if ', 'for ', 'while ', 'try:', 'except', 'async ', 'await ']

for prefix in tests:
    try:
        pos = raw_data.find(prefix)
        if pos >= 0:
            context_bytes = raw_data[:pos + len(prefix)].encode('utf-8')
            context_seqs = seg.segment(context_bytes)
            traj = [(s.codepoint, s.start) for s in context_seqs][-100:]
            
            probs = ct.continuation.continue_from_trajectory(traj)
            top = sorted(probs.items(), key=lambda x: -x[1])[:5]
            
            preds = []
            for cp, p in top:
                try:
                    c = chr(cp)
                    if c.isprintable() and c not in ['\n', '\r']:
                        preds.append(f"'{c}'")
                    else:
                        preds.append(f"U+{cp:04X}")
                except:
                    preds.append(f"U+{cp:04X}")
            
            print(f"  After '{prefix}' -> [{', '.join(preds)}]")
    except Exception as e:
        print(f"  '{prefix}' -> error: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
