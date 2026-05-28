"""
Quick BCS Test Suite - Focused on key functionality
"""
import sys
sys.path.insert(0, 'E:/arc')

import time
import numpy as np
from typing import List, Dict

print("=" * 80)
print("QUICK BCS TEST SUITE")
print("=" * 80)

# =========================================================================
# TEST 1: Script Detection
# =========================================================================
print("\n[TEST 1] Script Detection")
from bcs.information.character_manifold import CharacterManifold
m = CharacterManifold()

test_cases = [
    (0x41, 'latin'), (0x61, 'latin'), (0x5A, 'latin'), (0x7A, 'latin'),  # ASCII letters
    (0x41F, 'cyrillic'), (0x044F, 'cyrillic'),  # Cyrillic
    (0x20, 'ascii'), (0x21, 'ascii'),  # Punctuation
    (0x4E00, 'cjk'), (0x3040, 'hiragana'),  # CJK
]

passed = sum(1 for cp, exp in test_cases if m.get_script_for_cp(cp) == exp)
print(f"  Script detection: {passed}/{len(test_cases)} passed")
for cp, exp in test_cases:
    if m.get_script_for_cp(cp) != exp:
        print(f"    FAIL: U+{cp:04X} expected {exp}, got {m.get_script_for_cp(cp)}")

# =========================================================================
# TEST 2: Script Filtering in Continuation
# =========================================================================
print("\n[TEST 2] Script Filtering")
from bcs.perception.utf8_segmenter import UTF8Segmenter
from bcs.information.character_manifold import create_character_manifold
from bcs.information.character_continuation import CharacterGeometricContinuation

text = "Привіт Hello world Привіт"
data = text.encode('utf-8')
segmenter = UTF8Segmenter()
sequences = segmenter.segment(data)
seq_data = [(seq.bytes_data, seq.start) for seq in sequences]
manifold = create_character_manifold(seq_data, data)
cont = CharacterGeometricContinuation(manifold=manifold)

# Test after Cyrillic
cyrillic_traj = [(seq.codepoint, seq.start) for seq in sequences[:6]]  # 'Привіт '
active = manifold.get_active_script(cyrillic_traj)
probs = cont.continue_from_trajectory(cyrillic_traj)
top_cps = sorted(probs.items(), key=lambda x: -x[1])[:5]
top_scripts = [manifold.get_script_for_cp(cp) for cp, _ in top_cps]
cyrillic_count = top_scripts.count('cyrillic') + top_scripts.count('latin_ext') + top_scripts.count('ascii')
latin_count = top_scripts.count('latin')
print(f"  After 'Привіт ': active={active}, top predictions scripts: {top_scripts}")
print(f"    Cyrillic-related in top 5: {cyrillic_count}")

# Test after Latin
latin_traj = [(seq.codepoint, seq.start) for seq in sequences[7:13]]  # 'Hello '
active = manifold.get_active_script(latin_traj)
probs = cont.continue_from_trajectory(latin_traj)
top_cps = sorted(probs.items(), key=lambda x: -x[1])[:5]
top_scripts = [manifold.get_script_for_cp(cp) for cp, _ in top_cps]
cyrillic_count = top_scripts.count('cyrillic') + top_scripts.count('latin_ext') + top_scripts.count('ascii')
latin_count = top_scripts.count('latin')
print(f"  After 'Hello ': active={active}, top predictions scripts: {top_scripts}")
print(f"    Latin in top 5: {latin_count}, Cyrillic in top 5: {cyrillic_count}")

# =========================================================================
# TEST 3: UTF-8 Boundary Handling
# =========================================================================
print("\n[TEST 3] UTF-8 Boundaries")
from bcs.core.substrate import ByteSubstrate
from bcs.core.field import FieldSystemV6
from bcs.perception.organization import SelfOrganizerV4

def is_valid_utf8(start, end, data):
    if start > 0 and 0x80 <= data[start-1] <= 0xBF:
        return False
    if end < len(data) and 0x80 <= data[end] <= 0xBF:
        return False
    return True

texts = ["Hello", "Привіт", "你好", "Привіт Hello"]
all_safe = True
for text in texts:
    data = text.encode('utf-8')
    sub = ByteSubstrate(data)
    field = FieldSystemV6(substrate=sub, n_active_bytes=64)
    for _ in range(20):
        field.step()
    organizer = SelfOrganizerV4(field_system=field)
    clusters = organizer.detect_clusters()
    
    safe_count = sum(1 for c in clusters if is_valid_utf8(c['start'], c['end'], data))
    total = len(clusters)
    status = "✓" if safe_count == total else "✗"
    if safe_count != total:
        all_safe = False
    print(f"  {status} '{text}': {safe_count}/{total} clusters UTF-8 safe")

print(f"  Overall: {'ALL SAFE' if all_safe else 'SOME UNSAFE'}")

# =========================================================================
# BENCHMARK: Performance
# =========================================================================
print("\n" + "=" * 80)
print("PERFORMANCE BENCHMARKS")
print("=" * 80)

from bcs.perception.predictive import PredictiveCoding
from bcs.perception.token import EmergentTokenDiscovery

# Small data
print("\n[Benchmark 1] Small data (95 bytes)")
text = "Привіт світе! Hello world. Привіт всім на світі! How are you today?"
data = text.encode('utf-8')

times = []
for _ in range(3):
    t0 = time.perf_counter()
    sub = ByteSubstrate(data)
    field = FieldSystemV6(substrate=sub, n_active_bytes=64)
    for _ in range(100):
        field.step()
    organizer = SelfOrganizerV4(field_system=field)
    clusters = organizer.detect_clusters()
    td = EmergentTokenDiscovery(min_frequency=2, max_token_length=10, min_info_gain=0.01)
    tokens = td.discover(sub, clusters)
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"  Total time: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")
print(f"  Clusters: {len(clusters)}, Tokens: {len(tokens)}")

# Medium data (5KB)
print("\n[Benchmark 2] Medium data (~5KB)")
text_medium = (text + " ") * 50
data = text_medium.encode('utf-8')
print(f"  Size: {len(data)} bytes")

t0 = time.perf_counter()
sub = ByteSubstrate(data)
field = FieldSystemV6(substrate=sub, n_active_bytes=64)
for _ in range(50):  # Fewer steps
    field.step()
organizer = SelfOrganizerV4(field_system=field)
clusters = organizer.detect_clusters()
t1 = time.perf_counter()

print(f"  Total time: {(t1-t0)*1000:.1f}ms")
print(f"  Clusters: {len(clusters)}")

# Large data (20KB)
print("\n[Benchmark 3] Large data (~20KB)")
import random
chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
large = ''.join(random.choice(chars) for _ in range(20000))
data = large.encode('utf-8')
print(f"  Size: {len(data)} bytes")

t0 = time.perf_counter()
sub = ByteSubstrate(data)
field = FieldSystemV6(substrate=sub, n_active_bytes=64)
for _ in range(30):
    field.step()
organizer = SelfOrganizerV4(field_system=field)
clusters = organizer.detect_clusters()
t1 = time.perf_counter()

print(f"  Total time: {(t1-t0)*1000:.1f}ms")
print(f"  Clusters: {len(clusters)}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)