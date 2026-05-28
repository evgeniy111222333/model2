"""
WORD/TOKEN ANALYSIS - FIXED VERSION
Tests the full pipeline with proper UTF-8 handling and visual display.
"""
import sys, os
sys.path.insert(0, 'E:/arc')
os.chdir('E:/arc/bcs')

import numpy as np
from bcs.perception.utf8_segmenter import UTF8Segmenter, AdaptiveUTF8Segmenter
from bcs.information.character_manifold import CharacterManifold
from bcs.information.character_trajectory import CharacterTrajectory, create_character_trajectory
from bcs.perception.token import EmergentTokenDiscovery
from bcs.core.substrate import ByteSubstrate
from bcs.core.field import FieldSystemV6
from bcs.perception.organization import SelfOrganizerV4
from bcs.perception.predictive import PredictiveCoding


def safe_repr(token_bytes: bytes) -> str:
    """Display token, showing raw hex if not valid UTF-8."""
    try:
        s = token_bytes.decode('utf-8')
        if '\ufffd' in s:
            return f"<hex:{token_bytes.hex()}>"
        return f"'{s}'"
    except:
        return f"<hex:{token_bytes.hex()}>"


def cluster_repr(data: bytes, start: int, end: int) -> str:
    """Display cluster with UTF-8 safety."""
    seg_bytes = data[start:end]
    try:
        s = seg_bytes.decode('utf-8')
        if '\ufffd' in s:
            return f"hex:{seg_bytes.hex()}"
        return f"'{s}'"
    except:
        return f"hex:{seg_bytes.hex()}"


print("=" * 70)
print("  BCS WORD/TOKEN ANALYSIS")
print("=" * 70)

# === TEST DATA ===
test_text = "Привіт світе! Hello world. Привіт всім на світі! How are you today?"
test_data = test_text.encode('utf-8')
print(f"\nText: '{test_text}'")
print(f"Bytes: {len(test_data)}")

# =========================================================================
# BLOCK 1: UTF-8 Segmentation
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 1: UTF-8 SEGMENTATION")
print("=" * 70)

segmenter = UTF8Segmenter()
sequences = segmenter.segment(test_data)
print(f"Total sequences: {len(sequences)} (ASCII: {sum(1 for s in sequences if s.is_ascii)}, Multibyte: {sum(1 for s in sequences if s.is_multibyte)})")

print("\nAll characters:")
print("  " + " ".join(f"{s.char_str}" for s in sequences))

adaptive_seg = AdaptiveUTF8Segmenter()
adaptive_seg.learn_regions(test_data)
print(f"\nSelf-learned regions: {list(adaptive_seg.regions.keys())}")
for region, stats in adaptive_seg.region_stats.items():
    print(f"  {region}: {stats['count']} chars (cp U+{int(stats['min_cp']):04X}-U+{int(stats['max_cp']):04X})")

# =========================================================================
# BLOCK 2: CHARACTER MANIFOLD
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 2: CHARACTER MANIFOLD (self-learned)")
print("=" * 70)

seq_data = [(seq.bytes_data, seq.start) for seq in sequences]
manifold = CharacterManifold(embedding_dim=32)
manifold.add_sequences(seq_data)
manifold.learn_from_data(test_data, seq_data)

stats = manifold.get_stats()
print(f"Characters: {stats['n_characters']}, Regions: {stats['n_regions']}, Transitions: {stats['n_transitions']}")

print("\nCharacters by region:")
for name, r in manifold.regions.items():
    valid = [cp for cp in r.codepoints if 0x20 <= cp < 0x7F or cp >= 0x0410]
    chars_str = ''.join(chr(cp) for cp in sorted(valid) if chr(cp).isprintable())
    print(f"  {name}: {chars_str}")

# =========================================================================
# BLOCK 3: MANIFOLD PREDICTION
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 3: CHARACTER PREDICTION (from learned transitions)")
print("=" * 70)

context = [ord('П'), ord('р'), ord('и')]
print(f"\nContext: 'При'")
predictions = manifold.predict_next_chars(context, top_k=5)
for cp, prob in predictions:
    char = chr(cp) if cp < 0x110000 else '?'
    print(f"  '{char}' (U+{cp:04X}): {prob:.3f}")

context2 = [ord('H'), ord('e'), ord('l'), ord('l'), ord('o')]
print(f"\nContext: 'Hello'")
predictions2 = manifold.predict_next_chars(context2, top_k=5)
for cp, prob in predictions2:
    char = chr(cp) if cp < 0x110000 else '?'
    print(f"  '{char}' (U+{cp:04X}): {prob:.3f}")

# =========================================================================
# BLOCK 4: CHARACTER TRAJECTORY
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 4: CHARACTER TRAJECTORY (geometric continuation)")
print("=" * 70)

char_traj = create_character_trajectory(test_data)
traj_stats = char_traj.get_stats()
print(f"Trajectory: {traj_stats['n_characters']} chars, regions: {traj_stats.get('manifold', {}).get('n_regions', 'N/A')}")

print("\nContinuation tests:")
for prefix in ["Привіт ", "Hello ", "How "]:
    result = char_traj.continue_string(prefix, max_len=12)
    display = result if result else prefix
    valid = ''.join(c for c in display if c.isprintable())
    print(f"  '{prefix}' -> '{valid}'")

# =========================================================================
# BLOCK 5: FIELD + CLUSTERS -> EMERGENT TOKENS
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 5: FIELD EVOLUTION + CLUSTERS -> EMERGENT TOKENS")
print("=" * 70)

sub = ByteSubstrate(test_data)
field = FieldSystemV6(substrate=sub, n_active_bytes=64)

print("Running field evolution (100 steps)...")
for step in range(100):
    field.step()
    if step % 25 == 0:
        fe = field.compute_free_energy()
        print(f"  Step {step}: phi={field.Phi.mean():.4f}, F={fe:.4f}")

organizer = SelfOrganizerV4(field_system=field, predictive_coding=PredictiveCoding())
clusters = organizer.detect_clusters()

print(f"\nClusters found: {len(clusters)}")
for i, c in enumerate(clusters):
    seg_display = cluster_repr(test_data, c['start'], c['end'])
    print(f"  [{i}] byte[{c['start']}:{c['end']}] {seg_display} size={c['size']} q={c.get('quality_score', 0):.2f}")

# =========================================================================
# BLOCK 6: EMERGENT TOKEN DISCOVERY (KEY TEST)
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 6: EMERGENT TOKEN DISCOVERY")
print("=" * 70)

token_discovery = EmergentTokenDiscovery(
    min_frequency=2,
    max_token_length=8,
    min_info_gain=0.05,
)

u_field = field.u if hasattr(field, 'u') else None
pc = PredictiveCoding()
tokens = token_discovery.discover(sub, clusters, pc=pc, u_field=u_field)

print(f"\nTokens discovered: {len(tokens)}")
sig_tokens = [t for t in tokens if t['is_statistically_significant']]
print(f"Statistically significant: {len(sig_tokens)}")

print("\nTop 15 tokens by quality:")
print(f"  {'#':>2} {'Token':>20} {'Len':>3} {'Freq':>4} {'IG':>6} {'Pred':>6} {'p':>6} {'Sig':>3}")
print(f"  {'-'*2} {'-'*20} {'-'*3} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*3}")
for i, t in enumerate(tokens[:15]):
    token_display = safe_repr(t['token'])
    sig = "YES" if t['is_statistically_significant'] else "no"
    print(f"  {i+1:>2} {token_display:>20} {t['length']:>3} {t['frequency']:>4} "
          f"{t['info_gain']:>6.3f} {t['predictive_power']:>6.3f} {t['p_value']:>6.3f} {sig:>3}")

print("\nValid UTF-8 tokens (potential words):")
for t in tokens:
    try:
        s = t['token'].decode('utf-8')
        if '\ufffd' not in s and len(s) >= 3 and s.isprintable():
            sig = "***" if t['is_statistically_significant'] else "   "
            print(f"  '{s}' len={t['length']} freq={t['frequency']} IG={t['info_gain']:.3f} p={t['p_value']:.3f} {sig}")
    except:
        pass

# =========================================================================
# BLOCK 7: DEEP ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("  BLOCK 7: KEY INSIGHTS & DIAGNOSIS")
print("=" * 70)

valid_utf8 = 0
broken_utf8 = 0
for t in tokens:
    try:
        t['token'].decode('utf-8')
        valid_utf8 += 1
    except:
        broken_utf8 += 1

print(f"\nUTF-8 validity in discovered tokens: {valid_utf8} OK, {broken_utf8} broken")
print(f"\nField statistics:")
print(f"  Phi shape: {field.Phi.shape}")
print(f"  Phi range: [{field.Phi.min():.4f}, {field.Phi.max():.4f}]")
print(f"  u range: [{field.u.min():.4f}, {field.u.max():.4f}]")

print("\nUTF-8 boundary analysis in clusters:")
for c in clusters:
    starts_with_cont = c['start'] < len(test_data) and 0x80 <= test_data[c['start']] <= 0xBF
    ends_with_cont = c['end'] > 0 and c['end'] <= len(test_data) and 0x80 <= test_data[c['end']-1] <= 0xBF
    status = "UTF-8 SAFE" if not starts_with_cont and not ends_with_cont else "CROSSES BOUNDARY"
    print(f"  [{c['start']}:{c['end']}] {status}")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)