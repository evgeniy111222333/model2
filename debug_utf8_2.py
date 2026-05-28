import sys
sys.path.insert(0, 'E:/arc')
from bcs.core.substrate import ByteSubstrate
from bcs.core.field import FieldSystemV6
from bcs.perception.organization import SelfOrganizerV4, _snap_cluster_boundaries_to_utf8

text = 'Привіт'
data = text.encode('utf-8')
print(f'Data: {data.hex()} (len={len(data)})')

sub = ByteSubstrate(data)
field = FieldSystemV6(substrate=sub, n_active_bytes=64)
for _ in range(20):
    field.step()

organizer = SelfOrganizerV4(field_system=field)
clusters = organizer.detect_clusters()
print(f'Clusters after detect_clusters: {len(clusters)}')
for c in clusters:
    start, end = c['start'], c['end']
    print(f'  [{start}:{end}]')

# Test _snap_cluster_boundaries_to_utf8 directly
print('\nTesting _snap_cluster_boundaries_to_utf8 directly:')
test_clusters = [
    {'start': 0, 'end': 9, 'size': 9, 'positions': list(range(9))},
    {'start': 10, 'end': 11, 'size': 1, 'positions': [10]}
]
snapped = _snap_cluster_boundaries_to_utf8(test_clusters, data)
print(f'Snapped clusters: {len(snapped)}')
for c in snapped:
    start, end = c['start'], c['end']
    print(f'  [{start}:{end}]')

# Check the function exists and works
print('\nDirect call test:')
result = _snap_cluster_boundaries_to_utf8(clusters, data)
print(f'Result: {len(result)} clusters')