import sys
sys.path.insert(0, 'E:/arc')
from bcs.core.substrate import ByteSubstrate
from bcs.core.field import FieldSystemV6
from bcs.perception.organization import SelfOrganizerV4

text = 'Привіт'
data = text.encode('utf-8')
print(f'Text: {text}')
print(f'Data length: {len(data)} bytes')
print(f'Data hex: {data.hex()}')

sub = ByteSubstrate(data)
field = FieldSystemV6(substrate=sub, n_active_bytes=64)
for _ in range(20):
    field.step()

organizer = SelfOrganizerV4(field_system=field)
has_snap = hasattr(organizer, '_snap_boundaries')
print(f'Organizer has _snap_boundaries: {has_snap}')

clusters = organizer.detect_clusters()
print(f'Clusters found: {len(clusters)}')
for c in clusters:
    print(f'  [{c["start"]}:{c["end"]}] size={c["size"]}')

# Check boundaries
print('Boundary analysis:')
for c in clusters:
    start, end = c['start'], c['end']
    if start > 0:
        b_before = data[start-1]
        is_cont_before = 0x80 <= b_before <= 0xBF
        print(f'  [{start}:{end}] byte before start: 0x{b_before:02X} is_cont={is_cont_before}')
    if end < len(data):
        b_at = data[end]
        is_cont_at = 0x80 <= b_at <= 0xBF
        print(f'  [{start}:{end}] byte at end: 0x{b_at:02X} is_cont={is_cont_at}')

# Check last_boundaries
print(f'Last boundaries: {organizer.last_boundaries}')