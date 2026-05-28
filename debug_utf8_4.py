import sys
sys.path.insert(0, 'E:/arc')
from bcs.perception.organization import _is_utf8_lead_byte, _is_utf8_continuation_byte

data = b'\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd1\x96\xd1\x82'
print(f'Data: {data.hex()} (len={len(data)})')

def is_valid_utf8_boundary(start, end, data):
    """Check if [start:end] is a valid UTF-8 boundary."""
    N = len(data)
    # Start byte must not be a continuation
    if start > 0 and 0x80 <= data[start-1] <= 0xBF:
        return False, f"starts with continuation byte 0x{data[start-1]:02X}"
    # End byte (exclusive) - check byte before end
    if end > 0 and end <= N and 0x80 <= data[end-1] <= 0xBF:
        return False, f"ends with continuation byte 0x{data[end-1]:02X}"
    return True, "safe"

# Test clusters
test_cases = [
    (0, 9, "cluster 1"),
    (10, 12, "cluster 2"),
]

for start, end, name in test_cases:
    valid, reason = is_valid_utf8_boundary(start, end, data)
    if start > 0:
        byte_before = data[start-1]
    else:
        byte_before = None
    if end > 0 and end <= len(data):
        byte_at_end = data[end-1]
    else:
        byte_at_end = None
    print(f'\n{name} [{start}:{end}]:')
    print(f'  Byte before start: {hex(byte_before) if byte_before else "N/A"}')
    print(f'  Byte at end-1: {hex(byte_at_end) if byte_at_end else "N/A"}')
    print(f'  Valid: {valid} - {reason}')

# Now check what _snap_cluster_boundaries_to_utf8 actually does
print('\n--- Testing _snap_cluster_boundaries_to_utf8 ---')
from bcs.perception.organization import _snap_cluster_boundaries_to_utf8

clusters = [{'start': 0, 'end': 9, 'size': 9}, {'start': 10, 'end': 12, 'size': 2}]
print(f'Input clusters: {clusters}')

# Check each cluster's validity before
for c in clusters:
    valid, reason = is_valid_utf8_boundary(c['start'], c['end'], data)
    print(f'Before snap: [{c["start"]}:{c["end"]}] valid={valid}')

# Call snap function
snapped = _snap_cluster_boundaries_to_utf8(clusters, data)
print(f'Snapped clusters: {len(snapped)}')

# Check each cluster's validity after
for c in snapped:
    valid, reason = is_valid_utf8_boundary(c['start'], c['end'], data)
    start, end = c['start'], c['end']
    print(f'After snap: [{start}:{end}] valid={valid} - {reason}')