import sys
sys.path.insert(0, 'E:/arc')
import numpy as np
from bcs.perception.organization import _is_utf8_lead_byte, _is_utf8_continuation_byte

data = b'\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd1\x96\xd1\x82'
print(f'Data: {data.hex()} (len={len(data)})')
print(f'Bytes:')
for i, b in enumerate(data):
    print(f'  [{i}] 0x{b:02X} lead={_is_utf8_lead_byte(b)} cont={_is_utf8_continuation_byte(b)}')

# Cluster [0:9] analysis:
# - Starts at 0, so safe
# - Ends at 9, data[8] = 0x96 which is continuation (0x80-0xBF)
# So this cluster ends in the MIDDLE of a UTF-8 sequence!

print('\nCluster [0:9] analysis:')
print(f'  End byte (data[8]): 0x{data[8]:02X}')
print(f'  Is continuation: {0x80 <= data[8] <= 0xBF}')

# Cluster [10:11] analysis:
# - Starts at 10, data[9] = 0x96 (continuation)
# So this cluster STARTS with continuation!

print('\nCluster [10:11] analysis:')
print(f'  Start byte (data[10]): 0x{data[10]:02X}')
print(f'  Is continuation: {0x80 <= data[10] <= 0xBF}')

# Now test find_safe_end
def find_safe_end(end, data):
    N = len(data)
    if end == 0 or not (0x80 <= data[end - 1] <= 0xBF):
        return end
    while end > 0 and 0x80 <= data[end - 1] <= 0xBF:
        end -= 1
    return max(1, end)

def find_safe_start(start, data):
    N = len(data)
    if start == 0 or not (0x80 <= data[start] <= 0xBF):
        return start
    while start < N and 0x80 <= data[start] <= 0xBF:
        start += 1
    return min(start, N)

print('\nSafe boundary search:')
print(f'find_safe_end(9, data) = {find_safe_end(9, data)}')
print(f'find_safe_start(10, data) = {find_safe_start(10, data)}')