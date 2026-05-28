"""Fix corruption in field.py"""
with open('E:/arc/bcs/core/field.py', 'rb') as f:
    content = f.read()

idx_method = content.find(b'    def step(self, chunk_size')
idx_start = content.rfind(b'# CONCEPT FIX', 0, idx_method)
corrupt_len = idx_method - idx_start

print(f"Corruption: bytes {idx_start} to {idx_method - 1}, length={corrupt_len}")

# Clean replacement comment
clean = b'''# Phi field initialization: present bytes -> on-state,
# absent bytes -> off-state.

    def step(self, chunk_size: int = 16384):'''

# Compute the old method signature to know where to insert
# We need to find where the CORRUPTED method signature starts
old_sig = b'    def step(self, chunk_size: int = 16384):'
idx_sig = content.find(old_sig, idx_method)
print(f"Old signature at: {idx_sig}")

new = content[:idx_start] + clean + content[idx_sig:]
with open('E:/arc/bcs/core/field.py', 'wb') as f:
    f.write(new)
print("Done")
