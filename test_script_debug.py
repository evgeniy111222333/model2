# Force test the function
import sys
sys.path.insert(0, 'E:/arc')

# Direct test - load the module and test
from bcs.information.character_manifold import CharacterManifold

m = CharacterManifold()

print("Direct test of get_script_for_cp:")
print(f"  H (0x48): {m.get_script_for_cp(0x48)}")
print(f"  e (0x65): {m.get_script_for_cp(0x65)}")
print(f"  o (0x6F): {m.get_script_for_cp(0x6F)}")
print(f"  space (0x20): {m.get_script_for_cp(0x20)}")
print(f"  П (0x41F): {m.get_script_for_cp(0x41F)}")

# Check module location
import bcs.information.character_manifold as cm
print(f"\nModule loaded from: {cm.__file__}")
print(f"Module file hash check: {hash(open(cm.__file__).read())}")

# Try reloading
import importlib
importlib.reload(cm)

m2 = cm.CharacterManifold()
print(f"\nAfter reload:")
print(f"  H (0x48): {m2.get_script_for_cp(0x48)}")