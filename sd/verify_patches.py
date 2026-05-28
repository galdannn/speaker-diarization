"""
verify_patches.py
Checks all 4 required patches are in place before running baseline.py
Run: python verify_patches.py
"""

cd_path = (
    r"C:\Users\galda\AppData\Local\Programs\Python\Python310"
    r"\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py"
)
bl_path = r"C:\Users\galda\titanet\sd\baseline.py"

cd = open(cd_path, encoding="utf-8").read()
bl = open(bl_path, encoding="utf-8").read()

checks = [
    ("1. No deletion       ", "deletion disabled"              in cd),
    ("2. exist_ok=True     ", "exist_ok=True"                  in cd),
    ("3. Skip extraction   ", "Reusing cached embeddings"      in cd),
    ("4. save_embeddings x2", bl.count('"save_embeddings":      True') == 2),
]

all_good = True
for label, result in checks:
    status = "GOOD" if result else "MISSING"
    print(f"  {label}: {status}")
    if not result:
        all_good = False

print()
if all_good:
    print("All patches in place.")
    print("Run: python baseline.py")
    print("This will be the LAST extraction. Future runs reuse the cache.")
else:
    print("Some patches are missing — do NOT run baseline.py yet.")
    print("Paste this output and I will fix the missing patches.")