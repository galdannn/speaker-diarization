"""
fix_save_embeddings_b1.py
Sets save_embeddings=True for Baseline 1 in baseline.py
so its embeddings are cached like Baseline 2.

Run: python fix_save_embeddings_b1.py
"""
path = r"C:\Users\galda\titanet\sd\baseline.py"

content = open(path, encoding="utf-8").read()

# Baseline 1 speaker_embeddings block (first occurrence)
# Baseline 2 already has save_embeddings=True (second occurrence)
old = '"save_embeddings":      False'
new = '"save_embeddings":      True'

count = content.count(old)
print(f"Found 'save_embeddings: False': {count}")

if count == 0:
    print("Already set to True — nothing to do.")
else:
    # Only replace the FIRST occurrence (Baseline 1)
    content = content.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(content)
    print("Done — Baseline 1 will now save embeddings to disk.")
    print("Run: python baseline.py --skip_msdd")
    print("Then future runs will cache both baselines.")