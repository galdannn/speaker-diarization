"""
add_verbose.py
Adds verbose=False to Baseline 2's top-level cfg in baseline.py.
Run: python add_verbose.py
"""
path = r"C:\Users\galda\titanet\sd\baseline.py"

content = open(path, encoding="utf-8").read()

old = '        "batch_size":  64,         # top-level: used by internal ClusteringDiarizer for embedding extraction'
new = (
    '        "batch_size":  64,         # top-level: used by internal ClusteringDiarizer for embedding extraction\n'
    '        "verbose":     False,       # top-level: read by msdd_models.run_overlap_aware_eval'
)

count = content.count(old)
print(f"Found: {count}")

if count == 0:
    print("ERROR: pattern not found — check baseline.py manually")
else:
    content = content.replace(old, new)
    open(path, "w", encoding="utf-8").write(content)
    print("Done — verbose=False added to Baseline 2 cfg")