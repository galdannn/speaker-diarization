"""
patch_no_delete.py
Disables the speaker_outputs folder deletion in clustering_diarizer.py
so cached embedding pkl files are preserved between runs.

Run: python patch_no_delete.py
"""
path = (
    r"C:\Users\galda\AppData\Local\Programs\Python\Python310"
    r"\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py"
)

content = open(path, encoding="utf-8").read()

old = (
    '            logging.warning("Deleting previous clustering diarizer outputs.")\n'
    '            shutil.rmtree(self._speaker_dir, ignore_errors=True)'
)

new = '            pass  # deletion disabled — preserves cached embedding pkl files'

count = content.count(old)
print(f"Found: {count}")

if count == 0:
    print("ERROR: pattern not found. Check the file manually.")
else:
    content = content.replace(old, new)
    open(path, "w", encoding="utf-8").write(content)
    print("Done — speaker_dir deletion disabled.")
    print("Cached embeddings will now be reused on next run.")