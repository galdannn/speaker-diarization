"""
patch_skip_extraction.py
========================
Patches clustering_diarizer._extract_embeddings to skip extraction
if the embedding pkl file already exists on disk.

Run: python patch_skip_extraction.py
"""
import py_compile, sys
from pathlib import Path

path = Path(
    r"C:\Users\galda\AppData\Local\Programs\Python\Python310"
    r"\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py"
)

content = path.read_text(encoding="utf-8")

old = (
    "            # Embedding Extraction for the current scale (scale_idx)\n"
    "            self._extract_embeddings(self.subsegments_manifest_path, scale_idx, len(scales))"
)

new = (
    "            # Embedding Extraction for the current scale (scale_idx)\n"
    "            import os as _os\n"
    "            _pkl = _os.path.join(\n"
    "                self._speaker_dir, 'embeddings',\n"
    "                f'subsegments_scale{scale_idx}_embeddings.pkl'\n"
    "            )\n"
    "            if _os.path.isfile(_pkl):\n"
    "                logging.info(f'Reusing cached embeddings for scale {scale_idx}: {_pkl}')\n"
    "                import pickle\n"
    "                with open(_pkl, 'rb') as _f:\n"
    "                    self.embeddings = pickle.load(_f)\n"
    "            else:\n"
    "                self._extract_embeddings(self.subsegments_manifest_path, scale_idx, len(scales))"
)

count = content.count(old)
print(f"Found: {count}")

if count == 0:
    print("ERROR: pattern not found.")
    sys.exit(1)

content = content.replace(old, new)
path.write_text(content, encoding="utf-8")

try:
    py_compile.compile(str(path), doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)

print("Done — embeddings will be reused from cache on future runs.")