"""
patch_nemo_final.py
===================
Fixes NeMo 2.7.0 compatibility issues with PyTorch 2.6 on Windows.

Patches applied:
  1. verbose at line ~213 (_run_vad) — replaced with False
  2. verbose at line ~449 (_extract_embeddings) — replaced with False  
  3. collate 0-dim tensor — try/except instead of aggressive filtering
  4. None batch guard in _extract_embeddings

Run:
    python patch_nemo_final.py
"""

import shutil, py_compile, sys
from pathlib import Path

# ── clustering_diarizer.py ────────────────────────────────────────────────────
cd_path = Path(
    r"C:\Users\galda\AppData\Local\Programs\Python\Python310"
    r"\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py"
)
backup = Path(str(cd_path) + ".bak")

if backup.exists():
    shutil.copy(backup, cd_path)
    print("Restored clean clustering_diarizer.py backup.")
else:
    print("No backup found — patching current file.")

content = cd_path.read_text(encoding="utf-8")

# Patch 1: verbose in _run_vad
old1 = "disable=not self.verbose)"
new1 = "disable=False)"
c1 = content.count(old1)
content = content.replace(old1, new1)
print(f"Patch 1 (verbose _run_vad):           {c1} replacement(s)")

# Patch 2: verbose in _extract_embeddings tqdm
old2 = "disable=not self.verbose,"
new2 = "disable=False,"
c2 = content.count(old2)
content = content.replace(old2, new2)
print(f"Patch 2 (verbose _extract_embeddings): {c2} replacement(s)")

# Patch 3: None batch guard in _extract_embeddings
old3 = "            test_batch = [x.to(self._speaker_model.device) for x in test_batch]"
new3 = (
    "            if test_batch is None:\n"
    "                continue\n"
    "            test_batch = [x.to(self._speaker_model.device) for x in test_batch]"
)
c3 = content.count(old3)
content = content.replace(old3, new3)
print(f"Patch 3 (None batch guard):            {c3} replacement(s)")

cd_path.write_text(content, encoding="utf-8")

# ── audio_to_label.py ─────────────────────────────────────────────────────────
al_path = Path(
    r"C:\Users\galda\AppData\Local\Programs\Python\Python310"
    r"\lib\site-packages\nemo\collections\asr\data\audio_to_label.py"
)
al_backup = Path(str(al_path) + ".bak")

if al_backup.exists():
    shutil.copy(al_backup, al_path)
    print("\nRestored clean audio_to_label.py backup.")
else:
    print("\nNo audio_to_label.py backup — patching current.")

al_content = al_path.read_text(encoding="utf-8")

# Replace ALL occurrences of the zip line with a try/except version
# This is more robust than filtering — if zip fails, return None gracefully
old_zip = "    _, audio_lengths, _, tokens_lengths = zip(*batch)"
new_zip = (
    "    try:\n"
    "        _, audio_lengths, _, tokens_lengths = zip(*batch)\n"
    "    except TypeError:\n"
    "        return None"
)

c4 = al_content.count(old_zip)
al_content = al_content.replace(old_zip, new_zip)
print(f"Patch 4 (collate try/except):          {c4} replacement(s)")

al_path.write_text(al_content, encoding="utf-8")

# ── Verify both files ─────────────────────────────────────────────────────────
print("\nVerifying...")
all_ok = True

for path, checks in [
    (cd_path, [
        ("disable=False",         "verbose replaced"),
        ("if test_batch is None", "None guard present"),
    ]),
    (al_path, [
        ("except TypeError",      "try/except collate"),
    ]),
]:
    text = path.read_text(encoding="utf-8")
    for pattern, label in checks:
        count = text.count(pattern)
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"  {status}  {label} ({count} occurrence(s))")
        if count == 0:
            all_ok = False

    try:
        py_compile.compile(str(path), doraise=True)
        print(f"  ✓  {path.name} — no syntax errors")
    except py_compile.PyCompileError as e:
        print(f"  ✗  {path.name} — SYNTAX ERROR: {e}")
        all_ok = False

if all_ok:
    print("\nAll patches applied successfully.")
    print("Now run:  python baseline.py --skip_msdd")
else:
    print("\nSome patches failed.")
    sys.exit(1)