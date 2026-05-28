import os
import pandas as pd
import shutil
from collections import defaultdict
from tqdm import tqdm

CV_ROOT = r"C:\Users\galda\Capstone\cv-corpus-24.0-2025-12-05\mn"
CLIPS_DIR = os.path.join(CV_ROOT, "clips")
OUT_DIR = "data/cv_speakers_full"

TSV_FILES = ["validated.tsv", "other.tsv", "invalidated.tsv"]

MIN_CLIPS = 10
MAX_CLIPS = 120

os.makedirs(OUT_DIR, exist_ok=True)

spk2files = defaultdict(list)

# 🔥 Load ALL TSVs
for tsv in TSV_FILES:
    path = os.path.join(CV_ROOT, tsv)
    if not os.path.exists(path):
        continue

    print(f"Loading {tsv}")
    df = pd.read_csv(path, sep="\t")

    for _, row in df.iterrows():
        spk = str(row["client_id"])
        audio_path = os.path.join(CLIPS_DIR, row["path"])

        if os.path.exists(audio_path):
            spk2files[spk].append(audio_path)

# 🔥 Filter speakers
selected = {}
for spk, files in spk2files.items():
    if len(files) < MIN_CLIPS:
        continue

    # cap max clips
    files = files[:MAX_CLIPS]
    selected[spk] = files

print(f"Total speakers after filtering: {len(selected)}")

# 🔥 Write speaker folders
for i, (spk, files) in enumerate(tqdm(selected.items())):
    spk_id = f"spk_{i:05d}"
    out_spk = os.path.join(OUT_DIR, spk_id)
    os.makedirs(out_spk, exist_ok=True)

    for j, f in enumerate(files):
        dst = os.path.join(out_spk, f"{j:04d}.wav")
        shutil.copy(f, dst)

print("Done.")