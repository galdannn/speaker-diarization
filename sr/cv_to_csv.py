"""
cv_to_csv.py
============
Converts Common Voice Mongolian train.tsv into a CSV that
sr/prepare_dataset.py --layout C can read.

Common Voice TSV columns: client_id, path, text, up_votes, ...
Layout C CSV columns:     audio_filepath, speaker_id, start, duration

Run from titanet/sr/:
    python cv_to_csv.py

Then run:
    python prepare_dataset.py --layout C ^
      --csv_path data/raw/cv_mongolian.csv ^
      --output_dir data/processed ^
      --manifest_dir data/manifests
"""

import os
import pandas as pd
from pathlib import Path

# ── Paths — adjust if needed ─────────────────────────────────
CV_ROOT  = r"C:\Users\galda\Capstone\cv-corpus-24.0-2025-12-05\mn"
TSV_FILE = os.path.join(CV_ROOT, "train.tsv")
CLIPS_DIR = os.path.join(CV_ROOT, "clips")
OUT_CSV  = "data/raw/cv_mongolian.csv"

# ── Minimum clips per speaker to be included ─────────────────
MIN_CLIPS = 20

os.makedirs("data/raw", exist_ok=True)

print("Loading TSV...")
df = pd.read_csv(TSV_FILE, sep="\t", usecols=["client_id", "path"])
df = df.dropna()
print(f"  Total rows: {len(df)}")

# Filter to files that exist on disk
print("Checking files on disk...")
df["full_path"] = df["path"].apply(lambda p: os.path.join(CLIPS_DIR, p))
df = df[df["full_path"].apply(os.path.isfile)].reset_index(drop=True)
print(f"  Files found on disk: {len(df)}")

# Filter speakers with enough clips
counts = df["client_id"].value_counts()
eligible = counts[counts >= MIN_CLIPS].index
df = df[df["client_id"].isin(eligible)].reset_index(drop=True)
print(f"  After filtering (>={MIN_CLIPS} clips/speaker): {len(df)} clips, {len(eligible)} speakers")

# Build Layout C CSV
df_out = pd.DataFrame({
    "audio_filepath": df["full_path"],
    "speaker_id":     df["client_id"],
    "start":          0.0,
    "duration":       -1.0,   # -1 = use full file
})

df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"  Speakers : {df_out['speaker_id'].nunique()}")
print(f"  Clips    : {len(df_out)}")
print(f"\nNext step:")
print(f"  python prepare_dataset.py --layout C \\")
print(f"    --csv_path data/raw/cv_mongolian.csv \\")
print(f"    --output_dir data/processed \\")
print(f"    --manifest_dir data/manifests")
print(f"\nThen update configs/titanet_finetune.yaml:")
print(f"  model.num_classes: <number printed above>")