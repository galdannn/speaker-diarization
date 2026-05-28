"""
prepare_dataset.py
==================
Converts Mongolian speaker diarization dataset into NeMo-compatible
manifest format for TitaNet fine-tuning.

Supports:
  - Long audio files with RTTM/TextGrid/CSV annotations
  - Short audio clips (already segmented, one speaker per file)
  - Automatic train/val/test splits
  - Audio validation and resampling to 16kHz mono

Expected dataset layouts:
  Layout A (long audio + RTTM):
      data/raw/
          audio/  *.wav / *.mp3 / *.flac
          rttm/   *.rttm

  Layout B (short clips, labeled by folder):
      data/raw/
          speaker_001/  clip1.wav  clip2.wav ...
          speaker_002/  clip1.wav  ...
          ...

  Layout C (CSV manifest):
      data/raw/annotations.csv
          columns: audio_filepath, speaker_id, start, duration
"""

import os
import json
import random
import argparse
import logging
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
TARGET_SR        = 16000       # TitaNet expects 16kHz
MIN_DURATION     = 0.5         # seconds – skip clips shorter than this
MAX_DURATION     = 30.0        # seconds – split clips longer than this
SEGMENT_DURATION = 3.0         # seconds – chunk long-audio segments
SEGMENT_OVERLAP  = 0.5         # seconds – overlap when chunking
TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
# TEST_RATIO      = 0.10 (remainder)

# ─── Audio Utilities ─────────────────────────────────────────────────────────

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio, resample to 16kHz mono."""
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio, sr


def save_segment(audio: np.ndarray, path: str, sr: int = TARGET_SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


def get_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        audio, sr = librosa.load(path, sr=None, mono=True)
        return len(audio) / sr


def validate_audio(path: str) -> bool:
    try:
        info = sf.info(path)
        return info.duration >= MIN_DURATION
    except Exception:
        return False


# ─── RTTM Parser ─────────────────────────────────────────────────────────────

def parse_rttm(rttm_path: str) -> list[dict]:
    """
    Parse an RTTM file.
    Returns list of {speaker, start, duration}
    """
    segments = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            # SPEAKER <file> <chnl> <start> <dur> <ortho> <stype> <name> ...
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            if duration < MIN_DURATION:
                continue
            segments.append({
                "speaker": speaker,
                "start":   start,
                "duration": duration
            })
    return segments


# ─── Layout A: Long audio + RTTM ─────────────────────────────────────────────

def process_long_audio_rttm(
    audio_dir: str,
    rttm_dir: str,
    output_dir: str,
) -> list[dict]:
    """
    Slice long recordings into per-speaker segments using RTTM annotations.
    For segments > MAX_DURATION, further chunk them with overlap.
    """
    audio_dir  = Path(audio_dir)
    rttm_dir   = Path(rttm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    audio_files = [p for p in audio_dir.iterdir() if p.suffix.lower() in audio_exts]
    log.info(f"Found {len(audio_files)} audio files in {audio_dir}")

    for audio_path in sorted(audio_files):
        rttm_path = rttm_dir / (audio_path.stem + ".rttm")
        if not rttm_path.exists():
            log.warning(f"No RTTM for {audio_path.name}, skipping.")
            continue

        segments = parse_rttm(str(rttm_path))
        if not segments:
            log.warning(f"Empty RTTM: {rttm_path.name}")
            continue

        log.info(f"Processing {audio_path.name}: {len(segments)} segments")

        try:
            audio, sr = load_audio(str(audio_path))
        except Exception as e:
            log.error(f"Cannot load {audio_path}: {e}")
            continue

        for i, seg in enumerate(segments):
            start_sample = int(seg["start"] * sr)
            end_sample   = int((seg["start"] + seg["duration"]) * sr)
            end_sample   = min(end_sample, len(audio))
            clip         = audio[start_sample:end_sample]

            if len(clip) / sr < MIN_DURATION:
                continue

            speaker_id = seg["speaker"]
            spk_dir    = output_dir / speaker_id
            spk_dir.mkdir(exist_ok=True)

            # If segment is very long, chunk it
            if seg["duration"] > MAX_DURATION:
                sub_clips = chunk_audio(clip, sr, SEGMENT_DURATION, SEGMENT_OVERLAP)
                for j, sub in enumerate(sub_clips):
                    fname = f"{audio_path.stem}_seg{i:04d}_chunk{j:03d}.wav"
                    out_path = spk_dir / fname
                    save_segment(sub, str(out_path), sr)
                    manifest_entries.append({
                        "audio_filepath": str(out_path),
                        "duration":       round(len(sub) / sr, 4),
                        "label":          speaker_id,
                        "text":           "-",
                        "offset":         0.0
                    })
            else:
                fname    = f"{audio_path.stem}_seg{i:04d}.wav"
                out_path = spk_dir / fname
                save_segment(clip, str(out_path), sr)
                manifest_entries.append({
                    "audio_filepath": str(out_path),
                    "duration":       round(len(clip) / sr, 4),
                    "label":          speaker_id,
                    "text":           "-",
                    "offset":         0.0
                })

    log.info(f"Layout A: extracted {len(manifest_entries)} segments")
    return manifest_entries


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_sec: float,
    overlap_sec: float
) -> list[np.ndarray]:
    chunk_samples   = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step            = chunk_samples - overlap_samples
    chunks = []
    start  = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        if (end - start) / sr >= MIN_DURATION:
            chunks.append(audio[start:end])
        start += step
    return chunks


# ─── Layout B: Speaker folders ────────────────────────────────────────────────

def process_speaker_folders(
    data_dir: str,
    output_dir: str,
) -> list[dict]:
    """
    Each subfolder name = speaker ID, each file = one utterance.
    Copies/converts files to 16kHz WAV.
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    speaker_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    log.info(f"Found {len(speaker_dirs)} speaker folders in {data_dir}")

    for spk_dir in sorted(speaker_dirs):
        speaker_id = spk_dir.name
        audio_files = [f for f in spk_dir.iterdir() if f.suffix.lower() in audio_exts]

        out_spk_dir = output_dir / speaker_id
        out_spk_dir.mkdir(exist_ok=True)

        for audio_path in sorted(audio_files):
            try:
                audio, sr = load_audio(str(audio_path))
            except Exception as e:
                log.warning(f"Skipping {audio_path}: {e}")
                continue

            duration = len(audio) / sr
            if duration < MIN_DURATION:
                continue

            # Split long clips
            if duration > MAX_DURATION:
                chunks = chunk_audio(audio, sr, SEGMENT_DURATION, SEGMENT_OVERLAP)
                for j, chunk in enumerate(chunks):
                    fname    = f"{audio_path.stem}_chunk{j:03d}.wav"
                    out_path = out_spk_dir / fname
                    save_segment(chunk, str(out_path), sr)
                    manifest_entries.append({
                        "audio_filepath": str(out_path),
                        "duration":       round(len(chunk) / sr, 4),
                        "label":          speaker_id,
                        "text":           "-",
                        "offset":         0.0
                    })
            else:
                fname    = audio_path.stem + ".wav"
                out_path = out_spk_dir / fname
                save_segment(audio, str(out_path), sr)
                manifest_entries.append({
                    "audio_filepath": str(out_path),
                    "duration":       round(duration, 4),
                    "label":          speaker_id,
                    "text":           "-",
                    "offset":         0.0
                })

    log.info(f"Layout B: processed {len(manifest_entries)} utterances")
    return manifest_entries


# ─── Layout C: CSV Manifest ───────────────────────────────────────────────────

def process_csv_manifest(
    csv_path: str,
    output_dir: str,
) -> list[dict]:
    """
    CSV columns: audio_filepath, speaker_id, start (sec), duration (sec)
    start and duration are optional (0 and full file if missing).
    """
    df = pd.read_csv(csv_path)
    required = {"audio_filepath", "speaker_id"}
    assert required.issubset(df.columns), f"CSV must have columns: {required}"

    if "start"    not in df.columns: df["start"]    = 0.0
    if "duration" not in df.columns: df["duration"] = -1.0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for _, row in df.iterrows():
        audio_path = Path(row["audio_filepath"])
        speaker_id = str(row["speaker_id"])
        start      = float(row["start"])
        duration   = float(row["duration"])

        if not audio_path.exists():
            log.warning(f"File not found: {audio_path}")
            continue

        try:
            audio, sr = load_audio(str(audio_path))
        except Exception as e:
            log.warning(f"Cannot load {audio_path}: {e}")
            continue

        start_sample = int(start * sr)
        if duration > 0:
            end_sample = int((start + duration) * sr)
        else:
            end_sample = len(audio)
        clip = audio[start_sample:min(end_sample, len(audio))]

        actual_dur = len(clip) / sr
        if actual_dur < MIN_DURATION:
            continue

        out_spk_dir = output_dir / speaker_id
        out_spk_dir.mkdir(exist_ok=True)

        fname    = f"{audio_path.stem}_s{int(start*100):07d}.wav"
        out_path = out_spk_dir / fname
        save_segment(clip, str(out_path), sr)

        manifest_entries.append({
            "audio_filepath": str(out_path),
            "duration":       round(actual_dur, 4),
            "label":          speaker_id,
            "text":           "-",
            "offset":         0.0
        })

    log.info(f"Layout C: processed {len(manifest_entries)} utterances")
    return manifest_entries


# ─── Speaker ID Mapping ───────────────────────────────────────────────────────

def encode_speaker_ids(entries: list[dict]) -> tuple[list[dict], dict]:
    """
    Map string speaker IDs to integer indices.
    Returns updated entries + label_map dict.
    """
    speakers = sorted(set(e["label"] for e in entries))
    label_map = {spk: idx for idx, spk in enumerate(speakers)}
    log.info(f"Total unique speakers: {len(speakers)}")
    for e in entries:
        e["label"] = str(label_map[e["label"]])  # NeMo expects string label
    return entries, label_map


# ─── Split Dataset ────────────────────────────────────────────────────────────

def speaker_disjoint_split(
    entries: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed: int = 42
) -> tuple[list, list, list]:
    """
    Split so each speaker appears in ONLY ONE split (train OR val OR test).
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # Get unique speakers
    speakers = list(set(e["label"] for e in entries))
    random.shuffle(speakers)

    n_total = len(speakers)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_spk = set(speakers[:n_train])
    val_spk   = set(speakers[n_train:n_train + n_val])
    test_spk  = set(speakers[n_train + n_val:])

    train, val, test = [], [], []

    for e in entries:
        spk = e["label"]
        if spk in train_spk:
            train.append(e)
        elif spk in val_spk:
            val.append(e)
        else:
            test.append(e)

    log.info(f"Speakers → train: {len(train_spk)}, val: {len(val_spk)}, test: {len(test_spk)}")
    log.info(f"Samples  → train: {len(train)}, val: {len(val)}, test: {len(test)}")

    return train, val, test


# ─── Write Manifest ───────────────────────────────────────────────────────────

def write_manifest(entries: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(entries):,} entries → {path}")


def write_label_map(label_map: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    log.info(f"Label map ({len(label_map)} speakers) → {path}")


# ─── Dataset Statistics ───────────────────────────────────────────────────────

def print_stats(entries: list[dict], split_name: str):
    durations   = [e["duration"] for e in entries]
    n_speakers  = len(set(e["label"] for e in entries))
    total_hours = sum(durations) / 3600

    log.info(f"\n{'='*50}")
    log.info(f"  {split_name.upper()} SPLIT STATISTICS")
    log.info(f"{'='*50}")
    log.info(f"  Utterances   : {len(entries):,}")
    log.info(f"  Speakers     : {n_speakers}")
    log.info(f"  Total audio  : {total_hours:.2f} hours")
    log.info(f"  Dur mean     : {np.mean(durations):.2f}s")
    log.info(f"  Dur min/max  : {np.min(durations):.2f}s / {np.max(durations):.2f}s")
    log.info(f"  Dur std      : {np.std(durations):.2f}s")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Mongolian speaker dataset for TitaNet fine-tuning"
    )
    parser.add_argument(
        "--layout", choices=["A", "B", "C"], required=True,
        help="A=long+RTTM, B=speaker_folders, C=csv_manifest"
    )
    parser.add_argument("--data_dir",    default="data/raw",
                        help="Root of raw data")
    parser.add_argument("--audio_dir",   default="data/raw/audio",
                        help="[Layout A] Audio directory")
    parser.add_argument("--rttm_dir",    default="data/raw/rttm",
                        help="[Layout A] RTTM directory")
    parser.add_argument("--csv_path",    default="data/raw/annotations.csv",
                        help="[Layout C] CSV manifest path")
    parser.add_argument("--output_dir",  default="data/processed",
                        help="Where to save processed WAV segments")
    parser.add_argument("--manifest_dir",default="data/manifests",
                        help="Where to save JSON manifests")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    # ── Step 1: Collect entries ──────────────────────────────────────────────
    if args.layout == "A":
        entries = process_long_audio_rttm(
            args.audio_dir, args.rttm_dir, args.output_dir
        )
    elif args.layout == "B":
        entries = process_speaker_folders(args.data_dir, args.output_dir)
    else:  # C
        entries = process_csv_manifest(args.csv_path, args.output_dir)

    if not entries:
        log.error("No entries processed! Check your data paths.")
        return

    # ── Step 2: Encode speaker IDs ───────────────────────────────────────────
    entries, label_map = encode_speaker_ids(entries)

    # ── Step 3: Split ────────────────────────────────────────────────────────
    train, val, test = speaker_disjoint_split(
        entries, args.train_ratio, args.val_ratio, args.seed
    )

    # ── Step 4: Print statistics ─────────────────────────────────────────────
    print_stats(train, "train")
    print_stats(val,   "val")
    if test:
        print_stats(test, "test")

    # ── Step 5: Write manifests ──────────────────────────────────────────────
    write_manifest(train, os.path.join(args.manifest_dir, "train_manifest.json"))
    write_manifest(val,   os.path.join(args.manifest_dir, "val_manifest.json"))
    if test:
        write_manifest(test, os.path.join(args.manifest_dir, "test_manifest.json"))

    write_label_map(label_map, os.path.join(args.manifest_dir, "label_map.json"))

    log.info("\n✅ Dataset preparation complete!")
    log.info(f"   Manifests in: {args.manifest_dir}/")
    log.info(f"   Segments in:  {args.output_dir}/")
    log.info(f"\nUpdate configs/titanet_finetune.yaml with:")
    log.info(f"   num_classes: {len(label_map)}")
    log.info(f"   train_ds.manifest_filepath: {args.manifest_dir}/train_manifest.json")
    log.info(f"   validation_ds.manifest_filepath: {args.manifest_dir}/val_manifest.json")


if __name__ == "__main__":
    main()