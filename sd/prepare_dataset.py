"""
prepare_dataset.py  (sd/)
=========================
Generates NeMo MSDD manifests from the existing diarization dataset.

Respects the train/dev/test split already created by
build_diarization_dataset.py — does NOT re-split.

Input:
    data/diarization_dataset/
        train/audio/*.wav  train/rttm/*.rttm
        dev/audio/*.wav    dev/rttm/*.rttm
        test/audio/*.wav   test/rttm/*.rttm

Output:
    data/manifests/
        train_manifest.json
        val_manifest.json
        test_manifest.json

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --dataset_dir data/diarization_dataset
"""

import os
import json
import argparse
import logging
from pathlib import Path

import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def get_duration(audio_path: str) -> float:
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr


def count_speakers(rttm_path: str) -> int:
    speakers = set()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)


def build_manifest(audio_dir: str, rttm_dir: str) -> list:
    """Build manifest entries from a single split folder."""
    audio_dir = Path(audio_dir)
    rttm_dir  = Path(rttm_dir)

    entries  = []
    skipped  = 0

    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in {".wav", ".flac", ".mp3"}
    ])

    for audio_path in audio_files:
        rttm_path = rttm_dir / (audio_path.stem + ".rttm")
        if not rttm_path.exists():
            log.warning(f"No RTTM for {audio_path.name}, skipping")
            skipped += 1
            continue

        try:
            duration   = get_duration(str(audio_path))
            n_speakers = count_speakers(str(rttm_path))
        except Exception as e:
            log.warning(f"Skipping {audio_path.name}: {e}")
            skipped += 1
            continue

        entries.append({
            "audio_filepath": str(audio_path.resolve()),
            "rttm_filepath":  str(rttm_path.resolve()),
            "duration":       round(duration, 4),
            "num_speakers":   n_speakers,
            "offset":         0,
            "label":          "infer",
            "text":           "-",
            "uem_filepath":   None
        })

    log.info(f"  {audio_dir.parent.name}: {len(entries)} recordings, {skipped} skipped")
    return entries


def write_manifest(entries: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    total_hours = sum(e["duration"] for e in entries) / 3600
    log.info(f"  Wrote: {path} ({len(entries)} recordings, {total_hours:.3f} hours)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MSDD manifests from existing diarization dataset split"
    )
    parser.add_argument(
        "--dataset_dir", default="data/diarization_dataset",
        help="Root folder containing train/, dev/, test/ subdirs"
    )
    parser.add_argument(
        "--manifest_dir", default="data/manifests",
        help="Where to write the manifest JSON files"
    )
    args = parser.parse_args()

    dataset_dir  = Path(args.dataset_dir)
    manifest_dir = args.manifest_dir

    splits = {
        "train": "train_manifest.json",
        "dev":   "val_manifest.json",
        "test":  "test_manifest.json",
    }

    log.info(f"Building manifests from: {dataset_dir}")
    log.info("(Reusing existing train/dev/test split — not re-splitting)\n")

    for split_name, manifest_name in splits.items():
        audio_dir = dataset_dir / split_name / "audio"
        rttm_dir  = dataset_dir / split_name / "rttm"

        if not audio_dir.exists():
            log.warning(f"  {split_name}/audio not found, skipping")
            continue

        entries = build_manifest(str(audio_dir), str(rttm_dir))
        if entries:
            write_manifest(entries, os.path.join(manifest_dir, manifest_name))

    log.info("\nManifest generation complete.")
    log.info(f"  Manifests saved to: {manifest_dir}/")
    log.info("\nNext step: run train_msdd.py")


if __name__ == "__main__":
    main()