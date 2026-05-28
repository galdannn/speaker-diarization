"""
build_msdd_manifests.py
=======================
Builds NeMo MSDD-format manifests from the diarization dataset.

MSDD needs full conversation sessions with RTTM references, NOT
the individual speaker clips used for TitaNet training.

Input structure expected:
  data/diarization_dataset/
      train/audio/*.wav  +  train/rttm/*.rttm
      dev/audio/*.wav    +  dev/rttm/*.rttm
      test/audio/*.wav   +  test/rttm/*.rttm

Output:
  data/manifests/msdd_train_manifest.json
  data/manifests/msdd_val_manifest.json
  data/manifests/msdd_test_manifest.json

Usage:
  python build_msdd_manifests.py
  python build_msdd_manifests.py --data_dir data/diarization_dataset
"""

import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def build_manifest(
    audio_dir:     str,
    rttm_dir:      str,
    output_path:   str,
    num_speakers:  int = None,
) -> int:
    """
    Build one MSDD manifest from an audio + rttm directory pair.
    Each line = one full conversation session.
    Returns number of entries written.
    """
    audio_dir = Path(audio_dir)
    rttm_dir  = Path(rttm_dir)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    audio_exts  = {".wav", ".flac", ".mp3"}
    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_exts
    ])

    if not audio_files:
        log.error(f"No audio files found in {audio_dir}")
        return 0

    count = 0
    skipped = 0

    with open(output_path, "w") as f:
        for audio_path in audio_files:
            rttm_path = rttm_dir / (audio_path.stem + ".rttm")

            if not rttm_path.exists():
                log.warning(f"No RTTM for {audio_path.name} — skipping")
                skipped += 1
                continue

            # Count unique speakers in RTTM for reference
            speakers = set()
            with open(rttm_path) as rf:
                for line in rf:
                    parts = line.strip().split()
                    if parts and parts[0] == "SPEAKER":
                        speakers.add(parts[7])

            entry = {
                "audio_filepath": str(audio_path.resolve()),
                "rttm_filepath":  str(rttm_path.resolve()),
                "duration":       None,   # NeMo reads from file
                "offset":         0,
                "label":          "infer",
                "text":           "-",
                "num_speakers":   num_speakers if num_speakers else len(speakers),
                "uem_filepath":   None
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    log.info(f"Written: {output_path}  ({count} sessions, {skipped} skipped)")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build MSDD manifests from diarization dataset"
    )
    parser.add_argument("--data_dir",     default="data/diarization_dataset",
                        help="Root of diarization dataset")
    parser.add_argument("--manifest_dir", default="data/manifests",
                        help="Where to save manifest files")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Override speaker count (None = read from RTTM)")
    args = parser.parse_args()

    data_dir     = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": ("msdd_train_manifest.json", "train"),
        "val":   ("msdd_val_manifest.json",   "dev"),
        "test":  ("msdd_test_manifest.json",  "test"),
    }

    total = 0
    for split_name, (manifest_file, folder) in splits.items():
        audio_dir = data_dir / folder / "audio"
        rttm_dir  = data_dir / folder / "rttm"

        if not audio_dir.exists():
            log.warning(f"Skipping {split_name} — {audio_dir} not found")
            continue

        n = build_manifest(
            audio_dir=str(audio_dir),
            rttm_dir=str(rttm_dir),
            output_path=str(manifest_dir / manifest_file),
            num_speakers=args.num_speakers,
        )
        total += n

    log.info(f"\nTotal sessions written: {total}")
    log.info(f"Manifests saved to: {manifest_dir}/")
    log.info("\nUpdate msdd_finetune.yaml with:")
    log.info(f"  train_ds.manifest_filepath: {manifest_dir}/msdd_train_manifest.json")
    log.info(f"  validation_ds.manifest_filepath: {manifest_dir}/msdd_val_manifest.json")
    log.info(f"  test_ds.manifest_filepath: {manifest_dir}/msdd_test_manifest.json")


if __name__ == "__main__":
    main()