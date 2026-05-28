# fix_durations.py
import json, soundfile as sf
from pathlib import Path

manifests = [
    "data/manifests/msdd_train_manifest.json",
    "data/manifests/msdd_val_manifest.json",
    "data/manifests/msdd_test_manifest.json",
]

for manifest_path in manifests:
    lines = Path(manifest_path).read_text(encoding="utf-8").splitlines()
    fixed = []
    for line in lines:
        entry = json.loads(line)
        if entry.get("duration") is None:
            try:
                info = sf.info(entry["audio_filepath"])
                entry["duration"] = round(info.duration, 4)
            except Exception as e:
                print(f"  WARNING: could not read {entry['audio_filepath']}: {e}")
        fixed.append(json.dumps(entry))
    Path(manifest_path).write_text("\n".join(fixed), encoding="utf-8")
    print(f"Fixed {len(fixed)} entries in {manifest_path}")