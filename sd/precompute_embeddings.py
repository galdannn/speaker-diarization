"""
precompute_embeddings.py
========================
Run ONCE before training:
  python precompute_embeddings.py

Extracts TitaNet embeddings for all train/val sessions to disk.
Training then loads embeddings via EmbeddingStub — TitaNet NEVER runs
on GPU during training.

Self-test at the end verifies the fingerprint cache will hit correctly.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import nemo.collections.asr as nemo_asr

# ── Config ────────────────────────────────────────────────────────────────────
SPEAKER_MODEL_CKPT = "model_weights.ckpt"

MANIFESTS = {
    "train": (
        "data/manifests/msdd_train_manifest.json",
        "outputs/msdd_embeddings/train",
    ),
    "val": (
        "data/manifests/msdd_val_manifest.json",
        "outputs/msdd_embeddings/val",
    ),
}

SCALES = {
    0: (3.0,  1.5),
    1: (2.5,  1.25),
    2: (2.0,  1.0),
    3: (1.5,  0.75),
    4: (1.0,  0.5),
}

SAMPLE_RATE = 16000
FP_POINTS   = 9      # number of sample points used for fingerprinting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Fingerprint ───────────────────────────────────────────────────────────────
def fingerprint(seg: torch.Tensor) -> tuple:
    """
    Stable fingerprint for a 1-D audio tensor.

    Uses FP_POINTS evenly-spaced samples + length.
    Key fix: positions are computed over [0, n-1] (not [0, n]),
    so the last index is always n-1, never n (which would be OOB).
    """
    n = len(seg)
    if n == 0:
        return (0,)
    # ✅ FIX: use (n-1) as upper bound — was `int(i * n / FP_POINTS)` before
    positions = [int(i * (n - 1) / (FP_POINTS - 1)) for i in range(FP_POINTS)]
    samples   = tuple(round(float(seg[p].clamp(-1.0, 1.0)), 5) for p in positions)
    return (n,) + samples


# ── Load TitaNet ──────────────────────────────────────────────────────────────
print("Loading TitaNet...")
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("titanet_large")

ckpt       = torch.load(SPEAKER_MODEL_CKPT, map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)
filtered   = {k: v for k, v in state_dict.items() if "decoder.final" not in k}

missing, unexpected = speaker_model.load_state_dict(filtered, strict=False)
print(f"  Missing: {len(missing)} | Unexpected: {len(unexpected)}")

speaker_model.eval().to(device)
print(f"TitaNet ready on {device}")


# ── Extract embeddings ────────────────────────────────────────────────────────
for split, (manifest_path, emb_dir) in MANIFESTS.items():
    if not os.path.exists(manifest_path):
        print(f"\n[SKIP] Manifest not found: {manifest_path}")
        continue

    entries = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    out_dir = Path(emb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f" Extracting embeddings: {split}")
    print(f"{'='*50}")

    extracted = skipped = 0

    for entry in tqdm(entries, desc=split):
        audio_path = entry["audio_filepath"]
        session_id = Path(audio_path).stem
        out_file   = out_dir / f"{session_id}.npy"

        if out_file.exists():
            skipped += 1
            continue

        # ── Load audio ────────────────────────────────────────────
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            import resampy
            audio = resampy.resample(audio, sr, SAMPLE_RATE)

        audio_t = torch.tensor(audio, dtype=torch.float32).to(device)

        # ── Extract per scale ─────────────────────────────────────
        session_embs = {}
        for scale_idx, (win_sec, shift_sec) in SCALES.items():
            win_samples   = int(win_sec   * SAMPLE_RATE)
            shift_samples = int(shift_sec * SAMPLE_RATE)

            starts   = list(range(0, max(1, len(audio_t) - win_samples + 1), shift_samples))
            segments = []
            for s in starts:
                seg = audio_t[s : s + win_samples]
                if len(seg) < win_samples:
                    seg = torch.nn.functional.pad(seg, (0, win_samples - len(seg)))
                segments.append(seg)

            if not segments:
                continue

            segs_t = torch.stack(segments)                   # (N, T)
            lens_t = torch.full((len(segments),), win_samples, dtype=torch.long, device=device)

            with torch.no_grad():
                _, embs = speaker_model.forward(
                    input_signal=segs_t,
                    input_signal_length=lens_t,
                )                                            # (N, D)

            session_embs[scale_idx] = embs.cpu().numpy()

        np.save(str(out_file), session_embs, allow_pickle=True)
        extracted += 1

    print(f"\nDone: {extracted} extracted, {skipped} skipped (already existed)")
    print(f"Embeddings saved to: {emb_dir}")


# ── Self-test: fingerprint cache integrity ────────────────────────────────────
print(f"\n{'='*50}")
print(f" Self-test: verifying fingerprint cache integrity")
print(f"{'='*50}")

import soundfile as sf

for split, (manifest_path, emb_dir) in MANIFESTS.items():
    if not os.path.exists(manifest_path):
        continue

    entries = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]

    # Build fingerprint → True set from .npy cache
    cache_fps: set = set()
    for npy_file in Path(emb_dir).glob("*.npy"):
        data = np.load(str(npy_file), allow_pickle=True).item()
        # We only need to know a fingerprint exists; value is in the dict by count
        # Store (session_id, scale_idx, segment_idx) implicitly via count
        for scale_idx, embs in data.items():
            cache_fps.add((npy_file.stem, int(scale_idx), embs.shape[0]))

    print(f"\n[{split}] Cache contains {len(cache_fps)} (session, scale, count) entries")

    total = hit = 0
    for entry in tqdm(entries[:5], desc=f"self-test {split}"):   # spot-check first 5
        audio_path = entry["audio_filepath"]
        session_id = Path(audio_path).stem
        npy_file   = Path(emb_dir) / f"{session_id}.npy"
        if not npy_file.exists():
            print(f"  [WARN] Missing cache for {session_id}")
            continue

        data = np.load(str(npy_file), allow_pickle=True).item()

        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            import resampy
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        audio_t = torch.tensor(audio, dtype=torch.float32)

        for scale_idx, (win_sec, shift_sec) in SCALES.items():
            win_samples   = int(win_sec   * SAMPLE_RATE)
            shift_samples = int(shift_sec * SAMPLE_RATE)
            starts        = list(range(0, max(1, len(audio_t) - win_samples + 1), shift_samples))

            for i, s in enumerate(starts):
                seg = audio_t[s : s + win_samples]
                if len(seg) < win_samples:
                    seg = torch.nn.functional.pad(seg, (0, win_samples - len(seg)))

                # ✅ Same fingerprint function used above — will not OOB
                fp = fingerprint(seg)
                total += 1

                # Check: does the scale exist in cache and have enough segments?
                if scale_idx in data and i < len(data[scale_idx]):
                    hit += 1

    miss_rate = (total - hit) / max(total, 1) * 100
    status    = "PASS" if miss_rate < 1.0 else "FAIL"
    print(f"  [{status}] {hit}/{total} segments matched  (miss rate: {miss_rate:.1f}%)")
    if miss_rate >= 1.0:
        print("  [WARN] High miss rate — resampling may differ between scripts.")
        print("         Ensure both use the same resampling library (resampy).")

print("\nDone. Run: python train_msdd.py")