"""
evaluate_vad_nemo.py
Run VAD on your processed dataset using MarbleNet from NVIDIA NeMo.
Outputs per-file and per-speaker speech statistics in JSON.
"""

from pathlib import Path
import json
import torch
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/processed_cv")  # directory with speakers' WAV files
OUTPUT_JSON  = "vad_summary_nemo.json"
SAMPLE_LIMIT = None       # set to int for quick test, e.g., 50 files
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load pre-trained MarbleNet VAD ─────────────────────────────────────────
print("Loading MarbleNet VAD model...")
vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
    model_name="vad_marblenet"
)
vad_model = vad_model.to(DEVICE)
vad_model.eval()

# ── Helper: Run VAD on a single file ───────────────────────────────────────
def analyze_file(audio_path):
    try:
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # convert to mono

        # Run VAD: output is a 1D array of voice probabilities
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = vad_model.forward(input_signal=audio_tensor, input_signal_length=torch.tensor([audio_tensor.shape[1]]).to(DEVICE))[0]
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Convert probabilities to speech segments
        threshold = 0.5
        frame_size = 0.032  # 32 ms per frame
        segments = []
        start = None
        for i, p in enumerate(probs):
            if p > threshold and start is None:
                start = i * frame_size
            elif p <= threshold and start is not None:
                end = i * frame_size
                segments.append((start, end))
                start = None
        if start is not None:
            segments.append((start, len(audio)/sr))

        # Compute stats
        total_speech = sum(end - start for start, end in segments)
        dur = len(audio)/sr
        speech_ratio = total_speech / dur if dur > 0 else 0.0

        return {
            "audio_path": str(audio_path),
            "duration": dur,
            "speech_segments": len(segments),
            "speech_duration": total_speech,
            "speech_ratio": speech_ratio
        }

    except Exception as e:
        return {"audio_path": str(audio_path), "error": str(e)}

# ── Scan all files ─────────────────────────────────────────────────────────
all_files = list(DATA_DIR.rglob("*.wav"))
if SAMPLE_LIMIT:
    all_files = all_files[:SAMPLE_LIMIT]

print(f"Analyzing {len(all_files)} files with MarbleNet VAD...")

results = []
speaker_stats = {}

for f in all_files:
    res = analyze_file(f)
    results.append(res)

    # aggregate per speaker
    spk = f.parent.name
    if spk not in speaker_stats:
        speaker_stats[spk] = {"files": 0, "total_dur": 0.0, "speech_dur": 0.0}
    if "speech_duration" in res:
        speaker_stats[spk]["files"] += 1
        speaker_stats[spk]["total_dur"] += res["duration"]
        speaker_stats[spk]["speech_dur"] += res["speech_duration"]

# ── Print summary ──────────────────────────────────────────────────────────
print("\nPer-speaker summary:")
for spk, s in speaker_stats.items():
    speech_ratio = s["speech_dur"] / s["total_dur"] if s["total_dur"] > 0 else 0
    print(f"{spk}: {s['files']} files, total {s['total_dur']:.2f}s, "
          f"speech {s['speech_dur']:.2f}s ({speech_ratio*100:.1f}%)")

overall_speech = sum(s["speech_dur"] for s in speaker_stats.values())
overall_total  = sum(s["total_dur"] for s in speaker_stats.values())
print(f"\nOverall: {len(speaker_stats)} speakers, "
      f"{len(all_files)} files, "
      f"speech ratio: {overall_speech/overall_total*100:.1f}%")

# ── Save JSON ─────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"files": results, "per_speaker": speaker_stats}, f, indent=2, ensure_ascii=False)

print(f"\nSaved full VAD summary → {OUTPUT_JSON}")