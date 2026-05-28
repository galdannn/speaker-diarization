"""
server.py
=========
Local Flask server for the Mongolian Speaker Diarization Web Prototype.

Speed mode: 3 scales + ClusteringDiarizer only.
~30-45 seconds for a 2-minute file on RTX 3060.

After fine-tuning, switch to MSDD + 5 scales by setting:
    FAST_MODE = False

Install:
    pip install flask flask-cors

Run from titanet/sd/ (or titanet/prototype/):
    python server.py

Then open: http://localhost:5000
"""

import os
import sys
import json
import uuid
import shutil
import logging
import tempfile
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Storage ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("web_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()

# ── Model config ──────────────────────────────────────────────────────────────
FAST_MODE = True   # True = 3 scales + clustering (~40s). False = 5 scales + MSDD (~5min)

SPEAKER_MODEL = "titanet_large"
MSDD_MODEL    = "diar_msdd_telephonic"
VAD_MODEL     = "vad_multilingual_marblenet"

# After fine-tuning, swap to:
# SPEAKER_MODEL = "outputs/final/titanet_mongolian_finetuned.nemo"
# MSDD_MODEL    = "outputs/final/msdd_mongolian_finetuned.nemo"
# FAST_MODE     = False

WINDOW_LENGTHS_FAST = [3.0, 1.5, 0.5]
SHIFT_LENGTHS_FAST  = [1.5, 0.75, 0.25]
WINDOW_LENGTHS_FULL = [3.0, 2.5, 2.0, 1.5, 1.0]
SHIFT_LENGTHS_FULL  = [1.5, 1.25, 1.0, 0.75, 0.5]


import subprocess

def normalize_audio(src: str, dst: str) -> str:
    """
    Convert any audio format → 16kHz mono PCM WAV using ffmpeg.
    Returns dst path on success, raises on failure.
    """
    cmd = [
        "ffmpeg", "-y",           # overwrite output
        "-i", src,                # any input format
        "-ar", "16000",           # resample to 16kHz
        "-ac", "1",               # mono
        "-sample_fmt", "s16",     # 16-bit PCM
        "-vn",                    # drop video streams (some containers have them)
        dst
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg conversion failed:\n{result.stderr}")
    return dst
# ── Diarization worker ────────────────────────────────────────────────────────

def run_diarization(job_id: str, audio_path: str, num_speakers=None):
    try:
        with jobs_lock:
            jobs[job_id]["status"]   = "processing"
            jobs[job_id]["progress"] = "Loading VAD + TitaNet..."

        import torch
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import ClusteringDiarizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"[{job_id}] Starting on {device} | fast_mode={FAST_MODE}")

        win = WINDOW_LENGTHS_FAST if FAST_MODE else WINDOW_LENGTHS_FULL
        sft = SHIFT_LENGTHS_FAST  if FAST_MODE else SHIFT_LENGTHS_FULL
        weights = [1] * len(win)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            out_dir       = os.path.join(tmpdir, "output")
            os.makedirs(out_dir)

            entry = {
                "audio_filepath": str(Path(audio_path).resolve()),
                "offset": 0, "duration": None,
                "label": "infer", "text": "-",
                "num_speakers": num_speakers, "uem_filepath": None
            }
            with open(manifest_path, "w") as f:
                f.write(json.dumps(entry) + "\n")

            cfg_dict = {
                "device":      device,
                "num_workers": 0,
                "sample_rate": 16000,
                "batch_size":  64,
                "verbose":     False,
                "diarizer": {
                    "manifest_filepath": manifest_path,
                    "out_dir":           out_dir,
                    "oracle_vad":        False,
                    "collar":            0.25,
                    "ignore_overlap":    True,
                    "speaker_embeddings": {
                        "model_path": SPEAKER_MODEL,
                        "parameters": {
                            "window_length_in_sec": win,
                            "shift_length_in_sec":  sft,
                            "multiscale_weights":   weights,
                            "save_embeddings":      False,
                        }
                    },
                    "vad": {
                        "model_path": VAD_MODEL,
                        "parameters": {
                            "onset":               0.8,
                            "offset":              0.6,
                            "pad_onset":           0.1,
                            "pad_offset":          0.1,
                            "min_duration_on":     0.2,
                            "min_duration_off":    0.2,
                            "filter_speech_first": True,
                            "window_length_in_sec": 0.15,
                            "shift_length_in_sec":  0.01,
                            "smoothing":           False,
                            "overlap":             0.875,
                            "postprocessing_yaml": None,
                        }
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers":  num_speakers is not None,
                            "max_num_speakers":     8,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold":     0.25,
                            "sparse_search_volume": 30,
                        }
                    }
                }
            }

            # Add MSDD block if not in fast mode
            if not FAST_MODE:
                with jobs_lock:
                    jobs[job_id]["progress"] = "Running MSDD (full accuracy mode)..."
                cfg_dict["diarizer"]["msdd_model"] = {
                    "model_path": MSDD_MODEL,
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size":            25,
                        "sigmoid_threshold":           [0.7],
                        "seq_eval_mode":               False,
                        "split_infer":                 True,
                        "diar_eval_settings":          [[0.25, True], [0.25, False]],
                        "diar_window_length":          50,
                        "diar_batch_size":             25,
                        "diar_overlap_length":         1,
                    }
                }
                from nemo.collections.asr.models.msdd_models import NeuralDiarizer
                cfg = OmegaConf.create(cfg_dict)
                diarizer = NeuralDiarizer(cfg=cfg)
            else:
                with jobs_lock:
                    jobs[job_id]["progress"] = "Running VAD + embedding extraction..."
                cfg = OmegaConf.create(cfg_dict)
                diarizer = ClusteringDiarizer(cfg=cfg)

            diarizer.diarize()
            pred_dir = os.path.join(out_dir, "pred_rttms")

            stem     = Path(audio_path).stem
            rttm_src = os.path.join(pred_dir, stem + ".rttm")

            if not os.path.exists(rttm_src):
                raise FileNotFoundError(f"RTTM not generated: {rttm_src}")

            rttm_dst = str(UPLOAD_DIR / f"{job_id}.rttm")
            shutil.copy(rttm_src, rttm_dst)
            segments = parse_rttm(rttm_src)

        with jobs_lock:
            jobs[job_id]["status"]   = "done"
            jobs[job_id]["progress"] = "Complete"
            jobs[job_id]["segments"] = segments
            jobs[job_id]["rttm"]     = rttm_dst

        n_speakers = len(set(s["speaker"] for s in segments))
        log.info(f"[{job_id}] Done. {len(segments)} segments, {n_speakers} speakers")

    except Exception as e:
        log.error(f"[{job_id}] Error: {e}", exc_info=True)
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = str(e)


def parse_rttm(path: str) -> list:
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == "SPEAKER":
                segments.append({
                    "speaker":  parts[7],
                    "start":    round(float(parts[3]), 3),
                    "end":      round(float(parts[3]) + float(parts[4]), 3),
                    "duration": round(float(parts[4]), 3),
                })
    return sorted(segments, key=lambda x: x["start"])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f      = request.files["audio"]
    ext    = Path(f.filename).suffix.lower() or ".wav"
    job_id = str(uuid.uuid4())[:8]

    # Save the raw upload (any format)
    raw_path  = str(UPLOAD_DIR / f"{job_id}_raw{ext}")
    wav_path  = str(UPLOAD_DIR / f"{job_id}.wav")
    f.save(raw_path)

    # Normalize → clean 16kHz mono WAV (fixes WebM, OGG, MP3, bad WAV headers…)
    try:
        normalize_audio(raw_path, wav_path)
    except ValueError as e:
        os.remove(raw_path)
        return jsonify({"error": f"Unsupported audio format: {e}"}), 400
    finally:
        # Clean up the raw file regardless
        if os.path.exists(raw_path):
            os.remove(raw_path)

    ns = request.form.get("num_speakers")
    num_speakers = int(ns) if ns and ns.isdigit() else None

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "progress": "Queued...",
            "segments": [], "rttm": None, "error": None,
            "audio_path": wav_path          # always a clean WAV now
        }

    t = threading.Thread(
        target=run_diarization,
        args=(job_id, wav_path, num_speakers),
        daemon=True
    )
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job.get("progress", ""),
        "segments": job.get("segments", []),
        "error":    job.get("error"),
    })

@app.route("/audio/<job_id>")
def serve_audio(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return send_file(job["audio_path"])

@app.route("/download/<job_id>")
def download_rttm(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or not job.get("rttm"):
        return jsonify({"error": "RTTM not ready"}), 404
    return send_file(
        job["rttm"], as_attachment=True,
        download_name=f"diarization_{job_id}.rttm"
    )

if __name__ == "__main__":
    mode = "FAST (3 scales, clustering)" if FAST_MODE else "FULL (5 scales, MSDD)"
    log.info(f"Mongolian Speaker Diarization Server [{mode}]")
    log.info(f"Speaker model : {SPEAKER_MODEL}")
    log.info(f"Open: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)