"""
diarize.py
==========
Full speaker diarization pipeline for Mongolian audio using the
fine-tuned TitaNet model.

Pipeline:
  1. Voice Activity Detection (VAD)  – WebRTC or NeMo VAD
  2. Speaker Embedding Extraction    – fine-tuned TitaNet
  3. Spectral Clustering             – NME-SC or AHC
  4. RTTM Output                     – standard diarization format
  5. (Optional) DER evaluation       – pyannote.metrics

Usage:
  # Single file
  python diarize.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --audio path/to/audio.wav \
    --num_speakers 3 \
    --output output.rttm

  # Batch (directory)
  python diarize.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --audio_dir path/to/audio_dir/ \
    --output_dir outputs/rttm/

  # With known reference RTTM for DER evaluation
  python diarize.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --audio path/to/audio.wav \
    --reference_rttm path/to/reference.rttm
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf
import librosa
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

import nemo.collections.asr as nemo_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

TARGET_SR = 16000


# ─── VAD: WebRTC-based ───────────────────────────────────────────────────────

class WebRTCVAD:
    """
    Simple energy + WebRTC VAD for segmenting speech vs silence.
    Falls back to energy-based VAD if webrtcvad is not installed.
    """
    def __init__(self, aggressiveness: int = 2,
                 frame_ms: int = 30, sr: int = TARGET_SR):
        self.sr         = sr
        self.frame_ms   = frame_ms
        self.frame_size = int(sr * frame_ms / 1000)
        self.aggressiveness = aggressiveness

        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.use_webrtc = True
        except ImportError:
            log.warning("webrtcvad not installed, using energy VAD")
            self.use_webrtc = False

    def get_speech_segments(
        self,
        audio: np.ndarray,
        min_speech_sec: float = 0.3,
        min_silence_sec: float = 0.3,
        merge_gap_sec: float = 0.5
    ) -> list[tuple[float, float]]:
        """Returns list of (start_sec, end_sec) speech segments."""
        if self.use_webrtc:
            return self._webrtc_vad(audio, min_speech_sec,
                                    min_silence_sec, merge_gap_sec)
        else:
            return self._energy_vad(audio, min_speech_sec, merge_gap_sec)

    def _webrtc_vad(self, audio, min_speech_sec, min_silence_sec, merge_gap_sec):
        import webrtcvad

        # Convert to 16-bit PCM
        pcm = (audio * 32767).astype(np.int16)
        n_frames = len(pcm) // self.frame_size
        speech_frames = []

        for i in range(n_frames):
            frame = pcm[i * self.frame_size:(i + 1) * self.frame_size]
            frame_bytes = frame.tobytes()
            is_speech = self.vad.is_speech(frame_bytes, self.sr)
            speech_frames.append(is_speech)

        # Convert frame labels to time segments
        frame_dur = self.frame_ms / 1000.0
        raw_segs  = self._frames_to_segments(speech_frames, frame_dur)

        # Merge close segments
        merged = self._merge_segments(raw_segs, merge_gap_sec)

        # Filter short segments
        return [(s, e) for s, e in merged if e - s >= min_speech_sec]

    def _energy_vad(self, audio, min_speech_sec, merge_gap_sec):
        """Simple energy-based VAD."""
        hop      = self.frame_size // 2
        energy   = []
        for i in range(0, len(audio) - self.frame_size, hop):
            frame = audio[i:i + self.frame_size]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        energy    = np.array(energy)
        threshold = np.percentile(energy, 30)
        is_speech = energy > threshold

        frame_dur = hop / self.sr
        raw_segs  = self._frames_to_segments(is_speech.tolist(), frame_dur)
        merged    = self._merge_segments(raw_segs, merge_gap_sec)
        return [(s, e) for s, e in merged if e - s >= min_speech_sec]

    @staticmethod
    def _frames_to_segments(frames: list[bool], frame_dur: float):
        segs = []
        in_speech, start = False, 0.0
        for i, is_sp in enumerate(frames):
            t = i * frame_dur
            if is_sp and not in_speech:
                start, in_speech = t, True
            elif not is_sp and in_speech:
                segs.append((start, t))
                in_speech = False
        if in_speech:
            segs.append((start, len(frames) * frame_dur))
        return segs

    @staticmethod
    def _merge_segments(segs, gap_sec):
        if not segs:
            return []
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= gap_sec:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged


# ─── Embedding Extractor ──────────────────────────────────────────────────────

class TitaNetExtractor:
    def __init__(self, model_path: str):
        log.info(f"Loading fine-tuned TitaNet: {model_path}")
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_path, map_location="cpu"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_segment(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
        window_sec: float = 1.5,
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Extract embedding for a segment.
        For longer segments, averages embeddings over sliding windows.
        """
        duration = len(audio) / sr

        if duration <= window_sec * 1.5:
            # Short segment: extract directly
            return self._extract_single(audio)
        else:
            # Long segment: sliding window + mean-pool
            return self._extract_windowed(audio, sr, window_sec, overlap)

    def _extract_single(self, audio: np.ndarray) -> np.ndarray:
        audio_t = torch.tensor(audio).unsqueeze(0).to(self.device)
        len_t   = torch.tensor([len(audio)]).to(self.device)
        _, emb  = self.model.forward(
            input_signal=audio_t,
            input_signal_length=len_t
        )
        return emb.squeeze().cpu().numpy()

    def _extract_windowed(
        self,
        audio: np.ndarray,
        sr: int,
        window_sec: float,
        overlap: float
    ) -> np.ndarray:
        window  = int(window_sec * sr)
        step    = int(window * (1 - overlap))
        embs    = []
        for start in range(0, len(audio) - window // 2, step):
            end  = min(start + window, len(audio))
            clip = audio[start:end]
            if len(clip) / sr < 0.3:
                continue
            embs.append(self._extract_single(clip))
        if not embs:
            return self._extract_single(audio)
        return np.mean(embs, axis=0)


# ─── Spectral Clustering ──────────────────────────────────────────────────────

class SpectralClusterer:
    """
    Normalized Maximum Eigengap Spectral Clustering (NME-SC).
    Falls back to Agglomerative Hierarchical Clustering (AHC).
    """
    def __init__(self, method: str = "nme_sc"):
        self.method = method

    def cluster(
        self,
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
        max_speakers: int = 10,
        threshold: float = 0.015
    ) -> np.ndarray:
        """
        Returns cluster label for each embedding.
        If num_speakers is None, estimates it automatically.
        """
        n = len(embeddings)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([0])

        # Normalize embeddings (L2)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb   = embeddings / (norms + 1e-8)

        # Cosine similarity matrix
        sim_matrix = 1 - cdist(emb, emb, metric="cosine")
        np.fill_diagonal(sim_matrix, 0)
        sim_matrix = np.clip(sim_matrix, 0, 1)

        if num_speakers is not None:
            k = num_speakers
        else:
            k = self._estimate_num_speakers(
                sim_matrix, max_speakers, threshold
            )

        log.info(f"Clustering {n} segments into {k} speakers")

        if self.method == "nme_sc" and n >= k:
            return self._spectral_cluster(sim_matrix, k)
        else:
            return self._ahc_cluster(emb, k)

    @staticmethod
    def _estimate_num_speakers(
        sim_matrix: np.ndarray,
        max_k: int,
        threshold: float
    ) -> int:
        """Estimate k using eigengap heuristic."""
        from scipy.linalg import eigh

        n = len(sim_matrix)
        max_k = min(max_k, n - 1)

        # Normalized Laplacian
        D = np.diag(sim_matrix.sum(axis=1))
        L = D - sim_matrix
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(sim_matrix.sum(axis=1)) + 1e-8))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt

        eigenvalues = np.sort(eigh(L_sym, eigvals_only=True))

        # Find eigengap
        gaps = np.diff(eigenvalues[:max_k + 2])
        best_k = int(np.argmax(gaps)) + 1
        return max(2, best_k)

    @staticmethod
    def _spectral_cluster(sim_matrix: np.ndarray, k: int) -> np.ndarray:
        """Spectral clustering on similarity matrix."""
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=42,
            assign_labels="kmeans",
            n_init=10
        )
        return clustering.fit_predict(sim_matrix)

    @staticmethod
    def _ahc_cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
        """Agglomerative Hierarchical Clustering."""
        Z = linkage(embeddings, method="ward", metric="euclidean")
        return fcluster(Z, t=k, criterion="maxclust") - 1


# ─── RTTM Writer ─────────────────────────────────────────────────────────────

def write_rttm(
    segments: list[tuple[float, float]],
    labels: np.ndarray,
    filename: str,
    output_path: str
):
    """Write segments + cluster labels to RTTM format."""
    with open(output_path, "w") as f:
        for (start, end), label in zip(segments, labels):
            duration = end - start
            speaker  = f"SPEAKER_{int(label):02d}"
            f.write(
                f"SPEAKER {filename} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )
    log.info(f"RTTM written: {output_path}")


# ─── DER Evaluation ──────────────────────────────────────────────────────────

def compute_der(hypothesis_rttm: str, reference_rttm: str) -> dict:
    """Compute Diarization Error Rate using pyannote.metrics."""
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        def load_rttm(path):
            ann = Annotation()
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[0] != "SPEAKER":
                        continue
                    start    = float(parts[3])
                    duration = float(parts[4])
                    speaker  = parts[7]
                    ann[Segment(start, start + duration)] = speaker
            return ann

        ref  = load_rttm(reference_rttm)
        hyp  = load_rttm(hypothesis_rttm)
        metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        der   = metric(ref, hyp)

        details = metric.report()
        return {
            "DER":             round(float(der) * 100, 3),
            "missed_speech":   round(float(details.iloc[0]["missed detection"]) * 100, 3),
            "false_alarm":     round(float(details.iloc[0]["false alarm"]) * 100, 3),
            "speaker_error":   round(float(details.iloc[0]["confusion"]) * 100, 3),
        }
    except Exception as e:
        log.warning(f"DER computation failed: {e}")
        return {"DER": None, "error": str(e)}


# ─── Full Diarization Pipeline ────────────────────────────────────────────────

def diarize_file(
    audio_path: str,
    extractor: TitaNetExtractor,
    vad: WebRTCVAD,
    clusterer: SpectralClusterer,
    num_speakers: Optional[int] = None,
    max_speakers: int = 10,
    output_rttm: Optional[str] = None,
    reference_rttm: Optional[str] = None,
) -> dict:
    """
    Run the full diarization pipeline on one audio file.
    Returns dict with: segments, labels, RTTM path, (optional) DER.
    """
    filename = Path(audio_path).stem

    # 1. Load audio
    log.info(f"[{filename}] Loading audio...")
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration  = len(audio) / sr
    log.info(f"[{filename}] Duration: {duration:.1f}s")

    # 2. Voice Activity Detection
    log.info(f"[{filename}] Running VAD...")
    speech_segs = vad.get_speech_segments(audio)
    log.info(f"[{filename}] {len(speech_segs)} speech segments found")

    if not speech_segs:
        log.warning(f"[{filename}] No speech detected!")
        return {"segments": [], "labels": [], "der": None}

    # 3. Extract speaker embeddings per segment
    log.info(f"[{filename}] Extracting embeddings...")
    embeddings = []
    valid_segs = []
    for start, end in tqdm(speech_segs, desc="Embedding", leave=False):
        start_s = int(start * sr)
        end_s   = int(end * sr)
        clip    = audio[start_s:end_s]
        if len(clip) / sr < 0.3:
            continue
        emb = extractor.extract_segment(clip, sr)
        embeddings.append(emb)
        valid_segs.append((start, end))

    if not embeddings:
        log.warning(f"[{filename}] No valid embeddings!")
        return {"segments": [], "labels": [], "der": None}

    embeddings = np.array(embeddings)

    # 4. Cluster
    log.info(f"[{filename}] Clustering {len(embeddings)} segments...")
    labels = clusterer.cluster(
        embeddings,
        num_speakers=num_speakers,
        max_speakers=max_speakers
    )

    n_est = len(set(labels))
    log.info(f"[{filename}] Estimated {n_est} speakers")

    # 5. Write RTTM
    if output_rttm:
        os.makedirs(os.path.dirname(output_rttm) or ".", exist_ok=True)
        write_rttm(valid_segs, labels, filename, output_rttm)

    # 6. (Optional) DER
    der_result = None
    if reference_rttm and output_rttm:
        log.info(f"[{filename}] Computing DER...")
        der_result = compute_der(output_rttm, reference_rttm)
        log.info(f"[{filename}] DER: {der_result}")

    return {
        "filename":   filename,
        "duration":   duration,
        "n_segments": len(valid_segs),
        "n_speakers": n_est,
        "segments":   valid_segs,
        "labels":     labels.tolist(),
        "rttm_path":  output_rttm,
        "der":        der_result,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Speaker diarization with fine-tuned TitaNet"
    )
    parser.add_argument("--model", required=True,
                        help="Fine-tuned .nemo model path")
    # Input options
    parser.add_argument("--audio",      default=None, help="Single audio file")
    parser.add_argument("--audio_dir",  default=None, help="Directory of audio files")
    # Output options
    parser.add_argument("--output",     default=None, help="Single output RTTM")
    parser.add_argument("--output_dir", default="outputs/rttm",
                        help="Directory for batch RTTM outputs")
    # Diarization params
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Known number of speakers (None = auto-estimate)")
    parser.add_argument("--max_speakers", type=int, default=10,
                        help="Max speakers for auto-estimation")
    parser.add_argument("--clustering",
                        choices=["nme_sc", "ahc"], default="nme_sc")
    parser.add_argument("--vad_aggressiveness", type=int, default=2,
                        choices=[0, 1, 2, 3])
    # Evaluation
    parser.add_argument("--reference_rttm", default=None,
                        help="Reference RTTM for DER evaluation (single file)")
    parser.add_argument("--results_json", default="outputs/diarization_results.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize components
    extractor  = TitaNetExtractor(args.model)
    vad        = WebRTCVAD(aggressiveness=args.vad_aggressiveness)
    clusterer  = SpectralClusterer(method=args.clustering)

    results = []

    if args.audio:
        # Single file mode
        out_rttm = args.output or (
            Path(args.output_dir) / (Path(args.audio).stem + ".rttm")
        )
        result = diarize_file(
            audio_path=args.audio,
            extractor=extractor,
            vad=vad,
            clusterer=clusterer,
            num_speakers=args.num_speakers,
            max_speakers=args.max_speakers,
            output_rttm=str(out_rttm),
            reference_rttm=args.reference_rttm,
        )
        results.append(result)

        # Print timeline
        print(f"\n{'─'*55}")
        print(f"  DIARIZATION: {Path(args.audio).name}")
        print(f"{'─'*55}")
        print(f"  Duration   : {result['duration']:.1f}s")
        print(f"  Segments   : {result['n_segments']}")
        print(f"  Speakers   : {result['n_speakers']}")
        if result["der"]:
            print(f"  DER        : {result['der']['DER']:.2f}%")
        print(f"  RTTM saved : {out_rttm}\n")

    elif args.audio_dir:
        # Batch mode
        audio_dir  = Path(args.audio_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = sorted(
            [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_exts]
        )
        log.info(f"Processing {len(audio_files)} files...")

        for audio_path in tqdm(audio_files, desc="Diarizing"):
            out_rttm = output_dir / (audio_path.stem + ".rttm")
            try:
                result = diarize_file(
                    audio_path=str(audio_path),
                    extractor=extractor,
                    vad=vad,
                    clusterer=clusterer,
                    num_speakers=args.num_speakers,
                    max_speakers=args.max_speakers,
                    output_rttm=str(out_rttm),
                )
                results.append(result)
            except Exception as e:
                log.error(f"Failed {audio_path.name}: {e}")
                results.append({"filename": audio_path.stem, "error": str(e)})
    else:
        print("ERROR: Provide --audio or --audio_dir")
        sys.exit(1)

    # Save all results
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"\n✅ Results saved: {args.results_json}")


if __name__ == "__main__":
    main()