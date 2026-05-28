"""
baseline.py
===========
Runs speaker diarization using PRETRAINED NeMo models with NO fine-tuning.
This establishes the baseline DER to compare against after fine-tuning.

Two baseline systems are run automatically:
  1. Pretrained TitaNet + Spectral Clustering   (ClusteringDiarizer)
  2. Pretrained TitaNet + Pretrained MSDD        (NeuralDiarizer)

DER is reported at BOTH collar=0.0 and collar=0.25 (thesis requirement).
Results are saved to outputs/baseline_results.json for evaluate_der.py.

Usage:
  python baseline.py
  python baseline.py --num_speakers 3
  python baseline.py --skip_msdd
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/baseline.log")
    ]
)
log = logging.getLogger(__name__)


# ─── NeMo 2.7.0 Monkey Patch ─────────────────────────────────────────────────
# NeMo 2.7.0 broke ClusteringDiarizer by moving time_stamps out of the object.
# The diarizer runs VAD and extracts embeddings fine, then crashes at clustering
# because self.time_stamps no longer exists as a direct attribute.
# This patch restores the attribute lookup so diarize() can proceed.

def _apply_nemo_270_patch():
    try:
        import json as _json
        import torch.nn as nn
        from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer

        if hasattr(ClusteringDiarizer, '_nemo_270_patched'):
            return

        # ── What broke in NeMo 2.7.0 ─────────────────────────────────────────
        # _extract_embeddings() no longer populates self.time_stamps after it
        # runs (either fresh or from cache).  The diarize() loop then tries:
        #
        #   self.multiscale_embeddings_and_timestamps[i] = [self.embeddings,
        #                                                    self.time_stamps]
        #
        # and crashes.  The timestamps ARE on disk — NeMo writes a subsegment
        # JSON for every scale before extraction.  We wrap _extract_embeddings
        # to parse that file and rebuild self.time_stamps after each call.
        # ─────────────────────────────────────────────────────────────────────

        _orig_extract = ClusteringDiarizer._extract_embeddings

        def _patched_extract(self, manifest_file, *args, **kwargs):
            result = _orig_extract(self, manifest_file, *args, **kwargs)

            # Rebuild time_stamps from the subsegment manifest NeMo just wrote.
            # Format per line: {"uniq_id": "conv_0001", "offset": 0.5,
            #                   "duration": 1.5, ...}
            time_stamps = {}
            try:
                with open(manifest_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = _json.loads(line)
                        uid   = entry.get("uniq_id")
                        if uid is None:
                            # fall back to stem of audio_filepath
                            uid = entry.get("audio_filepath", "unknown")
                            uid = uid.replace("\\", "/").split("/")[-1]
                            uid = uid.rsplit(".", 1)[0]
                        start = float(entry.get("offset", 0.0))
                        end   = start + float(entry.get("duration", 0.0))
                        time_stamps.setdefault(uid, []).append([start, end])
            except Exception as e:
                log.warning(f"NeMo 2.7.0 patch: could not parse {manifest_file}: {e}")

            self.time_stamps = time_stamps
            return result

        ClusteringDiarizer._extract_embeddings = _patched_extract
        ClusteringDiarizer._nemo_270_patched   = True
        log.info("Applied NeMo 2.7.0 ClusteringDiarizer.time_stamps patch.")

    except Exception as e:
        log.warning(f"Could not apply NeMo 2.7.0 patch: {e}. Will attempt to run anyway.")

# Apply patch at import time so it covers both ClusteringDiarizer and
# NeuralDiarizer (which calls ClusteringDiarizer internally).
_apply_nemo_270_patch()


# ─── Build Inference Manifest ─────────────────────────────────────────────────

def build_manifest(
    audio_dir:     str,
    rttm_dir:      str,
    manifest_path: str,
    num_speakers:  Optional[int] = None
) -> int:
    """
    Build a NeMo diarization manifest from audio + rttm directories.
    Returns number of entries written.
    """
    audio_dir = Path(audio_dir)
    rttm_dir  = Path(rttm_dir)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in {".wav", ".flac", ".mp3"}
    ])

    if not audio_files:
        log.error(f"No audio files found in {audio_dir}")
        return 0

    count = 0
    with open(manifest_path, "w") as f:
        for audio_path in audio_files:
            rttm_path = rttm_dir / (audio_path.stem + ".rttm")
            if not rttm_path.exists():
                log.warning(f"No RTTM for {audio_path.name}, skipping")
                continue

            entry = {
                "audio_filepath": str(audio_path.resolve()),
                "rttm_filepath":  str(rttm_path.resolve()),
                "offset":         0,
                "duration":       None,
                "label":          "infer",
                "text":           "-",
                "num_speakers":   num_speakers,
                "uem_filepath":   None
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    log.info(f"Manifest written: {manifest_path} ({count} recordings)")
    return count


# ─── Baseline 1: ClusteringDiarizer ──────────────────────────────────────────

def run_baseline_clustering(
    manifest_path: str,
    out_dir:       str,
    num_speakers:  Optional[int] = None,
) -> str:
    """
    Baseline 1: Pretrained TitaNet + Spectral Clustering.
    No fine-tuning at all. Pure out-of-the-box NeMo.

    NeMo writes output RTTMs to: out_dir/pred_rttms/
    """
    log.info("\n" + "="*55)
    log.info("  BASELINE 1: Pretrained TitaNet + Spectral Clustering")
    log.info("="*55)

    from nemo.collections.asr.models import ClusteringDiarizer

    os.makedirs(out_dir, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_dict = {
        "device":      device,
        "num_workers": 0,          # top-level: Windows requires 0
        "sample_rate": 16000,      # top-level: accessed as self._cfg.sample_rate
        "batch_size":  64,         # top-level: written by diarize() to self._cfg.batch_size
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir":           out_dir,

            # Required fields checked by ClusteringDiarizer.__init__
            "oracle_vad":        False,
            "collar":            0.25,
            "ignore_overlap":    True,

            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [3.0, 2.5, 2.0, 1.5, 1.0],
                    "shift_length_in_sec":  [1.5, 1.25, 1.0, 0.75, 0.5],
                    "multiscale_weights":   [1, 1, 1, 1, 1],
                    "save_embeddings":      True
                }
            },

            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "onset":               0.4,   # lowered from 0.8 to reduce missed speech
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
                    "postprocessing_yaml": None
                }
            },

            "clustering": {
                "parameters": {
                    "oracle_num_speakers":  num_speakers is not None,
                    "max_num_speakers":     8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold":     0.25,
                    "sparse_search_volume": 30
                }
            }
        }
    }

    cfg = OmegaConf.create(cfg_dict)
    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    # NeMo writes RTTMs to out_dir/pred_rttms/
    pred_dir = os.path.join(out_dir, "pred_rttms")
    log.info(f"Baseline 1 complete. RTTMs in: {pred_dir}")
    return pred_dir


# ─── Baseline 2: NeuralDiarizer (pretrained MSDD) ────────────────────────────

def run_baseline_msdd(
    manifest_path: str,
    out_dir:       str,
    num_speakers:  Optional[int] = None,
) -> Optional[str]:
    """
    Baseline 2: Pretrained TitaNet + Pretrained MSDD.
    MSDD is pretrained on English telephony data — not fine-tuned on Mongolian.
    This shows the transfer learning gap your fine-tuning will close.

    NeMo writes output RTTMs to: out_dir/pred_rttms/
    """
    log.info("\n" + "="*55)
    log.info("  BASELINE 2: Pretrained TitaNet + Pretrained MSDD")
    log.info("="*55)

    os.makedirs(out_dir, exist_ok=True)

    try:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    except ImportError:
        log.warning("NeuralDiarizer not available in this NeMo version — skipping Baseline 2.")
        return None

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Look up the correct MSDD model name for this NeMo version
    from nemo.collections.asr.models import EncDecDiarLabelModel
    available = [m.pretrained_model_name for m in EncDecDiarLabelModel.list_available_models()]
    log.info(f"Available MSDD models in this NeMo version: {available}")

    # Prefer telephony model; fall back to first available
    msdd_model_name = next(
        (m for m in available if "telephony" in m.lower()),
        next((m for m in available if "msdd" in m.lower() or "diar" in m.lower()), None)
    )
    if msdd_model_name is None:
        log.warning(f"No suitable MSDD pretrained model found. Available: {available}")
        log.warning("Skipping Baseline 2.")
        return None
    log.info(f"Using MSDD model: {msdd_model_name}")

    cfg_dict = {
        # Required by NeuralDiarizer._init_msdd_model (map_location=cfg.device)
        "device":      device,
        "num_workers": 0,          # top-level: Windows requires 0
        "sample_rate": 16000,      # top-level: accessed as self._cfg.sample_rate
        "batch_size":  64,         # top-level: used by internal ClusteringDiarizer
        "verbose":     False,      # top-level: read by msdd_models.run_overlap_aware_eval
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir":           out_dir,

            # Required fields checked on init
            "oracle_vad":        False,
            "collar":            0.25,
            "ignore_overlap":    True,

            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [3.0, 2.5, 2.0, 1.5, 1.0],
                    "shift_length_in_sec":  [1.5, 1.25, 1.0, 0.75, 0.5],
                    "multiscale_weights":   [1, 1, 1, 1, 1],
                    "save_embeddings":      True
                }
            },

            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "onset":               0.4,   # lowered from 0.8 to reduce missed speech
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
                    "postprocessing_yaml": None
                }
            },

            # Pretrained MSDD — name resolved at runtime from NeMo registry
            "msdd_model": {
                "model_path": msdd_model_name,
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size":            25,
                    "sigmoid_threshold":           [0.7],
                    "seq_eval_mode":               False,
                    "split_infer":                 True,
                    "diar_eval_settings":          [[0.25, True], [0.25, False]],
                    # Required by NeuralDiarizer.__init__ line 1020
                    "diar_window_length":          50,
                    "diar_batch_size":             25,
                    "diar_overlap_length":         1
                }
            },

            "clustering": {
                "parameters": {
                    "oracle_num_speakers":  num_speakers is not None,
                    "max_num_speakers":     8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold":     0.25,
                    "sparse_search_volume": 30
                }
            }
        }
    }

    cfg = OmegaConf.create(cfg_dict)
    diarizer = NeuralDiarizer(cfg=cfg)
    diarizer.diarize()

    pred_dir = os.path.join(out_dir, "pred_rttms")
    log.info(f"Baseline 2 complete. RTTMs in: {pred_dir}")
    return pred_dir


# ─── DER Computation (dual collar — thesis requirement) ───────────────────────

def _load_rttm(path: str):
    """Parse an RTTM file into a pyannote Annotation."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann


def compute_der(
    hyp_dir:  str,
    ref_dir:  str,
    name:     str = "System"
) -> dict:
    """
    Compute DER at BOTH collar=0.0 and collar=0.25.
    Thesis requirement: always report both values.
    Also breaks down Miss / FA / Confusion separately.
    """
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        log.warning("pyannote.metrics not installed. Run: pip install pyannote.metrics")
        return {"system": name, "avg_der_0": None, "avg_der_25": None}

    hyp_dir = Path(hyp_dir)
    ref_dir = Path(ref_dir)

    hyp_rttms = sorted(hyp_dir.glob("*.rttm"))
    if not hyp_rttms:
        log.warning(f"No hypothesis RTTMs found in {hyp_dir}")
        return {"system": name, "avg_der_0": None, "avg_der_25": None}

    results = {}

    for collar in [0.0, 0.25]:
        metric   = DiarizationErrorRate(collar=collar, skip_overlap=False)
        per_file = {}
        totals   = {"der": 0, "miss": 0, "fa": 0, "conf": 0}
        count    = 0

        for hyp_rttm in hyp_rttms:
            ref_rttm = ref_dir / hyp_rttm.name
            if not ref_rttm.exists():
                log.warning(f"No reference RTTM for {hyp_rttm.name}, skipping")
                continue

            ref     = _load_rttm(str(ref_rttm))
            hyp     = _load_rttm(str(hyp_rttm))
            details = metric(ref, hyp, detailed=True)
            total   = details.get("total", 1.0) or 1.0

            der  = float(details["diarization error rate"]) * 100
            miss = float(details["missed detection"])  / total * 100
            fa   = float(details["false alarm"])       / total * 100
            conf = float(details["confusion"])         / total * 100

            per_file[hyp_rttm.stem] = {
                "der": round(der, 3), "miss": round(miss, 3),
                "fa":  round(fa,  3), "conf": round(conf, 3)
            }
            totals["der"]  += der
            totals["miss"] += miss
            totals["fa"]   += fa
            totals["conf"] += conf
            count += 1

        if count == 0:
            results[collar] = None
            continue

        summary = {
            "collar":         collar,
            "n_files":        count,
            "avg_der":        round(totals["der"]  / count, 3),
            "avg_miss":       round(totals["miss"] / count, 3),
            "avg_fa":         round(totals["fa"]   / count, 3),
            "avg_confusion":  round(totals["conf"] / count, 3),
            "per_file":       per_file
        }
        results[collar] = summary

        log.info(f"\n{'─'*55}")
        log.info(f"  {name}  [collar={collar}s]")
        log.info(f"{'─'*55}")
        log.info(f"  Files      : {count}")
        log.info(f"  DER        : {summary['avg_der']:.2f}%   ← record this in your thesis table")
        log.info(f"  Miss       : {summary['avg_miss']:.2f}%")
        log.info(f"  False Alarm: {summary['avg_fa']:.2f}%")
        log.info(f"  Confusion  : {summary['avg_confusion']:.2f}%")

    return {
        "system":       name,
        "collar_0.0":   results.get(0.0),
        "collar_0.25":  results.get(0.25),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline diarization — no fine-tuning"
    )
    parser.add_argument("--audio_dir",    default="data/diarization_dataset/test/audio",
                        help="Test audio directory")
    parser.add_argument("--rttm_dir",     default="data/diarization_dataset/test/rttm",
                        help="Reference RTTM directory")
    parser.add_argument("--output_dir",   default="outputs/baseline_onset04",
                        help="Where to save baseline RTTM outputs")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Known number of speakers (None = auto-estimate)")
    parser.add_argument("--skip_msdd",    action="store_true",
                        help="Skip Baseline 2 (pretrained MSDD)")
    parser.add_argument("--skip_clustering", action="store_true",
                        help="Skip Baseline 1 (pretrained TitaNet + Clustering)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Build manifest from test split
    manifest_path = "outputs/baseline_manifest.json"
    n = build_manifest(
        audio_dir=args.audio_dir,
        rttm_dir=args.rttm_dir,
        manifest_path=manifest_path,
        num_speakers=args.num_speakers
    )
    if n == 0:
        log.error("No valid audio+RTTM pairs found. Check --audio_dir and --rttm_dir.")
        sys.exit(1)

    all_results = []

    # ── Baseline 1: Pretrained TitaNet + Spectral Clustering ─────────────────
    if not args.skip_clustering:
        b1_out = os.path.join(args.output_dir, "clustering")
        try:
            pred_dir = run_baseline_clustering(
                manifest_path=manifest_path,
                out_dir=b1_out,
                num_speakers=args.num_speakers,
            )
            result1 = compute_der(
                hyp_dir=pred_dir,
                ref_dir=args.rttm_dir,
                name="Baseline 1: Pretrained TitaNet + Clustering"
            )
            all_results.append(result1)
        except Exception as e:
            log.error(f"Baseline 1 failed: {e}", exc_info=True)

    # ── Baseline 2: Pretrained TitaNet + Pretrained MSDD ─────────────────────
    if not args.skip_msdd:
        b2_out = os.path.join(args.output_dir, "msdd_pretrained")
        try:
            pred_dir = run_baseline_msdd(
                manifest_path=manifest_path,
                out_dir=b2_out,
                num_speakers=args.num_speakers,
            )
            if pred_dir:
                result2 = compute_der(
                    hyp_dir=pred_dir,
                    ref_dir=args.rttm_dir,
                    name="Baseline 2: Pretrained TitaNet + Pretrained MSDD"
                )
                all_results.append(result2)
        except Exception as e:
            log.error(f"Baseline 2 failed: {e}", exc_info=True)

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = "outputs/baseline_results_onset04.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"  BASELINE SUMMARY — fill these into your thesis table")
    log.info(f"{'='*55}")
    for r in all_results:
        log.info(f"\n  {r['system']}")
        for collar_key in ["collar_0.0", "collar_0.25"]:
            c = r.get(collar_key)
            if c:
                label = collar_key.replace("collar_", "collar=")
                log.info(f"    [{label}s]  DER={c['avg_der']:.2f}%  "
                         f"Miss={c['avg_miss']:.2f}%  "
                         f"FA={c['avg_fa']:.2f}%  "
                         f"Conf={c['avg_confusion']:.2f}%")
    log.info(f"\n{'='*55}")
    log.info(f"  Saved: {results_path}")
    log.info(f"  These are your BEFORE fine-tuning numbers.")
    log.info(f"  Run evaluate_der.py after fine-tuning to compare.")
    log.info(f"\nBaseline complete!")


if __name__ == "__main__":
    main()