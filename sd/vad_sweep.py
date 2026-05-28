"""
vad_sweep.py
============
Grid-search VAD parameters to find settings that minimize DER,
especially missed speech, before MSDD fine-tuning.

Runs ClusteringDiarizer (no MSDD) with different VAD thresholds
and reports a comparison table.

Usage:
  python vad_sweep.py
  python vad_sweep.py --skip_file conv_0058
  python vad_sweep.py --onset_values 0.30 0.35 0.40 --offset_values 0.45 0.50
"""

import os
import sys
import json
import logging
import argparse
import shutil
import itertools
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/vad_sweep.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _apply_nemo_patch():
    try:
        import json as _json
        from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer

        if hasattr(ClusteringDiarizer, "_nemo_270_patched"):
            return

        _orig_extract = ClusteringDiarizer._extract_embeddings

        def _patched_extract(self, manifest_file, *args, **kwargs):
            result = _orig_extract(self, manifest_file, *args, **kwargs)
            time_stamps = {}
            try:
                with open(manifest_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = _json.loads(line)
                        uid = entry.get("uniq_id")
                        if uid is None:
                            uid = entry.get("audio_filepath", "unknown")
                            uid = uid.replace("\\", "/").split("/")[-1]
                            uid = uid.rsplit(".", 1)[0]
                        start = float(entry.get("offset", 0.0))
                        end = start + float(entry.get("duration", 0.0))
                        time_stamps.setdefault(uid, []).append([start, end])
            except Exception as e:
                log.warning(f"Patch: could not parse {manifest_file}: {e}")
            self.time_stamps = time_stamps
            return result

        ClusteringDiarizer._extract_embeddings = _patched_extract
        ClusteringDiarizer._nemo_270_patched = True
        log.info("Applied NeMo 2.7.0 patch.")

    except Exception as e:
        log.warning(f"Could not apply patch: {e}")


_apply_nemo_patch()


def build_manifest(audio_dir, rttm_dir, manifest_path, skip_files=None):
    skip_files = set(skip_files or [])
    audio_dir = Path(audio_dir)
    rttm_dir = Path(rttm_dir)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for audio_path in sorted(audio_dir.glob("*.wav")):
            if audio_path.stem in skip_files:
                log.info(f"Skipping {audio_path.stem} (excluded)")
                continue
            rttm_path = rttm_dir / (audio_path.stem + ".rttm")
            if not rttm_path.exists():
                continue
            entry = {
                "audio_filepath": str(audio_path.resolve()),
                "rttm_filepath": str(rttm_path.resolve()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "uem_filepath": None,
            }
            f.write(json.dumps(entry) + "\n")
            count += 1
    return count


def run_one(manifest_path, out_dir, onset, offset, pad_onset, pad_offset,
            min_dur_on, speaker_model):
    """Run ClusteringDiarizer with given VAD params. Returns pred_rttms dir."""
    import torch
    from nemo.collections.asr.models import ClusteringDiarizer

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_dict = {
        "device": device,
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 64,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": out_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "speaker_embeddings": {
                "model_path": speaker_model,
                "parameters": {
                    "window_length_in_sec": [3.0, 2.5, 2.0, 1.5, 1.0],
                    "shift_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": True,
                },
            },
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "parameters": {
                    "onset": float(onset),
                    "offset": float(offset),
                    "pad_onset": float(pad_onset),
                    "pad_offset": float(pad_offset),
                    "min_duration_on": float(min_dur_on),
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": False,
                    "overlap": 0.875,
                    "postprocessing_yaml": None,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                },
            },
        },
    }

    cfg = OmegaConf.create(cfg_dict)
    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()
    return os.path.join(out_dir, "pred_rttms")


def compute_der(hyp_dir, ref_dir):
    from pyannote.metrics.diarization import DiarizationErrorRate

    hyp_dir = Path(hyp_dir)
    ref_dir = Path(ref_dir)
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)

    totals = {"der": 0, "miss": 0, "fa": 0, "conf": 0}
    count = 0

    for hyp_rttm in sorted(hyp_dir.glob("*.rttm")):
        ref_rttm = ref_dir / hyp_rttm.name
        if not ref_rttm.exists():
            continue

        ref = _load_rttm(str(ref_rttm))
        hyp = _load_rttm(str(hyp_rttm))
        details = metric(ref, hyp, detailed=True)
        total = details.get("total", 1.0) or 1.0

        totals["der"] += float(details["diarization error rate"]) * 100
        totals["miss"] += float(details["missed detection"]) / total * 100
        totals["fa"] += float(details["false alarm"]) / total * 100
        totals["conf"] += float(details["confusion"]) / total * 100
        count += 1

    if count == 0:
        return None

    return {k: round(v / count, 2) for k, v in totals.items()}


def _load_rttm(path):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            ann[Segment(float(parts[3]), float(parts[3]) + float(parts[4]))] = parts[7]
    return ann


def main():
    parser = argparse.ArgumentParser(description="Grid-search VAD settings to minimize DER")
    parser.add_argument("--audio_dir", default="data/diarization_dataset/test/audio")
    parser.add_argument("--rttm_dir", default="data/diarization_dataset/test/rttm")
    parser.add_argument("--output_dir", default="outputs/vad_sweep")
    parser.add_argument("--skip_file", default="conv_0058",
                        help="Filename stem to exclude (outlier)")
    parser.add_argument("--onset_range", nargs=3, type=float,
                        default=[0.20, 0.55, 0.05],
                        metavar=("START", "STOP", "STEP"),
                        help="Fallback onset sweep range if --onset_values is omitted")
    parser.add_argument("--onset_values", nargs="*", type=float, default=None)
    parser.add_argument("--offset_values", nargs="*", type=float, default=[0.50])
    parser.add_argument("--pad_onset_values", nargs="*", type=float, default=[0.20])
    parser.add_argument("--pad_offset_values", nargs="*", type=float, default=[0.20])
    parser.add_argument("--min_duration_on_values", nargs="*", type=float, default=[0.10])
    parser.add_argument("--speaker_model", default="titanet_large",
                        help="Speaker model for sweep (pretrained is fine)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    skip = [args.skip_file] if args.skip_file else []
    manifest_path = os.path.join(args.output_dir, "sweep_manifest.json")
    n = build_manifest(args.audio_dir, args.rttm_dir, manifest_path, skip_files=skip)
    log.info(f"Manifest: {n} files (excluding: {skip})")

    if args.onset_values:
        onsets = [round(float(x), 2) for x in args.onset_values]
    else:
        start, stop, step = args.onset_range
        onsets = [round(float(x), 2) for x in np.arange(start, stop + step / 2, step)]

    offsets = [round(float(x), 2) for x in args.offset_values]
    pad_onsets = [round(float(x), 2) for x in args.pad_onset_values]
    pad_offsets = [round(float(x), 2) for x in args.pad_offset_values]
    min_durs = [round(float(x), 2) for x in args.min_duration_on_values]
    grid = list(itertools.product(onsets, offsets, pad_onsets, pad_offsets, min_durs))

    log.info(
        f"\nGrid size: {len(grid)} | onset={onsets} offset={offsets} "
        f"pad_onset={pad_onsets} pad_offset={pad_offsets} min_duration_on={min_durs}\n"
    )

    results = []

    for onset, offset, pad_on, pad_off, min_dur in grid:
        label = (
            f"onset_{onset:.2f}_offset_{offset:.2f}"
            f"_pon_{pad_on:.2f}_poff_{pad_off:.2f}_minon_{min_dur:.2f}"
        )
        out_dir = os.path.join(args.output_dir, label)

        log.info(f"\n{'=' * 50}")
        log.info(
            f"  onset={onset:.2f}  offset={offset:.2f}  "
            f"pad_on={pad_on:.2f}  pad_off={pad_off:.2f}  min_dur={min_dur:.2f}"
        )
        log.info(f"{'=' * 50}")

        try:
            pred_dir = run_one(
                manifest_path=manifest_path,
                out_dir=out_dir,
                onset=onset,
                offset=offset,
                pad_onset=pad_on,
                pad_offset=pad_off,
                min_dur_on=min_dur,
                speaker_model=args.speaker_model,
            )
            metrics = compute_der(pred_dir, args.rttm_dir)
            if metrics:
                metrics["onset"] = onset
                metrics["offset"] = offset
                metrics["pad_onset"] = pad_on
                metrics["pad_offset"] = pad_off
                metrics["min_duration_on"] = min_dur
                results.append(metrics)
                log.info(
                    f"  DER={metrics['der']:.2f}%  "
                    f"Miss={metrics['miss']:.2f}%  "
                    f"FA={metrics['fa']:.2f}%  "
                    f"Conf={metrics['conf']:.2f}%"
                )
            else:
                log.warning(f"  No results for {label}")

        except Exception as e:
            log.error(f"  Failed {label}: {e}", exc_info=True)

    if not results:
        log.error("No results collected.")
        return

    log.info(f"\n{'=' * 65}")
    log.info(f"  VAD SWEEP SUMMARY  (collar=0.25, excluding {skip})")
    log.info(f"{'=' * 65}")
    log.info(
        f"  {'onset':>6}  {'offset':>6}  {'p_on':>6}  {'p_off':>6}  "
        f"{'min_on':>6}  {'DER':>7}  {'Miss':>7}  {'FA':>7}  {'Conf':>7}"
    )
    log.info(
        f"  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  "
        f"{'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}"
    )

    best = min(results, key=lambda x: x["der"])
    for r in results:
        marker = " <-- BEST" if r == best else ""
        log.info(
            f"  {r['onset']:>6.2f}  {r['offset']:>6.2f}  {r['pad_onset']:>6.2f}  "
            f"{r['pad_offset']:>6.2f}  {r['min_duration_on']:>6.2f}  "
            f"{r['der']:>6.2f}%  {r['miss']:>6.2f}%  "
            f"{r['fa']:>6.2f}%  {r['conf']:>6.2f}%{marker}"
        )

    log.info(
        f"\n  Best config: onset={best['onset']:.2f} offset={best['offset']:.2f} "
        f"pad_onset={best['pad_onset']:.2f} pad_offset={best['pad_offset']:.2f} "
        f"min_duration_on={best['min_duration_on']:.2f}"
    )
    log.info(
        f"  Best DER:   {best['der']:.2f}%  "
        f"(Miss={best['miss']:.2f}%  FA={best['fa']:.2f}%)"
    )
    log.info(f"{'=' * 65}")

    results_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    log.info(f"\nResults saved: {results_path}")
    log.info("\nOnce you have the best config, update msdd_finetune.yaml:")
    log.info(f"  vad.parameters.onset: {best['onset']:.2f}")
    log.info(f"  vad.parameters.offset: {best['offset']:.2f}")
    log.info(f"  vad.parameters.pad_onset: {best['pad_onset']:.2f}")
    log.info(f"  vad.parameters.pad_offset: {best['pad_offset']:.2f}")
    log.info(f"  vad.parameters.min_duration_on: {best['min_duration_on']:.2f}")


if __name__ == "__main__":
    main()
