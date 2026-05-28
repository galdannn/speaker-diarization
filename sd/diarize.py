"""
diarize.py
==========
Full speaker diarization inference using fine-tuned MSDD + TitaNet.

Pipeline:
  Audio → NeMo VAD → TitaNet Embeddings → MSDD → RTTM Output

Usage:
  # Single file
  python diarize.py \
    --audio path/to/recording.wav \
    --output outputs/rttm/recording.rttm

  # Known number of speakers
  python diarize.py \
    --audio path/to/recording.wav \
    --num_speakers 3

  # Batch
  python diarize.py \
    --audio_dir data/test_audio/ \
    --output_dir outputs/rttm/

  # With DER evaluation
  python diarize.py \
    --audio_dir data/test_audio/ \
    --rttm_dir  data/raw/rttm/ \
    --output_dir outputs/rttm/
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─── Fine-tuned TitaNet Compatibility Patch ───────────────────────────────────
#
# Fine-tuned TitaNet models have a custom decoder head size (e.g. 39 speakers
# for Mongolian data) that does NOT match the default titanet_large architecture
# (16681 VoxCeleb speakers).  For diarization inference we only use the encoder
# (embedding extractor) — the decoder head is irrelevant.
#
# NeMo's ClusteringDiarizer calls:
#   EncDecSpeakerLabelModel.restore_from(path, map_location=device)
# which eventually calls instance.load_state_dict(state_dict, strict=True),
# crashing on the decoder shape mismatch.
#
# This patch wraps SaveRestoreConnector.load_instance_with_state_dict to retry
# with strict=False on that specific RuntimeError, so the encoder weights load
# correctly while the mismatched decoder head is silently skipped.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_finetuned_titanet_patch():
    try:
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        from nemo.collections.asr.models import EncDecDiarLabelModel

        if hasattr(EncDecSpeakerLabelModel, '_finetuned_patch_applied'):
            return

        _orig_load_instance = SaveRestoreConnector.load_instance_with_state_dict

        def _patched_load_instance(self, instance, state_dict, strict):
            try:
                _orig_load_instance(self, instance, state_dict, strict)
            except RuntimeError as e:
                error_text = str(e)
                if isinstance(instance, EncDecDiarLabelModel) and (
                    "Missing key(s) in state_dict" in error_text
                    and "msdd._speaker_model." in error_text
                ):
                    log.warning(
                        "Fine-tuned MSDD patch: internal speaker-model weights are "
                        "missing from the checkpoint.\n"
                        "  Restoring MSDD with strict=False and using the diarizer's\n"
                        "  external speaker_embeddings model for inference."
                    )
                    instance.load_state_dict(state_dict, strict=False)
                elif "size mismatch" in error_text and "decoder" in error_text:
                    log.warning(
                        "Fine-tuned TitaNet patch: decoder head size mismatch detected.\n"
                        "  Filtering mismatched keys and reloading encoder only.\n"
                        "  This is expected for fine-tuned models — decoder is unused\n"
                        "  during diarization inference."
                    )
                    # strict=False alone doesn't help with size mismatches in PyTorch —
                    # we must manually remove keys whose shapes don't match the
                    # current model before calling load_state_dict.
                    model_state = instance.state_dict()
                    filtered_state = {
                        k: v for k, v in state_dict.items()
                        if k in model_state and v.shape == model_state[k].shape
                    }
                    skipped = [k for k in state_dict if k not in filtered_state]
                    if skipped:
                        log.warning(f"  Skipped {len(skipped)} mismatched key(s): {skipped}")
                    instance.load_state_dict(filtered_state, strict=False)
                else:
                    raise

        SaveRestoreConnector.load_instance_with_state_dict = _patched_load_instance
        EncDecSpeakerLabelModel._finetuned_patch_applied = True
        EncDecDiarLabelModel._finetuned_patch_applied = True
        log.info("Applied fine-tuned TitaNet compatibility patch (filters mismatched decoder keys).")

    except Exception as e:
        log.warning(f"Could not apply fine-tuned TitaNet patch: {e}. Will attempt to run anyway.")


_apply_finetuned_titanet_patch()


def _apply_nemo_270_patch():
    """Restore ClusteringDiarizer.time_stamps for NeMo 2.7 cached embeddings."""
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
                log.warning(f"NeMo 2.7.0 patch: could not parse {manifest_file}: {e}")

            self.time_stamps = time_stamps
            return result

        ClusteringDiarizer._extract_embeddings = _patched_extract
        ClusteringDiarizer._nemo_270_patched = True
        log.info("Applied NeMo 2.7.0 ClusteringDiarizer.time_stamps patch.")

    except Exception as e:
        log.warning(f"Could not apply NeMo 2.7.0 patch: {e}. Will attempt to run anyway.")


_apply_nemo_270_patch()


# ─── Build Inference Manifest ─────────────────────────────────────────────────

def build_inference_manifest(
    audio_paths:   list[str],
    rttm_paths:    Optional[list[str]] = None,
    num_speakers:  Optional[int]       = None,
    manifest_path: str                 = "temp_infer_manifest.json"
) -> str:
    """Create a temporary manifest file for NeMo diarization inference."""
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    with open(manifest_path, "w") as f:
        for i, audio_path in enumerate(audio_paths):
            entry = {
                "audio_filepath": str(Path(audio_path).resolve()),
                "offset":         0,
                "duration":       None,
                "label":          "infer",
                "text":           "-",
                "num_speakers":   num_speakers,
                "rttm_filepath":  str(Path(rttm_paths[i]).resolve())
                                  if rttm_paths else None,
                "uem_filepath":   None
            }
            f.write(json.dumps(entry) + "\n")

    log.info(f"Inference manifest: {manifest_path} ({len(audio_paths)} files)")
    return manifest_path


# ─── Shared config loader ─────────────────────────────────────────────────────

def _load_cfg(cfg_path: str) -> OmegaConf:
    """Load YAML and inject required top-level keys NeMo diarizers expect."""
    cfg = OmegaConf.load(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "device":      device,
        "num_workers": 0,       # must be 0 on Windows
        "batch_size":  64,
        "verbose":     False,
        "sample_rate": 16000,
    }))
    _normalize_diarizer_cfg(cfg)
    return cfg


def _normalize_diarizer_cfg(cfg: OmegaConf) -> None:
    """
    Add NeMo diarizer fields that may be absent from fine-tuning configs.

    NeMo accesses several `diarizer.vad.*` keys via attribute lookups while
    OmegaConf struct mode is enabled. If those keys are missing, oracle-VAD and
    some VAD code paths crash before inference starts.
    """
    if "diarizer" not in cfg:
        cfg.diarizer = OmegaConf.create({})

    OmegaConf.set_struct(cfg.diarizer, False)
    cfg.diarizer.setdefault("oracle_vad", False)
    cfg.diarizer.setdefault("collar", 0.25)
    cfg.diarizer.setdefault("ignore_overlap", True)
    cfg.diarizer.setdefault("manifest_filepath", None)
    cfg.diarizer.setdefault("out_dir", "outputs/diarization")

    if "vad" not in cfg.diarizer or cfg.diarizer.vad is None:
        cfg.diarizer.vad = OmegaConf.create({})

    OmegaConf.set_struct(cfg.diarizer.vad, False)
    cfg.diarizer.vad.setdefault("model_path", "vad_multilingual_marblenet")
    cfg.diarizer.vad.setdefault("external_vad_manifest", None)

    if "parameters" not in cfg.diarizer.vad or cfg.diarizer.vad.parameters is None:
        cfg.diarizer.vad.parameters = OmegaConf.create({})

    OmegaConf.set_struct(cfg.diarizer.vad.parameters, False)
    cfg.diarizer.vad.parameters.setdefault("onset", 0.4)
    cfg.diarizer.vad.parameters.setdefault("offset", 0.6)
    cfg.diarizer.vad.parameters.setdefault("pad_onset", 0.1)
    cfg.diarizer.vad.parameters.setdefault("pad_offset", 0.1)
    cfg.diarizer.vad.parameters.setdefault("min_duration_on", 0.2)
    cfg.diarizer.vad.parameters.setdefault("min_duration_off", 0.2)
    cfg.diarizer.vad.parameters.setdefault("filter_speech_first", True)
    cfg.diarizer.vad.parameters.setdefault("window_length_in_sec", 0.15)
    cfg.diarizer.vad.parameters.setdefault("shift_length_in_sec", 0.01)
    OmegaConf.set_struct(cfg.diarizer.vad.parameters, True)
    OmegaConf.set_struct(cfg.diarizer.vad, True)
    OmegaConf.set_struct(cfg.diarizer, True)


def _apply_vad_overrides(cfg: OmegaConf, *, oracle_vad: bool, vad_overrides: dict) -> None:
    cfg.diarizer.oracle_vad = bool(oracle_vad)

    if not vad_overrides:
        return

    OmegaConf.set_struct(cfg.diarizer.vad.parameters, False)
    for key, value in vad_overrides.items():
        if value is not None:
            cfg.diarizer.vad.parameters[key] = value
    OmegaConf.set_struct(cfg.diarizer.vad.parameters, True)


def _collect_vad_overrides(args: argparse.Namespace) -> dict:
    return {
        "onset": args.vad_onset,
        "offset": args.vad_offset,
        "pad_onset": args.vad_pad_onset,
        "pad_offset": args.vad_pad_offset,
        "min_duration_on": args.vad_min_duration_on,
        "min_duration_off": args.vad_min_duration_off,
    }


# ─── NeMo ClusteringDiarizer ──────────────────────────────────────────────────

def run_clustering_diarizer(
    manifest_path: str,
    out_dir:       str,
    cfg_path:      str,
    speaker_model_path: Optional[str] = None,
    num_speakers:  Optional[int] = None,
    oracle_vad: bool = False,
    vad_overrides: Optional[dict] = None,
) -> dict:
    """
    Run NeMo ClusteringDiarizer (TitaNet + Spectral Clustering).
    """
    from nemo.collections.asr.models import ClusteringDiarizer

    log.info("Running ClusteringDiarizer (TitaNet + Spectral Clustering)...")

    cfg = _load_cfg(cfg_path)
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir           = out_dir
    if speaker_model_path:
        cfg.diarizer.speaker_embeddings.model_path = speaker_model_path
    _apply_vad_overrides(cfg, oracle_vad=oracle_vad, vad_overrides=vad_overrides or {})

    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True

    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    log.info(f"Clustering diarization complete. RTTMs in: {out_dir}")
    return {"method": "clustering", "out_dir": out_dir}


# ─── NeMo NeuralDiarizer (MSDD) ──────────────────────────────────────────────

def run_msdd_diarizer(
    manifest_path: str,
    out_dir:       str,
    cfg_path:      str,
    msdd_model_path: Optional[str] = None,
    speaker_model_path: Optional[str] = None,
    num_speakers:  Optional[int] = None,
    oracle_vad: bool = False,
    vad_overrides: Optional[dict] = None,
) -> dict:
    try:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    except ImportError:
        from nemo.collections.asr.models import ClusteringDiarizer as NeuralDiarizer
        log.warning("NeuralDiarizer not found, falling back to ClusteringDiarizer")

    log.info("Running NeuralDiarizer (TitaNet + MSDD)...")

    # Monkey-patch _run_vad BEFORE constructing NeuralDiarizer.
    # NeMo's _run_vad reads smoothing/overlap/postprocessing_yaml but these
    # keys can't be in the YAML (OmegaConf struct mode rejects unknown keys
    # on load). Injecting them here at the method level is the safest fix.
    import nemo.collections.asr.models.clustering_diarizer as _cd
    from omegaconf import OmegaConf

    _orig_run_vad = _cd.ClusteringDiarizer._run_vad

    def _patched_run_vad(self, manifest_vad_input):
        OmegaConf.set_struct(self._vad_params, False)
        if not hasattr(self._vad_params, 'smoothing') or \
                'smoothing' not in self._vad_params:
            self._vad_params.smoothing = False
        if not hasattr(self._vad_params, 'overlap') or \
                'overlap' not in self._vad_params:
            self._vad_params.overlap = 0.875
        if not hasattr(self._vad_params, 'postprocessing_yaml') or \
                'postprocessing_yaml' not in self._vad_params:
            self._vad_params.postprocessing_yaml = None
        OmegaConf.set_struct(self._vad_params, True)
        return _orig_run_vad(self, manifest_vad_input)

    _cd.ClusteringDiarizer._run_vad = _patched_run_vad
    log.info("Patched ClusteringDiarizer._run_vad to inject VAD smoothing params.")

    cfg = _load_cfg(cfg_path)
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir           = out_dir
    if speaker_model_path:
        cfg.diarizer.speaker_embeddings.model_path = speaker_model_path
    _apply_vad_overrides(cfg, oracle_vad=oracle_vad, vad_overrides=vad_overrides or {})
    if msdd_model_path:
        cfg.diarizer.msdd_model.model_path = msdd_model_path
        # The fine-tuned MSDD checkpoint was trained with EmbeddingStub, so
        # inference should continue to use the configured TitaNet speaker model.
        cfg.diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = False

    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True

    diarizer = NeuralDiarizer(cfg=cfg)
    diarizer.diarize()

    log.info(f"MSDD diarization complete. RTTMs in: {out_dir}")
    return {"method": "msdd", "out_dir": out_dir}
# ─── DER Evaluation ──────────────────────────────────────────────────────────

def evaluate_der(
    hyp_rttm_dir: str,
    ref_rttm_dir: str,
    collar:       float = 0.25,
    skip_overlap: bool  = False
) -> dict:
    """
    Compute DER across all RTTM pairs in directories.
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        metric    = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        results   = {}
        total_der = 0.0
        count     = 0

        hyp_dir = Path(hyp_rttm_dir)
        ref_dir = Path(ref_rttm_dir)

        for hyp_rttm in sorted(hyp_dir.glob("*.rttm")):
            ref_rttm = ref_dir / hyp_rttm.name
            if not ref_rttm.exists():
                log.warning(f"No reference for: {hyp_rttm.name}")
                continue

            ref = _load_rttm(str(ref_rttm))
            hyp = _load_rttm(str(hyp_rttm))

            der = float(metric(ref, hyp)) * 100
            results[hyp_rttm.stem] = round(der, 3)
            total_der += der
            count     += 1

        avg_der = round(total_der / count, 3) if count > 0 else 0.0

        log.info(f"\n{'='*50}")
        log.info(f"  DER EVALUATION (collar={collar}s)")
        log.info(f"{'='*50}")
        log.info(f"  Files evaluated: {count}")
        log.info(f"  Average DER:     {avg_der:.2f}%")
        for fname, der in sorted(results.items()):
            log.info(f"  {fname}: {der:.2f}%")
        log.info(f"{'='*50}\n")

        return {"avg_der": avg_der, "per_file": results, "count": count}

    except ImportError:
        log.warning("pyannote.metrics not installed. pip install pyannote.metrics")
        return {"avg_der": None, "error": "pyannote.metrics not installed"}


def _load_rttm(path: str):
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


# ─── Print RTTM as Timeline ───────────────────────────────────────────────────

def print_timeline(rttm_path: str):
    """Print diarization result as a human-readable timeline."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            segments.append((start, start + duration, speaker))

    segments.sort(key=lambda x: x[0])

    print(f"\n{'─'*55}")
    print(f"  DIARIZATION TIMELINE: {Path(rttm_path).stem}")
    print(f"{'─'*55}")
    for start, end, speaker in segments:
        bar_len = int((end - start) * 3)
        bar     = "█" * max(1, bar_len)
        print(f"  {start:6.1f}s → {end:6.1f}s  [{speaker}]  {bar}")
    print(f"{'─'*55}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization with fine-tuned MSDD + TitaNet"
    )
    # Input
    parser.add_argument("--audio",      default=None, help="Single audio file")
    parser.add_argument("--audio_dir",  default=None, help="Directory of audio files")
    # Output
    parser.add_argument("--output",     default=None, help="Single output RTTM path")
    parser.add_argument("--output_dir", default="outputs/rttm")
    # Config
    parser.add_argument("--config",     default="msdd_finetune.yaml")
    parser.add_argument("--msdd_model_path", default=None,
                        help="Path to a fine-tuned MSDD .nemo to use for --method msdd")
    parser.add_argument("--speaker_model_path", default=None,
                        help="Optional override for the TitaNet speaker model path")
    # Options
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Known number of speakers (None = auto)")
    parser.add_argument("--oracle_vad", action="store_true",
                        help="Use manifest RTTMs as oracle VAD for diagnosis")
    parser.add_argument("--clean_output", action="store_true",
                        help="Delete the output directory before running")
    parser.add_argument("--method",
                        choices=["msdd", "clustering"], default="msdd",
                        help="msdd = full neural diarizer, clustering = baseline")
    parser.add_argument("--collar", type=float, default=0.25,
                        help="DER collar in seconds")
    parser.add_argument("--vad_onset", type=float, default=None)
    parser.add_argument("--vad_offset", type=float, default=None)
    parser.add_argument("--vad_pad_onset", type=float, default=None)
    parser.add_argument("--vad_pad_offset", type=float, default=None)
    parser.add_argument("--vad_min_duration_on", type=float, default=None)
    parser.add_argument("--vad_min_duration_off", type=float, default=None)
    # Evaluation
    parser.add_argument("--rttm_dir",     default=None,
                        help="Reference RTTM directory for DER evaluation")
    parser.add_argument("--results_json", default="outputs/diarization_results.json")
    args = parser.parse_args()

    if args.oracle_vad and not args.rttm_dir:
        log.error("--oracle_vad requires --rttm_dir so manifest RTTMs are available.")
        sys.exit(1)

    if args.clean_output and os.path.exists(args.output_dir):
        log.info(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs",       exist_ok=True)

    # Collect audio files
    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    if args.audio:
        audio_files = [args.audio]
    elif args.audio_dir:
        audio_files = sorted([
            str(f) for f in Path(args.audio_dir).iterdir()
            if f.suffix.lower() in audio_exts
        ])
        log.info(f"Found {len(audio_files)} audio files")
    else:
        log.error("Provide --audio or --audio_dir")
        sys.exit(1)

    if not audio_files:
        log.error("No audio files found.")
        sys.exit(1)

    # Match reference RTTMs if provided
    rttm_paths = None
    if args.rttm_dir:
        rttm_paths = []
        for af in audio_files:
            rttm = Path(args.rttm_dir) / (Path(af).stem + ".rttm")
            rttm_paths.append(str(rttm) if rttm.exists() else None)

    # Build inference manifest
    manifest_path = "outputs/infer_manifest.json"
    build_inference_manifest(
        audio_paths=audio_files,
        rttm_paths=rttm_paths,
        num_speakers=args.num_speakers,
        manifest_path=manifest_path
    )

    # Run diarization
    out_dir = args.output_dir
    vad_overrides = _collect_vad_overrides(args)
    active_vad = {k: v for k, v in vad_overrides.items() if v is not None}
    if args.oracle_vad:
        log.info("Oracle VAD enabled for this run.")
    elif active_vad:
        log.info(f"Applying VAD overrides: {active_vad}")

    if args.method == "msdd":
        result = run_msdd_diarizer(
            manifest_path=manifest_path,
            out_dir=out_dir,
            cfg_path=args.config,
            msdd_model_path=args.msdd_model_path,
            speaker_model_path=args.speaker_model_path,
            num_speakers=args.num_speakers,
            oracle_vad=args.oracle_vad,
            vad_overrides=vad_overrides,
        )
    else:
        result = run_clustering_diarizer(
            manifest_path=manifest_path,
            out_dir=out_dir,
            cfg_path=args.config,
            speaker_model_path=args.speaker_model_path,
            num_speakers=args.num_speakers,
            oracle_vad=args.oracle_vad,
            vad_overrides=vad_overrides,
        )

    result["oracle_vad"] = args.oracle_vad
    result["vad_overrides"] = active_vad

    # Print timelines — NeMo writes RTTMs into a pred_rttms/ subdir
    rttm_search_dirs = [
        Path(out_dir) / "pred_rttms",
        Path(out_dir),
    ]
    printed = False
    for rttm_dir in rttm_search_dirs:
        if rttm_dir.exists():
            for rttm_file in sorted(rttm_dir.glob("*.rttm")):
                print_timeline(str(rttm_file))
                printed = True
        if printed:
            break
    if not printed:
        log.warning(f"No RTTM files found under {out_dir}. Check NeMo output path.")

    # DER evaluation — NeMo writes predicted RTTMs to pred_rttms/ subdir
    der_results = None
    if args.rttm_dir:
        pred_rttm_dir = Path(out_dir) / "pred_rttms"
        if not pred_rttm_dir.exists():
            pred_rttm_dir = Path(out_dir)   # fallback

        log.info("Evaluating DER against reference RTTMs...")
        der_results = evaluate_der(
            hyp_rttm_dir=str(pred_rttm_dir),
            ref_rttm_dir=args.rttm_dir,
            collar=args.collar
        )
        result["der"] = der_results

    # Save results
    with open(args.results_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ Diarization complete!")
    log.info(f"   RTTMs saved to: {out_dir}/pred_rttms/")
    log.info(f"   Results saved:  {args.results_json}")
    if der_results and der_results.get("avg_der") is not None:
        log.info(f"   Average DER:    {der_results['avg_der']:.2f}%")


if __name__ == "__main__":
    main()
