"""
train_msdd.py
=============
Fine-tunes NeMo's Multi-Scale Diarization Decoder (MSDD) on the
Mongolian diarization dataset.

Usage:
  python train_msdd.py
  python train_msdd.py --config configs/msdd_finetune.yaml
  python train_msdd.py --resume
"""
import os
import sys
import io
import json
import logging
import argparse
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecDiarLabelModel

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train_msdd.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("medium")

# ── Constants (must match precompute_embeddings.py) ───────────────────────────
SAMPLE_RATE = 16000
FP_POINTS   = 9

SCALES = {
    0: (3.0,  1.5),
    1: (2.5,  1.25),
    2: (2.0,  1.0),
    3: (1.5,  0.75),
    4: (1.0,  0.5),
}


# ── Fingerprint (identical to precompute_embeddings.py) ──────────────────────
def _fingerprint(seg):
    seg = seg.view(-1)  # force 1D

    positions = [
        0,
        len(seg) // 4,
        len(seg) // 2,
        3 * len(seg) // 4,
        len(seg) - 1,
    ]

    samples = tuple(
        round(seg[p].clamp(-1.0, 1.0).item(), 5)
        for p in positions
    )

    return samples


# ── EmbeddingStub ─────────────────────────────────────────────────────────────
class EmbeddingStub(nn.Module):
    """
    Drop-in replacement for TitaNet inside MSDD.

    At startup it builds a fingerprint → embedding lookup from your
    pre-computed .npy files.  When NeMo calls forward_for_export with
    a batch of audio segments, the stub does a ~1 ms dict lookup per
    segment instead of running TitaNet.

    TitaNet never touches the GPU after startup.
    """

    def __init__(self, manifest_paths: list[str], emb_dirs: list[str]):
        super().__init__()
        # Dummy param so Lightning doesn't complain about empty modules
        
        self._cache: dict[tuple, np.ndarray] = {}
        self._build_cache(manifest_paths, emb_dirs)

    # ── Cache build ───────────────────────────────────────────────
    def _build_cache(self, manifest_paths: list[str], emb_dirs: list[str]):
        import soundfile as sf

        for manifest_path, emb_dir in zip(manifest_paths, emb_dirs):
            if not os.path.exists(manifest_path):
                log.warning(f"EmbeddingStub: manifest not found: {manifest_path}")
                continue

            entries = [
                json.loads(line)
                for line in open(manifest_path, encoding="utf-8")
            ]
            log.info(
                f"Building fingerprint cache from {manifest_path} "
                f"({len(entries)} sessions)..."
            )

            for entry in entries:
                audio_path = entry["audio_filepath"]
                session_id = Path(audio_path).stem
                npy_file   = Path(emb_dir) / f"{session_id}.npy"

                if not npy_file.exists():
                    log.warning(f"  Missing .npy for session: {session_id}")
                    continue

                npy_data = np.load(str(npy_file), allow_pickle=True).item()

                # Load audio (CPU only — one time at startup)
                audio, sr = sf.read(audio_path)
                if sr != SAMPLE_RATE:
                    try:
                        import resampy
                        audio = resampy.resample(audio, sr, SAMPLE_RATE)
                    except ImportError:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

                audio_t = torch.tensor(audio, dtype=torch.float32)

                # Build fingerprint → embedding row for every segment × scale
                for scale_idx, (win_sec, shift_sec) in SCALES.items():
                    if scale_idx not in npy_data:
                        continue

                    win_samples   = int(win_sec   * SAMPLE_RATE)
                    shift_samples = int(shift_sec * SAMPLE_RATE)
                    starts        = list(
                        range(0, max(1, len(audio_t) - win_samples + 1), shift_samples)
                    )
                    embs = npy_data[scale_idx]   # shape (N_segs, D)

                    for i, s in enumerate(starts):
                        if i >= len(embs):
                            break
                        seg = audio_t[s : s + win_samples]
                        if len(seg) < win_samples:
                            seg = torch.nn.functional.pad(
                                seg, (0, win_samples - len(seg))
                            )
                        fp = _fingerprint(seg)
                        self._cache[fp] = embs[i]   # numpy row, shape (D,)

        log.info(
            f"EmbeddingStub ready: {len(self._cache):,} unique segment fingerprints cached"
        )

    # ── forward_for_export ────────────────────────────────────────
    # NeMo calls: speaker_model.forward_for_export(
    #     input_signal=..., input_signal_length=...
    # )
    # We intercept and return cached embeddings.
    def forward_for_export(self, *args, **kwargs):
        """
        Fully compatible with NeMo speaker_model API variants.
        """

        # ── Case 1: positional ─────────────────────────
        if len(args) >= 2:
            input_signal = args[0]
            input_signal_length = args[1]

        # ── Case 2: standard ───────────────────────────
        elif "input_signal" in kwargs:
            input_signal = kwargs["input_signal"]
            input_signal_length = kwargs["input_signal_length"]

        # ── Case 3: processed ──────────────────────────
        elif "processed_signal" in kwargs:
            input_signal = kwargs["processed_signal"]
            input_signal_length = kwargs["processed_signal_len"]

        # ── Case 4: audio_signal (your current failure) ─
        elif "audio_signal" in kwargs:
            input_signal = kwargs["audio_signal"]
            input_signal_length = kwargs["length"]

        else:
            raise RuntimeError(
                f"EmbeddingStub: unknown call signature: {list(kwargs.keys())}"
            )

        device = input_signal.device
        batch = input_signal.cpu()

        rows = []
        miss = 0

        for i in range(batch.shape[0]):
            length = int(input_signal_length[i].item())
            seg = batch[i]

            # Handle channel dimension explicitly
            if seg.dim() == 2:
                # (C, T) → take first channel (safe for mono diarization)
                seg = seg[0]

            # Now trim length
            seg = seg[:length]

            # Ensure strict 1D
            seg = seg.contiguous().view(-1)

            fp = _fingerprint(seg)

            if fp in self._cache:
                rows.append(torch.tensor(self._cache[fp], dtype=torch.float32))
            else:
                D = next(iter(self._cache.values())).shape[0] if self._cache else 192
                rows.append(torch.zeros(D, dtype=torch.float32))
                miss += 1
        if miss:
            log.debug(f"EmbeddingStub: {miss}/{batch.shape[0]} cache misses")

        embs = torch.stack(rows).to(device)
        dummy_logits = torch.zeros(embs.shape[0], 1, device=device)

        return dummy_logits, embs
    # ── Absorb any NeMo attribute access ─────────────────────────
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"{name} not implemented in EmbeddingStub") 


# ── DER Callback ──────────────────────────────────────────────────────────────
class DERCallback(pl.Callback):
    def __init__(self, val_manifest: str, out_dir: str, collar: float = 0.25):
        super().__init__()
        self.val_manifest = val_manifest
        self.out_dir      = out_dir
        self.collar       = collar

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            der = self._compute_der(trainer.current_epoch)
            pl_module.log("val_der", der, prog_bar=True, on_epoch=True)
            log.info(f"Epoch {trainer.current_epoch} | DER: {der:.2f}%")
        except Exception as e:
            log.warning(f"DER callback failed: {e}")

    def _compute_der(self, epoch: int) -> float:
        try:
            from pyannote.core import Annotation, Segment
            from pyannote.metrics.diarization import DiarizationErrorRate

            rttm_dir = Path(self.out_dir) / f"epoch_{epoch:03d}"
            rttm_dir.mkdir(parents=True, exist_ok=True)
            entries  = [json.loads(l) for l in open(self.val_manifest)]
            metric   = DiarizationErrorRate(collar=self.collar, skip_overlap=False)

            total_der = 0.0
            count     = 0
            for entry in entries:
                hyp_rttm = rttm_dir / (Path(entry["audio_filepath"]).stem + ".rttm")
                ref_rttm = entry.get("rttm_filepath")
                if not hyp_rttm.exists() or not ref_rttm:
                    continue
                ref        = self._load_rttm(ref_rttm)
                hyp        = self._load_rttm(str(hyp_rttm))
                total_der += float(metric(ref, hyp))
                count     += 1

            return (total_der / count * 100) if count > 0 else 0.0
        except Exception as e:
            log.warning(f"DER computation error: {e}")
            return 0.0

    @staticmethod
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


class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy over valid frames only.

    NeMo's MSDD training step calls the loss with `(probs, labels, target_lens)`.
    Some restored checkpoints expose `loss` as config rather than an instantiated
    callable, so we provide a small compatible fallback here.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(self, probs, labels, target_lens):
        target_lens = target_lens.to(device=probs.device)

        max_frames = min(probs.shape[1], labels.shape[1])
        probs = probs[:, :max_frames, :]
        labels = labels[:, :max_frames, :].to(probs.device)
        target_lens = target_lens.clamp(max=max_frames)

        time_mask = (
            torch.arange(max_frames, device=probs.device).unsqueeze(0)
            < target_lens.unsqueeze(1)
        )
        probs_fp32 = probs.float().clamp(self.eps, 1.0 - self.eps)
        labels_fp32 = labels.float()
        frame_mask = time_mask.unsqueeze(-1).expand_as(probs_fp32)

        with torch.cuda.amp.autocast(enabled=False):
            loss = torch.nn.functional.binary_cross_entropy(
                probs_fp32,
                labels_fp32,
                reduction="none",
            )
        loss = loss * frame_mask

        denom = frame_mask.sum().float().clamp_min(1.0)
        return loss.sum() * self.weight / denom


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(config_path: str, overrides: dict) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val, merge=True)
    log.info(f"Config loaded: {config_path}")
    return cfg


def validate_paths(cfg: DictConfig):
    checks = {
        "Train manifest": cfg.model.train_ds.manifest_filepath,
        "Val manifest":   cfg.model.validation_ds.manifest_filepath,
        "Speaker model":  cfg.speaker_model_path,
    }
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        log.info(f"  {'[OK]' if exists else '[MISSING]'} {name}: {path}")
        if not exists:
            all_ok = False
    if not all_ok:
        raise FileNotFoundError("Some required files are missing.")


def prepare_msdd_runtime(model: EncDecDiarLabelModel, cfg: DictConfig) -> EncDecDiarLabelModel:
    # NeMo expects these runtime attributes to exist before data loader setup.
    model.multiscale_args_dict = OmegaConf.to_container(
        cfg.model.multiscale_args_dict, resolve=True
    )
    log.info(f"multiscale_args_dict set: {model.multiscale_args_dict}")

    msdd_defaults = {
        "emb_batch_size": cfg.model.train_ds.get("emb_batch_size", 4),
        "scale_n": cfg.model.scale_n,
        "num_spks": cfg.model.train_ds.get("num_spks", 2),
        "soft_label_thres": cfg.model.train_ds.get("soft_label_thres", 0.5),
        "session_len_sec": cfg.model.train_ds.get("session_len_sec", 20),
        "emb_sess_train_dict": {},
        "emb_sess_val_dict": {},
        "emb_seq_train": {},
        "emb_seq_val": {},
        "clus_label_dict": {},
        "msdd_train_count_dict": {},
        "msdd_val_count_dict": {},
    }
    for attr, val in msdd_defaults.items():
        if not hasattr(model, attr):
            setattr(model, attr, val)

    return model


def ensure_msdd_loss(model: EncDecDiarLabelModel, cfg: DictConfig) -> EncDecDiarLabelModel:
    loss_obj = getattr(model, "loss", None)
    if callable(loss_obj) and not isinstance(loss_obj, DictConfig):
        return model

    loss_cfg = getattr(getattr(model, "cfg", None), "loss", None)
    if isinstance(loss_cfg, DictConfig) and loss_cfg.get("_target_"):
        from hydra.utils import instantiate

        model.loss = instantiate(loss_cfg)
        log.info(f"Instantiated NeMo loss from checkpoint config: {loss_cfg.get('_target_')}")
        return model

    loss_weight = cfg.model.get("loss", {}).get("weight", 1.0)
    model.loss = MaskedBCELoss(weight=loss_weight)
    log.info(f"Using fallback MaskedBCELoss(weight={loss_weight})")
    return model


# ── Build MSDD Model ──────────────────────────────────────────────────────────
def build_msdd_model(cfg: DictConfig) -> EncDecDiarLabelModel:
    log.info("Initializing MSDD model...")

    # ── Load pretrained MSDD ──────────────────────────────────────
    try:
        model = EncDecDiarLabelModel.from_pretrained("diar_msdd_telephonic")
        log.info("Loaded pretrained MSDD: diar_msdd_telephonic")
    except Exception:
        try:
            model = EncDecDiarLabelModel.from_pretrained("diar_msdd_meeting")
            log.info("Loaded pretrained MSDD: diar_msdd_meeting")
        except Exception as e:
            log.warning(f"Could not load pretrained MSDD: {e}")
            model = EncDecDiarLabelModel(cfg=cfg.model)

    prepare_msdd_runtime(model, cfg)

    # ── Load fine-tuned TitaNet weights (only for initialisation) ─
    ckpt_path = "model_weights.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing extracted checkpoint: {ckpt_path}")

    log.info(f"Loading fine-tuned TitaNet weights from {ckpt_path}...")
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    filtered   = {k: v for k, v in state_dict.items() if "decoder.final" not in k}

    missing, unexpected = model.msdd._speaker_model.load_state_dict(
        filtered, strict=False
    )
    log.info(f"TitaNet weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
   
    
    # ── Freeze speaker encoder ────────────────────────────────────
    for name, param in model.named_parameters():
        if "speaker_model" in name or "_speaker_model" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}")
   
    # ── Replace TitaNet with EmbeddingStub ────────────────────────
    # Pre-computed .npy files are loaded once here; TitaNet is then
    # deleted from GPU entirely — only the 3.4M MSDD transformer runs.
    log.info("Building EmbeddingStub fingerprint cache (one-time startup cost)...")

    train_manifest = cfg.model.train_ds.manifest_filepath
    val_manifest   = cfg.model.validation_ds.manifest_filepath
    train_emb_dir  = cfg.model.train_ds.emb_dir
    val_emb_dir    = cfg.model.validation_ds.emb_dir

    stub = EmbeddingStub(
        manifest_paths=[train_manifest, val_manifest],
        emb_dirs       =[train_emb_dir,  val_emb_dir],
    )

    # Keep reference BEFORE deleting
    original_speaker = model.msdd._speaker_model

    # Extract required components
    preprocessor = original_speaker.preprocessor

    # Delete heavy model
    del model.msdd._speaker_model
    torch.cuda.empty_cache()

    # Attach required interface to stub
    stub.preprocessor = preprocessor

    # Install stub
    model.msdd._speaker_model = stub
    log.info("TitaNet removed from GPU. EmbeddingStub installed.")

    return model
# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    pl.seed_everything(args.seed, workers=True)

    overrides = {}
    if args.batch_size: overrides["model.train_ds.batch_size"]  = args.batch_size
    if args.max_epochs: overrides["trainer.max_epochs"]         = args.max_epochs
    if args.lr:         overrides["model.optim.lr"]             = args.lr
    if args.devices:    overrides["trainer.devices"]            = args.devices

    cfg = load_config(args.config, overrides)
    validate_paths(cfg)

    model = build_msdd_model(cfg)

    # ── Trainer ───────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(save_dir="outputs/", name="msdd-mongolian")

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="msdd-mongolian-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    der_cb     = DERCallback(
        val_manifest=cfg.model.validation_ds.manifest_filepath,
        out_dir="outputs/diarization_val",
        collar=0.25,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.get("strategy", "auto"),
        precision=cfg.trainer.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 4),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
        num_sanity_val_steps=0,
        callbacks=[checkpoint_cb, early_stop, lr_monitor, der_cb],
        logger=tb_logger,
        enable_progress_bar=True,
    )

    model._trainer = trainer

    log.info("Setting up data loaders...")
    ensure_msdd_loss(model, cfg)

    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)

    ckpt_path = None
    last_ckpt = Path("checkpoints/last.ckpt")
    if last_ckpt.exists() and args.resume:
        ckpt_path = str(last_ckpt)
        log.info(f"Resuming from: {ckpt_path}")

    log.info(f"\n{'='*55}")
    log.info(f"  MSDD FINE-TUNING SUMMARY")
    log.info(f"{'='*55}")
    log.info(f"  Speaker model  : {cfg.speaker_model_path} (stub — GPU free)")
    log.info(f"  Train manifest : {cfg.model.train_ds.manifest_filepath}")
    log.info(f"  Val manifest   : {cfg.model.validation_ds.manifest_filepath}")
    log.info(f"  Max epochs     : {cfg.trainer.max_epochs}")
    log.info(f"  Batch size     : {cfg.model.train_ds.batch_size}")
    log.info(f"  Scales         : {cfg.model.scale_n}")
    log.info(f"{'='*55}\n")

    log.info("Starting MSDD fine-tuning...")
    trainer.fit(model, ckpt_path=ckpt_path)

    os.makedirs("outputs/final", exist_ok=True)
    final_path = "outputs/final/msdd_mongolian_finetuned.nemo"
    model.save_to(final_path)
    log.info(f"Fine-tuning complete. Final model: {final_path}")

    # ── Test evaluation ────────────────────────────────────────────
    test_manifest = cfg.model.test_ds.manifest_filepath
    if args.run_test and os.path.exists(test_manifest):
        try:
            for attr in [
                "emb_sess_test_dict", "emb_seq_test",
                "clus_test_label_dict", "msdd_test_count_dict",
            ]:
                if not hasattr(model, attr):
                    setattr(model, attr, {})
            model.setup_test_data(cfg.model.test_ds)
            trainer.test(model, ckpt_path="best")
        except Exception as e:
            log.warning(f"Test evaluation skipped after training due to setup error: {e}")
    elif args.run_test:
        log.warning(f"Test evaluation requested but manifest was not found: {test_manifest}")
    else:
        log.info("Skipping test evaluation. Use --run_test to enable it.")

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune NeMo MSDD for Mongolian speaker diarization"
    )
    p.add_argument("--config",     default="configs/msdd_finetune.yaml")
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--max_epochs", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--devices",    type=int,   default=None)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--run_test",   action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
