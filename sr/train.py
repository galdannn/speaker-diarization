"""
train.py
========
Fine-tunes TitaNet-Large on a Mongolian speaker recognition dataset
using NeMo + PyTorch Lightning.

Features:
  - Phased training: freeze encoder → partial unfreeze → full fine-tune
  - ArcFace (AAM-Softmax) loss
  - Mixed precision (AMP)
  - TensorBoard + optional W&B logging
  - Auto-resume from last checkpoint
  - Equal Error Rate (EER) validation callback

Usage:
  python train.py \
    --config configs/titanet_finetune.yaml \
    --num_classes 120 \
    --batch_size 64 \
    --max_epochs 50

  # Multi-GPU:
  python train.py --config configs/titanet_finetune.yaml --devices 2
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping
)
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
from omegaconf import OmegaConf, DictConfig
import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner
from nemo.utils import logging as nemo_log
from nemo.utils.exp_manager import exp_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log")
    ]
)
log = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)


# ─── EER Callback ────────────────────────────────────────────────────────────

class EERCallback(pl.Callback):
    """
    Computes Equal Error Rate on the validation set after each epoch.
    Uses cosine similarity between speaker embeddings.
    """
    def __init__(self, val_manifest: str, n_pairs: int = 5000):
        super().__init__()
        self.val_manifest = val_manifest
        self.n_pairs      = n_pairs

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            eer = self._compute_eer(pl_module)
            pl_module.log("val_eer", eer, prog_bar=True, on_epoch=True)
            log.info(f"Epoch {trainer.current_epoch} | EER: {eer:.4f}")
        except Exception as e:
            log.warning(f"EER computation failed: {e}")

    def _compute_eer(self, model) -> float:
        from sklearn.metrics import roc_curve

        model.eval()
        device = next(model.parameters()).device

        # Load manifest
        entries = []
        with open(self.val_manifest) as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        # Group by speaker
        by_speaker = {}
        for e in entries:
            spk = e["label"]
            if spk not in by_speaker:
                by_speaker[spk] = []
            by_speaker[spk].append(e["audio_filepath"])

        speakers = list(by_speaker.keys())
        if len(speakers) < 2:
            return 0.0

        # Generate pairs
        import random
        pairs, labels = [], []
        rng = random.Random(42)

        n_pos = self.n_pairs // 2
        n_neg = self.n_pairs - n_pos

        # Positive pairs (same speaker)
        for _ in range(n_pos):
            spk   = rng.choice(speakers)
            files = by_speaker[spk]
            if len(files) < 2:
                continue
            a, b = rng.sample(files, 2)
            pairs.append((a, b))
            labels.append(1)

        # Negative pairs (different speakers)
        for _ in range(n_neg):
            spk_a, spk_b = rng.sample(speakers, 2)
            a = rng.choice(by_speaker[spk_a])
            b = rng.choice(by_speaker[spk_b])
            pairs.append((a, b))
            labels.append(0)

        # Extract embeddings
        scores = []
        with torch.no_grad():
            for path_a, path_b in pairs:
                emb_a = self._get_embedding(model, path_a, device)
                emb_b = self._get_embedding(model, path_b, device)
                if emb_a is None or emb_b is None:
                    scores.append(0.0)
                    continue
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_a, emb_b, dim=-1
                ).item()
                scores.append(cos_sim)

        # Compute EER
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        return eer

    @staticmethod
    def _get_embedding(model, audio_path: str, device) -> torch.Tensor | None:
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
            audio_len    = torch.tensor([len(audio)]).to(device)

            with torch.no_grad():
                logits, embs = model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len
                )
            return embs
        except Exception:
            return None


# ─── Phased Freezing Callback ─────────────────────────────────────────────────

class PhasedFreezeCallback(pl.Callback):
    """
    Implements a 3-phase fine-tuning strategy:
      Phase 1: Freeze encoder → train only decoder/head
      Phase 2: Unfreeze upper encoder layers
      Phase 3: Full fine-tuning with low LR
    """
    def __init__(self, freeze_cfg: DictConfig):
        super().__init__()
        self.cfg = freeze_cfg

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        p1_end = self.cfg.phase1.epochs
        p2_end = self.cfg.phase2.epochs

        if epoch < p1_end:
            if epoch == 0:
                log.info("── Phase 1: Encoder FROZEN, training decoder only ──")
                self._set_encoder_frozen(pl_module, frozen=True)
                self._update_lr(trainer, self.cfg.phase1.lr)

        elif epoch < p2_end:
            if epoch == p1_end:
                log.info("── Phase 2: Partially unfreezing encoder ──")
                self._set_encoder_frozen(pl_module, frozen=False,
                    unfreeze_from=self.cfg.phase2.get("unfreeze_from_layer", None))
                self._update_lr(trainer, self.cfg.phase2.lr)

        else:
            if epoch == p2_end:
                log.info("── Phase 3: Full fine-tuning ──")
                self._unfreeze_all(pl_module)
                self._update_lr(trainer, self.cfg.phase3.lr)

        # Log frozen/unfrozen summary every 10 epochs
        if epoch % 10 == 0:
            self._log_frozen_summary(pl_module)

    @staticmethod
    def _set_encoder_frozen(model, frozen: bool, unfreeze_from: str = None):
        for name, param in model.named_parameters():
            if "encoder" in name:
                if unfreeze_from and unfreeze_from in name:
                    # Unfreeze from this layer onwards
                    frozen = False
                param.requires_grad = not frozen

    @staticmethod
    def _unfreeze_all(model):
        for param in model.parameters():
            param.requires_grad = True

    @staticmethod
    def _update_lr(trainer, new_lr: float):
        for pg in trainer.optimizers[0].param_groups:
            pg["lr"] = new_lr
        log.info(f"   LR updated to {new_lr}")

    @staticmethod
    def _log_frozen_summary(model):
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        total  = sum(1 for p in model.parameters())
        log.info(f"   Parameters: {total - frozen}/{total} trainable "
                 f"({100*(total-frozen)/total:.1f}%)")


# ─── Load & Patch Config ─────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    # Apply CLI overrides
    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val, merge=True)
    log.info(f"Loaded config: {config_path}")
    return cfg


def patch_num_classes(model, num_classes: int):
    """Replace the classifier head to match the new number of speakers."""
    in_features = model.decoder.final.in_features
    model.decoder.final = torch.nn.Linear(in_features, num_classes)
    log.info(f"Decoder head replaced: {in_features} → {num_classes} classes")


# ─── Main Training Function ───────────────────────────────────────────────────

def train(args):
    pl.seed_everything(args.seed, workers=True)

    # ── Load Config ──────────────────────────────────────────────────────────
    overrides = {}
    if args.num_classes:  overrides["model.num_classes"]           = args.num_classes
    if args.batch_size:   overrides["model.train_ds.batch_size"]   = args.batch_size
    if args.max_epochs:   overrides["trainer.max_epochs"]          = args.max_epochs
    if args.devices:      overrides["trainer.devices"]             = args.devices
    if args.lr:           overrides["model.optim.lr"]              = args.lr

    cfg = load_config(args.config, overrides)

    # ── Validate manifests exist ─────────────────────────────────────────────
    train_manifest = cfg.model.train_ds.manifest_filepath
    val_manifest   = cfg.model.validation_ds.manifest_filepath
    for path in [train_manifest, val_manifest]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Manifest not found: {path}\n"
                "Run prepare_dataset.py first."
            )

    # ── Load pretrained model ────────────────────────────────────────────────
    log.info(f"Loading pretrained model: {cfg.init_from_nemo_model}")
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
        cfg.init_from_nemo_model,
        override_config_path=None,
        map_location="cpu"
    )

    # ── Replace decoder head for new speaker count ───────────────────────────
    num_classes = cfg.model.num_classes
    log.info(f"Replacing classifier head for {num_classes} Mongolian speakers")
    # NeMo 2.7.0: change_labels does not exist — rebuild decoder head manually
    import torch.nn as nn
    decoder = model.decoder
    final_layer, final_name = None, None
    for name, module in decoder.named_modules():
        if isinstance(module, nn.Linear):
            final_layer, final_name = module, name
    if final_layer is not None:
        in_features = final_layer.in_features
        new_linear = nn.Linear(in_features, num_classes)
        parts = final_name.split('.')
        parent = decoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_linear.to(next(model.parameters()).device))
        log.info(f'Decoder head replaced: {in_features} -> {num_classes} classes')
    else:
        log.warning('Could not find final Linear layer in decoder')

    # ── Update data config ───────────────────────────────────────────────────
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)
    if os.path.exists(cfg.model.test_ds.manifest_filepath):
        model.setup_test_data(cfg.model.test_ds)

    # ── Callbacks ────────────────────────────────────────────────────────────
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="titanet-mongolian-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    eer_cb = EERCallback(
        val_manifest=val_manifest,
        n_pairs=2000
    )

    freeze_cb = PhasedFreezeCallback(cfg.freeze_strategy)

    callbacks = [lr_monitor, checkpoint_cb, early_stop, eer_cb, freeze_cb]

    # ── Logger ───────────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(
        save_dir="outputs/",
        name="titanet-mongolian",
        version=None  # auto-increment
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.get("strategy", "auto"),
        precision=cfg.trainer.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 1.0),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
        callbacks=callbacks,
        logger=tb_logger,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
    )

    # ── Resume from checkpoint if exists ─────────────────────────────────────
    ckpt_path = None
    last_ckpt = Path("checkpoints/last.ckpt")
    if last_ckpt.exists() and args.resume:
        ckpt_path = str(last_ckpt)
        log.info(f"Resuming from: {ckpt_path}")

    # ── Print trainable parameters summary ───────────────────────────────────
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"\n{'='*55}")
    log.info(f"  MODEL SUMMARY")
    log.info(f"{'='*55}")
    log.info(f"  Total parameters   : {total_params:,}")
    log.info(f"  Trainable params   : {trainable_params:,}")
    log.info(f"  Frozen params      : {total_params - trainable_params:,}")
    log.info(f"  Speakers           : {num_classes}")
    log.info(f"  Train manifest     : {train_manifest}")
    log.info(f"  Val manifest       : {val_manifest}")
    log.info(f"{'='*55}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("Starting fine-tuning...")
    trainer.fit(model, ckpt_path=ckpt_path)

    # ── Save final .nemo model ────────────────────────────────────────────────
    os.makedirs("outputs/final", exist_ok=True)
    final_path = "outputs/final/titanet_mongolian_finetuned.nemo"
    model.save_to(final_path)
    log.info(f"\n✅ Fine-tuning complete!")
    log.info(f"   Final model saved: {final_path}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    if os.path.exists(cfg.model.test_ds.manifest_filepath):
        log.info("Running test set evaluation...")
        trainer.test(model, ckpt_path="best")

    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune TitaNet-Large on Mongolian speaker data"
    )
    parser.add_argument(
        "--config", default="configs/titanet_finetune.yaml",
        help="Path to Hydra config YAML"
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="Number of unique speakers (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--devices", type=int, default=None,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints/last.ckpt if it exists"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = train(args)