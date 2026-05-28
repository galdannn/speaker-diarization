
"""
finetune_titanet.py  -- Manual training loop (optimized)
Stable version for Common Voice speaker recognition (252 speakers)
"""

import os, sys, logging, time
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ---- Setup -------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs/final", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# ---- Config ------------------------------------------------------------------
PRETRAINED_NEMO  = "checkpoints/titanet_large_pretrained.nemo"
TRAIN_MANIFEST   = "data/manifests_cv/train_manifest.json"
VAL_MANIFEST     = "data/manifests_cv/val_manifest.json"

NUM_CLASSES      = 237
BATCH_SIZE       = 32
MAX_EPOCHS       = 50
SAMPLE_RATE      = 16000

LR               = 3e-5          # ↓ lower for finetuning
WEIGHT_DECAY     = 1e-3
PATIENCE         = 10
SAVE_TOP_K       = 3

FINAL_MODEL_PATH = "outputs/final/titanet_mongolian_finetuned_V2.nemo"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")

# ---- Load model --------------------------------------------------------------
import nemo.collections.asr as nemo_asr

log.info(f"Loading: {PRETRAINED_NEMO}")
model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
    PRETRAINED_NEMO, map_location="cpu"
)

# ---- Freeze encoder (important) ----------------------------------------------
for p in model.encoder.parameters():
    p.requires_grad = False
log.info("Encoder frozen.")

# ---- Replace decoder head ----------------------------------------------------
decoder = model.decoder
replaced = False

for name, module in decoder.named_modules():
    if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
        old_w = module.weight
        if old_w.dim() == 2:
            old_classes, emb_dim = old_w.shape
            module.weight = nn.Parameter(
                torch.empty(NUM_CLASSES, emb_dim)
            )
            nn.init.normal_(module.weight, std=0.01)

            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                module.bias = nn.Parameter(torch.zeros(NUM_CLASSES))

            log.info(
                f"Replaced angular weight in '{name}': "
                f"({old_classes}, {emb_dim}) -> ({NUM_CLASSES}, {emb_dim})"
            )
            replaced = True
            break

assert replaced, "Could not find angular softmax weight in decoder"

model = model.to(device)

# ---- Data loaders ------------------------------------------------------------
train_cfg = OmegaConf.create({
    "manifest_filepath": TRAIN_MANIFEST,
    "sample_rate": SAMPLE_RATE,
    "labels": None,
    "batch_size": BATCH_SIZE,
    "shuffle": True,
    "time_length": 3.0,
    "num_workers": 0,
    "pin_memory": False,
})

val_cfg = OmegaConf.create({
    "manifest_filepath": VAL_MANIFEST,
    "sample_rate": SAMPLE_RATE,
    "labels": None,
    "batch_size": BATCH_SIZE,
    "shuffle": False,
    "time_length": 3.0,
    "num_workers": 0,
    "pin_memory": False,
})

model.setup_training_data(train_cfg)
model.setup_validation_data(val_cfg)

train_loader = model._train_dl
val_loader   = model._validation_dl

log.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

# ---- Optimizer ---------------------------------------------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # only trainable params
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.98)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-7
)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# ---- Training loop -----------------------------------------------------------
best_val_loss  = float("inf")
patience_count = 0
saved_ckpts    = []

log.info("=" * 55)
log.info(f"Epochs={MAX_EPOCHS}  Batch={BATCH_SIZE}  LR={LR}")
log.info("=" * 55)

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    # ---- Train --------------------------------------------------------------
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0

    for i, batch in enumerate(train_loader):
        audio, audio_len, labels, _ = batch

        audio     = audio.to(device)
        audio_len = audio_len.to(device)
        labels    = labels.to(device).long().squeeze(-1)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, _ = model.forward(
                input_signal=audio,
                input_signal_length=audio_len
            )
            loss = model.loss(logits=logits, labels=labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        tr_loss += loss.item()
        tr_correct += (logits.argmax(-1) == labels).sum().item()
        tr_total += labels.size(0)

        if (i + 1) % 10 == 0:
            log.info(f"ep{epoch:02d} [{i+1}/{len(train_loader)}] loss={loss.item():.4f}")

    # ---- Validate -----------------------------------------------------------
    model.eval()
    vl_loss, vl_correct, vl_total = 0.0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            audio, audio_len, labels, _ = batch

            audio     = audio.to(device)
            audio_len = audio_len.to(device)
            labels    = labels.to(device).long().squeeze(-1)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model.forward(
                    input_signal=audio,
                    input_signal_length=audio_len
                )
                loss = model.loss(logits=logits, labels=labels)  # ✅ FIXED

            vl_loss += loss.item()
            vl_correct += (logits.argmax(-1) == labels).sum().item()
            vl_total += labels.size(0)

    # ---- Metrics ------------------------------------------------------------
    tr_loss /= len(train_loader)
    vl_loss /= len(val_loader)

    tr_acc = tr_correct / tr_total * 100
    vl_acc = vl_correct / vl_total * 100

    elapsed = time.time() - t0

    log.info(
        f"Epoch {epoch:02d}/{MAX_EPOCHS}  "
        f"train_loss={tr_loss:.4f} acc={tr_acc:.1f}%  "
        f"val_loss={vl_loss:.4f} acc={vl_acc:.1f}%  "
        f"[{elapsed:.0f}s]"
    )

    scheduler.step()

    # ---- Save checkpoint ----------------------------------------------------
    ckpt = f"checkpoints/titanet-mn-v2-ep{epoch:02d}-vl{vl_loss:.4f}.nemo"
    model.save_to(ckpt)

    saved_ckpts.append((vl_loss, ckpt))
    saved_ckpts.sort(key=lambda x: x[0])

    while len(saved_ckpts) > SAVE_TOP_K:
        _, old = saved_ckpts.pop()
        if os.path.exists(old):
            os.remove(old)

    # ---- Early stopping -----------------------------------------------------
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_count = 0
        model.save_to("checkpoints/titanet-mn-best_v2.nemo")
        log.info(f"New best: {best_val_loss:.4f}")
    else:
        patience_count += 1
        log.info(f"No improvement {patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch}")
            break

# ---- Final save --------------------------------------------------------------
model.save_to(FINAL_MODEL_PATH)

log.info(f"Final model saved: {FINAL_MODEL_PATH}")
log.info(f"Best val_loss: {best_val_loss:.4f}")

