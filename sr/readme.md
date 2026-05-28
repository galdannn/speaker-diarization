# TitaNet Fine-Tuning for Mongolian Speaker Diarization

Full pipeline to fine-tune NVIDIA's **TitaNet-Large** speaker embedding model on a Mongolian speech dataset and use it for speaker diarization.

---

## Project Structure

```
titanet_mongolian/
├── setup.sh                  # Install all dependencies + download pretrained model
├── prepare_dataset.py        # Convert raw audio+annotations → NeMo manifests
├── train.py                  # Fine-tune TitaNet with phased freezing strategy
├── evaluate.py               # EER / MinDCF / t-SNE evaluation
├── diarize.py                # Full diarization inference pipeline
├── requirements.txt
├── configs/
│   └── titanet_finetune.yaml # Training config (Hydra/OmegaConf)
├── data/
│   ├── raw/                  # ← put your audio + annotations here
│   ├── processed/            # Resampled 16kHz WAV segments (auto-generated)
│   └── manifests/            # NeMo JSON manifests (auto-generated)
├── checkpoints/              # Model checkpoints during training
├── outputs/                  # TensorBoard logs, final .nemo model, eval plots
└── logs/                     # Training logs
```

---

## Quick Start

### 1. Setup Environment
```bash
bash setup.sh
```

### 2. Prepare Your Dataset

**Layout A** – Long recordings + RTTM annotations:
```
data/raw/
    audio/  recording1.wav  recording2.wav ...
    rttm/   recording1.rttm recording2.rttm ...
```
```bash
python prepare_dataset.py --layout A \
    --audio_dir data/raw/audio \
    --rttm_dir  data/raw/rttm \
    --output_dir data/processed \
    --manifest_dir data/manifests
```

**Layout B** – Short clips in per-speaker folders:
```
data/raw/
    speaker_001/  utt1.wav  utt2.wav ...
    speaker_002/  utt1.wav  ...
```
```bash
python prepare_dataset.py --layout B \
    --data_dir data/raw \
    --output_dir data/processed \
    --manifest_dir data/manifests
```

**Layout C** – CSV with columns `audio_filepath, speaker_id, start, duration`:
```bash
python prepare_dataset.py --layout C \
    --csv_path data/raw/annotations.csv \
    --output_dir data/processed \
    --manifest_dir data/manifests
```

After running, **update `configs/titanet_finetune.yaml`**:
```yaml
model:
  num_classes: <NUMBER FROM prepare_dataset.py OUTPUT>
```

### 3. Fine-Tune TitaNet
```bash
python train.py \
    --config configs/titanet_finetune.yaml \
    --num_classes 120 \
    --batch_size 64 \
    --max_epochs 50
```

Multi-GPU (e.g., 2 GPUs):
```bash
python train.py --config configs/titanet_finetune.yaml --devices 2
```

Resume interrupted training:
```bash
python train.py --config configs/titanet_finetune.yaml --resume
```

### 4. Evaluate
```bash
python evaluate.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --manifest data/manifests/test_manifest.json \
    --output_dir outputs/eval
```
Produces: `metrics.json`, `det_curve.png`, `score_distribution.png`, `tsne_embeddings.png`

### 5. Run Diarization
```bash
# Single file
python diarize.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --audio meeting.wav \
    --num_speakers 3 \
    --output meeting.rttm

# With DER evaluation (if you have a reference RTTM)
python diarize.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --audio meeting.wav \
    --reference_rttm meeting_ref.rttm

# Batch processing
python diarize.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --audio_dir data/test_audio/ \
    --output_dir outputs/rttm/
```

---

## Training Strategy

The pipeline uses a **3-phase phased fine-tuning** approach:

| Phase | Epochs | Frozen      | LR       | Goal                          |
|-------|--------|-------------|----------|-------------------------------|
| 1     | 0–9    | Encoder     | 5e-4     | Adapt classifier to Mongolian |
| 2     | 10–29  | Lower layers| 1e-4     | Fine-tune upper encoder       |
| 3     | 30–49  | Nothing     | 3e-5     | Global fine-tuning            |

---

## Architecture: TitaNet-Large

```
Input Audio (16kHz)
       ↓
Mel Filterbank (80-dim)
       ↓
DepthSeparable Conv Blocks  ← pretrained on VoxCeleb2 (7,205 speakers)
  + Squeeze-Excitation
       ↓
Attention Pooling           ← aggregates frame embeddings
       ↓
Speaker Embedding (192-dim)
       ↓
ArcFace Classifier          ← fine-tuned for Mongolian speakers
       ↓
AAM-Softmax Loss
```

---

## Key Files to Update

| File | What to change |
|------|---------------|
| `configs/titanet_finetune.yaml` | `model.num_classes` after running `prepare_dataset.py` |
| `configs/titanet_finetune.yaml` | `trainer.devices`, `trainer.max_epochs`, batch sizes |
| `configs/titanet_finetune.yaml` | `freeze_strategy.phase*.epochs` based on your dataset size |

---

## Expected Output Files

```
outputs/
├── final/
│   └── titanet_mongolian_finetuned.nemo   ← Your fine-tuned model
├── eval/
│   ├── metrics.json                        ← EER, MinDCF
│   ├── det_curve.png
│   ├── score_distribution.png
│   └── tsne_embeddings.png
├── rttm/
│   └── *.rttm                              ← Diarization outputs
└── titanet-mongolian/
    └── version_N/                          ← TensorBoard logs
```

---

## Metrics Reference

| Metric | Description | Target |
|--------|-------------|--------|
| **EER** | Equal Error Rate (verification) | < 5% |
| **MinDCF** | Min Detection Cost (NIST SRE) | < 0.2 |
| **DER** | Diarization Error Rate | < 15% |

---

## Tips for Mongolian Data

- **Short utterances**: Min 0.5s, ideal 2–5s for training
- **Long recordings**: Automatically chunked into 3s segments with 0.5s overlap
- **Speaker balance**: Aim for ≥ 20 utterances per speaker
- **Data augmentation**: Speed perturbation (0.9–1.1×) + background noise is pre-configured
- **Batch size**: Start with 64; reduce to 32 or 16 if GPU OOM