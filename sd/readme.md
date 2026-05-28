# Mongolian Speaker Diarization — Approach 2 (MSDD Fine-Tuning)

Fine-tunes NeMo's full **Multi-Scale Diarization Decoder (MSDD)** pipeline
on Mongolian conversational speech data.

---

## Project Structure

```
titanet/
├── sr/                              ← Approach 1 (speaker recognition)
│   └── ...                            fine-tuned TitaNet lives here
│
└── sd/                              ← Approach 2 (this folder)
    ├── prepare_dataset.py           ← audio + RTTM → NeMo manifests
    ├── train_msdd.py                ← fine-tune MSDD pipeline
    ├── diarize.py                   ← run diarization on new audio
    ├── evaluate_der.py              ← compute DER / Miss / FA / Confusion
    ├── requirements.txt
    ├── configs/
    │   └── msdd_finetune.yaml       ← full pipeline config
    ├── data/
    │   ├── raw/
    │   │   ├── audio/               ← put .wav files here
    │   │   └── rttm/                ← put .rttm files here
    │   ├── manifests/               ← auto-generated
    │   └── processed/               ← auto-generated
    ├── checkpoints/                 ← saved during training
    └── outputs/
        ├── final/                   ← final .nemo model
        ├── rttm/                    ← diarization outputs
        └── eval/                    ← DER plots and metrics
```

---

## Dataset Format

### Audio files → `data/raw/audio/`
Any `.wav`, `.flac`, or `.mp3` files. Multi-speaker recordings.

### RTTM files → `data/raw/rttm/`
One `.rttm` file per audio file, same stem name.
```
SPEAKER recording1 1 0.000 3.450 <NA> <NA> Speaker_A <NA> <NA>
SPEAKER recording1 1 4.100 2.200 <NA> <NA> Speaker_B <NA> <NA>
SPEAKER recording1 1 6.500 5.800 <NA> <NA> Speaker_A <NA> <NA>
```

---

## Step-by-Step Commands

### 1. Prepare dataset
```powershell
python prepare_dataset.py `
  --audio_dir data\raw\audio `
  --rttm_dir  data\raw\rttm `
  --manifest_dir data\manifests
```

### 2. Train MSDD
```powershell
python train_msdd.py `
  --config configs\msdd_finetune.yaml `
  --batch_size 8 `
  --max_epochs 50
```

### 3. Run diarization
```powershell
# Approach 1: fine-tuned TitaNet + clustering
python diarize.py `
  --audio_dir data\diarization_dataset\test\audio `
  --rttm_dir data\diarization_dataset\test\rttm `
  --config msdd_finetune.yaml `
  --method clustering `
  --output_dir outputs\finetuned_titanet_clustering `
  --results_json outputs\finetuned_titanet_clustering_results.json

# Approach 2: fine-tuned TitaNet + fine-tuned MSDD
python diarize.py `
  --audio_dir data\diarization_dataset\test\audio `
  --rttm_dir data\diarization_dataset\test\rttm `
  --config msdd_finetune.yaml `
  --method msdd `
  --msdd_model_path outputs\final\msdd_mongolian_finetuned.nemo `
  --output_dir outputs\finetuned_msdd `
  --results_json outputs\finetuned_msdd_results.json
```

### 4. Evaluate DER
```powershell
python evaluate_der.py `
  --ref_rttm_dir data\diarization_dataset\test\rttm `
  --baseline_clustering_dir outputs\baseline_onset04\clustering `
  --baseline_msdd_dir outputs\baseline_onset04\msdd_pretrained `
  --finetuned_clustering_dir outputs\finetuned_titanet_clustering `
  --finetuned_msdd_dir outputs\finetuned_msdd `
  --output_dir   outputs\eval\
```

---

## Full Pipeline Architecture

```
Long Mongolian Recording
          ↓
    NeMo VAD (MarbleNet)
    detects speech regions
          ↓
    TitaNet (fine-tuned)          ← loaded from ../sr/
    extracts embeddings at
    5 different time scales
    [3.0s, 2.0s, 1.5s, 1.0s, 0.5s]
          ↓
    MSDD Transformer (fine-tuned)
    models speaker activity
    across all scales jointly
          ↓
    Per-frame speaker probabilities
          ↓
    RTTM: "who spoke when"
```

---

## Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **DER** | Miss + FA + Confusion | < 15% |
| **Miss** | Speech not detected | < 5% |
| **FA** | Non-speech labeled as speech | < 5% |
| **Confusion** | Wrong speaker assigned | < 10% |

---

## Connection to Approach 1 (sr/)

The fine-tuned TitaNet from `sr/` is loaded into the MSDD pipeline automatically.
The path is configured in `configs/msdd_finetune.yaml`:

```yaml
speaker_model_path: "../sr/outputs/final/titanet_mongolian_finetuned.nemo"
```

This means Approach 2 **builds on top of** Approach 1 — better speaker
embeddings from fine-tuned TitaNet → better MSDD diarization results.

---

## Thesis Comparison Table

Current exact numbers already saved under `outputs/eval/full_comparison.*`
use the common 14-file subset at collar `0.25`.
The first two rows below are verified from that artifact.
The last row should be filled after running true fine-tuned MSDD inference with
`--msdd_model_path outputs\final\msdd_mongolian_finetuned.nemo`.

| System | EER | DER | Miss | FA | Confusion |
|--------|-----|-----|------|----|-----------|
| Pretrained TitaNet + Clustering (baseline) | - | 21.846 | 17.937 | 0.022 | 3.887 |
| Fine-tuned TitaNet + Clustering (Approach 1) | - | 19.464 | 17.919 | 0.293 | 1.252 |
| Fine-tuned TitaNet + Fine-tuned MSDD (Approach 2) | - | - | - | - | - |
