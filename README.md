# Speaker Recognition and Speaker Diarization: Bachelor Thesis

GitHub repository: <https://github.com/galdannn/thesis.git>

This repository contains the source code, thesis document sources, configuration files, and prototype application for a bachelor thesis on **speaker recognition** and **speaker diarization** for Mongolian speech.

## Thesis Information

| Field | Details |
| --- | --- |
| Thesis title | Speaker Recognition ба Speaker Diarization: Бакалаврын Судалгааны Ажил |
| English title | Speaker Recognition and Speaker Diarization: Bachelor Thesis |
| Author | Мөнхжаргалын Золжаргал |
| University | Монгол Улсын Их Сургууль |
| Faculty | Мэдээллийн технологи, электроникийн сургууль |
| Department | Мэдээлэл, компьютерын ухааны тэнхим |
| Program | Мэдээллийн технологи (D061303) |
| Degree work | Бакалаврын судалгааны ажил |
| Supervisor | Др. Ч.Алтангэрэл |
| Co-supervisor | Др. Г.Амарсанаа |
| Location/date | Улаанбаатар, 2026 оны 12 дугаар сар |

## Research Summary

The thesis studies speaker recognition and speaker diarization pipelines for Mongolian speech. It covers data preparation, speaker embedding extraction, model fine-tuning, diarization inference, evaluation, and result analysis.

Main goals:

- Fine-tune and evaluate speaker recognition models for Mongolian speech.
- Build a diarization pipeline that identifies **who spoke when** in multi-speaker recordings.
- Compare pretrained and fine-tuned approaches using metrics such as **EER** and **DER**.
- Analyze diarization error components: missed speech, false alarm, and speaker confusion.

## Repository Structure

```text
titanet/
├── README.md                    # Project overview
├── analyse_dataset.py           # Dataset and experiment analysis script
├── document/                    # LaTeX thesis source files
├── prototype/                   # Lightweight web prototype for inference/demo
├── sd/                          # Speaker diarization experiments
└── sr/                          # Speaker recognition / TitaNet experiments
```

## Main Components

### `sr/` - Speaker Recognition

The `sr` module fine-tunes NVIDIA NeMo **TitaNet-Large** for Mongolian speaker embeddings.

Important files:

- `sr/prepare_dataset.py` - prepares raw audio and annotations into NeMo manifests.
- `sr/train.py` - fine-tunes TitaNet with phased freezing.
- `sr/evaluate.py` - evaluates speaker verification metrics such as EER and MinDCF.
- `sr/diarize.py` - runs diarization using the fine-tuned speaker embedding model.
- `sr/configs/titanet_finetune.yaml` - training configuration.
- `sr/readme.md` - detailed speaker recognition pipeline documentation.

### `sd/` - Speaker Diarization

The `sd` module builds on the fine-tuned TitaNet embeddings and fine-tunes NeMo's **Multi-Scale Diarization Decoder (MSDD)** pipeline.

Important files:

- `sd/prepare_dataset.py` - prepares diarization audio and RTTM annotations.
- `sd/train_msdd.py` - fine-tunes the MSDD diarization model.
- `sd/diarize.py` - runs clustering or MSDD-based diarization.
- `sd/evaluate_der.py` - computes DER, miss, false alarm, and confusion.
- `sd/configs/msdd_finetune.yaml` - MSDD pipeline configuration.
- `sd/readme.md` - detailed diarization pipeline documentation.

### `document/` - Thesis Document

Contains the LaTeX thesis sources:

- `document/main.tex` - main thesis document.
- `document/abstract.tex` - thesis abstract.
- `document/main-pre.tex`, `document/dics.sty`, `document/writing.tex` - template and formatting support files.

### `prototype/` - Demo Prototype

Contains a small web prototype:

- `prototype/server.py` - backend server.
- `prototype/static/index.html` - browser interface.

Generated uploads are ignored by Git and should stay local.

## Evaluation Metrics

| Metric | Meaning |
| --- | --- |
| EER | Equal Error Rate for speaker verification |
| MinDCF | Minimum Detection Cost Function |
| DER | Diarization Error Rate |
| Miss | Reference speech not detected by the system |
| FA | Non-speech incorrectly labeled as speech |
| Confusion | Speech assigned to the wrong speaker |

Current diarization comparison notes are documented in `sd/readme.md`. The tracked source repository excludes large model checkpoints, audio datasets, logs, and generated outputs.

## Setup

Install dependencies separately for each experiment track:

```bash
pip install -r sr/requirements.txt
pip install -r sd/requirements.txt
```

For the speaker recognition setup script:

```bash
cd sr
bash setup.sh
```

## Typical Workflow

1. Prepare speaker recognition data and fine-tune TitaNet in `sr/`.
2. Evaluate speaker embeddings with EER and related metrics.
3. Use the fine-tuned TitaNet model as the speaker embedding model for diarization in `sd/`.
4. Fine-tune or run MSDD diarization.
5. Evaluate diarization results with DER and its components.
6. Document results in the LaTeX thesis files under `document/`.

## Data and Model Artifacts

Large or generated artifacts are intentionally not committed to this repository:

- Audio datasets: `sd/data/`, `sr/data/`
- Model checkpoints: `sd/checkpoints/`, `sr/checkpoints/`
- Generated outputs: `sd/outputs/`, `sr/outputs/`, `outputs/`
- Logs and web uploads
- `.ckpt`, `.nemo`, and audio files

This keeps the GitHub repository focused on source code, configuration, documentation, and reproducible project structure.
