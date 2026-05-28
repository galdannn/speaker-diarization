#!/bin/bash
# ============================================================
# TitaNet Fine-Tuning Setup for Mongolian Speaker Diarization
# ============================================================

set -e

echo "======================================"
echo " NeMo TitaNet Environment Setup"
echo "======================================"

# 1. System dependencies
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    git \
    wget \
    2>/dev/null || true

# 2. Python dependencies
echo "[2/6] Installing Python packages..."
pip install --quiet --upgrade pip

# Core NeMo with ASR (includes speaker recognition)
pip install --quiet \
    "nemo_toolkit[asr]>=1.23.0" \
    "hydra-core>=1.3.0" \
    "omegaconf>=2.3.0"

# Audio processing
pip install --quiet \
    librosa \
    soundfile \
    pydub \
    webrtcvad \
    pyannote.audio \
    pyannote.core

# Training utilities
pip install --quiet \
    pytorch-lightning>=2.0.0 \
    tensorboard \
    wandb \
    tqdm \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn

# Evaluation metrics
pip install --quiet \
    pyannote.metrics \
    jiwer

echo "[3/6] Verifying NeMo installation..."
python -c "import nemo; print(f'NeMo version: {nemo.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "[4/6] Creating project directory structure..."
mkdir -p titanet_mongolian/{data/{raw,manifests,processed},configs,checkpoints,logs,outputs,scripts}

echo "[5/6] Downloading TitaNet-Large pretrained model..."
python - <<'EOF'
import nemo.collections.asr as nemo_asr
print("Downloading titanet-large pretrained weights...")
model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/titanet-large")
model.save_to("titanet_mongolian/checkpoints/titanet_large_pretrained.nemo")
print("Saved to: titanet_mongolian/checkpoints/titanet_large_pretrained.nemo")
EOF

echo "[6/6] Setup complete!"
echo ""
echo "Project structure:"
find titanet_mongolian -type d | sort
echo ""
echo "Next steps:"
echo "  1. Place your audio files in: titanet_mongolian/data/raw/"
echo "  2. Run: python prepare_dataset.py"
echo "  3. Run: python train.py"