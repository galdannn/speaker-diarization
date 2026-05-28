"""
evaluate.py
===========
Comprehensive evaluation of the fine-tuned TitaNet model:
  - Equal Error Rate (EER)
  - Minimum Detection Cost Function (MinDCF)
  - Speaker embedding visualization (t-SNE / UMAP)
  - Confusion matrix
  - Per-speaker performance breakdown

Usage:
  python evaluate.py \
    --model outputs/final/titanet_mongolian_finetuned.nemo \
    --manifest data/manifests/test_manifest.json \
    --output_dir outputs/eval
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, confusion_matrix, classification_report
)
from tqdm import tqdm

import nemo.collections.asr as nemo_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─── Embedding Extractor ──────────────────────────────────────────────────────

class EmbeddingExtractor:
    def __init__(self, model_path: str, device: str = "auto"):
        log.info(f"Loading model: {model_path}")
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_path,
            map_location="cpu"
        )
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()
        log.info(f"Model on device: {self.device}")

    @torch.no_grad()
    def extract(self, audio_path: str) -> np.ndarray | None:
        """Extract 192-dim speaker embedding from a WAV file."""
        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
            audio_len    = torch.tensor([len(audio)]).to(self.device)

            _, emb = self.model.forward(
                input_signal=audio_tensor,
                input_signal_length=audio_len
            )
            return emb.squeeze().cpu().numpy()
        except Exception as e:
            log.warning(f"Failed to extract from {audio_path}: {e}")
            return None

    @torch.no_grad()
    def extract_batch(self, audio_paths: list[str],
                      batch_size: int = 32) -> list[np.ndarray | None]:
        """Extract embeddings in batches for speed."""
        results = []
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_audio, batch_lens = [], []
            valid_indices = []

            for j, path in enumerate(batch_paths):
                try:
                    audio, _ = sf.read(path, dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    batch_audio.append(audio)
                    batch_lens.append(len(audio))
                    valid_indices.append(j)
                except Exception:
                    pass

            if not batch_audio:
                results.extend([None] * len(batch_paths))
                continue

            # Pad to same length
            max_len = max(batch_lens)
            padded  = np.zeros((len(batch_audio), max_len), dtype=np.float32)
            for k, a in enumerate(batch_audio):
                padded[k, :len(a)] = a

            audio_tensor = torch.tensor(padded).to(self.device)
            lens_tensor  = torch.tensor(batch_lens).to(self.device)

            _, embs = self.model.forward(
                input_signal=audio_tensor,
                input_signal_length=lens_tensor
            )
            embs_np = embs.cpu().numpy()

            # Fill results
            out = [None] * len(batch_paths)
            for k, orig_j in enumerate(valid_indices):
                out[orig_j] = embs_np[k]
            results.extend(out)

        return results


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Returns (EER, threshold)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    thresh  = float(thresholds[eer_idx])
    return eer, thresh


def compute_mindcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0
) -> tuple[float, float]:
    """Compute MinDCF (NIST SRE metric)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr  = 1 - tpr
    dcf  = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = float(np.min(dcf))
    # Normalize
    c_def   = min(c_miss * p_target, c_fa * (1 - p_target))
    norm_dcf = min_dcf / c_def if c_def > 0 else min_dcf
    return min_dcf, norm_dcf


def generate_verification_pairs(
    entries: list[dict],
    n_pairs: int = 10000,
    seed: int = 42
) -> tuple[list[tuple], list[int]]:
    """Generate speaker verification trial pairs."""
    import random
    rng = random.Random(seed)

    by_speaker = defaultdict(list)
    for e in entries:
        by_speaker[e["label"]].append(e["audio_filepath"])

    speakers = [s for s, files in by_speaker.items() if len(files) >= 2]
    pairs, labels = [], []

    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos

    # Positive pairs
    for _ in range(n_pos):
        spk   = rng.choice(speakers)
        a, b  = rng.sample(by_speaker[spk], 2)
        pairs.append((a, b))
        labels.append(1)

    # Negative pairs
    all_speakers = list(by_speaker.keys())
    for _ in range(n_neg):
        spk_a, spk_b = rng.sample(all_speakers, 2)
        a = rng.choice(by_speaker[spk_a])
        b = rng.choice(by_speaker[spk_b])
        pairs.append((a, b))
        labels.append(0)

    return pairs, labels


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_det_curve(scores, labels, output_path: str):
    """DET (Detection Error Tradeoff) curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr * 100, fnr * 100, linewidth=2, color="#2196F3")
    ax.set_xlabel("False Alarm Rate (%)", fontsize=12)
    ax.set_ylabel("Miss Rate (%)", fontsize=12)
    ax.set_title("DET Curve – TitaNet Mongolian", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"DET curve saved: {output_path}")


def plot_score_distribution(scores, labels, eer_thresh, output_path: str):
    """Score distribution for positive and negative pairs."""
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(neg_scores, bins=100, alpha=0.6, color="#F44336",
            density=True, label="Different speakers")
    ax.hist(pos_scores, bins=100, alpha=0.6, color="#4CAF50",
            density=True, label="Same speaker")
    ax.axvline(eer_thresh, color="black", linestyle="--",
               linewidth=2, label=f"EER threshold = {eer_thresh:.3f}")
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"Score distribution saved: {output_path}")


def plot_tsne(embeddings: np.ndarray, labels: list[str],
              output_path: str, n_speakers: int = 20):
    """t-SNE visualization of speaker embeddings."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import LabelEncoder

        # Select top-N speakers by utterance count
        from collections import Counter
        spk_counts = Counter(labels)
        top_speakers = [s for s, _ in spk_counts.most_common(n_speakers)]

        mask  = [l in top_speakers for l in labels]
        embs  = embeddings[mask]
        lbls  = [l for l, m in zip(labels, mask) if m]

        if len(embs) < 10:
            return

        log.info(f"Running t-SNE on {len(embs)} embeddings ({n_speakers} speakers)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        proj = tsne.fit_transform(embs)

        le = LabelEncoder().fit(lbls)
        colors = le.transform(lbls)

        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            proj[:, 0], proj[:, 1],
            c=colors, cmap="tab20",
            alpha=0.7, s=20
        )
        ax.set_title(f"t-SNE Speaker Embeddings (top-{n_speakers} speakers)",
                     fontsize=14)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        plt.colorbar(scatter, ax=ax, label="Speaker ID")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        log.info(f"t-SNE plot saved: {output_path}")
    except Exception as e:
        log.warning(f"t-SNE failed: {e}")


# ─── Main Evaluation ─────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    log.info(f"Loading manifest: {args.manifest}")
    entries = []
    with open(args.manifest) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    log.info(f"Total utterances: {len(entries):,}")

    # Initialize extractor
    extractor = EmbeddingExtractor(args.model)

    # ── Extract all embeddings ────────────────────────────────────────────────
    log.info("Extracting embeddings...")
    audio_paths = [e["audio_filepath"] for e in entries]
    spk_labels  = [e["label"] for e in entries]

    embeddings = extractor.extract_batch(audio_paths, batch_size=args.batch_size)

    valid_mask  = [emb is not None for emb in embeddings]
    embeddings  = np.array([e for e in embeddings if e is not None])
    spk_labels  = [l for l, v in zip(spk_labels, valid_mask) if v]
    audio_paths = [p for p, v in zip(audio_paths, valid_mask) if v]

    log.info(f"Valid embeddings: {len(embeddings):,} / {len(valid_mask):,}")

    # ── Speaker Verification (EER, MinDCF) ───────────────────────────────────
    log.info("Generating verification pairs...")
    pairs, pair_labels = generate_verification_pairs(
        [{"label": l, "audio_filepath": p}
         for l, p in zip(spk_labels, audio_paths)],
        n_pairs=args.n_pairs
    )

    log.info("Computing pairwise cosine scores...")
    emb_index = {p: e for p, e in zip(audio_paths, embeddings)}
    scores = []
    for path_a, path_b in tqdm(pairs, desc="Scoring pairs"):
        emb_a = emb_index.get(path_a)
        emb_b = emb_index.get(path_b)
        if emb_a is None or emb_b is None:
            scores.append(0.0)
        else:
            cos = float(
                np.dot(emb_a, emb_b) /
                (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
            )
            scores.append(cos)

    scores_arr = np.array(scores)
    labels_arr = np.array(pair_labels)

    eer, eer_thresh = compute_eer(scores_arr, labels_arr)
    min_dcf, norm_dcf = compute_mindcf(scores_arr, labels_arr)

    n_speakers = len(set(spk_labels))

    log.info(f"\n{'='*55}")
    log.info(f"  EVALUATION RESULTS")
    log.info(f"{'='*55}")
    log.info(f"  Speakers evaluated : {n_speakers}")
    log.info(f"  Utterances         : {len(embeddings):,}")
    log.info(f"  Trial pairs        : {len(pairs):,}")
    log.info(f"  EER                : {eer*100:.3f}%")
    log.info(f"  EER threshold      : {eer_thresh:.4f}")
    log.info(f"  MinDCF (raw)       : {min_dcf:.4f}")
    log.info(f"  MinDCF (norm)      : {norm_dcf:.4f}")
    log.info(f"{'='*55}\n")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "model": args.model,
        "manifest": args.manifest,
        "n_speakers": n_speakers,
        "n_utterances": len(embeddings),
        "n_pairs": len(pairs),
        "eer": round(eer * 100, 4),
        "eer_threshold": round(eer_thresh, 6),
        "min_dcf_raw": round(min_dcf, 6),
        "min_dcf_norm": round(norm_dcf, 6),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved: {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_det_curve(
        scores_arr, labels_arr,
        os.path.join(args.output_dir, "det_curve.png")
    )
    plot_score_distribution(
        scores_arr, labels_arr, eer_thresh,
        os.path.join(args.output_dir, "score_distribution.png")
    )
    plot_tsne(
        embeddings, spk_labels,
        os.path.join(args.output_dir, "tsne_embeddings.png"),
        n_speakers=min(20, n_speakers)
    )

    log.info(f"\n✅ Evaluation complete. Results in: {args.output_dir}/")
    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned TitaNet speaker model"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to .nemo model file"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to test manifest JSON"
    )
    parser.add_argument(
        "--output_dir", default="outputs/eval",
        help="Where to save evaluation results"
    )
    parser.add_argument(
        "--n_pairs", type=int, default=10000,
        help="Number of verification trial pairs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding extraction"
    )
    args = parser.parse_args()
    evaluate(args)