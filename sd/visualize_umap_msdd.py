"""
visualize_umap_msdd.py
======================

Create a 2D visualization from cached speaker embeddings produced during
diarization inference. By default it reads the fine-tuned MSDD run outputs,
aligns each subsegment with RTTM speaker labels, and saves a scatter plot.

Examples:
  py -3 visualize_umap_msdd.py
  py -3 visualize_umap_msdd.py --run_dir outputs\oracle_finetuned_msdd
  py -3 visualize_umap_msdd.py --scale 4 --color_by session
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MSDD speaker embeddings with UMAP or PCA."
    )
    parser.add_argument(
        "--run_dir",
        default="outputs/finetuned_msdd",
        help="Inference output directory containing speaker_outputs/.",
    )
    parser.add_argument(
        "--rttm_dir",
        default="data/diarization_dataset/test/rttm",
        help="Reference RTTM directory used for speaker coloring.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help="Embedding scale to visualize. Scale 4 is usually the clearest.",
    )
    parser.add_argument(
        "--color_by",
        choices=["speaker", "session"],
        default="speaker",
        help="Color points by RTTM speaker label or by session name.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=3000,
        help="Maximum number of points to plot for readability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and dimensionality reduction.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults under outputs/eval/.",
    )
    return parser.parse_args()


def load_embeddings(emb_path: Path) -> np.ndarray:
    with emb_path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        return np.asarray(data)

    raise TypeError(f"Unsupported embedding container in {emb_path}: {type(data)!r}")


def load_subsegments(meta_path: Path) -> pd.DataFrame:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    meta = pd.DataFrame(rows)
    meta["session"] = meta["audio_filepath"].apply(lambda p: Path(p).stem)
    meta["segment_start"] = meta["offset"].astype(float)
    meta["segment_end"] = meta["segment_start"] + meta["duration"].astype(float)
    return meta


def load_rttm_annotations(rttm_dir: Path) -> dict[str, list[tuple[float, float, str]]]:
    annotations: dict[str, list[tuple[float, float, str]]] = {}

    for rttm_path in sorted(rttm_dir.glob("*.rttm")):
        session = rttm_path.stem
        segments: list[tuple[float, float, str]] = []

        with rttm_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != "SPEAKER":
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append((start, start + duration, speaker))

        annotations[session] = segments

    return annotations


def assign_speaker_label(
    session: str,
    seg_start: float,
    seg_end: float,
    annotations: dict[str, list[tuple[float, float, str]]],
) -> str:
    overlaps: dict[str, float] = {}

    for ann_start, ann_end, speaker in annotations.get(session, []):
        overlap = max(0.0, min(seg_end, ann_end) - max(seg_start, ann_start))
        if overlap > 0:
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

    if not overlaps:
        return "UNK"

    return max(overlaps.items(), key=lambda kv: kv[1])[0]


def fit_2d_projection(X: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    try:
        import umap.umap_ as umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(X), "UMAP"
    except ImportError:
        X_centered = X - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        coords = X_centered @ vt[:2].T
        return coords, "PCA fallback"


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    run_dir = (base_dir / args.run_dir).resolve()
    rttm_dir = (base_dir / args.rttm_dir).resolve()

    emb_path = run_dir / "speaker_outputs" / "embeddings" / f"subsegments_scale{args.scale}_embeddings.pkl"
    meta_path = run_dir / "speaker_outputs" / f"subsegments_scale{args.scale}.json"

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = (
            base_dir
            / "outputs"
            / "eval"
            / f"{run_dir.name}_scale{args.scale}_{args.color_by}_umap.png"
        ).resolve()

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Subsegment metadata not found: {meta_path}")

    X = load_embeddings(emb_path)
    meta = load_subsegments(meta_path)

    if len(X) != len(meta):
        raise ValueError(
            f"Embedding rows ({len(X)}) do not match metadata rows ({len(meta)})."
        )

    if args.color_by == "speaker":
        annotations = load_rttm_annotations(rttm_dir)
        meta["speaker"] = meta.apply(
            lambda row: assign_speaker_label(
                session=row["session"],
                seg_start=row["segment_start"],
                seg_end=row["segment_end"],
                annotations=annotations,
            ),
            axis=1,
        )
        color_column = "speaker"
    else:
        color_column = "session"

    if args.sample_size and len(meta) > args.sample_size:
        rng = np.random.default_rng(args.seed)
        keep_idx = np.sort(rng.choice(len(meta), size=args.sample_size, replace=False))
        X = X[keep_idx]
        meta = meta.iloc[keep_idx].reset_index(drop=True)

    coords, method_name = fit_2d_projection(X, args.seed)

    plot_df = meta.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 9))
    categories = list(plot_df[color_column].astype(str).unique())
    cmap = plt.get_cmap("tab20", max(len(categories), 1))

    for idx, category in enumerate(categories):
        group = plot_df[plot_df[color_column].astype(str) == category]
        plt.scatter(
            group["x"],
            group["y"],
            s=12,
            alpha=0.72,
            color=cmap(idx),
            label=category,
        )

    plt.title(
        f"{method_name} of {run_dir.name} embeddings (scale {args.scale}, color={color_column})"
    )
    plt.xlabel(f"{method_name} 1")
    plt.ylabel(f"{method_name} 2")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")

    counts = plot_df[color_column].value_counts().to_dict()
    print(f"Saved plot: {output_path}")
    print(f"Projection: {method_name}")
    print(f"Points plotted: {len(plot_df)}")
    print(f"Groups: {counts}")


if __name__ == "__main__":
    main()
