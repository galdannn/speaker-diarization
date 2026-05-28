"""
evaluate_der.py
===============
Compare diarization systems by DER, Miss, False Alarm, and Confusion.

This script is thesis-oriented:
  - resolves NeMo-style output folders automatically (`pred_rttms/`)
  - reports both collar=0.0 and collar=0.25
  - saves JSON, CSV, and a simple comparison plot

Example:
  python evaluate_der.py ^
    --ref_rttm_dir data/diarization_dataset/test/rttm ^
    --baseline_clustering_dir outputs/baseline_onset04/clustering ^
    --finetuned_clustering_dir outputs/finetuned_titanet_clustering ^
    --finetuned_msdd_dir outputs/rttm_finetuned_titanet_pretrained_msdd
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_rttm(path: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann


def resolve_rttm_dir(path_str: str) -> Path | None:
    path = Path(path_str)
    if not path.exists():
        return None

    pred_dir = path / "pred_rttms"
    if pred_dir.exists() and list(pred_dir.glob("*.rttm")):
        return pred_dir

    if path.is_dir() and list(path.glob("*.rttm")):
        return path

    return None


def evaluate_system(hyp_dir: str, ref_dir: str, name: str, collar: float) -> dict:
    from pyannote.metrics.diarization import DiarizationErrorRate

    hyp_path = Path(hyp_dir)
    ref_path = Path(ref_dir)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)

    totals = {"der": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0}
    per_file = {}
    count = 0

    for hyp_rttm in sorted(hyp_path.glob("*.rttm")):
        ref_rttm = ref_path / hyp_rttm.name
        if not ref_rttm.exists():
            log.warning(f"[{name}] Missing reference RTTM: {ref_rttm.name}")
            continue

        ref = load_rttm(str(ref_rttm))
        hyp = load_rttm(str(hyp_rttm))
        details = metric(ref, hyp, detailed=True)
        total = details.get("total", 1.0) or 1.0

        der = float(details["diarization error rate"]) * 100.0
        miss = float(details["missed detection"]) / total * 100.0
        fa = float(details["false alarm"]) / total * 100.0
        conf = float(details["confusion"]) / total * 100.0

        per_file[hyp_rttm.stem] = {
            "der": round(der, 3),
            "miss": round(miss, 3),
            "fa": round(fa, 3),
            "conf": round(conf, 3),
        }
        totals["der"] += der
        totals["miss"] += miss
        totals["fa"] += fa
        totals["conf"] += conf
        count += 1

    if count == 0:
        return {
            "system": name,
            "collar": collar,
            "n_files": 0,
            "resolved_hyp_dir": str(hyp_path),
            "avg_der": None,
            "avg_miss": None,
            "avg_fa": None,
            "avg_confusion": None,
            "per_file": {},
        }

    return {
        "system": name,
        "collar": collar,
        "n_files": count,
        "resolved_hyp_dir": str(hyp_path),
        "avg_der": round(totals["der"] / count, 3),
        "avg_miss": round(totals["miss"] / count, 3),
        "avg_fa": round(totals["fa"] / count, 3),
        "avg_confusion": round(totals["conf"] / count, 3),
        "per_file": per_file,
    }


def build_summary_rows(results_by_system: list[dict]) -> list[dict]:
    rows = []
    for system_result in results_by_system:
        system_name = system_result["system"]
        vad_type = system_result.get("vad_type", "normal")
        for collar_key, metrics in system_result["metrics"].items():
            if metrics["avg_der"] is None:
                continue
            rows.append(
                {
                    "System": system_name,
                    "VAD": vad_type,
                    "Collar (s)": metrics["collar"],
                    "Files": metrics["n_files"],
                    "DER (%)": metrics["avg_der"],
                    "Miss (%)": metrics["avg_miss"],
                    "FA (%)": metrics["avg_fa"],
                    "Confusion (%)": metrics["avg_confusion"],
                }
            )
    return rows


def print_tables(df: pd.DataFrame):
    for vad_type in ["normal", "oracle"]:
        vad_df = df[df["VAD"] == vad_type]
        if vad_df.empty:
            continue
        for collar in sorted(vad_df["Collar (s)"].unique()):
            subset = vad_df[vad_df["Collar (s)"] == collar].sort_values("DER (%)")
            log.info("")
            log.info("=" * 80)
            log.info(f"DER comparison at collar={collar} ({vad_type} VAD)")
            log.info("=" * 80)
            log.info(f"\n{subset.to_string(index=False)}\n")


def plot_comparison(df: pd.DataFrame, output_path: str):
    collar_025 = df[(df["Collar (s)"] == 0.25) & (df["VAD"] == "normal")].sort_values("DER (%)")
    if collar_025.empty:
        log.warning("Skipping plot because no normal-VAD collar=0.25 results were found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    x = range(len(collar_025))
    miss = collar_025["Miss (%)"].tolist()
    fa = collar_025["FA (%)"].tolist()
    conf = collar_025["Confusion (%)"].tolist()
    der = collar_025["DER (%)"].tolist()
    labels = collar_025["System"].tolist()

    axes[0].bar(x, miss, label="Miss", color="#d9534f")
    axes[0].bar(x, fa, bottom=miss, label="FA", color="#f0ad4e")
    axes[0].bar(
        x,
        conf,
        bottom=[m + f for m, f in zip(miss, fa)],
        label="Confusion",
        color="#5bc0de",
    )
    axes[0].set_title("DER Breakdown (collar=0.25)")
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(x, der, color="#4c72b0")
    axes[1].set_title("Total DER (collar=0.25)")
    axes[1].set_ylabel("DER (%)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, der):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle("Mongolian Speaker Diarization Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare diarization DER across systems")
    parser.add_argument(
        "--ref_rttm_dir",
        default="data/diarization_dataset/test/rttm",
        help="Reference RTTM directory",
    )
    parser.add_argument("--output_dir", default="outputs/eval", help="Directory for reports")
    parser.add_argument(
        "--baseline_clustering_dir",
        default="outputs/baseline_onset04/clustering",
        help="Pretrained TitaNet + clustering output directory",
    )
    parser.add_argument(
        "--finetuned_clustering_dir",
        default="outputs/finetuned_titanet_clustering",
        help="Fine-tuned TitaNet + clustering output directory",
    )
    parser.add_argument(
        "--finetuned_msdd_dir",
        default="outputs/rttm_finetuned_titanet_pretrained_msdd",
        help="Fine-tuned TitaNet + fine-tuned MSDD output directory",
    )
    parser.add_argument(
        "--oracle_baseline_clustering_dir",
        default=None,
        help="Optional oracle-VAD pretrained TitaNet + clustering output directory",
    )
    parser.add_argument(
        "--oracle_finetuned_pretrained_msdd_dir",
        default=None,
        help="Optional oracle-VAD fine-tuned TitaNet + pretrained MSDD output directory",
    )
    parser.add_argument(
        "--oracle_vad_msdd_dir",
        default=None,
        help="Optional oracle-VAD fine-tuned MSDD output directory",
    )
    parser.add_argument(
        "--oracle_finetuned_msdd_dir",
        default=None,
        help="Optional oracle-VAD fine-tuned TitaNet + fine-tuned MSDD output directory",
    )
    parser.add_argument(
        "--skip_baseline_msdd",
        action="store_true",
        help="Do not include pretrained MSDD comparison row",
    )
    parser.add_argument(
        "--baseline_msdd_dir",
        default="outputs/baseline_onset04/msdd_pretrained",
        help="Pretrained TitaNet + pretrained MSDD output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    systems = [
        ("Pretrained TitaNet + Clustering (baseline)", args.baseline_clustering_dir, "normal"),
        ("Fine-tuned TitaNet + Clustering (Approach 1)", args.finetuned_clustering_dir, "normal"),
        ("Fine-tuned TitaNet + Fine-tuned MSDD (Approach 2)", args.finetuned_msdd_dir, "normal"),
    ]
    if not args.skip_baseline_msdd:
        systems.insert(1, ("Pretrained TitaNet + Pretrained MSDD", args.baseline_msdd_dir, "normal"))
    if args.oracle_baseline_clustering_dir:
        systems.append(
            (
                "Pretrained TitaNet + Clustering (baseline, Oracle VAD)",
                args.oracle_baseline_clustering_dir,
                "oracle",
            )
        )
    if args.oracle_finetuned_pretrained_msdd_dir:
        systems.append(
            (
                "Fine-tuned TitaNet + Pretrained MSDD (Oracle VAD)",
                args.oracle_finetuned_pretrained_msdd_dir,
                "oracle",
            )
        )
    if args.oracle_finetuned_msdd_dir:
        systems.append(
            (
                "Fine-tuned TitaNet + Fine-tuned MSDD (Oracle VAD)",
                args.oracle_finetuned_msdd_dir,
                "oracle",
            )
        )
    elif args.oracle_vad_msdd_dir:
        systems.append(
            (
                "Fine-tuned TitaNet + Fine-tuned MSDD (Oracle VAD)",
                args.oracle_vad_msdd_dir,
                "oracle",
            )
        )

    all_results = []
    for system_name, requested_dir, vad_type in systems:
        resolved = resolve_rttm_dir(requested_dir)
        if resolved is None:
            log.warning(f"Skipping {system_name}: no RTTMs found under {requested_dir}")
            continue

        log.info(f"Evaluating {system_name}")
        metrics_00 = evaluate_system(str(resolved), args.ref_rttm_dir, system_name, collar=0.0)
        metrics_025 = evaluate_system(str(resolved), args.ref_rttm_dir, system_name, collar=0.25)
        all_results.append(
            {
                "system": system_name,
                "vad_type": vad_type,
                "requested_dir": requested_dir,
                "resolved_dir": str(resolved),
                "metrics": {
                    "collar_0.0": metrics_00,
                    "collar_0.25": metrics_025,
                },
            }
        )

    if not all_results:
        log.error("No comparable RTTM outputs were found.")
        return

    rows = build_summary_rows(all_results)
    df = pd.DataFrame(rows)
    print_tables(df)

    json_path = Path(args.output_dir) / "comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"JSON saved: {json_path}")

    csv_path = Path(args.output_dir) / "comparison_summary.csv"
    df.sort_values(["VAD", "Collar (s)", "DER (%)"]).to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")

    thesis_normal_path = Path(args.output_dir) / "thesis_table_normal_vad_collar_0_25.csv"
    thesis_normal_df = (
        df[(df["Collar (s)"] == 0.25) & (df["VAD"] == "normal")]
        .sort_values("DER (%)")
        .loc[:, ["System", "DER (%)", "Miss (%)", "FA (%)", "Confusion (%)", "Files"]]
    )
    thesis_normal_df.to_csv(thesis_normal_path, index=False)
    log.info(f"Normal-VAD thesis table CSV saved: {thesis_normal_path}")

    thesis_oracle_path = Path(args.output_dir) / "thesis_table_oracle_vad_collar_0_25.csv"
    thesis_oracle_df = (
        df[(df["Collar (s)"] == 0.25) & (df["VAD"] == "oracle")]
        .sort_values("DER (%)")
        .loc[:, ["System", "DER (%)", "Miss (%)", "FA (%)", "Confusion (%)", "Files"]]
    )
    thesis_oracle_df.to_csv(thesis_oracle_path, index=False)
    log.info(f"Oracle-VAD thesis table CSV saved: {thesis_oracle_path}")

    plot_comparison(df, str(Path(args.output_dir) / "comparison_plot.png"))


if __name__ == "__main__":
    main()
