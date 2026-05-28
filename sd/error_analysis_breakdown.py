"""
error_analysis_breakdown.py
===========================
Break down diarization errors by:
  - segment length bin
  - number of speakers
  - audio condition (optional metadata)

This script reuses the per-file metrics already saved in outputs/eval/full_comparison.json
and joins them with test-manifest metadata. If an audio-condition CSV is provided,
it also aggregates by that label. Otherwise, audio_condition is set to "unknown".

Example:
  python error_analysis_breakdown.py ^
    --comparison_json outputs/eval/full_comparison.json ^
    --manifest data/manifests/msdd_test_manifest.json ^
    --output_dir outputs/eval/error_analysis

Optional:
  python error_analysis_breakdown.py ^
    --comparison_json outputs/eval/full_comparison.json ^
    --manifest data/manifests/msdd_test_manifest.json ^
    --conditions_csv data/manifests/test_audio_conditions.csv ^
    --output_dir outputs/eval/error_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_LENGTH_BINS = [0.0, 90.0, 110.0, 125.0, float("inf")]
DEFAULT_LENGTH_LABELS = ["short", "medium", "long", "very_long"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = Path(item["audio_filepath"])
            rows.append(
                {
                    "recording_id": audio_path.stem,
                    "duration_sec": item.get("duration"),
                    "num_speakers": item.get("num_speakers"),
                }
            )
    return pd.DataFrame(rows)


def load_conditions_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["recording_id", "audio_condition"])

    df = pd.read_csv(path)
    required = {"recording_id", "audio_condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )

    return df[["recording_id", "audio_condition"]].copy()


def comparison_json_to_rows(results: list[dict]) -> pd.DataFrame:
    rows = []
    for system_result in results:
        system = system_result["system"]
        metrics = system_result.get("metrics", {})
        for collar_key, metric_block in metrics.items():
            collar = metric_block.get("collar")
            per_file = metric_block.get("per_file", {})
            for recording_id, values in per_file.items():
                rows.append(
                    {
                        "system": system,
                        "collar": collar,
                        "recording_id": recording_id,
                        "der": values.get("der"),
                        "miss": values.get("miss"),
                        "fa": values.get("fa"),
                        "conf": values.get("conf"),
                    }
                )
    return pd.DataFrame(rows)


def add_length_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["segment_length_bin"] = pd.cut(
        df["duration_sec"],
        bins=DEFAULT_LENGTH_BINS,
        labels=DEFAULT_LENGTH_LABELS,
        right=False,
    )
    df["segment_length_bin"] = df["segment_length_bin"].astype("object").fillna("unknown")
    return df


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(["system", "collar", group_col], dropna=False)
        .agg(
            files=("recording_id", "count"),
            avg_der=("der", "mean"),
            avg_miss=("miss", "mean"),
            avg_fa=("fa", "mean"),
            avg_conf=("conf", "mean"),
        )
        .reset_index()
    )

    for col in ["avg_der", "avg_miss", "avg_fa", "avg_conf"]:
        summary[col] = summary[col].round(3)

    return summary.sort_values(["collar", "system", group_col]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Break down diarization errors by length, speakers, and conditions"
    )
    parser.add_argument(
        "--comparison_json",
        default="outputs/eval/full_comparison.json",
        help="Path to full_comparison.json produced by evaluate_der.py",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/msdd_test_manifest.json",
        help="Test manifest with duration and num_speakers",
    )
    parser.add_argument(
        "--conditions_csv",
        default=None,
        help="Optional CSV with columns: recording_id,audio_condition",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/eval/error_analysis",
        help="Directory to save joined tables and grouped summaries",
    )
    args = parser.parse_args()

    comparison_json = Path(args.comparison_json)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_json(comparison_json)
    metrics_df = comparison_json_to_rows(results)
    manifest_df = load_manifest(manifest_path)
    conditions_df = load_conditions_csv(Path(args.conditions_csv)) if args.conditions_csv else load_conditions_csv(None)

    merged = metrics_df.merge(manifest_df, on="recording_id", how="left")
    merged = merged.merge(conditions_df, on="recording_id", how="left")
    merged["audio_condition"] = merged["audio_condition"].fillna("unknown")
    merged = add_length_bins(merged)

    merged.to_csv(output_dir / "per_file_error_analysis.csv", index=False, encoding="utf-8-sig")

    by_length = summarize(merged, "segment_length_bin")
    by_speakers = summarize(merged, "num_speakers")
    by_condition = summarize(merged, "audio_condition")

    by_length.to_csv(output_dir / "by_segment_length.csv", index=False, encoding="utf-8-sig")
    by_speakers.to_csv(output_dir / "by_num_speakers.csv", index=False, encoding="utf-8-sig")
    by_condition.to_csv(output_dir / "by_audio_condition.csv", index=False, encoding="utf-8-sig")

    print(f"Saved: {output_dir / 'per_file_error_analysis.csv'}")
    print(f"Saved: {output_dir / 'by_segment_length.csv'}")
    print(f"Saved: {output_dir / 'by_num_speakers.csv'}")
    print(f"Saved: {output_dir / 'by_audio_condition.csv'}")

    print("\nSegment length breakdown")
    print(by_length.to_string(index=False))
    print("\nSpeaker-count breakdown")
    print(by_speakers.to_string(index=False))
    print("\nAudio-condition breakdown")
    print(by_condition.to_string(index=False))


if __name__ == "__main__":
    main()
