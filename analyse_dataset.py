"""
analyse_dataset.py
==================
Comprehensive dataset analysis for:
  1. Mozilla Common Voice Mongolian (raw + filtered)
  2. TitaNet training manifests (sr/data/manifests/)
  3. Synthesized diarization dataset (sd/data/diarization_dataset/)
  4. MSDD training manifests (sd/data/manifests/)

Outputs:
  - Console tables with statistics
  - outputs/analysis/ folder with CSV exports
  - outputs/analysis/figures/ folder with plots

Run from anywhere:
    python analyse_dataset.py

Or with custom paths:
    python analyse_dataset.py --cv_root "C:/path/to/cv" --sd_root "C:/path/to/sd"
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import defaultdict

# ── Install check ──────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")   # headless — saves to file
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import soundfile as sf
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install numpy pandas matplotlib soundfile tqdm")
    exit(1)

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_CV_ROOT  = r"C:\Users\galda\Capstone\cv-corpus-24.0-2025-12-05\mn"
DEFAULT_SR_ROOT  = r"C:\Users\galda\titanet\sr"
DEFAULT_SD_ROOT  = r"C:\Users\galda\titanet\sd"
OUTPUT_DIR       = Path("outputs/analysis")
FIGURES_DIR      = OUTPUT_DIR / "figures"

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#eee",
    "grid.color":       "#2a2d37",
    "grid.alpha":       0.5,
    "axes.titlecolor":  "#4fffb0",
    "axes.titlesize":   13,
    "axes.labelsize":   10,
    "font.family":      "monospace",
})

ACCENT   = "#4fffb0"
ACCENT2  = "#00c8ff"
WARN     = "#ff8c42"
DANGER   = "#ff5f5f"
COLORS   = [ACCENT, ACCENT2, WARN, DANGER, "#c084fc", "#fbbf24"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(title):
    w = 70
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def sep():
    print(f"{'─'*70}")


def fmt_hours(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def get_audio_duration(path):
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return None


def parse_rttm(path):
    segments = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == "SPEAKER":
                    segments.append({
                        "speaker":  parts[7],
                        "start":    float(parts[3]),
                        "duration": float(parts[4]),
                        "end":      float(parts[3]) + float(parts[4]),
                    })
    except Exception:
        pass
    return segments


def save_csv(df, name):
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {path}")


def save_fig(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMMON VOICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_common_voice(cv_root):
    banner("1. MOZILLA COMMON VOICE MONGOLIAN v24.0")
    cv_root = Path(cv_root)
    tsv_path = cv_root / "train.tsv"
    clips_dir = cv_root / "clips"

    if not tsv_path.exists():
        print(f"  [SKIP] train.tsv not found at {tsv_path}")
        return None, None

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"\n  Columns: {list(df.columns)}")
    print(f"  Total rows in train.tsv: {len(df):,}")

    # Check which files exist on disk
    print("\n  Checking files on disk...")
    df["full_path"] = df["path"].apply(lambda p: str(clips_dir / p))
    df["exists"]    = df["full_path"].apply(os.path.isfile)
    df_exists       = df[df["exists"]].copy()
    print(f"  Files found on disk: {len(df_exists):,} / {len(df):,}")

    # Speaker counts
    spk_counts = df_exists["client_id"].value_counts()
    print(f"\n  Total unique speakers: {len(spk_counts):,}")
    print(f"  Avg clips/speaker:     {spk_counts.mean():.1f}")
    print(f"  Median clips/speaker:  {spk_counts.median():.0f}")
    print(f"  Min clips/speaker:     {spk_counts.min()}")
    print(f"  Max clips/speaker:     {spk_counts.max()}")

    # Filter tiers
    for threshold in [10, 20, 50, 100]:
        eligible = spk_counts[spk_counts >= threshold]
        total_clips = spk_counts[spk_counts >= threshold].sum()
        print(f"  >= {threshold:3d} clips: {len(eligible):3d} speakers, {total_clips:,} clips")

    sep()
    # Get durations for eligible speakers (>=20 clips)
    eligible_ids   = spk_counts[spk_counts >= 20].index
    df_eligible    = df_exists[df_exists["client_id"].isin(eligible_ids)].copy()
    print(f"\n  After >=20 clip filter:")
    print(f"  Speakers: {df_eligible['client_id'].nunique()}")
    print(f"  Clips:    {len(df_eligible):,}")

    print("\n  Sampling audio durations (first 200 clips)...")
    sample = df_eligible.head(200)
    durations = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), leave=False):
        d = get_audio_duration(row["full_path"])
        if d: durations.append(d)

    if durations:
        d = np.array(durations)
        print(f"  Sample mean duration:  {d.mean():.2f}s")
        print(f"  Sample median:         {np.median(d):.2f}s")
        print(f"  Sample min/max:        {d.min():.2f}s / {d.max():.2f}s")
        # Extrapolate to full dataset
        total_est = d.mean() * len(df_eligible) / 3600
        print(f"  Estimated total hours (eligible): {total_est:.2f}h")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Common Voice Mongolian — Яригчийн дистрибуц", fontsize=14, color=ACCENT)

    # 1. Clips per speaker histogram
    ax = axes[0]
    ax.hist(spk_counts.values, bins=40, color=ACCENT, alpha=0.8, edgecolor="#000")
    ax.axvline(20, color=DANGER, linestyle="--", linewidth=1.5, label="20-clip threshold")
    ax.set_title("Яригч бүрийн клипийн тоо")
    ax.set_xlabel("Клипийн тоо")
    ax.set_ylabel("Яригчийн тоо")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 2. Eligible vs total speakers
    ax = axes[1]
    thresholds = [5, 10, 20, 30, 50, 100]
    counts = [len(spk_counts[spk_counts >= t]) for t in thresholds]
    bars = ax.bar(range(len(thresholds)), counts, color=ACCENT2, alpha=0.8)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f">={t}" for t in thresholds])
    ax.set_title("Шүүлтүүрийн нөлөөлөл")
    ax.set_xlabel("Хамгийн бага клипийн тоо")
    ax.set_ylabel("Тохирох яригчийн тоо")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=9, color="#eee")
    ax.grid(True, axis="y")

    # 3. Duration distribution (sample)
    ax = axes[2]
    if durations:
        ax.hist(durations, bins=30, color=WARN, alpha=0.8, edgecolor="#000")
        ax.axvline(np.mean(durations), color=ACCENT, linestyle="--",
                   linewidth=1.5, label=f"Mean {np.mean(durations):.1f}s")
        ax.set_title("Аудио клипийн хугацааны дистрибуц")
        ax.set_xlabel("Хугацаа (секунд)")
        ax.set_ylabel("Клипийн тоо")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    save_fig(fig, "01_common_voice_distribution")

    # CSV export
    summary = pd.DataFrame({
        "threshold": thresholds,
        "eligible_speakers": counts,
        "pct_of_total": [c/len(spk_counts)*100 for c in counts]
    })
    save_csv(summary, "cv_speaker_filter_summary")

    return df_eligible, spk_counts


# ══════════════════════════════════════════════════════════════════════════════
# 2. TITANET MANIFEST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_titanet_manifests(sr_root):
    banner("2. TITANET SУРГАЛТЫН MANIFEST ШИНЖИЛГЭЭ")
    sr_root = Path(sr_root)
    manifest_dir = sr_root / "data" / "manifests"

    splits = {
        "train": "train_manifest.json",
        "val":   "val_manifest.json",
        "test":  "test_manifest.json",
    }

    all_entries = []
    split_stats = []

    for split_name, fname in splits.items():
        fpath = manifest_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fpath} not found")
            continue

        entries = [json.loads(l) for l in open(fpath, encoding="utf-8")]
        for e in entries:
            e["split"] = split_name
        all_entries.extend(entries)

        durations  = [e["duration"] for e in entries]
        speakers   = set(e["label"] for e in entries)
        total_h    = sum(durations) / 3600

        print(f"\n  [{split_name.upper()}]")
        print(f"    Entries:    {len(entries):,}")
        print(f"    Speakers:   {len(speakers)}")
        print(f"    Total:      {fmt_hours(sum(durations))}")
        print(f"    Dur mean:   {np.mean(durations):.2f}s")
        print(f"    Dur min/max:{min(durations):.2f}s / {max(durations):.2f}s")

        split_stats.append({
            "split": split_name, "entries": len(entries),
            "speakers": len(speakers), "total_hours": total_h,
            "mean_dur": np.mean(durations), "min_dur": min(durations),
            "max_dur": max(durations)
        })

    if not all_entries:
        return

    df = pd.DataFrame(all_entries)
    save_csv(pd.DataFrame(split_stats), "titanet_split_stats")

    # Per-speaker clip count
    spk_clips = df.groupby("label").size().sort_values(ascending=False)
    spk_dur   = df.groupby("label")["duration"].sum().sort_values(ascending=False)

    print(f"\n  ЯРИГЧ ДАХЬ НИЙТ ({df['label'].nunique()} яригч):")
    print(f"    Avg clips/speaker:  {spk_clips.mean():.1f}")
    print(f"    Avg minutes/speaker:{spk_dur.mean()/60:.1f} мин")
    print(f"    Min clips/speaker:  {spk_clips.min()}")
    print(f"    Max clips/speaker:  {spk_clips.max()}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TitaNet Сургалтын Манифест — Шинжилгээ", fontsize=14, color=ACCENT)

    # 1. Clips per speaker (all splits)
    ax = axes[0]
    ax.bar(range(len(spk_clips)), spk_clips.values, color=ACCENT, alpha=0.8)
    ax.set_title("Яригч бүрийн клипийн тоо (нийт хуваарилалт)")
    ax.set_xlabel("Яригчийн индекс")
    ax.set_ylabel("Клипийн тоо")
    ax.grid(True, axis="y")

    # 2. Duration per split
    ax = axes[1]
    s_df = pd.DataFrame(split_stats)
    if len(s_df):
        bars = ax.bar(s_df["split"], s_df["total_hours"],
                      color=[ACCENT, ACCENT2, WARN][:len(s_df)], alpha=0.85)
        for bar, val in zip(bars, s_df["total_hours"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}h", ha="center", va="bottom", fontsize=10, color="#eee")
        ax.set_title("Хуваарилалт тус бүрийн нийт цаг")
        ax.set_xlabel("Хуваарилалт")
        ax.set_ylabel("Цаг")
        ax.grid(True, axis="y")

    # 3. Duration histogram
    ax = axes[2]
    durations = df["duration"].values
    ax.hist(durations, bins=40, color=ACCENT2, alpha=0.8, edgecolor="#000")
    ax.axvline(np.mean(durations), color=ACCENT, linestyle="--",
               linewidth=1.5, label=f"Mean {np.mean(durations):.1f}s")
    ax.set_title("Клипийн хугацааны дистрибуц")
    ax.set_xlabel("Хугацаа (секунд)")
    ax.set_ylabel("Клипийн тоо")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    save_fig(fig, "02_titanet_manifest_analysis")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SYNTHESIZED DIARIZATION DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_synthesis_dataset(sd_root):
    banner("3. НИЙЛЭГЖҮҮЛСЭН ЯРИА ЯЛГАЛТЫН ӨГӨГДЛИЙН САНГИЙН ШИНЖИЛГЭЭ")
    sd_root  = Path(sd_root)
    diar_dir = sd_root / "data" / "diarization_dataset"

    if not diar_dir.exists():
        print(f"  [SKIP] {diar_dir} not found")
        return

    splits = ["train", "dev", "test"]
    all_conv_stats = []

    for split in splits:
        audio_dir = diar_dir / split / "audio"
        rttm_dir  = diar_dir / split / "rttm"

        if not audio_dir.exists():
            continue

        audio_files = sorted(audio_dir.glob("*.wav"))
        print(f"\n  [{split.upper()}]  {len(audio_files)} яриа")

        conv_stats = []
        for af in tqdm(audio_files, desc=f"  {split}", leave=False):
            rttm_path = rttm_dir / (af.stem + ".rttm")
            duration  = get_audio_duration(af)
            segments  = parse_rttm(rttm_path) if rttm_path.exists() else []

            if not segments or not duration:
                continue

            speakers        = set(s["speaker"] for s in segments)
            n_speakers      = len(speakers)
            n_turns         = len(segments)
            total_speech    = sum(s["duration"] for s in segments)
            speech_ratio    = total_speech / duration if duration else 0

            # Detect overlaps
            segs_sorted = sorted(segments, key=lambda x: x["start"])
            overlaps = 0
            overlap_dur = 0.0
            for i in range(1, len(segs_sorted)):
                prev_end  = segs_sorted[i-1]["end"]
                curr_start = segs_sorted[i]["start"]
                if curr_start < prev_end:
                    overlaps += 1
                    overlap_dur += prev_end - curr_start

            # Silence gaps
            gaps = []
            for i in range(1, len(segs_sorted)):
                prev_end   = segs_sorted[i-1]["end"]
                curr_start = segs_sorted[i]["start"]
                if curr_start > prev_end:
                    gaps.append(curr_start - prev_end)

            conv_stats.append({
                "file":            af.stem,
                "split":           split,
                "duration":        round(duration, 3),
                "n_speakers":      n_speakers,
                "n_turns":         n_turns,
                "total_speech":    round(total_speech, 3),
                "speech_ratio":    round(speech_ratio, 4),
                "n_overlaps":      overlaps,
                "overlap_ratio":   round(overlaps / n_turns if n_turns else 0, 4),
                "overlap_dur":     round(overlap_dur, 3),
                "mean_gap":        round(np.mean(gaps) if gaps else 0, 3),
                "mean_turn_dur":   round(np.mean([s["duration"] for s in segments]), 3),
            })

        all_conv_stats.extend(conv_stats)

        df = pd.DataFrame(conv_stats)
        if len(df) == 0:
            continue

        print(f"    Нийт хугацаа:        {fmt_hours(df['duration'].sum())}")
        print(f"    Дундаж яриа хугацаа: {df['duration'].mean():.1f}с")
        print(f"    Яригчийн тоо:        {df['n_speakers'].value_counts().to_dict()}")
        print(f"    Дундаж эргэлтийн тоо:{df['n_turns'].mean():.1f}")
        print(f"    Давхцлын дундаж хувь:{df['overlap_ratio'].mean()*100:.1f}%")
        print(f"    Яриаг эзлэх хувь:    {df['speech_ratio'].mean()*100:.1f}%")
        print(f"    Дундаж завсар:        {df['mean_gap'].mean():.3f}с")

    if not all_conv_stats:
        return

    df_all = pd.DataFrame(all_conv_stats)
    save_csv(df_all, "synthesis_conversation_stats")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Нийлэгжүүлсэн Яриа Ялгалтын Өгөгдлийн Санн Шинжилгээ",
                 fontsize=14, color=ACCENT)

    # 1. Conversation duration distribution
    ax = axes[0, 0]
    for split, color in zip(splits, [ACCENT, ACCENT2, WARN]):
        d = df_all[df_all["split"] == split]["duration"]
        if len(d):
            ax.hist(d, bins=20, alpha=0.7, color=color, label=split, edgecolor="#000")
    ax.set_title("Ярианы хугацааны дистрибуц")
    ax.set_xlabel("Хугацаа (секунд)")
    ax.set_ylabel("Ярианы тоо")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 2. Speaker count distribution
    ax = axes[0, 1]
    spk_counts = df_all["n_speakers"].value_counts().sort_index()
    bars = ax.bar(spk_counts.index.astype(str), spk_counts.values,
                  color=[ACCENT, ACCENT2, WARN, DANGER][:len(spk_counts)], alpha=0.85)
    for bar, val in zip(bars, spk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=11, color="#eee")
    ax.set_title("Яриа дахь яригчийн тооны дистрибуц")
    ax.set_xlabel("Яригчийн тоо")
    ax.set_ylabel("Ярианы тоо")
    ax.grid(True, axis="y")

    # 3. Overlap ratio distribution
    ax = axes[0, 2]
    ax.hist(df_all["overlap_ratio"] * 100, bins=25, color=WARN, alpha=0.85, edgecolor="#000")
    ax.axvline(12, color=DANGER, linestyle="--", linewidth=2, label="Зорилтот 12%")
    ax.axvline(df_all["overlap_ratio"].mean() * 100, color=ACCENT,
               linestyle="--", linewidth=1.5,
               label=f"Дундаж {df_all['overlap_ratio'].mean()*100:.1f}%")
    ax.set_title("Давхцлын хувийн дистрибуц")
    ax.set_xlabel("Давхцлын хувь (%)")
    ax.set_ylabel("Ярианы тоо")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 4. Silence gap distribution
    ax = axes[1, 0]
    gaps = df_all["mean_gap"]
    ax.hist(gaps, bins=25, color=ACCENT2, alpha=0.85, edgecolor="#000")
    ax.axvline(0.3, color=WARN, linestyle="--", linewidth=1.5, label="Min 0.3с")
    ax.axvline(1.5, color=WARN, linestyle="--", linewidth=1.5, label="Max 1.5с")
    ax.axvline(gaps.mean(), color=ACCENT, linestyle="--",
               linewidth=1.5, label=f"Дундаж {gaps.mean():.2f}с")
    ax.set_title("Дундаж чимээгүй завсарын дистрибуц")
    ax.set_xlabel("Завсар (секунд)")
    ax.set_ylabel("Ярианы тоо")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 5. Speech ratio distribution
    ax = axes[1, 1]
    ax.hist(df_all["speech_ratio"] * 100, bins=25, color="#c084fc", alpha=0.85, edgecolor="#000")
    ax.axvline(df_all["speech_ratio"].mean() * 100, color=ACCENT,
               linestyle="--", linewidth=1.5,
               label=f"Дундаж {df_all['speech_ratio'].mean()*100:.1f}%")
    ax.set_title("Яриаг эзлэх хувийн дистрибуц")
    ax.set_xlabel("Яриаг эзлэх хувь (%)")
    ax.set_ylabel("Ярианы тоо")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 6. Split comparison bar chart
    ax = axes[1, 2]
    split_dur = df_all.groupby("split")["duration"].sum() / 3600
    split_dur = split_dur.reindex(["train", "dev", "test"])
    bars = ax.bar(split_dur.index, split_dur.values,
                  color=[ACCENT, ACCENT2, WARN], alpha=0.85)
    for bar, val in zip(bars, split_dur.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}h", ha="center", va="bottom", fontsize=10, color="#eee")
    ax.set_title("Хуваарилалт тус бүрийн нийт цаг")
    ax.set_xlabel("Хуваарилалт")
    ax.set_ylabel("Цаг")
    ax.grid(True, axis="y")

    plt.tight_layout()
    save_fig(fig, "03_synthesis_dataset_analysis")

    # ── Per-conversation timeline plot (first 5 from test) ─────────────────────
    test_convs = df_all[df_all["split"] == "test"].head(5)
    if len(test_convs):
        _plot_conversation_timelines(
            diar_dir / "test" / "rttm",
            list(test_convs["file"])
        )


def _plot_conversation_timelines(rttm_dir, files):
    """Plot speaker timelines for a few example conversations."""
    from matplotlib.patches import FancyBboxPatch

    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(16, n * 2 + 1))
    if n == 1:
        axes = [axes]
    fig.suptitle("Ярианы тайман дараалал (туршилт хуваарилалтын жишээ)",
                 fontsize=13, color=ACCENT)

    spk_palette = [ACCENT, ACCENT2, WARN, DANGER, "#c084fc", "#fbbf24"]

    for ax, fname in zip(axes, files):
        rttm_path = Path(rttm_dir) / (fname + ".rttm")
        segs = parse_rttm(rttm_path)
        if not segs:
            continue

        speakers  = sorted(set(s["speaker"] for s in segs))
        spk_color = {spk: spk_palette[i % len(spk_palette)]
                     for i, spk in enumerate(speakers)}
        max_t = max(s["end"] for s in segs)

        for seg in segs:
            color = spk_color[seg["speaker"]]
            rect = FancyBboxPatch(
                (seg["start"], 0.1), seg["duration"], 0.8,
                boxstyle="round,pad=0.01",
                linewidth=0.5, edgecolor="#000",
                facecolor=color, alpha=0.85
            )
            ax.add_patch(rect)
            # Label if segment is wide enough
            if seg["duration"] > max_t * 0.04:
                ax.text(seg["start"] + seg["duration"]/2, 0.5,
                        seg["speaker"][-1],  # last char of speaker ID
                        ha="center", va="center", fontsize=7,
                        color="#000", fontweight="bold")

        ax.set_xlim(0, max_t)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Цаг (секунд)", fontsize=8)
        ax.set_title(fname, fontsize=9, color="#aaa", loc="left")

        # Legend
        patches = [mpatches.Patch(color=spk_color[s], label=s) for s in speakers]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  framealpha=0.3, ncol=len(speakers))
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "04_conversation_timelines")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MSDD MANIFEST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_msdd_manifests(sd_root):
    banner("4. MSDD MANIFEST ШИНЖИЛГЭЭ")
    sd_root      = Path(sd_root)
    manifest_dir = sd_root / "data" / "manifests"

    splits = {
        "train": "train_manifest.json",
        "val":   "val_manifest.json",
        "test":  "test_manifest.json",
    }

    split_stats = []
    for split_name, fname in splits.items():
        fpath = manifest_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] {fpath}")
            continue

        entries = [json.loads(l) for l in open(fpath, encoding="utf-8")]
        durations  = [e["duration"] for e in entries]
        n_speakers = [e.get("num_speakers", 0) for e in entries]

        print(f"\n  [{split_name.upper()}]")
        print(f"    Бичлэгийн тоо:   {len(entries)}")
        print(f"    Нийт хугацаа:    {fmt_hours(sum(durations))}")
        print(f"    Дундаж хугацаа:  {np.mean(durations):.1f}с")
        print(f"    Яригчийн тоо дистрибуц: {pd.Series(n_speakers).value_counts().to_dict()}")

        # MSDD training sample estimate
        step_count   = 50
        shift        = 0.25  # base scale
        n_windows    = int(sum(durations) / shift)
        n_samples    = n_windows // step_count
        print(f"    MSDD дееж (step={step_count}): ~{n_samples:,}")

        split_stats.append({
            "split": split_name, "entries": len(entries),
            "total_hours": sum(durations)/3600,
            "mean_dur": np.mean(durations),
            "msdd_samples_est": n_samples,
        })

    if split_stats:
        save_csv(pd.DataFrame(split_stats), "msdd_manifest_stats")

    # MSDD training samples calculation visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("MSDD Сургалтын Дееж Тооцоо", fontsize=13, color=ACCENT)

    n_convs_range  = range(10, 201, 10)
    avg_dur        = 120  # 2 min avg
    samples_range  = [int(n * avg_dur / 0.25 / 50) for n in n_convs_range]

    ax.plot(list(n_convs_range), samples_range, color=ACCENT, linewidth=2, marker="o", markersize=4)
    ax.axvline(100, color=DANGER,  linestyle="--", linewidth=2, label="Манай: 100 яриа")
    ax.axhline(648, color=ACCENT2, linestyle="--", linewidth=1.5,
               label=f"Манай: ~648 дееж")
    ax.fill_between(list(n_convs_range), 0, samples_range, alpha=0.1, color=ACCENT)
    ax.set_xlabel("Нийлэгжүүлсэн ярианы тоо")
    ax.set_ylabel("MSDD сургалтын дееж (step_count=50)")
    ax.set_title("Яагаад яг 100 яриа? — MSDD сургалтын тооцоо")
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    save_fig(fig, "05_msdd_training_samples")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SYNTHESIS QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_synthesis_quality(sd_root):
    banner("5. НИЙЛЭГЖҮҮЛЭЛТИЙН ЧАНАРЫН БАТАЛГААЖУУЛАЛТ")
    sd_root  = Path(sd_root)
    diar_dir = sd_root / "data" / "diarization_dataset"

    issues = {"rttm_missing": 0, "duration_mismatch": 0, "single_speaker": 0,
              "overlap_exceeds": 0, "empty_segments": 0}
    checked = 0

    for split in ["train", "dev", "test"]:
        audio_dir = diar_dir / split / "audio"
        rttm_dir  = diar_dir / split / "rttm"
        if not audio_dir.exists():
            continue

        for af in tqdm(sorted(audio_dir.glob("*.wav")), desc=f"  {split}", leave=False):
            checked += 1
            rttm_path = rttm_dir / (af.stem + ".rttm")

            if not rttm_path.exists():
                issues["rttm_missing"] += 1
                continue

            segments = parse_rttm(rttm_path)
            if not segments:
                issues["empty_segments"] += 1
                continue

            duration = get_audio_duration(af)
            rttm_end = max(s["end"] for s in segments)

            if duration and abs(duration - rttm_end) > 0.5:
                issues["duration_mismatch"] += 1

            if len(set(s["speaker"] for s in segments)) < 2:
                issues["single_speaker"] += 1

            # Check overlap
            segs_sorted = sorted(segments, key=lambda x: x["start"])
            n_overlaps = sum(
                1 for i in range(1, len(segs_sorted))
                if segs_sorted[i]["start"] < segs_sorted[i-1]["end"]
            )
            if n_overlaps / len(segments) > 0.3:
                issues["overlap_exceeds"] += 1

    print(f"\n  Шалгасан файлуудын тоо: {checked}")
    print(f"\n  Олдсон асуудлууд:")
    all_ok = True
    for issue, count in issues.items():
        icon = "✓" if count == 0 else "⚠"
        print(f"    {icon} {issue:<30}: {count}")
        if count > 0:
            all_ok = False
    if all_ok:
        print("\n  ✓ Бүх файлууд чанарын шалгалтыг давсан!")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    banner("НИЙТ ХУРААНГУЙ — ОНОЛ ДИПЛОМЫН ХҮСНЭГТЭД ОРОХ ҮР ДҮН")

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │  COMMON VOICE МОНГОЛ v24.0                                  │
  │  Нийт бичлэг:        96,308    Яригч:    606                │
  │  Нийт цаг:           140.56h   Баталгаажсан: 49.19h (35%)  │
  │  >=20 шүүлтүүрийн дараа:  39 яригч, 2,180 бичлэг           │
  ├─────────────────────────────────────────────────────────────┤
  │  TITANET СУРГАЛТЫН ӨГӨГДӨЛ                                  │
  │  Train: 1,730 бичлэг (2.46h, 39 яригч)                     │
  │  Val:     202 бичлэг (0.29h, 39 яригч)                     │
  │  Test:    247 бичлэг (0.36h, 39 яригч)                     │
  ├─────────────────────────────────────────────────────────────┤
  │  НИЙЛЭГЖҮҮЛСЭН ЯРИА ЯЛГАЛТЫН ӨГӨГДЛИЙН САН               │
  │  Нийт: 100 яриа (3.21 цаг)  70/15/15 хуваарилалт          │
  │  Давхцал: ~12%  Завсар: 0.3-1.5с  Яригч: 2-4/яриа         │
  │  MSDD сургалтын дееж: ~648 (step_count=50)                  │
  ├─────────────────────────────────────────────────────────────┤
  │  СУУРЬ DER ҮР ДҮН (тест хуваарилалт, collar=0.25с)        │
  │  TitaNet + Бүлэглэл: DER=25.56% (Miss=20.17%, Conf=5.37%) │
  │  TitaNet + MSDD:      DER=23.97% (Miss=20.07%, Conf=3.66%) │
  └─────────────────────────────────────────────────────────────┘
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dataset analysis for Mongolian diarization thesis")
    parser.add_argument("--cv_root", default=DEFAULT_CV_ROOT,
                        help="Path to Common Voice Mongolian root folder")
    parser.add_argument("--sr_root", default=DEFAULT_SR_ROOT,
                        help="Path to titanet/sr/ folder")
    parser.add_argument("--sd_root", default=DEFAULT_SD_ROOT,
                        help="Path to titanet/sd/ folder")
    parser.add_argument("--skip_cv",  action="store_true", help="Skip Common Voice analysis")
    parser.add_argument("--skip_dur", action="store_true", help="Skip audio duration sampling (faster)")
    args = parser.parse_args()

    # Create output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Run analyses
    if not args.skip_cv:
        analyse_common_voice(args.cv_root)
    else:
        print("\n[SKIP] Common Voice analysis (--skip_cv)")

    analyse_titanet_manifests(args.sr_root)
    analyse_synthesis_dataset(args.sd_root)
    analyse_msdd_manifests(args.sd_root)
    validate_synthesis_quality(args.sd_root)
    print_summary()

    banner("ДУУСГАЛТ")
    print(f"  Зургууд:  {FIGURES_DIR.resolve()}")
    print(f"  CSV файл: {OUTPUT_DIR.resolve()}")
    print(f"\n  Зургуудыг дипломын ажлын дараах хэсгүүдэд оруулна:")
    print(f"    01_common_voice   → Бүлэг 2.1 (Common Voice шинжилгээ)")
    print(f"    02_titanet        → Бүлэг 2.1 (TitaNet өгөгдлийн хуваарилалт)")
    print(f"    03_synthesis      → Бүлэг 2.2 (Нийлэгжүүлэлтийн чанарын шинжилгээ)")
    print(f"    04_timelines      → Бүлэг 2.2 (Ярианы тайман дараалал)")
    print(f"    05_msdd_samples   → Бүлэг 2.2 (Яагаад 100 яриа)")
    print()


if __name__ == "__main__":
    main()