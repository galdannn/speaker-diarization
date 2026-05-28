"""
umap_visualize.py
=================
Монгол хэлний яригч танилт — UMAP визуализаци
Fine-tuning өмнө болон дараах TitaNet embedding-уудыг харьцуулна.

Шаардлагатай package-ууд:
    pip install umap-learn plotly tqdm soundfile

Ашиглалт:
    python umap_visualize.py \
        --pretrained  path/to/titanet_large.nemo \
        --finetuned   path/to/titanet_mongolian_finetuned.nemo \
        --audio_dir   path/to/cv_mongolian_test_wavs \
        --manifest    path/to/test_manifest.json \
        --output      umap_comparison.html

Manifest формат (NeMo стандарт JSON lines):
    {"audio_filepath": "...", "label": "spk_001", "duration": 4.2}
    {"audio_filepath": "...", "label": "spk_002", "duration": 3.8}
    ...
    (sr/prepare_dataset.py-ийн гаралт шууд ашиглаж болно)

Хэрэв manifest байхгүй бол --audio_dir дотор хавтасны нэрийг
яригчийн label болгон ашиглана:
    audio_dir/
        spk_001/
            clip_001.wav
            clip_002.wav
        spk_002/
            clip_001.wav
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Командын мөрийн аргументууд
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="UMAP visualization: pretrained vs fine-tuned TitaNet"
    )
    parser.add_argument(
        "--pretrained", required=True,
        help="Урьдчилан сургасан TitaNet .nemo файл"
    )
    parser.add_argument(
        "--finetuned", required=True,
        help="Fine-tuned TitaNet .nemo файл"
    )
    parser.add_argument(
        "--audio_dir", default=None,
        help="Аудио файлуудын үндсэн хавтас (manifest байхгүй тохиолдолд)"
    )
    parser.add_argument(
        "--manifest", default=None,
        help="NeMo JSON lines manifest файл"
    )
    parser.add_argument(
        "--output", default="umap_comparison.html",
        help="Гаралтын HTML файлын нэр"
    )
    parser.add_argument(
        "--max_per_speaker", type=int, default=30,
        help="Яригч тус бүрээс авах хамгийн их клип (default: 30)"
    )
    parser.add_argument(
        "--min_speakers", type=int, default=5,
        help="Харуулах хамгийн бага яригчийн тоо (default: 5)"
    )
    parser.add_argument(
        "--umap_neighbors", type=int, default=15,
        help="UMAP n_neighbors (default: 15)"
    )
    parser.add_argument(
        "--umap_min_dist", type=float, default=0.1,
        help="UMAP min_dist (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Аудио файлуудын жагсаалт үүсгэх
# ──────────────────────────────────────────────────────────────────────────────

def build_file_list(audio_dir=None, manifest=None, max_per_speaker=30):
    """
    Returns:
        files  : list of audio file paths (str)
        labels : list of speaker labels (str)
    """
    files, labels = [], []

    if manifest and os.path.isfile(manifest):
        print(f"[INFO] Manifest уншиж байна: {manifest}")
        from collections import defaultdict
        per_spk = defaultdict(list)
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                spk = entry.get("label", entry.get("speaker_id", "unknown"))
                per_spk[spk].append(entry["audio_filepath"])
        for spk, clips in per_spk.items():
            chosen = clips[:max_per_speaker]
            files.extend(chosen)
            labels.extend([spk] * len(chosen))

    elif audio_dir and os.path.isdir(audio_dir):
        print(f"[INFO] audio_dir-с файлуудыг уншиж байна: {audio_dir}")
        from collections import defaultdict
        per_spk = defaultdict(list)
        for root, dirs, fnames in os.walk(audio_dir):
            spk = os.path.basename(root)
            for fn in fnames:
                if fn.lower().endswith((".wav", ".mp3", ".flac")):
                    per_spk[spk].append(os.path.join(root, fn))
        if not per_spk:
            raise ValueError(
                "audio_dir дотор WAV/MP3/FLAC файл олдсонгүй. "
                "Хавтасны бүтэц: audio_dir/speaker_id/clip.wav"
            )
        for spk, clips in per_spk.items():
            chosen = clips[:max_per_speaker]
            files.extend(chosen)
            labels.extend([spk] * len(chosen))
    else:
        raise ValueError("--manifest эсвэл --audio_dir аргументуудын аль нэгийг өгнө үү.")

    print(f"[INFO] Нийт {len(files)} клип, {len(set(labels))} яригч")
    return files, labels


# ──────────────────────────────────────────────────────────────────────────────
# 3. TitaNet embedding гаргалт
# ──────────────────────────────────────────────────────────────────────────────

def load_titanet(nemo_path):
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("NeMo олдсонгүй. `pip install nemo_toolkit[asr]`")

    import tarfile, tempfile, yaml

    print(f"[INFO] Загвар ачаалж байна: {nemo_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path, "r") as tar:
            tar.extractall(tmpdir)

        # 1️⃣ Weights уншина
        weights_file = None
        for fname in os.listdir(tmpdir):
            if fname.endswith(".ckpt") or fname.endswith(".pt"):
                weights_file = os.path.join(tmpdir, fname)
                break
        if weights_file is None:
            raise FileNotFoundError(f"{nemo_path} дотор .ckpt/.pt файл олдсонгүй.")

        state_dict = torch.load(weights_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # 2️⃣ decoder.final.weight-ийг шууд хайна (classification head)
        DECODER_KEY = "decoder.final.weight"
        if DECODER_KEY in state_dict:
            actual_num_classes = state_dict[DECODER_KEY].shape[0]
            print(f"[INFO] Checkpoint num_classes: {actual_num_classes}")
        else:
            # fallback: decoder-ийн хамгийн том weight-ийг хайна
            decoder_keys = [k for k in state_dict if k.startswith("decoder") and "weight" in k]
            print(f"[DEBUG] Decoder keys олдсон: {decoder_keys}")
            sizes = {k: state_dict[k].shape[0] for k in decoder_keys}
            DECODER_KEY = max(sizes, key=sizes.get)
            actual_num_classes = sizes[DECODER_KEY]
            print(f"[INFO] Fallback decoder key: {DECODER_KEY} → {actual_num_classes} класс")

        # 3️⃣ cfg уншиж засварлана
        config_path = os.path.join(tmpdir, "model_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        cfg = OmegaConf.create(cfg_dict)
        OmegaConf.set_struct(cfg, False)
        cfg.pop("train_ds", None)
        cfg.pop("validation_ds", None)
        cfg.pop("test_ds", None)

        # decoder.num_classes-ийг checkpoint-тай яг тааруулна
        if OmegaConf.select(cfg, "decoder.num_classes") is not None:
            cfg.decoder.num_classes = actual_num_classes
            print(f"[INFO] cfg.decoder.num_classes → {actual_num_classes}")

        # 4️⃣ Загвар үүсгэнэ
        model = EncDecSpeakerLabelModel(cfg=cfg)

        # 5️⃣ Weights ачаална
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys  ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

    model.eval()
    model = model.to(device)
    print(f"[INFO]  → {device} дээр ажиллана")
    return model
def extract_embeddings(model, file_list, batch_size=16):
    """
    NeMo EncDecSpeakerLabelModel-оор embedding гаргана.
    Returns numpy array (N, D).
    """
    import soundfile as sf
    import torch.nn.functional as F

    device = next(model.parameters()).device
    all_embs = []

    for i in tqdm.tqdm(range(0, len(file_list), batch_size),
                       desc="  Embedding", unit="batch"):
        batch_files = file_list[i : i + batch_size]

        # Аудио уншиж, padding хийнэ
        waveforms = []
        lengths = []
        for fp in batch_files:
            try:
                wav, sr = sf.read(fp, dtype="float32", always_2d=False)
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                # 16kHz-д resample шаардлагатай бол
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                waveforms.append(torch.tensor(wav))
                lengths.append(len(wav))
            except Exception as e:
                print(f"[WARN] Файл алдаа ({fp}): {e}")
                waveforms.append(torch.zeros(16000))
                lengths.append(16000)

        # Pad to max length
        max_len = max(lengths)
        padded = torch.zeros(len(waveforms), max_len)
        for j, (wav, l) in enumerate(zip(waveforms, lengths)):
            padded[j, :l] = wav
        lens_tensor = torch.tensor(lengths, dtype=torch.long).to(device)
        padded = padded.to(device)

        with torch.no_grad():
            # NeMo forward: logits, embs
            _, embs = model.forward(
                input_signal=padded,
                input_signal_length=lens_tensor
            )
            # L2 normalize
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())

    return np.vstack(all_embs)


# ──────────────────────────────────────────────────────────────────────────────
# 4. UMAP тооцоолол
# ──────────────────────────────────────────────────────────────────────────────

def run_umap(embeddings, n_neighbors=15, min_dist=0.1, seed=42):
    try:
        import umap
    except ImportError:
        raise ImportError("`pip install umap-learn` ажиллуулна уу.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False
    )
    print("  [INFO] UMAP тооцоолж байна...")
    return reducer.fit_transform(embeddings)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Plotly интерактив HTML үүсгэх
# ──────────────────────────────────────────────────────────────────────────────

# 39 яригчийн хувьд хангалттай тод, ялгаатай өнгний палитр
PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4",
    "#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9a6324","#fffac8",
    "#800000","#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ffffff",
    "#e6beff","#0075dc","#993f00","#4c005c","#191919","#005c31","#2bce48",
    "#ffcc99","#808080","#00998f","#740aff","#ff5005","#ce3175","#94ffb5",
    "#8f7c00","#9dcc00","#c20088","#003380",
]

def build_html(coords_pre, coords_ft, labels, output_path,
               umap_neighbors, umap_min_dist):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("`pip install plotly` ажиллуулна уу.")

    unique_speakers = sorted(set(labels))
    color_map = {spk: PALETTE[i % len(PALETTE)]
                 for i, spk in enumerate(unique_speakers)}
    colors = [color_map[l] for l in labels]

    # Hover текст: яригч + файлын дугаар
    spk_counts = {}
    hover_texts = []
    for l in labels:
        spk_counts[l] = spk_counts.get(l, 0) + 1
        hover_texts.append(f"Яригч: {l}<br>Клип #{spk_counts[l]}")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Урьдчилан сургасан TitaNet (pretrained)",
            "Fine-tuned TitaNet (монгол хэлэнд)"
        ],
        horizontal_spacing=0.08
    )

    # Яригч бүрт нэг trace — legend-д нэг удаа гарах
    for i, spk in enumerate(unique_speakers):
        mask = [j for j, l in enumerate(labels) if l == spk]
        show_legend = True

        for col, coords in [(1, coords_pre), (2, coords_ft)]:
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=spk,
                    legendgroup=spk,
                    showlegend=show_legend,
                    marker=dict(
                        color=color_map[spk],
                        size=7,
                        opacity=0.85,
                        line=dict(width=0.5, color="white")
                    ),
                    text=[hover_texts[j] for j in mask],
                    hovertemplate="%{text}<extra></extra>"
                ),
                row=1, col=col
            )
            show_legend = False  # зөвхөн эхний subplot-д legend харуулна

    fig.update_layout(
        title=dict(
            text=(
                "UMAP: Яригчийн embedding — fine-tuning өмнө болон дараа<br>"
                f"<sup>{len(unique_speakers)} яригч · {len(labels)} клип · "
                f"UMAP n_neighbors={umap_neighbors}, min_dist={umap_min_dist} · "
                f"metric=cosine</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Arial")
        ),
        legend=dict(
            title=dict(text="Яригч", font=dict(size=12)),
            itemsizing="constant",
            font=dict(size=11),
            bordercolor="#cccccc",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font=dict(color="#e0e0e0", family="Arial"),
        height=650,
        margin=dict(l=40, r=20, t=100, b=40),
    )

    # Axis шинж
    axis_style = dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        showticklabels=False,
        title=""
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # Subplot цонхны гарчгийн өнгө
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.color = "#c0c0e0"

    # Clustered-байдлын тайлбар annotation
    fig.add_annotation(
        text=(
            "💡 Яригч бүрийн цэгүүд ойр бөөгнөрвөл fine-tuning амжилттай. "
            "Өнгөт бүлгүүд тодорхой тусгаарлагдах нь DER бууралттай холбоотой."
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.04,
        showarrow=False,
        font=dict(size=11, color="#a0a0c0"),
        xanchor="center"
    )

    html_content = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "umap_comparison",
                "scale": 2
            }
        }
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ HTML хадгалагдлаа: {output_path}")
    print(f"   Хөтөч дээр нээхийн тулд: file:///{os.path.abspath(output_path)}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Үндсэн функц
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Файлуудын жагсаалт ──
    files, labels = build_file_list(
        audio_dir=args.audio_dir,
        manifest=args.manifest,
        max_per_speaker=args.max_per_speaker
    )

    # Хэт цөөн яригчтай тохиолдолд анхааруулга
    unique = set(labels)
    if len(unique) < args.min_speakers:
        print(f"[WARN] Зөвхөн {len(unique)} яригч олдлоо "
              f"(шаардлага: {args.min_speakers}+). "
              "UMAP-ийн дүрслэл хангалттай биш байж болно.")

    # ── Pretrained загвар ──
    print("\n── Урьдчилан сургасан загварын embedding ──")
    model_pre = load_titanet(args.pretrained)
    embs_pre = extract_embeddings(model_pre, files)
    del model_pre
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Fine-tuned загвар ──
    print("\n── Fine-tuned загварын embedding ──")
    model_ft = load_titanet(args.finetuned)
    embs_ft = extract_embeddings(model_ft, files)
    del model_ft
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[INFO] Embedding хэмжүүр: {embs_pre.shape[1]}-dim "
          f"({embs_pre.shape[0]} вектор)")

    # ── UMAP ──
    print("\n── UMAP тооцоолол ──")
    print("  [1/2] Pretrained...")
    coords_pre = run_umap(embs_pre, args.umap_neighbors,
                          args.umap_min_dist, args.seed)
    print("  [2/2] Fine-tuned...")
    coords_ft  = run_umap(embs_ft,  args.umap_neighbors,
                          args.umap_min_dist, args.seed)

    # ── HTML үүсгэх ──
    print("\n── HTML үүсгэж байна ──")
    build_html(
        coords_pre, coords_ft, labels,
        output_path=args.output,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist
    )


if __name__ == "__main__":
    main()