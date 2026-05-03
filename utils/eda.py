# utils/eda.py
# ============================================================
# Analyse Exploratoire des Données (EDA)
# Usage : python utils/eda.py
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    HAM10000_META, HAM10000_PART1, HAM10000_PART2,
    PH2_IMG_DIR, PH2_MASK_DIR,
    LABEL_NAMES, OUTPUTS_DIR
)

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _find_ham_image(image_id):
    for part in [HAM10000_PART1, HAM10000_PART2]:
        p = os.path.join(part, f"{image_id}.jpg")
        if os.path.exists(p): return p
    return None


def plot_class_distribution(metadata: pd.DataFrame):
    """4 graphiques sur le déséquilibre de classes."""
    counts      = metadata["dx"].value_counts()
    percentages = counts / counts.sum() * 100

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Analyse du Déséquilibre des Classes — HAM10000",
                 fontsize=16, fontweight="bold")

    # 1 — Barplot counts
    bars = axes[0,0].bar(counts.index, counts.values,
                         color=sns.color_palette("Set2", len(counts)))
    for bar, val in zip(bars, counts.values):
        axes[0,0].text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 20, str(val),
                       ha="center", fontsize=10, fontweight="bold")
    axes[0,0].set_title("Nombre d'images par classe")
    axes[0,0].tick_params(axis="x", rotation=30)

    # 2 — Pie chart
    axes[0,1].pie(counts.values, labels=counts.index,
                  autopct="%1.1f%%", startangle=140,
                  colors=sns.color_palette("Set2", len(counts)))
    axes[0,1].set_title("Répartition en pourcentage")

    # 3 — Barplot horizontal noms complets
    full_names = [f"{k}\n({LABEL_NAMES[k]})" for k in counts.index]
    sns.barplot(x=counts.values, y=full_names, ax=axes[1,0],
                palette="Set2", orient="h")
    axes[1,0].set_title("Distribution (noms complets)")
    for i, val in enumerate(counts.values):
        axes[1,0].text(val + 10, i,
                       f"{val} ({percentages[counts.index[i]]:.1f}%)",
                       va="center", fontsize=9)

    # 4 — Ratio déséquilibre
    ratio  = counts.max() / counts
    colors = ["red" if r > 5 else "orange" if r > 2 else "green" for r in ratio.values]
    bars2  = axes[1,1].bar(ratio.index, ratio.values, color=colors)
    axes[1,1].axhline(1, color="blue", linestyle="--", alpha=0.7, label="Équilibre parfait")
    axes[1,1].set_title("Ratio de déséquilibre vs classe majoritaire")
    for bar, val in zip(bars2, ratio.values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.1,
                       f"x{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    axes[1,1].legend()
    axes[1,1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Sauvegardé : {out}")


def plot_representative_grid(metadata: pd.DataFrame, n_samples: int = 5):
    """Grille d'images représentatives par catégorie."""
    classes = list(LABEL_NAMES.keys())
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(16, 22))
    fig.suptitle("Grille d'images représentatives par catégorie — HAM10000",
                 fontsize=15, fontweight="bold")

    for row, cls in enumerate(classes):
        samples = metadata[metadata["dx"] == cls].sample(n=n_samples, random_state=42)
        for col, (_, r) in enumerate(samples.iterrows()):
            ax = axes[row, col]
            p  = _find_ham_image(r["image_id"])
            if p:
                ax.imshow(Image.open(p))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"{cls}\n{LABEL_NAMES[cls]}", fontsize=9,
                              fontweight="bold", rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "representative_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Sauvegardé : {out}")


def analyze_image_quality_ham(metadata: pd.DataFrame):
    """Distribution dimensions + luminosité HAM10000."""
    print("⏳ Analyse qualité HAM10000...")
    widths, heights, lum_vals, corrupted = [], [], [], []
    class_lum = defaultdict(list)
    id_to_cls = dict(zip(metadata["image_id"], metadata["dx"]))

    for part in [HAM10000_PART1, HAM10000_PART2]:
        if not os.path.exists(part): continue
        for fname in os.listdir(part):
            if not fname.endswith(".jpg"): continue
            try:
                with Image.open(os.path.join(part, fname)) as img:
                    w, h = img.size
                    arr  = np.array(img.convert("L"))
                    if arr.size == 0: raise ValueError("vide")
                    lum  = arr.mean()
                    widths.append(w); heights.append(h); lum_vals.append(lum)
                    cls  = id_to_cls.get(os.path.splitext(fname)[0], "?")
                    class_lum[cls].append(lum)
            except Exception:
                corrupted.append(fname)

    print(f"  ✅ Analysées : {len(widths)}  |  ❌ Corrompues : {len(corrupted)}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Qualité des Images — HAM10000", fontsize=14, fontweight="bold")

    axes[0].scatter(widths, heights, alpha=0.3, s=5, color="steelblue")
    axes[0].set_title("Distribution des dimensions")

    axes[1].hist(lum_vals, bins=40, color="coral", edgecolor="white")
    axes[1].axvline(np.mean(lum_vals), color="black", linestyle="--",
                    label=f"Moy: {np.mean(lum_vals):.1f}")
    axes[1].set_title("Distribution de la luminosité")
    axes[1].legend()

    cls_means = {k: np.mean(v) for k, v in class_lum.items()}
    cls_stds  = {k: np.std(v)  for k, v in class_lum.items()}
    sorted_cls = sorted(cls_means, key=cls_means.get)
    axes[2].barh(sorted_cls, [cls_means[c] for c in sorted_cls],
                 xerr=[cls_stds[c] for c in sorted_cls],
                 color=sns.color_palette("Set2", len(sorted_cls)))
    axes[2].set_title("Luminosité moyenne par classe")

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "ham_image_quality.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
    print(f"💾 Sauvegardé : {out}")


def analyze_image_quality_ph2():
    """Distribution dimensions + luminosité + couverture masque PH2."""
    widths, heights, lum_vals, coverage = [], [], [], []

    for fname in os.listdir(PH2_IMG_DIR):
        if not fname.endswith(".bmp"): continue
        with Image.open(os.path.join(PH2_IMG_DIR, fname)) as img:
            w, h = img.size
            arr  = np.array(img.convert("L"))
            widths.append(w); heights.append(h); lum_vals.append(arr.mean())
        mask_p = os.path.join(PH2_MASK_DIR, fname)
        if os.path.exists(mask_p):
            mask_arr = np.array(Image.open(mask_p).convert("L"))
            coverage.append((mask_arr > 127).sum() / mask_arr.size * 100)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Qualité des Images — PH2", fontsize=14, fontweight="bold")

    axes[0].scatter(widths, heights, alpha=0.5, color="steelblue")
    axes[0].set_title("Distribution des dimensions")

    axes[1].hist(lum_vals, bins=20, color="coral", edgecolor="white")
    axes[1].axvline(np.mean(lum_vals), color="black", linestyle="--",
                    label=f"Moy: {np.mean(lum_vals):.1f}")
    axes[1].set_title("Distribution de la luminosité"); axes[1].legend()

    axes[2].hist(coverage, bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].axvline(np.mean(coverage), color="black", linestyle="--",
                    label=f"Moy: {np.mean(coverage):.1f}%")
    axes[2].set_title("Couverture de la lésion (% pixels)"); axes[2].legend()

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "ph2_image_quality.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
    print(f"💾 Sauvegardé : {out}")


if __name__ == "__main__":
    meta = pd.read_csv(HAM10000_META)
    plot_class_distribution(meta)
    plot_representative_grid(meta)
    analyze_image_quality_ham(meta)
    analyze_image_quality_ph2()
