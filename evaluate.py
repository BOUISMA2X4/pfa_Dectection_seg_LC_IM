# evaluate.py
# ============================================================
# Evaluation finale sur le test set
# Classification HAM10000 + Segmentation PH2
# Usage : python evaluate.py
# ============================================================

import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from config.config import (
    CLASSIFIER_CKPT, SEGMENTOR_CKPT,
    OUTPUTS_DIR, NUM_CLASSES_HAM, LABEL_NAMES
)
from models.classifier import build_classifier
from models.segmentor  import build_segmentor
from utils.metrics     import get_classification_metrics, compute_dice
from preprocessing.datasets import get_ham10000_loaders, get_ph2_loaders

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Classification ───────────────────────────────────────────

def evaluate_classifier(
    model     : torch.nn.Module,
    loader    : DataLoader,
    device    : torch.device
) -> dict:

    metrics = get_classification_metrics(
        num_classes=NUM_CLASSES_HAM,
        device=device
    )

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            metrics.update(outputs, labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    results = metrics.compute()

    logger.info("=" * 50)
    logger.info("EVALUATION CLASSIFICATION — HAM10000 TEST SET")
    logger.info("=" * 50)
    logger.info(f"F1 Macro   : {results['f1_macro'].item():.4f}")
    logger.info(f"AUROC      : {results['auroc'].item():.4f}")
    logger.info(f"Precision  : {results['precision'].item():.4f}")
    logger.info(f"Recall     : {results['recall'].item():.4f}")

    return {
        "f1_macro"  : results['f1_macro'].item(),
        "auroc"     : results['auroc'].item(),
        "precision" : results['precision'].item(),
        "recall"    : results['recall'].item(),
        "preds"     : all_preds,
        "labels"    : all_labels
    }


def plot_confusion_matrix(preds: list, labels: list) -> None:

    cm        = confusion_matrix(labels, preds)
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    class_names = list(LABEL_NAMES.keys())

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        "Matrice de Confusion — HAM10000 Test Set",
        fontsize=14, fontweight='bold'
    )

    # Matrice absolue
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues', ax=axes[0]
    )
    axes[0].set_title("Valeurs absolues")
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")

    # Matrice normalisée
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues', ax=axes[1],
        vmin=0, vmax=1
    )
    axes[1].set_title("Valeurs normalisées")
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Réel")

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Matrice de confusion sauvegardee : {out}")


# ── Segmentation ─────────────────────────────────────────────

def evaluate_segmentor(
    model  : torch.nn.Module,
    loader : DataLoader,
    device : torch.device
) -> dict:

    model.eval()
    dice_scores = []
    iou_scores  = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images)

            # Dice Score
            dice = compute_dice(preds, masks.unsqueeze(1).float())
            dice_scores.append(dice)

            # IoU
            preds_bin    = (torch.sigmoid(preds) > 0.5).float()
            intersection = (preds_bin * masks.unsqueeze(1)).sum().item()
            union        = (preds_bin + masks.unsqueeze(1)).clamp(0, 1).sum().item()
            iou          = intersection / (union + 1e-6)
            iou_scores.append(iou)

    mean_dice = float(np.mean(dice_scores))
    mean_iou  = float(np.mean(iou_scores))

    logger.info("=" * 50)
    logger.info("EVALUATION SEGMENTATION — PH2 TEST SET")
    logger.info("=" * 50)
    logger.info(f"Dice Score : {mean_dice:.4f}")
    logger.info(f"IoU        : {mean_iou:.4f}")

    return {"dice": mean_dice, "iou": mean_iou}


def plot_segmentation_samples(
    model  : torch.nn.Module,
    loader : DataLoader,
    device : torch.device,
    n      : int = 4
) -> None:

    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(images))
        preds_bin = (preds > 0.5).float()

    fig, axes = plt.subplots(3, n, figsize=(16, 10))
    fig.suptitle(
        "Segmentation PH2 — Image / Masque réel / Masque prédit",
        fontsize=13, fontweight='bold'
    )

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(n):
        # Image originale dénormalisée
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(std * img + mean, 0, 1)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(masks[i].cpu().numpy(), cmap='gray')
        axes[1, i].set_title("Masque réel")
        axes[1, i].axis('off')

        axes[2, i].imshow(preds_bin[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title("Masque prédit")
        axes[2, i].axis('off')

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "segmentation_samples.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Exemples segmentation sauvegardes : {out}")


# ── Point d'entrée ───────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device : {device}")

    # ── Classification ───────────────────────────────────────
    logger.info("Chargement du classifieur...")
    _, _, test_loader_ham, _ = get_ham10000_loaders()

    classifier = build_classifier(num_classes=NUM_CLASSES_HAM).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=device))
    logger.info(f"Poids charges depuis : {CLASSIFIER_CKPT}")

    clf_results = evaluate_classifier(classifier, test_loader_ham, device)
    plot_confusion_matrix(clf_results['preds'], clf_results['labels'])

    # ── Segmentation ─────────────────────────────────────────
    logger.info("Chargement du segmenteur...")
    _, _, test_loader_ph2 = get_ph2_loaders()

    segmentor = build_segmentor(num_classes=1).to(device)
    segmentor.load_state_dict(torch.load(SEGMENTOR_CKPT, map_location=device))
    logger.info(f"Poids charges depuis : {SEGMENTOR_CKPT}")

    seg_results = evaluate_segmentor(segmentor, test_loader_ph2, device)
    plot_segmentation_samples(segmentor, test_loader_ph2, device)

    # ── Résumé final ─────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("RESUME FINAL")
    logger.info("=" * 50)
    logger.info(f"Classification F1 Macro : {clf_results['f1_macro']:.4f}")
    logger.info(f"Classification AUROC    : {clf_results['auroc']:.4f}")
    logger.info(f"Segmentation Dice       : {seg_results['dice']:.4f}")
    logger.info(f"Segmentation IoU        : {seg_results['iou']:.4f}")


if __name__ == "__main__":
    main()