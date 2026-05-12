# predict.py
# ============================================================
# Inference sur une nouvelle image
# Classification HAM10000 + Segmentation PH2
# Usage : python predict.py --image path/to/image.jpg
# ============================================================

import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Tuple

from config.config import (
    CLASSIFIER_CKPT, SEGMENTOR_CKPT,
    OUTPUTS_DIR, LABEL_NAMES,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
)
from models.classifier import build_classifier
from models.segmentor  import build_segmentor

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/predict.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Transformation inference ──────────────────────────────────

infer_transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])


def load_image(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    img_pil = Image.open(path).convert("RGB")
    img_np  = np.array(img_pil)
    tensor  = infer_transform(image=img_np)['image']
    return tensor.unsqueeze(0), img_np   # (1, 3, 224, 224)


# ── Classification ───────────────────────────────────────────

def predict_class(
    image_path : str,
    device     : torch.device
) -> dict:

    tensor, img_np = load_image(image_path)
    tensor         = tensor.to(device)

    classifier = build_classifier(num_classes=7).to(device)
    classifier.load_state_dict(
        torch.load(CLASSIFIER_CKPT, map_location=device)
    )
    classifier.eval()

    with torch.no_grad():
        outputs     = classifier(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        pred_idx    = probs.argmax().item()

    class_names = list(LABEL_NAMES.keys())
    pred_class  = class_names[pred_idx]
    pred_name   = LABEL_NAMES[pred_class]
    confidence  = probs[pred_idx].item()

    logger.info(f"Image            : {image_path}")
    logger.info(f"Classe predite   : {pred_class} ({pred_name})")
    logger.info(f"Confiance        : {confidence:.4f} ({confidence*100:.1f}%)")

    # Afficher toutes les probabilités
    logger.info("Probabilites par classe :")
    for i, (cls, name) in enumerate(LABEL_NAMES.items()):
        logger.info(f"  {cls:6s} ({name:30s}) : {probs[i].item():.4f}")

    return {
        "class"      : pred_class,
        "name"       : pred_name,
        "confidence" : confidence,
        "probs"      : {cls: probs[i].item() for i, cls in enumerate(class_names)},
        "image"      : img_np
    }


def plot_classification_result(result: dict, image_path: str) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Prediction : {result['class']} — {result['name']} "
        f"({result['confidence']*100:.1f}%)",
        fontsize=13, fontweight='bold'
    )

    # Image
    axes[0].imshow(result['image'])
    axes[0].set_title("Image d'entrée")
    axes[0].axis('off')

    # Probabilités
    classes = list(result['probs'].keys())
    probs   = list(result['probs'].values())
    colors  = ['green' if c == result['class'] else 'steelblue' for c in classes]

    axes[1].barh(classes, probs, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probabilité")
    axes[1].set_title("Distribution des probabilités")
    for i, (cls, p) in enumerate(zip(classes, probs)):
        axes[1].text(p + 0.01, i, f"{p:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "prediction_classification.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Resultat sauvegarde : {out}")


# ── Segmentation ─────────────────────────────────────────────

def predict_mask(
    image_path : str,
    device     : torch.device
) -> dict:

    tensor, img_np = load_image(image_path)
    tensor         = tensor.to(device)

    segmentor = build_segmentor(num_classes=1).to(device)
    segmentor.load_state_dict(
        torch.load(SEGMENTOR_CKPT, map_location=device)
    )
    segmentor.eval()

    with torch.no_grad():
        pred     = segmentor(tensor)
        prob_map = torch.sigmoid(pred)[0, 0].cpu().numpy()
        mask_bin = (prob_map > 0.5).astype(np.uint8)

    coverage = mask_bin.sum() / mask_bin.size * 100
    logger.info(f"Image            : {image_path}")
    logger.info(f"Couverture lesion: {coverage:.1f}% des pixels")

    return {
        "mask"     : mask_bin,
        "prob_map" : prob_map,
        "coverage" : coverage,
        "image"    : img_np
    }


def plot_segmentation_result(result: dict, image_path: str) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Segmentation — Couverture lésion : {result['coverage']:.1f}%",
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(result['image'])
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    axes[1].imshow(result['prob_map'], cmap='hot')
    axes[1].set_title("Carte de probabilité")
    axes[1].axis('off')

    axes[2].imshow(result['mask'], cmap='gray')
    axes[2].set_title("Masque prédit (seuil 0.5)")
    axes[2].axis('off')

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "prediction_segmentation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Resultat sauvegarde : {out}")


# ── Point d'entrée ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference sur une image dermoscopique"
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Chemin vers l image a analyser'
    )
    parser.add_argument(
        '--task', type=str, default='both',
        choices=['classification', 'segmentation', 'both'],
        help='Tache a effectuer (defaut: both)'
    )
    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device : {device}")

    if not os.path.exists(args.image):
        logger.error(f"Image introuvable : {args.image}")
        return

    if args.task in ('classification', 'both'):
        result = predict_class(args.image, device)
        plot_classification_result(result, args.image)

    if args.task in ('segmentation', 'both'):
        result = predict_mask(args.image, device)
        plot_segmentation_result(result, args.image)


if __name__ == "__main__":
    main()