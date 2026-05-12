# preprocessing/transforms.py
# ============================================================
# Transformations et augmentations avec Albumentations
# HAM10000 (classification) + PH2 (segmentation synchronisee)
# ============================================================

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

H, W = IMAGE_SIZE

# ── HAM10000 — Train ─────────────────────────────────────────
ham_train_transform = A.Compose([
    A.Resize(H, W),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Affine remplace ShiftScaleRotate (plus recent)
    A.Affine(
        translate_percent = 0.1,
        scale             = (0.8, 1.2),
        rotate            = (-45, 45),
        p                 = 0.5
    ),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(hue_shift_limit=20, p=0.3),
    # CoarseDropout — syntaxe correcte version recente
    A.CoarseDropout(
        num_holes_range = (1, 8),
        hole_height_range = (16, 32),
        hole_width_range  = (16, 32),
        p = 0.3
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# ── HAM10000 — Val / Test ────────────────────────────────────
ham_val_transform = A.Compose([
    A.Resize(H, W),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# ── PH2 — Train (synchronisation native image + masque) ──────
ph2_train_transform = A.Compose([
    A.Resize(H, W),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent = 0.1,
        scale             = (0.8, 1.2),
        rotate            = (-45, 45),
        p                 = 0.5
    ),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.CLAHE(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# ── PH2 — Val / Test ─────────────────────────────────────────
ph2_val_transform = A.Compose([
    A.Resize(H, W),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})