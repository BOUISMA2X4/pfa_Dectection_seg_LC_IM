# SUPPRIME : toutes les classes custom PH2
# SUPPRIME : import torchvision.transforms
# SUPPRIME : import PIL, numpy (plus nécessaires)
# AJOUTE   : albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

# ── HAM10000 — REMPLACE les transforms torchvision ───────────
ham_train_transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.2,
        rotate_limit=45, p=0.5
    ),
    # NOUVEAU — augmentations médicales
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(hue_shift_limit=20, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

ham_val_transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# ── PH2 — REMPLACE les classes custom par synchronisation native
ph2_train_transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.CLAHE(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})  # synchronisation native

ph2_val_transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})