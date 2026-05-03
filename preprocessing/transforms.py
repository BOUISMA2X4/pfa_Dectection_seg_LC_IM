# preprocessing/transforms.py
# ============================================================
# Transformations pour HAM10000 (classification)
# et PH2 (segmentation — image + masque synchronisés)
# ============================================================

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ── HAM10000 — Classification ────────────────────────────────

ham_train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

ham_val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ── PH2 — Segmentation (image + masque synchronisés) ─────────

class ResizePH2:
    def __init__(self, size=IMAGE_SIZE): self.size = size
    def __call__(self, img, mask):
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class RandomHFlipPH2:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomVFlipPH2:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img  = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class RandomRotationPH2:
    def __init__(self, degrees=20): self.degrees = degrees
    def __call__(self, img, mask):
        angle = np.random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR), mask.rotate(angle, resample=Image.NEAREST)

class ToTensorPH2:
    def __call__(self, img, mask):
        img_t  = transforms.ToTensor()(img)
        mask_t = torch.from_numpy(np.array(mask)).long()
        mask_t = (mask_t > 0).long()
        return img_t, mask_t

class ComposePH2:
    def __init__(self, tfms): self.tfms = tfms
    def __call__(self, img, mask):
        for t in self.tfms:
            img, mask = t(img, mask)
        return img, mask


ph2_train_transform = ComposePH2([
    ResizePH2(),
    RandomHFlipPH2(0.5),
    RandomVFlipPH2(0.5),
    RandomRotationPH2(20),
    ToTensorPH2()
])

ph2_val_transform = ComposePH2([ResizePH2(), ToTensorPH2()])
