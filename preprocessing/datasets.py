# preprocessing/datasets.py
# ============================================================
# Classes Dataset PyTorch pour HAM10000 et PH2
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    HAM10000_META, HAM10000_PART1, HAM10000_PART2,
    PH2_IMG_DIR, PH2_MASK_DIR,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE, BATCH_SIZE, NUM_WORKERS
)
from preprocessing.transforms import (
    ham_train_transform, ham_val_transform,
    ph2_train_transform, ph2_val_transform
)


# ── HAM10000 ─────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels    = labels
        self.transform = transform
        self.classes   = sorted(list(set(labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img   = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def _find_ham_image(image_id: str) -> str | None:
    for part in [HAM10000_PART1, HAM10000_PART2]:
        p = os.path.join(part, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None


def get_ham10000_loaders():
    """Retourne (train_loader, val_loader, test_loader, class_weights)."""
    meta = pd.read_csv(HAM10000_META)

    paths  = [_find_ham_image(iid) for iid in meta["image_id"]]
    labels = meta["dx"].tolist()

    # Filtrer les images introuvables
    valid = [(p, l) for p, l in zip(paths, labels) if p is not None]
    paths, labels = zip(*valid)

    # Splits stratifiés
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE, stratify=y_tmp, random_state=RANDOM_STATE)

    train_ds = HAM10000Dataset(X_tr,  y_tr,  ham_train_transform)
    val_ds   = HAM10000Dataset(X_val, y_val, ham_val_transform)
    test_ds  = HAM10000Dataset(X_te,  y_te,  ham_val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Poids de classes pour CrossEntropyLoss
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weights = torch.tensor(weights, dtype=torch.float)

    print(f"HAM10000 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_weights


# ── PH2 ──────────────────────────────────────────────────────

class PH2Dataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


def get_ph2_loaders():
    """Retourne (train_loader, val_loader, test_loader)."""
    imgs  = sorted([os.path.join(PH2_IMG_DIR,  f) for f in os.listdir(PH2_IMG_DIR)  if f.endswith(".bmp")])
    masks = sorted([os.path.join(PH2_MASK_DIR, f) for f in os.listdir(PH2_MASK_DIR) if f.endswith(".bmp")])

    # Assertion alignement
    assert len(imgs) == len(masks), "Mismatch images/masques PH2 !"
    for ip, mp in zip(imgs, masks):
        assert os.path.basename(ip) == os.path.basename(mp), f"Désalignement : {ip} vs {mp}"

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        imgs, masks, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    train_ds = PH2Dataset(X_tr,  y_tr,  ph2_train_transform)
    val_ds   = PH2Dataset(X_val, y_val, ph2_val_transform)
    test_ds  = PH2Dataset(X_te,  y_te,  ph2_val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"PH2 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
