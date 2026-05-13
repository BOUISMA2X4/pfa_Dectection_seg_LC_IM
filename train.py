# train.py
# ============================================================
# Boucle d'entrainement complete
# Classification HAM10000 + Segmentation PH2
# Usage : python train.py
# ============================================================

import os
import logging
import torch
import torch.nn as nn

from utils.seed      import set_seed
from utils.callbacks import EarlyStopping
from utils.tracking  import start_run, log_epoch, log_model
from utils.metrics   import get_classification_metrics, compute_dice

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers = [
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Classification HAM10000 ──────────────────────────────────

def train_classifier():

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device : {device}")

    # Donnees
    from preprocessing.datasets import get_ham10000_loaders
    train_loader, val_loader, test_loader, class_weights = get_ham10000_loaders()

    # Modele
    from models.classifier import build_classifier
    model = build_classifier().to(device)

    # Loss + Optimizer + Scheduler
    from models.losses import FocalLoss
    criterion = FocalLoss(gamma=2.0, weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = 1e-4,
        weight_decay = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = 30,
        eta_min = 1e-6
    )

    # CORRECTION — nouvelle syntaxe PyTorch 2.0+
    scaler = torch.amp.GradScaler(device.type)

    early_stop = EarlyStopping(patience=10)
    metrics    = get_classification_metrics(device=device)

    params = {
        "model"        : "ResNet50",
        "loss"         : "FocalLoss",
        "gamma"        : 2.0,
        "lr"           : 1e-4,
        "weight_decay" : 1e-4,
        "batch_size"   : 8,
        "scheduler"    : "CosineAnnealing"
    }

    with start_run("resnet50_focal_v1", params):
        for epoch in range(30):

            # ── Train ────────────────────────────────────────
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # CORRECTION — nouvelle syntaxe autocast
                with torch.amp.autocast(device.type):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            # ── Validation ───────────────────────────────────
            model.eval()
            val_loss = 0.0
            metrics.reset()

            with torch.no_grad():
                for images, labels in val_loader:
                    images  = images.to(device)
                    labels  = labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    metrics.update(outputs, labels)

            results   = metrics.compute()
            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)

            log_epoch({
                "train_loss" : avg_train,
                "val_loss"   : avg_val,
                "val_f1"     : results['f1_macro'].item(),
                "val_auroc"  : results['auroc'].item(),
                "lr"         : scheduler.get_last_lr()[0]
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1:03d} | "
                f"Train: {avg_train:.4f} | "
                f"Val: {avg_val:.4f} | "
                f"F1: {results['f1_macro']:.4f} | "
                f"AUROC: {results['auroc']:.4f}"
            )

            scheduler.step()
            early_stop(avg_val, model, 'checkpoints/classifier_best.pth')

            if early_stop.stop:
                logger.info(f"Early stopping a l'epoch {epoch+1}")
                break

        log_model(model, "classifier")

    logger.info("Entrainement classificateur termine")


# ── Segmentation PH2 ─────────────────────────────────────────

def train_segmentor():

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device : {device}")

    # Donnees
    from preprocessing.datasets import get_ph2_loaders
    train_loader, val_loader, test_loader = get_ph2_loaders()

    # Modele
    from models.segmentor import build_segmentor
    model = build_segmentor().to(device)

    # Loss + Optimizer + Scheduler
    from models.losses import DiceBCELoss
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = 1e-4,
        weight_decay = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = 30,
        eta_min = 1e-6
    )

    # CORRECTION — nouvelle syntaxe PyTorch 2.0+
    scaler = torch.amp.GradScaler(device.type)

    early_stop = EarlyStopping(patience=10)

    params = {
        "model"        : "UNet_ResNet34",
        "loss"         : "DiceBCELoss",
        "lr"           : 1e-4,
        "weight_decay" : 1e-4,
        "batch_size"   : 8,
        "scheduler"    : "CosineAnnealing"
    }

    with start_run("unet_dicebce_v1", params):
        for epoch in range(30):

            # ── Train ────────────────────────────────────────
            model.train()
            train_loss = 0.0

            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.unsqueeze(1).float().to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # CORRECTION — nouvelle syntaxe autocast
                with torch.amp.autocast(device.type):
                    outputs = model(images)
                    loss    = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            # ── Validation ───────────────────────────────────
            model.eval()
            val_loss   = 0.0
            dice_scores = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks  = masks.unsqueeze(1).float().to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
                    dice_scores.append(compute_dice(outputs, masks))

            avg_train = train_loss  / len(train_loader)
            avg_val   = val_loss    / len(val_loader)
            avg_dice  = sum(dice_scores) / len(dice_scores)

            log_epoch({
                "train_loss" : avg_train,
                "val_loss"   : avg_val,
                "val_dice"   : avg_dice,
                "lr"         : scheduler.get_last_lr()[0]
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1:03d} | "
                f"Train: {avg_train:.4f} | "
                f"Val: {avg_val:.4f} | "
                f"Dice: {avg_dice:.4f}"
            )

            scheduler.step()
            early_stop(avg_val, model, 'checkpoints/segmentor_best.pth')

            if early_stop.stop:
                logger.info(f"Early stopping a l'epoch {epoch+1}")
                break

        log_model(model, "segmentor")

    logger.info("Entrainement segmenteur termine")


# ── Point d'entree ───────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    logger.info("=" * 55)
    logger.info("  Phase 2 — Entrainement des modeles")
    logger.info("=" * 55)

    logger.info("Debut entrainement Classification HAM10000")
    train_classifier()

    logger.info("Debut entrainement Segmentation PH2")
    train_segmentor()