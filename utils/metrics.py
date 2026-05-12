# Toutes les métriques d'évaluation
# F1 Macro, AUROC pour classification
# Dice, IoU pour segmentation

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
from torchmetrics.segmentation import MeanIoU
import torch

def get_classification_metrics(num_classes: int = 7, device: torch.device = 'cpu'):
    return MetricCollection({
        'f1_macro'  : MulticlassF1Score(num_classes=num_classes, average='macro'),
        'auroc'     : MulticlassAUROC(num_classes=num_classes,   average='macro'),
        'precision' : MulticlassPrecision(num_classes=num_classes, average='macro'),
        'recall'    : MulticlassRecall(num_classes=num_classes,  average='macro'),
    }).to(device)

def get_segmentation_metrics(device: torch.device = 'cpu'):
    return MetricCollection({
        'iou' : MeanIoU(num_classes=2),
    }).to(device)

def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred      = (torch.sigmoid(pred) > 0.5).float()
    intersect = (pred * target).sum()
    return (2 * intersect / (pred.sum() + target.sum() + 1e-6)).item()