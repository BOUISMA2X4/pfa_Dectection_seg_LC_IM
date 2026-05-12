import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Classification ───────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(inputs, dim=-1)
        prob     = torch.exp(log_prob)
        loss     = F.nll_loss(log_prob, targets, weight=self.weight, reduction='none')
        focal    = ((1 - prob.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma) * loss
        return focal.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int = 7, smoothing: float = 0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot    = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_val
        log_prob   = F.log_softmax(inputs, dim=-1)
        return -(smooth_one_hot * log_prob).sum(dim=-1).mean()


# ── Segmentation ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred         = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice         = 1 - (2 * intersection + 1) / (
                           pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1)
        return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(pred, target.float())
        dice = self.dice(pred, target)
        return bce + dice