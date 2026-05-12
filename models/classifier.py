# models/classifier.py
# ============================================================
# ResNet50 fine-tuné pour la classification HAM10000
# 7 classes de lésions cutanées
# ============================================================

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DROPOUT_P, NUM_CLASSES_HAM


class ResNet50Classifier(nn.Module):

    def __init__(
        self,
        num_classes : int   = NUM_CLASSES_HAM,
        dropout_p   : float = DROPOUT_P,
        pretrained  : bool  = True
    ):
        super().__init__()

        # Charger ResNet50 preentraine ImageNet
        weights       = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone      = models.resnet50(weights=weights)

        # Garder tout sauf la derniere couche FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Nouvelle tete de classification adaptee a 7 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)         # (B, 2048, 1, 1)
        output   = self.classifier(features) # (B, num_classes)
        return output


def build_classifier(
    num_classes : int  = NUM_CLASSES_HAM,
    pretrained  : bool = True
) -> nn.Module:
    """
    Construit et retourne le classifieur ResNet50.
    torch.compile() desactive par defaut sur CPU Windows
    car necessite un compilateur C++ compatible.
    Activer sur GPU Colab uniquement.
    """
    model = ResNet50Classifier(
        num_classes = num_classes,
        pretrained  = pretrained
    )

    # torch.compile() uniquement si GPU disponible
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
        except Exception:
            pass  # fallback si compile non disponible

    return model
