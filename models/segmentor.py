# models/segmentor.py
# ============================================================
# UNet avec encodeur ResNet34 preentraine
# Segmentation binaire des lésions cutanées PH2
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import NUM_CLASSES_PH2


class ConvBlock(nn.Module):
    """
    Bloc de convolution double :
    Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetResNet34(nn.Module):
    """
    UNet avec encodeur ResNet34 preentraine ImageNet.

    Architecture :
      Encodeur : ResNet34 preentraine
        enc1 : (B, 64,  112, 112)
        enc2 : (B, 64,   56,  56)
        enc3 : (B, 128,  28,  28)
        enc4 : (B, 256,  14,  14)
        enc5 : (B, 512,   7,   7)

      Decodeur avec skip connections :
        dec4 : (B, 256, 14, 14)
        dec3 : (B, 128, 28, 28)
        dec2 : (B, 64,  56, 56)
        dec1 : (B, 64, 112, 112)

      Sortie : (B, num_classes, 224, 224)
    """

    def __init__(
        self,
        num_classes : int  = NUM_CLASSES_PH2,
        pretrained  : bool = True
    ):
        super().__init__()

        # Encodeur ResNet34 preentraine
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = models.resnet34(weights=weights)

        # Extraire les couches de l'encodeur
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1   # 64  canaux
        self.enc3 = resnet.layer2   # 128 canaux
        self.enc4 = resnet.layer3   # 256 canaux
        self.enc5 = resnet.layer4   # 512 canaux

        # Decodeur — skip connections
        # Concat(Upsample(enc_n), enc_{n-1}) → ConvBlock
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64,  64)
        self.dec1 = ConvBlock(64  + 64,  64)

        # Upsample bilineaire x2
        self.up = nn.Upsample(
            scale_factor  = 2,
            mode          = 'bilinear',
            align_corners = True
        )

        # Couche de sortie finale
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encodeur
        e1 = self.enc1(x)              # (B, 64,  112, 112)
        e2 = self.enc2(self.pool(e1))  # (B, 64,   56,  56)
        e3 = self.enc3(e2)             # (B, 128,  28,  28)
        e4 = self.enc4(e3)             # (B, 256,  14,  14)
        e5 = self.enc5(e4)             # (B, 512,   7,   7)

        # Decodeur + skip connections
        d4 = self.dec4(torch.cat([self.up(e5), e4], dim=1))  # (B, 256, 14, 14)
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))  # (B, 128, 28, 28)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))  # (B, 64,  56, 56)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # (B, 64, 112, 112)

        # Sortie finale
        out = self.final(self.up(d1))  # (B, num_classes, 224, 224)
        return out


def build_segmentor(
    num_classes : int  = NUM_CLASSES_PH2,
    pretrained  : bool = True
) -> nn.Module:
    """
    Construit et retourne le segmenteur UNet ResNet34.
    torch.compile() uniquement sur GPU.
    """
    model = UNetResNet34(
        num_classes = num_classes,
        pretrained  = pretrained
    )

    # torch.compile() uniquement si GPU disponible
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model
