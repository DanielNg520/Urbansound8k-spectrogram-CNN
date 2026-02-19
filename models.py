"""
CNN architectures for mel spectrogram classification.
Input: (B, 1, 128, 128) — single-channel log-mel spectrogram.
Output: (B, 10) — class logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional MaxPool"""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class UrbanCNN(nn.Module):
    """
    4-block CNN for UrbanSound8K.
    Designed for (1, 128, 128) input.
    ~1.2M parameters — trains well even on CPU.
    """
    def __init__(self, n_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),    # → (32, 64, 64)
            ConvBlock(32, 64, pool=True),    # → (64, 32, 32)
            ConvBlock(64, 128, pool=True),   # → (128, 16, 16)
            ConvBlock(128, 256, pool=True),  # → (256, 8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LightCNN(nn.Module):
    """
    Lighter variant for Colab free tier / CPU.
    ~300K parameters.
    """
    def __init__(self, n_classes=10, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  16, pool=True),   # → (16, 64, 64)
            ConvBlock(16, 32, pool=True),   # → (32, 32, 32)
            ConvBlock(32, 64, pool=True),   # → (64, 16, 16)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
