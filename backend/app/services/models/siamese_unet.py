"""
backend/app/services/models/siamese_unet.py
============================================
Lightweight Siamese encoder-decoder (Siamese U-Net) for binary change detection.

Architecture overview
---------------------
Input:  Two 3-channel RGB images (img_A, img_B) at time T1 and T2.
        Concatenated to a 6-channel tensor: [img_A | img_B] of shape (B, 6, H, W).

Encoder (shared/independent — we use a simple independent encoder per pair):
  Block 1:  Conv(6→16)  + BN + ReLU + Conv(16→16)  + BN + ReLU  → skip1
  Block 2:  MaxPool + Conv(16→32) + BN + ReLU + Conv(32→32)  + BN + ReLU  → skip2
  Block 3:  MaxPool + Conv(32→64) + BN + ReLU + Conv(64→64)  + BN + ReLU  → bottleneck

Decoder:
  Up1: Bilinear ×2 + Concat(skip2) → Conv(64+32→32) + BN + ReLU + Conv(32→32) + BN + ReLU
  Up2: Bilinear ×2 + Concat(skip1) → Conv(32+16→16) + BN + ReLU + Conv(16→16) + BN + ReLU
  Head: Conv(16→1) → logit map (no sigmoid here — use BCEWithLogitsLoss during training)

Output: (B, 1, H, W) logit tensor.
        Apply torch.sigmoid for probability map.
        Threshold at 0.5 for binary change mask.

Parameter count: ~490K — runnable on CPU in reasonable time for inference.

Design decisions
----------------
- Input concatenation (6-channel) is simpler and faster than weight-shared Siamese
  branches for the baseline. It is the most common approach in LEVIR-CD papers.
- No pretrained weights needed — trained from scratch on LEVIR-CD.
- BatchNorm included for training stability.
- Dropout is omitted in this baseline for simplicity (can be added later).
- Resolution preserved: uses same padding so output H×W == input H×W.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU (inplace)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DoubleConv(nn.Module):
    """Two sequential ConvBNReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """MaxPool2d(2) → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    """
    Bilinear upsampling × 2 → concatenate skip → DoubleConv.
    in_ch  = channels from decoder path (before concat)
    skip_ch = channels from skip connection
    out_ch = output channels after DoubleConv
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Handle potential size mismatch after interpolation (odd dimensions)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SiameseUNet(nn.Module):
    """
    Lightweight Siamese U-Net for pixel-level binary change detection.

    Input
    -----
    x : torch.Tensor of shape (B, 6, H, W)
        Concatenation of img_A and img_B along the channel dimension.
        Both images must be normalised to [0, 1].

    Output
    ------
    logits : torch.Tensor of shape (B, 1, H, W)
        Raw (un-activated) logit scores.
        Apply sigmoid for probabilities; threshold at 0.5 for binary mask.
        Use BCEWithLogitsLoss during training (numerically stable).
    """

    def __init__(self, in_channels: int = 6, base_filters: int = 16):
        """
        Parameters
        ----------
        in_channels  : Number of input channels. Default 6 = 3 (A) + 3 (B).
        base_filters : Number of filters in the first conv block. Doubles at each depth.
                       Default 16 → [16, 32, 64] at depths 1/2/3.
        """
        super().__init__()

        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4

        # Encoder
        self.enc1 = _DoubleConv(in_channels, f1)   # (B, f1, H, W)
        self.enc2 = _Down(f1, f2)                   # (B, f2, H/2, W/2)
        self.enc3 = _Down(f2, f3)                   # (B, f3, H/4, W/4)

        # Decoder
        self.dec2 = _Up(f3, f2, f2)                 # (B, f2, H/2, W/2)
        self.dec1 = _Up(f2, f1, f1)                 # (B, f1, H, W)

        # Classification head
        self.head = nn.Conv2d(f1, 1, kernel_size=1)  # (B, 1, H, W) logits

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)   # skip at full resolution
        s2 = self.enc2(s1)  # skip at 1/2 resolution
        b = self.enc3(s2)   # bottleneck at 1/4 resolution

        # Decoder
        d2 = self.dec2(b, s2)
        d1 = self.dec1(d2, s1)

        # Head
        return self.head(d1)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_model(base_filters: int = 16) -> SiameseUNet:
    """Create a new, randomly-initialised SiameseUNet."""
    return SiameseUNet(in_channels=6, base_filters=base_filters)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_change_mask(
    model: SiameseUNet,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference and return both probability map and binary mask.

    Parameters
    ----------
    model    : Trained SiameseUNet (eval mode assumed).
    img_a    : (B, 3, H, W) float32 tensor, values in [0, 1].
    img_b    : (B, 3, H, W) float32 tensor, values in [0, 1].
    threshold: Probability threshold for binary mask. Default 0.5.

    Returns
    -------
    prob_map    : (B, 1, H, W) float32 probability map [0, 1].
    binary_mask : (B, 1, H, W) binary float32 tensor {0.0, 1.0}.
    """
    model.eval()
    with torch.no_grad():
        x = torch.cat([img_a, img_b], dim=1)  # (B, 6, H, W)
        logits = model(x)
        prob_map = torch.sigmoid(logits)
        binary_mask = (prob_map >= threshold).float()
    return prob_map, binary_mask


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable Dice loss for binary segmentation.

    Parameters
    ----------
    pred   : (B, 1, H, W) probability map (after sigmoid), values in [0, 1].
    target : (B, 1, H, W) binary mask {0.0, 1.0}.
    smooth : Laplace smoothing to avoid division by zero.

    Returns
    -------
    Scalar Dice loss = 1 - Dice coefficient.
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_coeff = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return 1.0 - dice_coeff.mean()


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable Tversky loss for binary segmentation with class imbalance.

    When beta > alpha (e.g. alpha=0.3, beta=0.7), false negatives are penalized
    more heavily than false positives, which boosts minority class recall.

    Parameters
    ----------
    pred   : (B, 1, H, W) probability map (after sigmoid), values in [0, 1].
    target : (B, 1, H, W) binary mask {0.0, 1.0}.
    alpha  : Weight for False Positives (FP).
    beta   : Weight for False Negatives (FN).
    smooth : Laplace smoothing constant.
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)

    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_index.mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary Focal Loss with logits input.

    Downweights well-classified background pixels, focusing gradients on
    hard-to-classify change boundaries.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    focal = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return focal.mean()


def hybrid_imbalance_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    bce_weight: float = 0.35,
    tversky_weight: float = 0.45,
    focal_weight: float = 0.20,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    """
    Hybrid loss engineered for extreme class-imbalanced change detection:
    1. Weighted BCE: provides strong positive gradient pull.
    2. Tversky (alpha=0.3, beta=0.7): penalizes false negatives (recall booster).
    3. Focal Loss: focuses on hard boundary examples.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    tversky = tversky_loss(prob, target, alpha=tversky_alpha, beta=tversky_beta)
    focal = focal_loss(logits, target, alpha=0.25, gamma=2.0)
    return bce_weight * bce + tversky_weight * tversky + focal_weight * focal


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combined BCE + Dice loss (standard for binary segmentation).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    return bce_weight * bce + dice_weight * dice


def enhanced_hybrid_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    bce_weight: float = 0.25,
    dice_weight: float = 0.30,
    tversky_weight: float = 0.30,
    focal_weight: float = 0.15,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    """
    Enhanced 4-component hybrid loss engineered for Experiment 02:
    1. Weighted BCE (0.25): provides calibrated pixel supervision & positive pull.
    2. Soft Dice Loss (0.30): directly optimizes global region IoU/F1 overlap.
    3. Asymmetric Tversky (0.30, alpha=0.3, beta=0.7): penalizes false negatives to boost change recall.
    4. Focal Loss (0.15, gamma=2.0): focuses on hard edge boundaries.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    tversky = tversky_loss(prob, target, alpha=tversky_alpha, beta=tversky_beta)
    focal = focal_loss(logits, target, alpha=0.25, gamma=2.0)
    return bce_weight * bce + dice_weight * dice + tversky_weight * tversky + focal_weight * focal


# Make Optional importable at module level for type signatures
from typing import Optional

