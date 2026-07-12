"""
Training and evaluation utilities.

Provides metric tracking (:class:`AverageMeter`), registration evaluation
functions (ZNCC, MSE, NMI, foreground Dice, Jacobian), and 2D Dice for
segmentation masks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# ======================================================================
# AverageMeter — training metric tracker
# ======================================================================


class AverageMeter:
    """Tracks the average, current value, and std of a scalar metric.

    Usage::

        meter = AverageMeter()
        for batch in loader:
            loss = ...
            meter.update(loss.item(), n=batch_size)
        print(f"Avg: {meter.avg:.4f} ± {meter.std:.4f}")
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.vals: list[float] = []
        self.std: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = float(np.std(self.vals))


# ======================================================================
# Image similarity metrics (NumPy, for evaluation)
# ======================================================================


def compute_zncc(
    I: np.ndarray,
    J: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Zero-mean Normalised Cross-Correlation (ZNCC).

    Range [-1, 1]; higher is better.  Robust to linear intensity shifts.
    """
    I_mean, J_mean = float(np.mean(I)), float(np.mean(J))
    cross = np.sum((I - I_mean) * (J - J_mean))
    I_var = np.sum((I - I_mean) ** 2)
    J_var = np.sum((J - J_mean) ** 2)
    return float(cross / (np.sqrt(I_var * J_var) + eps))


def compute_mse(I: np.ndarray, J: np.ndarray) -> float:
    """Mean Squared Error (lower is better)."""
    return float(np.mean((I - J) ** 2))


def compute_nmi(I: np.ndarray, J: np.ndarray, bins: int = 256) -> float:
    """Normalised Mutual Information (NMI).

    Range typically [1.0, 2.0]; higher is better.  Robust to nonlinear
    intensity mappings.
    """
    hist_2d, _, _ = np.histogram2d(I.ravel(), J.ravel(), bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_nz, py_nz = px[px > 0], py[py > 0]
    pxy_nz = pxy[pxy > 0]

    hx = -np.sum(px_nz * np.log(px_nz + 1e-10))
    hy = -np.sum(py_nz * np.log(py_nz + 1e-10))
    hxy = -np.sum(pxy_nz * np.log(pxy_nz + 1e-10))

    return float((hx + hy) / hxy) if hxy > 0 else 0.0


def compute_foreground_dice(
    I: np.ndarray,
    J: np.ndarray,
    threshold: float = 0.01,
    dilate_ks: int = 5,
) -> float:
    """Foreground Dice overlap, with optional morphological dilation.

    Parameters
    ----------
    I, J : np.ndarray
        Images in [0, 1] range.
    threshold : float
        Intensity threshold for foreground.
    dilate_ks : int
        Dilation kernel size (1 = no dilation).
    """
    from scipy.ndimage import maximum_filter

    m_I = I > threshold
    m_J = J > threshold

    if dilate_ks > 1:
        m_I = maximum_filter(m_I, size=dilate_ks)
        m_J = maximum_filter(m_J, size=dilate_ks)

    intersection = np.sum(m_I & m_J)
    union = np.sum(m_I) + np.sum(m_J)
    return float(2.0 * intersection / (union + 1e-8))


# ======================================================================
# Flow quality metrics
# ======================================================================


def compute_epe(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Endpoint Error (EPE) — mean L2 distance between flow fields.

    Parameters
    ----------
    pred_flow : (B, 2, H, W)
    gt_flow : (B, 2, H, W)
    valid_mask : (B, 1, H, W) or None
        If given, EPE is computed only within valid pixels.
    """
    diff = pred_flow - gt_flow
    epe = torch.norm(diff, p=2, dim=1, keepdim=True)  # (B, 1, H, W)
    if valid_mask is not None:
        masked = epe * valid_mask
        return float(masked.sum() / (valid_mask.sum() + 1e-8))
    return float(epe.mean())


def compute_folding_ratio(flow: np.ndarray) -> float:
    """Fraction of pixels with a negative Jacobian determinant (folding).

    Parameters
    ----------
    flow : np.ndarray
        Shape (2, H, W) with ``indexing='ij'`` (dy, dx).
    """
    H, W = flow.shape[1], flow.shape[2]
    dy, dx = flow[0], flow[1]
    dy_dy, dy_dx = np.gradient(dy)
    dx_dy, dx_dx = np.gradient(dx)
    jac = (1.0 + dx_dx) * (1.0 + dy_dy) - dx_dy * dy_dx
    return float(np.mean(jac <= 0) * 100.0)


# ======================================================================
# 2D Dice for segmentation masks
# ======================================================================


def dice_score_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Per-class Dice coefficient for 2D segmentation.

    Parameters
    ----------
    pred : (B, 1, H, W)  integer tensor.
    target : (B, 1, H, W)  integer tensor.
    num_classes : int

    Returns
    -------
    (B, num_classes) tensor.
    """
    pred_oh = F.one_hot(pred.long(), num_classes=num_classes)
    if pred_oh.dim() == 5:
        pred_oh = pred_oh.squeeze(1)
    pred_oh = pred_oh.permute(0, 3, 1, 2).contiguous()

    tgt_oh = F.one_hot(target.long(), num_classes=num_classes)
    if tgt_oh.dim() == 5:
        tgt_oh = tgt_oh.squeeze(1)
    tgt_oh = tgt_oh.permute(0, 3, 1, 2).contiguous()

    intersection = (pred_oh * tgt_oh).sum(dim=[2, 3])
    cardinality = pred_oh.sum(dim=[2, 3]) + tgt_oh.sum(dim=[2, 3])
    return (2.0 * intersection) / (cardinality + 1e-5)


# ======================================================================
# Misc helpers
# ======================================================================


def build_foreground_mask(
    img: torch.Tensor,
    threshold: float = 0.01,
    dilate_ks: int = 5,
) -> torch.Tensor:
    """Build a binary foreground mask from an image.

    Parameters
    ----------
    img : (B, 1, H, W) in [0, 1].
    threshold : float
    dilate_ks : int
        Dilation kernel size (1 = no dilation).

    Returns
    -------
    (B, 1, H, W) float32 mask.
    """
    mask = (img > threshold).float()
    if dilate_ks > 1:
        pad = dilate_ks // 2
        mask = F.max_pool2d(mask, kernel_size=dilate_ks, stride=1, padding=pad)
    return mask
