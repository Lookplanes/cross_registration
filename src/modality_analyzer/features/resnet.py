"""
ResNet18 deep feature extractor for modality analysis.

Provides a lightweight wrapper around torchvision's ResNet18 (ImageNet
pretrained) that outputs 512-dim feature vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEAT_DIM: int = 512

# ImageNet normalization (RGB)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Default ResNet preprocessing pipeline
_RESNET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_resnet_extractor(device: str | torch.device = "cpu") -> nn.Module:
    """Build a ResNet18 feature extractor (512-dim output).

    Uses ``IMAGENET1K_V1`` pretrained weights.  The final ``fc`` layer is
    replaced with ``nn.Identity()`` so that ``forward(x)`` returns the
    512-dim avgpool vector.

    Args:
        device: ``"cpu"``, ``"cuda"``, or a ``torch.device``.

    Returns:
        ResNet18 in eval mode, on the requested device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> torch.Tensor:
    """Load an image, convert to RGB, apply ResNet18 preprocessing.

    Returns:
        Tensor of shape ``(3, 224, 224)``, normalised to ImageNet stats.
    """
    img = Image.open(path).convert("RGB")
    return _RESNET_TRANSFORM(img)
