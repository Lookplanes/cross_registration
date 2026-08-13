"""
TransCUT — cross-modality translation with shared Swin-Transformer encoder.

Provides:
- :class:`TransCUT`         — full training/inference model
- :class:`TransCUTConfig`   — dataclass configuration
- :func:`load_transcut_encoder` — extract encoder weights for TransMorph sharing
"""

from .transcut_model import TransCUT, TransCUTConfig
from .highres_decoder import HighResolutionContentDecoder
from .fullres_residual_decoder import FullResolutionResidualDecoder

__all__ = [
    "TransCUT",
    "TransCUTConfig",
    "HighResolutionContentDecoder",
    "FullResolutionResidualDecoder",
]
