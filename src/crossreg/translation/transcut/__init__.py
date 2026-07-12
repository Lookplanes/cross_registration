"""
TransCUT — cross-modality translation with shared Swin-Transformer encoder.

Provides:
- :class:`TransCUT`         — full training/inference model
- :class:`TransCUTConfig`   — dataclass configuration
- :func:`load_transcut_encoder` — extract encoder weights for TransMorph sharing
"""

from .transcut_model import TransCUT, TransCUTConfig
