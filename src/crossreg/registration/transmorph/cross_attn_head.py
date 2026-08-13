"""Compatibility import for the deprecated Cross-Attention RegHead.

New code should import :class:`ModalityConditionedTransMorph` from
``crossreg.registration.transmorph``.  This path remains so historical
checkpoints and experiment scripts do not break.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cross_attn_head is deprecated; use ModalityConditionedTransMorph for new "
    "registration training",
    DeprecationWarning,
    stacklevel=2,
)

from .deprecated_cross_attn_head import (  # noqa: E402,F401
    CrossAttentionFusion,
    CrossAttentionRegHead,
    MultiScaleCrossAttentionFuser,
)

__all__ = [
    "CrossAttentionFusion",
    "CrossAttentionRegHead",
    "MultiScaleCrossAttentionFuser",
]
