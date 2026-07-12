"""
I/O utilities: image discovery, checkpoint persistence, feature CSV helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Supported image extensions (lowercase)
IMAGE_EXT: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(
    data_root: str | Path,
    source_dir: str,
    glob_pattern: str = "**/*",
    label: str = "unknown",
    is_center: bool = False,
) -> list[dict[str, Any]]:
    """Discover images under ``data_root / source_dir`` matching glob & extensions.

    Args:
        data_root: Root directory containing per-modality subdirectories.
        source_dir: Subdirectory name (or relative path) under data_root.
        glob_pattern: Glob relative to source_dir (default ``**/*``).
        label: Modality display name to attach to each record.
        is_center: Whether this modality is a center/hub modality.

    Returns:
        List of dicts with keys ``path``, ``modality_name``, ``is_center``.
    """
    root = Path(data_root) / source_dir
    if not root.exists():
        return []

    files: list[Path] = []
    for ext in IMAGE_EXT:
        files.extend(root.glob(f"{glob_pattern}{ext}"))
        files.extend(root.glob(f"{glob_pattern}{ext.upper()}"))
    files = sorted(set(files))

    images: list[dict[str, Any]] = []
    for f in files:
        if f.suffix.lower() not in IMAGE_EXT:
            continue
        images.append({
            "path": str(f),
            "modality_name": label,
            "is_center": is_center,
        })
    return images


# ---------------------------------------------------------------------------
# Checkpoint (resume) helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str | Path) -> set[str]:
    """Load set of already-processed unique keys from a JSON checkpoint file."""
    p = Path(path)
    if p.exists():
        return set(json.loads(p.read_text()))
    return set()


def save_checkpoint(path: str | Path, processed: set[str]) -> None:
    """Save processed keys to a JSON checkpoint file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(list(processed)))


# ---------------------------------------------------------------------------
# Feature CSV helpers
# ---------------------------------------------------------------------------

def ensure_is_center_column(
    df: pd.DataFrame,
    center_modalities: list[str],
) -> pd.DataFrame:
    """Add ``is_center`` column if missing, based on modality_name membership."""
    if "is_center" not in df.columns:
        df["is_center"] = df["modality_name"].isin(center_modalities)
    return df
