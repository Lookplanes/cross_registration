"""Stable modality registries for multi-domain translation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModalitySpec:
    """A stable integer modality ID and its image directory."""

    id: int
    name: str
    path: str
    channels: int = 1


def load_modality_registry(path: str | Path) -> list[ModalitySpec]:
    """Load and validate a YAML modality registry.

    Relative image paths are resolved relative to the registry file. IDs must
    be unique and contiguous from zero so they can safely index ``nn.Embedding``.
    """
    registry_path = Path(path).expanduser().resolve()
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("modalities")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("modality registry must contain a 'modalities' list with at least two entries")

    specs: list[ModalitySpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"modalities[{index}] must be a mapping")
        unknown = set(entry) - {"id", "name", "path", "channels"}
        if unknown:
            raise ValueError(f"unknown keys in modalities[{index}]: {sorted(unknown)}")
        try:
            modality_id = int(entry["id"])
            name = str(entry["name"]).strip()
            image_path = Path(str(entry["path"])).expanduser()
            channels = int(entry.get("channels", 1))
        except KeyError as exc:
            raise ValueError(f"modalities[{index}] is missing {exc.args[0]!r}") from exc
        if not name:
            raise ValueError(f"modalities[{index}].name must not be empty")
        if not image_path.is_absolute():
            image_path = registry_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_dir():
            raise ValueError(f"modality directory does not exist: {image_path}")
        if channels not in {1, 3}:
            raise ValueError(f"modality {name!r} channels must be 1 or 3")
        specs.append(ModalitySpec(modality_id, name, str(image_path), channels))

    ids = [spec.id for spec in specs]
    names = [spec.name for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("modality IDs must be unique")
    if len(set(names)) != len(names):
        raise ValueError("modality names must be unique")
    if sorted(ids) != list(range(len(specs))):
        raise ValueError("modality IDs must be contiguous from 0 to N-1")
    specs.sort(key=lambda spec: spec.id)
    if len({spec.channels for spec in specs}) != 1:
        raise ValueError("all modalities must currently use the same channel count")
    return specs


def save_modality_registry(specs: list[ModalitySpec], path: str | Path) -> None:
    """Write the resolved registry used by an experiment."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"modalities": [asdict(spec) for spec in specs]},
            handle, sort_keys=False, allow_unicode=True,
        )
