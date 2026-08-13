"""Reusable validation visualizations for displacement-based registration."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _rgb(image: torch.Tensor) -> np.ndarray:
    """Convert one Stage-1-range ``(C,H,W)`` tensor to display RGB."""
    array = image.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte()
    if array.shape[0] == 1:
        array = array.expand(3, -1, -1)
    if array.shape[0] != 3:
        raise ValueError(f"expected one or three image channels, got {array.shape[0]}")
    return array.permute(1, 2, 0).numpy()


def _scalar_color(value: np.ndarray, limit: float) -> np.ndarray:
    """Map zero to black and increasing scalar error to yellow/red."""
    scaled = np.clip(value / max(limit, 1e-6), 0.0, 1.0)
    red = np.clip(2.0 * scaled, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - scaled), 0.0, 1.0) * scaled
    blue = np.zeros_like(scaled)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def _flow_panel(flow: np.ndarray, limit: float, arrow_step: int) -> Image.Image:
    dy, dx = flow
    magnitude = np.sqrt(dy ** 2 + dx ** 2)
    panel = Image.fromarray(_scalar_color(magnitude, limit), mode="RGB")
    draw = ImageDraw.Draw(panel)
    height, width = magnitude.shape
    scale = arrow_step / max(limit, 1e-6) * 0.7
    for y in range(arrow_step // 2, height, arrow_step):
        for x in range(arrow_step // 2, width, arrow_step):
            end_x = x + float(dx[y, x]) * scale
            end_y = y + float(dy[y, x]) * scale
            draw.line((x, y, end_x, end_y), fill="white", width=2)
            angle = math.atan2(end_y - y, end_x - x)
            for offset in (2.5, -2.5):
                draw.line(
                    (
                        end_x, end_y,
                        end_x - 4 * math.cos(angle + offset),
                        end_y - 4 * math.sin(angle + offset),
                    ),
                    fill="white", width=1,
                )
    return panel


def _titled(image: np.ndarray | Image.Image, title: str) -> Image.Image:
    panel = image if isinstance(image, Image.Image) else Image.fromarray(image)
    canvas = Image.new("RGB", (panel.width, panel.height + 34), "white")
    canvas.paste(panel.convert("RGB"), (0, 34))
    ImageDraw.Draw(canvas).text((4, 5), title, fill="black")
    return canvas


@dataclass
class RegistrationValidationSample:
    """CPU snapshot of one fixed validation example."""

    sample_id: str
    direction: str
    moving: torch.Tensor
    fixed: torch.Tensor
    warped: torch.Tensor
    target_flow: torch.Tensor
    predicted_flow: torch.Tensor
    valid_mask: torch.Tensor


class RegistrationSampleCollector:
    """Collect a deterministic quota per direction and render one sheet."""

    def __init__(self, samples_per_direction: int = 1) -> None:
        if samples_per_direction < 1:
            raise ValueError("samples_per_direction must be positive")
        self.samples_per_direction = samples_per_direction
        self.samples: list[RegistrationValidationSample] = []
        self._counts: dict[str, int] = {}

    def wants(self, direction: str) -> bool:
        return self._counts.get(direction, 0) < self.samples_per_direction

    def add(
        self, *, sample_id: str, direction: str, moving: torch.Tensor,
        fixed: torch.Tensor, warped: torch.Tensor, target_flow: torch.Tensor,
        predicted_flow: torch.Tensor, valid_mask: torch.Tensor,
    ) -> None:
        if not self.wants(direction):
            return
        self.samples.append(RegistrationValidationSample(
            sample_id=sample_id,
            direction=direction,
            moving=moving.detach().cpu(),
            fixed=fixed.detach().cpu(),
            warped=warped.detach().cpu(),
            target_flow=target_flow.detach().float().cpu(),
            predicted_flow=predicted_flow.detach().float().cpu(),
            valid_mask=valid_mask.detach().cpu(),
        ))
        self._counts[direction] = self._counts.get(direction, 0) + 1

    def save(
        self, output: str | Path, *, flow_limit: float = 15.0,
        arrow_step: int = 32,
    ) -> Path:
        if not self.samples:
            raise RuntimeError("no registration validation samples were collected")
        if flow_limit <= 0 or arrow_step < 1:
            raise ValueError("flow_limit and arrow_step must be positive")
        rows: list[list[Image.Image]] = []
        for sample in sorted(self.samples, key=lambda item: (item.direction, item.sample_id)):
            target = sample.target_flow.numpy()
            prediction = sample.predicted_flow.numpy()
            mask = sample.valid_mask.squeeze().numpy() > 0.5
            error = np.linalg.norm(prediction - target, axis=0)
            valid_epe = float(error[mask].mean()) if np.any(mask) else float("nan")
            zero_epe = float(np.linalg.norm(target, axis=0)[mask].mean()) if np.any(mask) else float("nan")
            rows.append([
                _titled(_rgb(sample.moving), f"{sample.direction} | moving"),
                _titled(_rgb(sample.fixed), f"{sample.sample_id} | fixed"),
                _titled(_rgb(sample.warped), "warped moving (prediction)"),
                _titled(_flow_panel(target, flow_limit, arrow_step), "GT flow"),
                _titled(_flow_panel(prediction, flow_limit, arrow_step), "predicted flow"),
                _titled(
                    _scalar_color(error, flow_limit),
                    f"endpoint error EPE={valid_epe:.2f}px (zero={zero_epe:.2f})",
                ),
                _titled(
                    np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2),
                    "valid supervision mask",
                ),
            ])

        width, height = rows[0][0].size
        sheet = Image.new("RGB", (width * len(rows[0]), height * len(rows)), "white")
        for row_index, panels in enumerate(rows):
            for column_index, panel in enumerate(panels):
                sheet.paste(panel, (column_index * width, row_index * height))
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, optimize=True)
        return destination


def update_sample_alias(source: str | Path, alias: str | Path) -> Path:
    """Copy a rendered snapshot to a stable ``latest`` or ``best`` name."""
    source_path = Path(source).expanduser().resolve()
    alias_path = Path(alias).expanduser().resolve()
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, alias_path)
    return alias_path


def prune_sample_snapshots(directory: str | Path, keep: int) -> None:
    """Retain only the newest numbered epoch sheets; aliases are untouched."""
    if keep <= 0:
        return
    root = Path(directory).expanduser().resolve()
    snapshots = sorted(
        root.glob("epoch_*.png"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    for path in snapshots[:-keep]:
        path.unlink()
