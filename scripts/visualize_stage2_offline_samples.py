#!/usr/bin/env python3
"""Create one diagnostic sheet for offline Stage 2 samples."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--flow-limit", type=float, default=15.0)
    parser.add_argument("--arrow-step", type=int, default=32)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def color_scalar(value: np.ndarray, limit: float, diverging: bool) -> np.ndarray:
    if diverging:
        scaled = np.clip(value / limit, -1.0, 1.0)
        red = np.where(scaled >= 0, 255, 255 * (1 + scaled))
        blue = np.where(scaled <= 0, 255, 255 * (1 - scaled))
        green = 255 * (1 - np.abs(scaled))
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)
    scaled = np.clip(value / limit, 0.0, 1.0)
    red = np.clip(2 * scaled, 0, 1)
    blue = np.clip(2 * (1 - scaled), 0, 1)
    green = np.clip(1 - np.abs(2 * scaled - 1), 0, 1)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def flow_panel(flow: np.ndarray, limit: float, step: int) -> Image.Image:
    dy, dx = flow
    magnitude = np.sqrt(dy ** 2 + dx ** 2)
    image = Image.fromarray(color_scalar(magnitude, limit, False), mode="RGB")
    draw = ImageDraw.Draw(image)
    height, width = magnitude.shape
    scale = step / max(limit, 1e-6) * 0.7
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            end_x = x + float(dx[y, x]) * scale
            end_y = y + float(dy[y, x]) * scale
            draw.line((x, y, end_x, end_y), fill="white", width=2)
            angle = math.atan2(end_y - y, end_x - x)
            head = 4
            for offset in (2.5, -2.5):
                draw.line(
                    (end_x, end_y,
                     end_x - head * math.cos(angle + offset),
                     end_y - head * math.sin(angle + offset)),
                    fill="white", width=1,
                )
    return image


def titled(array: np.ndarray | Image.Image, title: str) -> Image.Image:
    image = array if isinstance(array, Image.Image) else Image.fromarray(array)
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image.convert("RGB"), (0, 34))
    ImageDraw.Draw(canvas).text((4, 5), title, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    with Path(args.manifest).expanduser().resolve().open(
        "r", encoding="utf-8", newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))[:args.max_samples]
    if not rows:
        raise RuntimeError("manifest contains no samples")

    panel_rows: list[list[Image.Image]] = []
    for row in rows:
        if not row.get("aligned_target_path"):
            raise ValueError(
                "manifest lacks aligned_target_path; rebuild preview with --save-diagnostics"
            )
        moving = load_rgb(root / row["moving_path"])
        aligned = load_rgb(root / row["aligned_target_path"])
        fixed = load_rgb(root / row["fixed_path"])
        flow = np.load(root / row["flow_path"], allow_pickle=False).astype(np.float32)
        with Image.open(root / row["valid_mask_path"]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        difference = np.clip(
            np.abs(aligned.astype(np.int16) - fixed.astype(np.int16)) * 3,
            0, 255,
        ).astype(np.uint8)
        magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
        direction = row["translation_direction"]
        panel_rows.append([
            titled(moving, f"{direction} | source/moving"),
            titled(aligned, "G(source), aligned"),
            titled(fixed, "warped target / fixed"),
            titled(difference, "3x |aligned-fixed|"),
            titled(
                flow_panel(flow, args.flow_limit, args.arrow_step),
                f"flow magnitude+vectors mean={magnitude.mean():.2f}px",
            ),
            titled(color_scalar(flow[0], args.flow_limit, True), "dy: blue(-) red(+)"),
            titled(color_scalar(flow[1], args.flow_limit, True), "dx: blue(-) red(+)"),
            titled(np.repeat(mask[..., None], 3, axis=2), "valid mask"),
        ])

    panel_width = panel_rows[0][0].width
    panel_height = panel_rows[0][0].height
    sheet = Image.new(
        "RGB", (panel_width * len(panel_rows[0]), panel_height * len(panel_rows)),
        "white",
    )
    for row_index, panels in enumerate(panel_rows):
        for column_index, panel in enumerate(panels):
            sheet.paste(panel, (column_index * panel_width, row_index * panel_height))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    print(f"Diagnostic sheet: {output}")


if __name__ == "__main__":
    main()
