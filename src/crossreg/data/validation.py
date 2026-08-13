"""Read-only dataset preflight checks, independent from training code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .translation import IMAGE_EXTENSIONS


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    task: str
    root: str
    statistics: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def error(self, code: str, message: str, path: Path | str | None = None) -> None:
        self.issues.append(ValidationIssue("error", code, message, str(path) if path else None))

    def warning(self, code: str, message: str, path: Path | str | None = None) -> None:
        self.issues.append(ValidationIssue("warning", code, message, str(path) if path else None))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.valid else "failed",
            "task": self.task,
            "root": self.root,
            "statistics": self.statistics,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _check_images(paths: list[Path], report: ValidationReport,
                  max_samples: int | None) -> None:
    shapes: dict[str, int] = {}
    selected = paths if max_samples is None else paths[:max_samples]
    for path in selected:
        try:
            with Image.open(path) as image:
                image.load()
                key = f"{image.mode}:{image.height}x{image.width}"
                shapes[key] = shapes.get(key, 0) + 1
        except Exception as exc:
            report.error("unreadable_image", str(exc), path)
    if shapes:
        report.statistics.setdefault("sampled_image_shapes", {}).update(shapes)


def validate_translation_dataset(
    root: str, pairing_mode: str = "unpaired",
    max_samples: int | None = 100,
) -> ValidationReport:
    """Validate two-domain ``trainA/trainB`` or ``A/B`` translation data."""
    report = ValidationReport("translation", root)
    base = Path(root)
    dir_a, dir_b = base / "trainA", base / "trainB"
    if not dir_a.is_dir() or not dir_b.is_dir():
        dir_a, dir_b = base / "A", base / "B"
    paths_a, paths_b = _image_paths(dir_a), _image_paths(dir_b)
    report.statistics.update({"domain_a": len(paths_a), "domain_b": len(paths_b)})
    if not paths_a:
        report.error("empty_domain", "domain A is missing or empty", dir_a)
    if not paths_b:
        report.error("empty_domain", "domain B is missing or empty", dir_b)
    _check_images(paths_a, report, max_samples)
    _check_images(paths_b, report, max_samples)
    if pairing_mode == "paired" and paths_a and paths_b:
        stems_a, stems_b = {p.stem for p in paths_a}, {p.stem for p in paths_b}
        common = stems_a & stems_b
        report.statistics["matched_stems"] = len(common)
        if stems_a != stems_b:
            report.error(
                "pair_mismatch",
                f"paired mode requires identical stems; A-only={len(stems_a-stems_b)}, "
                f"B-only={len(stems_b-stems_a)}",
            )
    elif pairing_mode != "unpaired":
        report.error("invalid_pairing_mode", f"unsupported pairing mode: {pairing_mode}")
    return report


def validate_multidomain_translation_dataset(
    root: str, pairing_mode: str = "unpaired",
    max_samples: int | None = 100,
) -> ValidationReport:
    """Validate a root whose immediate subdirectories are modality domains."""
    report = ValidationReport("multidomain_translation", root)
    base = Path(root)
    domains = sorted(path for path in base.iterdir() if path.is_dir()) if base.is_dir() else []
    if len(domains) < 2:
        report.error("insufficient_domains", "at least two modality directories are required", base)
        return report
    stem_sets = []
    counts = {}
    for domain in domains:
        paths = _image_paths(domain)
        counts[domain.name] = len(paths)
        if not paths:
            report.error("empty_domain", "modality directory contains no images", domain)
        _check_images(paths, report, max_samples)
        stem_sets.append({path.stem for path in paths})
    report.statistics["modalities"] = counts
    if pairing_mode == "paired" and stem_sets:
        common = set.intersection(*stem_sets)
        report.statistics["common_stems"] = len(common)
        if not common:
            report.error("no_common_stems", "paired mode requires common stems across all modalities")
        if any(stems != stem_sets[0] for stems in stem_sets[1:]):
            report.warning("partial_pairing", "modality stem sets are not identical; only their intersection is usable")
    elif pairing_mode != "unpaired":
        report.error("invalid_pairing_mode", f"unsupported pairing mode: {pairing_mode}")
    return report


def _registration_tasks(root: Path) -> list[Path]:
    if (root / "moving").is_dir() and (root / "fixed").is_dir():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "moving").is_dir() and (path / "fixed").is_dir()
    )


def validate_registration_dataset(
    root: str, require_flow: bool = False, require_mask: bool = False,
    max_samples: int | None = 100,
) -> ValidationReport:
    """Validate image pairing plus optional flow and mask arrays."""
    report = ValidationReport("registration", root)
    tasks = _registration_tasks(Path(root))
    if not tasks:
        report.error("invalid_layout", "no moving/fixed registration task found", root)
        return report
    total_pairs = 0
    for task in tasks:
        moving_paths = _image_paths(task / "moving")
        fixed_paths = _image_paths(task / "fixed")
        moving = {path.stem: path for path in moving_paths}
        fixed = {path.stem: path for path in fixed_paths}
        common = sorted(moving.keys() & fixed.keys())
        total_pairs += len(common)
        if moving.keys() != fixed.keys():
            report.error(
                "pair_mismatch",
                f"moving-only={len(moving.keys()-fixed.keys())}, fixed-only={len(fixed.keys()-moving.keys())}",
                task,
            )
        _check_images([moving[key] for key in common], report, max_samples)
        _check_images([fixed[key] for key in common], report, max_samples)
        selected = common if max_samples is None else common[:max_samples]
        for stem in selected:
            flow_path = task / "gt_flow" / f"{stem}.npy"
            if flow_path.is_file():
                try:
                    flow = np.load(flow_path, mmap_mode="r")
                    if flow.ndim != 3 or (flow.shape[0] != 2 and flow.shape[-1] != 2):
                        report.error("invalid_flow_shape", f"expected (2,H,W) or (H,W,2), got {flow.shape}", flow_path)
                    elif not np.isfinite(flow).all():
                        report.error("nonfinite_flow", "flow contains NaN or Inf", flow_path)
                except Exception as exc:
                    report.error("unreadable_flow", str(exc), flow_path)
            elif require_flow:
                report.error("missing_flow", "required flow is missing", flow_path)

            mask_candidates = list((task / "valid_mask").glob(f"{stem}.*"))
            if mask_candidates:
                try:
                    with Image.open(mask_candidates[0]) as mask:
                        mask.load()
                        if mask.mode not in {"1", "L", "I", "I;16"}:
                            report.warning("mask_mode", f"unusual mask mode: {mask.mode}", mask_candidates[0])
                except Exception as exc:
                    report.error("unreadable_mask", str(exc), mask_candidates[0])
            elif require_mask:
                report.error("missing_mask", "required valid mask is missing", task / "valid_mask" / stem)
    report.statistics.update({"tasks": len(tasks), "paired_images": total_pairs})
    if total_pairs == 0:
        report.error("empty_dataset", "no matching moving/fixed images found")
    return report
