"""Tests for the standalone, read-only dataset preflight module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _image(path: Path) -> None:
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(path)


def test_unpaired_translation_allows_different_stems(tmp_path: Path) -> None:
    from crossreg.data.validation import validate_translation_dataset

    (tmp_path / "trainA").mkdir()
    (tmp_path / "trainB").mkdir()
    _image(tmp_path / "trainA" / "source.png")
    _image(tmp_path / "trainB" / "target.png")
    assert validate_translation_dataset(str(tmp_path), "unpaired").valid
    assert not validate_translation_dataset(str(tmp_path), "paired").valid


def test_registration_validation_reports_bad_flow(tmp_path: Path) -> None:
    from crossreg.data.validation import validate_registration_dataset

    for directory in ("moving", "fixed", "gt_flow", "valid_mask"):
        (tmp_path / directory).mkdir()
    _image(tmp_path / "moving" / "a.png")
    _image(tmp_path / "fixed" / "a.png")
    _image(tmp_path / "valid_mask" / "a.png")
    np.save(tmp_path / "gt_flow" / "a.npy", np.zeros((3, 16, 16), dtype=np.float32))

    report = validate_registration_dataset(
        str(tmp_path), require_flow=True, require_mask=True,
    )
    assert not report.valid
    assert any(issue.code == "invalid_flow_shape" for issue in report.issues)
